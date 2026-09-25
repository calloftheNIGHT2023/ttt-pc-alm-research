"""Read-only diagnosis of recorded failures; no rescue or altered benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import independent_hybrid_memory as memory
from run_independent_hybrid_memory import observations


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';out=root/'results/independent_hybrid/failure_audit'
    p=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    failures=json.loads((inp/'failures.json').read_text()) if (inp/'failures.json').exists() else [];records=[]
    for failure in failures:
        seed=failure['seed'];name=failure['method'];rep=failure['repetition'];cfg=next(c for c in p['configs'] if c['name']==name)
        xx,vv=observations(seed);state=None;previous_n=0;artifacts=[]
        for n in p['stages']:
            path=inp/f'state_{seed}_{name}_{rep}_{n}.npz'
            if not path.exists():break
            with np.load(path) as z:
                state=memory.State(z['anchor'].copy(),z['points'].copy())
                assert np.array_equal(z['x'],xx[:n]) and np.array_equal(z['v'],vv[:n])
                assert np.max(abs(memory.capture.model.light.forward_many(xx[:n],state.samples)[1][:,-1]-vv[:n]))<=.001+1e-8
                assert np.max(abs(memory.posterior.make_predict(state.samples)(z['q'])-z['prediction']))<1e-12
            artifacts.append(dict(n=n,file=path.name,sha256=sha(path)));previous_n=n
        failed_n=n;assert state is not None and failed_n>previous_n
        data,meta=memory.prepare(xx[:failed_n],vv[:failed_n],state,cfg)
        bank,discovery=memory.neighbor.discover(xx[:failed_n],vv[:failed_n],state,cfg)
        raw=memory.neighbor.previous.interface.archived.signatures(xx[:failed_n],bank)
        keys=sorted({reg.tobytes() for reg in raw});statuses=[];witnesses=[]
        for key in keys:
            reg=np.frombuffer(key,np.uint8).reshape(4,failed_n);_,_,a,r=memory.neighbor.pattern_matrix(xx[:failed_n],vv[:failed_n],reg)
            lp=linprog(np.zeros(4),A_ub=a,b_ub=r,bounds=[(-.12,.12)]*4,options={'primal_feasibility_tolerance':1e-9})
            statuses.append(int(lp.status))
            if lp.success:witnesses.append(dict(pattern=key.hex(),point=lp.x.tolist(),max_violation=float(np.max(a@lp.x-r))))
        # Evaluation-only known valid witness, reconstructed after all runs.
        teacher=np.random.default_rng(seed).uniform(-.12,.12,4)
        true_key=memory.base.pattern(xx[:failed_n],teacher).astype(np.uint8).tobytes()
        first_true=memory.base.pattern(xx[:4],teacher).astype(np.uint8).tobytes().hex()
        truth_error=float(np.max(abs(memory.base.forward(xx[:failed_n],teacher)-vv[:failed_n])));assert truth_error<=.001
        detail=json.loads((inp/f'detail_{seed}_{name}.json').read_text())
        if cfg['sampler']=='geometry':first_keys=detail['positive_mode_keys']
        else:first_keys=detail['surviving_pattern_keys']
        records.append(dict(seed=seed,method=name,repetition=rep,last_completed_n=previous_n,failed_n=failed_n,
            partial_artifacts=artifacts,raw_candidate_cells=len(keys),raw_lp_statuses=statuses,
            lp_feasible_witnesses=witnesses,after_c20_pool=meta['pool'],sampler_polys=len(data['polys']),
            all_raw_cells_lp_infeasible=bool(statuses and all(s==2 for s in statuses)),
            evaluation_teacher_support_error=truth_error,teacher_cell_present_at_failure=true_key in keys,
            teacher_first_cell_in_recorded_first_union=first_true in first_keys,
            scope='Teacher is evaluator-only; its missing cell is one witness, not a completeness proof over all undiscovered cells'))
        print(json.dumps(dict(seed=seed,method=name,rep=rep,failed_n=failed_n,raw_cells=len(keys),lp_status_counts={str(s):statuses.count(s) for s in set(statuses)})),flush=True)
    result=dict(passed=True,failures=len(records),partial_state_replays=sum(len(r['partial_artifacts']) for r in records),
        records=records,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='Failure diagnosis only, no changed parameters, new trajectories, invented failure loss, or retroactive rescue')
    out.mkdir(parents=True,exist_ok=True);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,failures=len(records),output=str(out/'summary.json'))),flush=True)


if __name__=='__main__':main()
