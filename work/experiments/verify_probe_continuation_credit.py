"""283 primitive: golden probeALM, independent scalar controls and first-step identity."""
import argparse,json,os,time
from pathlib import Path
from collections import Counter
import numpy as np
import probe_continuation_credit as model
import probe_continuation_reference as independent
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from audit_confirmation_posterior_moments import piecewise_forward
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/probe_continuation_credit/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    parent=root/'results/round_282_audit.json';old=json.loads(parent.read_text());assert old['passed'];hashes=dict(old['source_sha256'])
    for n in ['probe_continuation_credit.py','probe_continuation_reference.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    design='283_probe_continuation_credit_protocol.md';assert sha(root/'outputs/ttt-pc-alm-research'/design)==old['report_sha256'][design]
    original=root/'results/certificate_activity_attribution/development';rr={(r['seed'],r['method']):r for r in json.loads((original/'rows.json').read_text())}
    p=dict(source_sha256=hashes,parent_audit_sha256=sha(parent),design_sha256=old['report_sha256'][design],seeds=[5910000,5910001,5910053,5910063],configs=model.CONFIGS,golden=model.GOLD,phase_accesses_query_targets=False)
    dump(out/'protocol.json',p);rows=[];counts=Counter();begin=time.perf_counter();maxforward=0.
    with discovery_box(.12):
        for seed in p['seeds']:
            row=rr[seed,model.PRIMARY];assert sha(original/row['file'])==row['sha256']
            with np.load(original/row['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy();gold={k:z[k].copy() for k in z.files if k not in ['x_observed','v_observed','q_observed']}
            identity={}
            for cfg in [model.GOLD]+model.CONFIGS:
                refs,cc=independent.reference(cfg,x,v);counts.update(cc);repairs=[]
                with conditioned.geometry_scope(repairs):a,meta=model.fit(cfg,x,v,q,seed,trace=True)
                _,cc=independent.verify(cfg,x,v,a,meta,refs);counts.update(cc)
                if cfg['solver'] in ['alm','nodual']:identity[cfg['solver']]={key:a['prefix_'+key][1].copy() for key in ['b','h','best']}
                if cfg['name']==model.PRIMARY:
                    for key,value in gold.items():assert a[key].tobytes()==value.tobytes(),(seed,key);counts['golden_arrays']+=1
                    assert meta['positive_modes']==row['metadata']['positive_modes'];counts['golden_predictors']+=1
                # Same trial pool and selected atomic state are shared by ALL controls.
                for key in ['initial_b','initial_h','initial_u','initial_best','atomic_trial_b','atomic_trial_origins']:
                    assert a[key].tobytes()==gold[key].tobytes();counts['shared_initial_arrays']+=1
                gap=float(np.max(abs(piecewise_forward(q,a['points']).mean(0)-a['prediction'])));assert gap<1e-12;maxforward=max(maxforward,gap)
                if not meta['fallback']:assert np.max(abs(piecewise_forward(x,a['points'])-v))<=.001+1e-7
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=sha(out/fn),metadata=meta,repairs=repairs));counts['predictors']+=1
            for key in ['b','h','best']:assert identity['alm'][key].tobytes()==identity['nodual'][key].tobytes();counts['first_primal_step_identities']+=1
            print(json.dumps(dict(tasks_done=seed,predictors=len(rows),seconds=time.perf_counter()-begin)),flush=True)
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,maximum_independent_forward_gap=maxforward,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
