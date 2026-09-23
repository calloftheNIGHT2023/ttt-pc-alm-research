"""283 predictions before quality: four controls after identical zero-dual trials."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import probe_continuation_credit as model
import conditioned_mode_geometry as conditioned
import matched_budget_suite as guard
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_continuation_credit';pre=base/'primitive';out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    ss=json.loads((pre/'summary.json').read_text());assert ss['passed'];p0=json.loads((pre/'protocol.json').read_text())
    assert sha(pre/'protocol.json')==ss['protocol_sha256'] and sha(pre/'rows.json')==ss['rows_sha256']
    hashes=dict(p0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/283_probe_continuation_credit_protocol.md';assert sha(design)==p0['design_sha256']
    original=root/'results/certificate_activity_attribution/development';manifest=json.loads((original/'before_evaluation_manifest.json').read_text());assert sha(original/'rows.json')==manifest['rows_sha256']
    originals={(r['seed'],r['method']):r for r in json.loads((original/'rows.json').read_text())};seeds=list(range(5910000,5910064))
    p=dict(source_sha256=hashes,design_sha256=sha(design),preflight_summary_sha256=sha(pre/'summary.json'),seeds=seeds,configs=model.CONFIGS,primary=model.PRIMARY,
        discovery_bound=.12,preparation_sweeps=16,restarts=33,particles=2048,repetition_seed=249911,query_points=257,
        phase_accesses_query_targets=False,prior_query_results_exist=True,scope='Development, shared probe pool, no matched-wall-time claim',
        failure_policy='failed attempt charged then fixed zero-bias fallback; no retries or task removal')
    dump(out/'protocol.json',p);rows=[];files={};begin=time.perf_counter()
    def runner(cfg,x,v,q,seed,loaded):return model.fit(cfg,x,v,q,seed,trace=True)
    with discovery_box(.12):
        for seed in seeds:
            old=originals[seed,model.PRIMARY];assert sha(original/old['file'])==old['sha256']==manifest['prediction_files'][old['file']]
            with np.load(original/old['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            for cfg in model.CONFIGS:
                repairs=[];start=time.perf_counter()
                with conditioned.geometry_scope(repairs):a,m=guard.guarded_fit(cfg,x,v,q,seed,None,runner=runner)
                m.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs),charged_complete_seconds=time.perf_counter()-start)
                if m['execution_failed']:m['positive_modes']=[];m['fallback']=True
                else:assert m['atomic_calls']==33 and m['scans']['scanned_states']==0 and m['trace_enabled']
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);files[fn]=sha(out/fn)
                rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=files[fn],metadata=m))
            dump(out/'rows.json',rows)
            if (seed-seeds[0]+1)%4==0:print(json.dumps(dict(tasks_done=seed-seeds[0]+1,predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin)),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),prediction_files=files,phase_accesses_query_targets=False,prior_query_results_exist=True)
    dump(out/'before_evaluation_manifest.json',before)
    ans=dict(passed=True,tasks=64,predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin,before_evaluation_manifest_sha256=sha(out/'before_evaluation_manifest.json'),phase_accesses_query_targets=False,next='All-task independent trajectories and geometry before69-method quality')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
