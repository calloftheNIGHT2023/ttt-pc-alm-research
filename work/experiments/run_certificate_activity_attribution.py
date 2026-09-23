"""281 development predictions only; fixed origin horizon and fixed1088 updates."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import certificate_activity_attribution as fork
import conditioned_mode_geometry as conditioned
import matched_budget_suite as guard
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/certificate_activity_attribution';pre=base/'primitive';out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1'
    aa=json.loads((pre/'summary.json').read_text());assert aa['passed'];pp=json.loads((pre/'protocol.json').read_text())
    assert sha(pre/'protocol.json')==aa['protocol_sha256'] and sha(pre/'rows.json')==aa['rows_sha256']
    hashes=dict(pp['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    design=root/'outputs/ttt-pc-alm-research/281_certificate_activity_attribution_protocol.md';assert sha(design)==pp['design_sha256']
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';before=json.loads((old/'before_query_manifest.json').read_text());assert sha(old/'rows.json')==before['rows_sha256']
    originals={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())};seeds=list(range(5910000,5910064))
    p=dict(source_sha256=hashes,design_sha256=sha(design),preflight_summary_sha256=sha(pre/'summary.json'),seeds=seeds,configs=fork.CONFIGS,
        primary=fork.PRIMARY,discovery_bound=.12,preparation_sweeps=16,anchor_sweeps=64,other_sweeps=32,restarts=33,total_restart_sweeps=1088,particles=2048,
        repetition_seed=249911,query_points=257,phase_accesses_query_targets=False,prior_query_results_exist=True,
        scope='Development only: these64 contexts were previously evaluated; trace-enabled timings are not deployment resource comparisons',
        failure_policy='same frozen numerical guard: failed attempt charged, fixed zero-bias fallback retained; no retries or task removal')
    dump(out/'protocol.json',p);rows=[];files={};begin=time.perf_counter()
    def runner(cfg,x,v,q,seed,loaded):return fork.fit(cfg,x,v,q,seed,trace=True)
    with discovery_box(.12):
        for seed in seeds:
            original=originals[seed,'minimum_dual_alm64'];assert sha(old/original['file'])==original['sha256']==before['prediction_files'][original['file']]
            with np.load(old/original['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            for cfg in fork.CONFIGS:
                repairs=[];start=time.perf_counter()
                with conditioned.geometry_scope(repairs):arrays,meta=guard.guarded_fit(cfg,x,v,q,seed,None,runner=runner)
                meta.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs),charged_complete_seconds=time.perf_counter()-start)
                if meta['execution_failed']:meta['positive_modes']=[];meta['fallback']=True
                else:assert meta['atomic_calls']==33 and meta['total_restart_sweeps']==1088 and meta['trace_enabled']
                filename=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/filename,x_observed=x,v_observed=v,q_observed=q,**arrays);files[filename]=sha(out/filename)
                rows.append(dict(seed=seed,method=cfg['name'],file=filename,sha256=files[filename],metadata=meta))
            dump(out/'rows.json',rows)
            print(json.dumps(dict(tasks_done=(seed-seeds[0]+1),total=64,predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin)),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    manifest=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),prediction_files=files,
        phase_accesses_query_targets=False,prior_query_results_exist=True,scope='New predictions frozen before their evaluation; old contexts, not a blind confirmation')
    dump(out/'before_evaluation_manifest.json',manifest)
    ans=dict(passed=True,tasks=64,predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin,
        before_evaluation_manifest_sha256=sha(out/'before_evaluation_manifest.json'),phase_accesses_query_targets=False,
        next='Independent all64 protected-anchor/prefix, geometry and draw audit, then development evaluation against all59 old comparators; no new blind claim')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
