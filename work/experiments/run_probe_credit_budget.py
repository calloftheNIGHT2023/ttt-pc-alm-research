"""285 selected-budget development predictions only; old69 immutable."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
import torch
import probe_credit_resource_suite as suite
import conditioned_mode_geometry as conditioned
from posterior_confirmation_pipeline import discovery_box
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    resources=root/'results/probe_credit_resources';gate=resources/'audit';cal=resources/'calibration';out=root/'results/probe_credit_budget/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS')==os.environ.get('OMP_NUM_THREADS')=='1';torch.set_num_threads(1);torch.set_num_interop_threads(1)
    audit=json.loads((gate/'summary.json').read_text());assert audit['passed'] and audit['may_run_selected_budget_development'];gp=json.loads((gate/'protocol.json').read_text());hashes=dict(gp['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    cs=json.loads((cal/'summary.json').read_text());assert cs['passed'] and sha(cal/'summary.json')==gp['calibration_summary_sha256']
    for n,h in cs['outputs_sha256'].items():assert sha(cal/n)==h,n
    selection=json.loads((cal/'selected_configs.json').read_text());previous=root/'results/probe_continuation_credit/evaluation';pp=json.loads((previous/'protocol.json').read_text());assert json.loads((previous.parent/'evaluation_audit/summary.json').read_text())['passed']
    configs=[c for c in selection['configs'] if c['name'] not in pp['methods']];assert len({c['name'] for c in configs})==len(configs)
    if not configs:raise RuntimeError('No new selected configuration; evaluate existing resource-quality frontier without duplicating predictors')
    loaded,manifest=suite.legacy.oldfit.meta.load(root);cp=json.loads((cal/'protocol.json').read_text());assert manifest==cp['checkpoint_manifest'];index=suite.frozen_inputs(root);seeds=list(range(5910000,5910064))
    p=dict(source_sha256=hashes,resource_audit_sha256=sha(gate/'summary.json'),selected_configs_sha256=sha(cal/'selected_configs.json'),design_sha256=cp['design_sha256'],checkpoint_manifest=manifest,
        configs=configs,all_selected_configs=selection['configs'],primary=suite.PRIMARY,seeds=seeds,previous_methods=pp['methods'],previous_evaluation_summary_sha256=sha(previous/'summary.json'),
        budget_seconds=selection['budget_seconds'],within_budget=selection['within_budget'],over_budget_background=selection['over_budget_background'],
        phase_accesses_query_targets=False,prior_query_results_exist=True,trace=True,scope='Selected only by full untraced resource calibration; all64 contexts are development, not blind confirmation')
    dump(out/'protocol.json',p);rows=[];files={};begin=time.perf_counter()
    def runner(cfg,x,v,q,seed,loaded):return suite.fit(cfg,x,v,q,seed,loaded,trace=True)
    with discovery_box(.12):
        for seed in seeds:
            directory,original=index[seed,suite.PRIMARY];assert sha(directory/original['file'])==original['sha256']
            with np.load(directory/original['file']) as z:x=z['x_observed'].copy();v=z['v_observed'].copy();q=z['q_observed'].copy()
            for cfg in configs:
                repairs=[];start=time.perf_counter()
                with conditioned.geometry_scope(repairs):a,m=suite.legacy.guarded_fit(cfg,x,v,q,seed,loaded,runner=runner)
                m.update(geometry_repair_log=repairs,geometry_repair_count=len(repairs),charged_complete_seconds=time.perf_counter()-start,
                    method_kind='head' if cfg['family']=='prior' or cfg['group']=='shallow' else 'mode_pool')
                if m['execution_failed']:m['positive_modes']=[];m['fallback']=True
                fn=f'{seed}_{cfg["name"]}.npz';np.savez_compressed(out/fn,x_observed=x,v_observed=v,q_observed=q,**a);files[fn]=sha(out/fn)
                rows.append(dict(seed=seed,method=cfg['name'],file=fn,sha256=files[fn],metadata=m))
            dump(out/'rows.json',rows)
            if (seed-seeds[0]+1)%4==0:print(json.dumps(dict(tasks_done=seed-seeds[0]+1,configs=len(configs),predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin)),flush=True)
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=dict(protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),prediction_files=files,phase_accesses_query_targets=False,prior_query_results_exist=True)
    dump(out/'before_evaluation_manifest.json',before)
    ans=dict(passed=True,tasks=64,configs=len(configs),predictions=len(rows),failures=sum(r['metadata']['execution_failed'] for r in rows),seconds=time.perf_counter()-begin,
        before_evaluation_manifest_sha256=sha(out/'before_evaluation_manifest.json'),phase_accesses_query_targets=False,next='Independent selected-horizon trajectories and geometry before quality; original69 retained')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
