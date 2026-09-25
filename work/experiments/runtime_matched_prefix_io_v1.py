"""405 same scientific registry, corrected setup, explicit reused preflights."""
import numpy as np
import prefix_deadline_pilot_io_v1 as previous
import run_prefix_deadline_bound_v1 as binding

BASE='results/runtime_matched_prefix'
DESIGN='outputs/ttt-pc-alm-research/405_runtime_matched_prefix_protocol_v1.md'
PRIMARY='regional_active512_dfs_farthest_x'
PRIMARY_BUDGET=.5
SEEDS=list(previous.SEEDS)
STAGES=list(previous.STAGES)
BUDGETS=list(previous.BUDGETS)
old=previous.old
read,save,sha,load_arrays,observations=previous.read,previous.save,previous.sha,previous.load_arrays,previous.observations
configs=previous.configs


def dependencies(root):
    value=binding.dependencies(root)
    proof={}
    for folder_name in ['results/prefix_deadline_pilot/preflight_v1',
                        'results/prefix_deadline_pilot/audit_preflight_v1',
                        'results/portfolio_runtime_setup/preflight_v1']:
        folder=root/folder_name;summary=read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists()
        for name,digest in summary['outputs_sha256'].items():assert sha(folder/name)==digest
        if isinstance(summary.get('source_sha256'),dict):old.verify_hashes(root,summary['source_sha256'])
        proof[folder_name+'/summary.json']=sha(folder/'summary.json')
    source=root/'results/prefix_deadline_pilot/preflight_v1'
    checked=read(root/'results/prefix_deadline_pilot/audit_preflight_v1/summary.json')
    assert checked['prediction_summary_sha256']==sha(source/'summary.json')
    old.verify_hashes(root,read(source/'protocol.json')['source_sha256'])
    repaired=read(root/'results/portfolio_runtime_setup/preflight_v1/summary.json')
    assert repaired['calls']==32 and repaired['saved_arrays_replayed']==1120 and repaired['event_predictions_replayed']==160
    for path in ['results/prefix_deadline_pilot/preflight_v1/rows.json',
                 'results/portfolio_runtime_setup/preflight_v1/rows.json']:
        for row in read(root/path):
            for name,digest in row['files'].items():assert sha(root/row['directory']/name)==digest
    value['source_sha256'][DESIGN]=sha(root/DESIGN)
    value.pop('scientific_code_unchanged_from_passed_prefix_preflight',None)
    value.update(reused_preflight_summary_sha256=proof,new_wide_preflight_calls=0,
                 portfolio_task_free_module_preload=True,original_calculate_unchanged=True)
    return value


def jobs(configurations,seeds,stages,budgets):
    for ci in np.random.default_rng(405101).permutation(len(configurations)):
        ci=int(ci);cfg=configurations[ci]
        all_jobs=[(seed,n,budget) for seed in seeds for n in stages for budget in budgets]
        order=np.random.default_rng(np.random.SeedSequence([405103,ci])).permutation(len(all_jobs))
        yield ci,cfg,[all_jobs[int(j)] for j in order]
