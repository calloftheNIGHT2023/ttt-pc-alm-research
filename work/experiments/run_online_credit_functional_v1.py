"""326 full live calls for all 64 old tasks and twelve controls, no targets."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import online_credit_branch_search_v1 as candidate
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    started=time.perf_counter();raw=root/'results/certificate_activity_attribution/development'
    index=read(raw/'before_evaluation_manifest.json');assert sha(raw/'rows.json')==index['rows_sha256']
    originals={r['seed']:r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    tested=read(root/'results/online_credit_branch_search/tests_v1/summary.json');assert tested['passed']
    for n,d in tested['source_sha256'].items():assert sha(root/'work/experiments'/n)==d
    deps=read(root/'results/round_287_audit_v6.json')['source_sha256']
    for n,d in deps.items():assert sha(root/'work/experiments'/n)==d
    names=[Path(__file__).name,'online_credit_branch_search_v1.py','support_consistency_trigger_v1.py',
           'branch_image_chain_v1.py','factorized_dual_branch_search_v1.py','run_factorized_dual_branch_search_v1.py',
           'complete_credit_mode_geometry_v1.py','conditioned_mode_geometry.py','shared_mode_readout.py']
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in names},
        frozen_original_dependency_sha256=deps,design_sha256={n:sha(root/'outputs/ttt-pc-alm-research'/n) for n in ['326_online_credit_search_design_v1.md','326_functional_execution_protocol_v1.md']},
        tests_sha256=sha(root/'results/online_credit_branch_search/tests_v1/summary.json'),
        states324_sha256=sha(root/'results/support_consistency_trigger/states_v1/summary.json'),
        proposals324_sha256=sha(root/'results/support_consistency_trigger/development_v1/summary.json'),
        geometry324_sha256=sha(root/'results/support_consistency_trigger/geometry_v1/summary.json'),
        seeds=list(range(5910000,5910064)),policies=candidate.POLICIES,channels=candidate.CHANNELS,
        trace_functional_only=True,query_targets_accessed=False,posterior_reference_accessed=False,resources_matched=False)
    save(out/'protocol.json',protocol);rows=[];files={};counts=Counter()
    with discovery_box(.12):
        for seed in protocol['seeds']:
            original=originals[seed];path=raw/original['file'];assert sha(path)==original['sha256']
            with np.load(path,allow_pickle=False) as z:
                x,v=z['x_observed'],z['v_observed']
                preserved={n:z[n] for n in z.files if n.startswith(('prefix_','anchor_','initial_','effective_','atomic_')) or n in ['selected_b','best_bank','point_prediction']}
            q=np.linspace(0,1,257)
            for policy in protocol['policies']:
                state=read(root/'results/support_consistency_trigger/states_v1'/f'{seed}_{policy}.json')
                expected=read(root/'results/support_consistency_trigger/development_v1'/f'{seed}_{policy}.json')
                geo=read(root/'results/support_consistency_trigger/geometry_v1'/f'{seed}_geometry.json')
                for channel in protocol['channels']:
                    arrays,meta=candidate.fit(x,v,q,seed,policy=policy,channel=channel,trace=True)
                    for n,value in preserved.items():assert arrays[n].tobytes()==value.tobytes(),(seed,policy,channel,n);counts['preserved_arrays']+=1
                    for field in ['b','h','u']:assert arrays['trigger_'+field].tobytes()==np.array(state[field]).tobytes();counts['selected_state_arrays']+=1
                    assert arrays['trigger_credit'].tobytes()==np.array(state['credits'][channel]).tobytes();counts['credit_arrays']+=1
                    for field in ['location','original_mode','exact_support_loss','support_fit']:assert meta['selected_state'][field]==state[field];counts['trigger_fields']+=1
                    assert meta['exact_trigger_forward_calls']==state['exact_selection_forward_calls']
                    assert meta['proposal']['proposals']==expected['results'][channel]['proposals'];counts['proposal_records']+=len(meta['proposal']['proposals'])
                    assert meta['original_positive_modes']==original['metadata']['positive_modes']
                    assert meta['original_visited_modes']==original['metadata']['visited_modes']
                    assert set(meta['positive_modes'])==set(original['metadata']['positive_modes'])|set(geo['methods'][policy+'/'+channel]['positive_modes']);counts['positive_pools']+=1
                    assert meta['cached_trigger_numeric_bytes']==288
                    assert np.isfinite(arrays['prediction']).all() and np.isfinite(arrays['points']).all()
                    stem=f'{seed}_{policy}_{channel}';filename=stem+'.npz';metafile=stem+'.json'
                    with (out/filename).open('xb') as stream:np.savez_compressed(stream,**arrays)
                    save(out/metafile,meta);files[filename]=sha(out/filename);files[metafile]=sha(out/metafile)
                    rows.append(dict(seed=seed,policy=policy,channel=channel,file=filename,sha256=files[filename],metadata_file=metafile,
                                     metadata_sha256=files[metafile],source_npz_sha256=original['sha256'],functional_seconds=meta['total_seconds']))
                    counts['complete_fit_calls']+=1;counts['no_bp_guard_calls']+=int(meta['no_global_bp_guard_enabled'])
            if (seed-5910000+1)%2==0:print(dict(tasks=seed-5910000+1,calls=counts['complete_fit_calls'],seconds=time.perf_counter()-started),flush=True)
    assert counts['complete_fit_calls']==768 and counts['no_bp_guard_calls']==640
    for n,d in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==d
    for n,d in deps.items():assert sha(root/'work/experiments'/n)==d
    save(out/'rows.json',rows);files['rows.json']=sha(out/'rows.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_evaluation_manifest.json',dict(files_sha256=files,predictions_sealed=True,query_targets_accessed=False,posterior_reference_accessed=False))
    result=dict(passed=True,counts=dict(counts),tasks=64,methods=12,seconds=time.perf_counter()-started,
                manifest_sha256=sha(out/'before_evaluation_manifest.json'),query_targets_accessed=False,posterior_reference_accessed=False,
                trace_functional_only=True,resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
