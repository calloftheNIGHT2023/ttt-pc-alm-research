"""324 freeze two context-only trigger states for every old development task."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
from support_consistency_trigger_v1 import choose_first_fit, exact_forward, support_loss, locations, uniform_index
from run_factorized_dual_branch_search_v1 import signals, bp_control, forward
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha

CHANNELS=['dual','dual_plus_residual','residual','bp','random_sign','zero']


def run(root,out):
    start=time.perf_counter();raw=root/'results/certificate_activity_attribution/development'
    test=read(root/'results/support_consistency_trigger/tests_v1/summary.json');assert test['passed']
    for name,digest in test['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    # Only source identities are used, not the row's geometry/outcome metadata.
    sources={r['seed']:{k:r[k] for k in ['seed','file','sha256']} for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    names=[Path(__file__).name,'support_consistency_trigger_v1.py','run_factorized_dual_branch_search_v1.py','batched_bp_discovery.py']
    protocol=dict(design_sha256=sha(root/'outputs/ttt-pc-alm-research/324_support_consistency_trigger_protocol_v1.md'),
                  source_sha256={n:sha(root/'work/experiments'/n) for n in names},
                  input_index_sha256=sha(raw/'rows.json'),tests_sha256=sha(root/'results/support_consistency_trigger/tests_v1/summary.json'),
                  policies=['first_fit','uniform_state'],seeds=list(range(5910000,5910064)),channels=CHANNELS,
                  allowed_arrays=['x_observed','v_observed','prefix_b','prefix_h','prefix_u','anchor_b','anchor_h','anchor_u'],
                  query_targets_accessed=False,geometry_accessed=False,cached_prefix_not_free=True,resources_matched=False)
    save(out/'protocol.json',protocol);files={};rows=[];counts=Counter();locs=locations()
    for seed in protocol['seeds']:
        src=raw/sources[seed]['file'];assert sha(src)==sources[seed]['sha256']
        tick=time.perf_counter()
        with np.load(src,allow_pickle=False) as z:a={k:z[k] for k in protocol['allowed_arrays']}
        assert a['prefix_b'].shape==(33,33,4) and a['anchor_b'].shape==(33,1,4)
        bs=np.array([a[('prefix' if phase==0 else 'anchor')+'_b'][t,r] for phase,t,r in locs])
        x,v=a['x_observed'],a['v_observed'];first=choose_first_fit(bs,x,v)
        first_seconds=time.perf_counter()-tick
        tick=time.perf_counter();idx=uniform_index(seed);random_seconds=time.perf_counter()-tick
        for policy,choice,selection_seconds in [('first_fit',first,first_seconds),('uniform_state',dict(index=idx),random_seconds)]:
            index=choice['index'];phase,t,r=locs[index];prefix='prefix' if phase==0 else 'anchor'
            b=a[prefix+'_b'][t,r];h=a[prefix+'_h'][t,:,r];u=a[prefix+'_u'][t,:,r]
            loss=support_loss(b,x,v);_,pattern=exact_forward(b,x);original=np.array(pattern,dtype=np.uint8)
            floatmode,_,_=forward(b,x);tick=time.perf_counter();credit=signals(b,h,u,x,seed,locs[index]);local_seconds=time.perf_counter()-tick
            tick=time.perf_counter();credit['bp'],gap=bp_control(b,x,v);bp_seconds=time.perf_counter()-tick
            note=dict(seed=seed,policy=policy,location=list(locs[index]),trajectory_index=index,
                      x_observed=x.tolist(),v_observed=v.tolist(),b=b.tolist(),h=h.tolist(),u=u.tolist(),
                      source_file=sources[seed]['file'],source_sha256=sources[seed]['sha256'],
                      original_mode=original.tobytes().hex(),float_mode=floatmode.tobytes().hex(),
                      exact_support_loss=str(loss),support_fit=loss==0,fallback=choice.get('fallback',False),
                      exact_selection_forward_calls=choice.get('exact_forward_calls',0),extra_diagnostic_forward_calls=1,
                      selection_seconds=selection_seconds,local_credit_seconds=local_seconds,bp_credit_seconds=bp_seconds,
                      bp_gradient_check_gap=gap,credits={n:credit[n].tolist() for n in CHANNELS},
                      credit_norms={n:float(np.linalg.norm(credit[n])) for n in CHANNELS},
                      retained_state_numeric_bytes=b.nbytes+h.nbytes+u.nbytes,full_archive_not_online_state=True)
            filename=f'{seed}_{policy}.json';save(out/filename,note);files[filename]=sha(out/filename)
            rows.append(dict(seed=seed,policy=policy,file=filename,sha256=files[filename]))
            counts[policy+'_states']+=1;counts[policy+'_support_fit']+=int(loss==0)
            counts[policy+'_fallback']+=int(note['fallback']);counts[policy+'_nonzero_dual']+=int(np.any(u))
            counts[policy+'_exact_float_mode_disagreements']+=int(note['original_mode']!=note['float_mode'])
            counts[policy+'_exact_selection_forward_calls']+=note['exact_selection_forward_calls']
        if (seed-5910000+1)%16==0:print(dict(tasks=seed-5910000+1,seconds=time.perf_counter()-start),flush=True)
    for name,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/name)==digest
    save(out/'rows.json',rows);files['rows.json']=sha(out/'rows.json');files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_search_manifest.json',dict(files_sha256=files,all_states_sealed=True,geometry_accessed=False,query_targets_accessed=False))
    result=dict(passed=True,counts=dict(counts),tasks=64,states=128,seconds=time.perf_counter()-start,
                manifest_sha256=sha(out/'before_search_manifest.json'),query_targets_accessed=False,geometry_accessed=False,
                resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
