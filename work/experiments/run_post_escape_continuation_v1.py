"""319 full five-control continuation, sealed before geometry or outcomes."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from post_escape_continuation_v1 import entries,rollout,forward,VARIANTS
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def verified(folder,manifest_name=None):
    summary=read(folder/'summary.json');assert summary['passed'] and not (folder/'failure.json').exists()
    if manifest_name:
        manifest=read(folder/manifest_name);assert sha(folder/manifest_name)==summary['manifest_sha256'];files=manifest['files_sha256']
    else:files=summary['outputs_sha256']
    for n,digest in files.items():assert sha(folder/n)==digest,(folder,n)
    return summary,read(folder/'protocol.json')


def run(root,out):
    begin=time.perf_counter();base=root/'results/solver_policy_recurrence/development_v1'
    line=root/'results/invariant_dual_drift/development_v1';esc=root/'results/primal_stasis_escape/development_v1'
    names={Path(__file__).name,'evaluate_complete_credit_mode_geometry_v1.py'};ancestors={}
    roots=[(base,None),(root/'results/affine_root_proposal/development_v1','before_aggregation_manifest.json'),
           (line,'before_guard_validation_manifest.json'),(esc,'before_geometry_manifest.json')]
    for folder,manifest in roots:
        summary,protocol=verified(folder,manifest);ancestors[str(folder.relative_to(root))]=sha(folder/'summary.json')
        for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest;names.add(n)
    tested=read(root/'results/post_escape_continuation/tests_v1/summary.json');assert tested['passed']
    for n,digest in tested['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest;names.add(n)
    docs=root/'outputs/ttt-pc-alm-research'
    protocol=dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},ancestor_summary_sha256=ancestors,
                  tests_sha256=sha(root/'results/post_escape_continuation/tests_v1/summary.json'),
                  design_sha256={n:sha(docs/n) for n in ['319_post_escape_continuation_protocol_v1.md','319_execution_details_v1.md']},
                  cases=393,variants=VARIANTS,trajectories=1965,expected_valid_updates=125760,
                  tasks=64,empty_tasks=read(base/'summary.json')['empty_tasks'],query_targets_accessed=False,
                  geometry_accessed=False,resources_matched=False,cached_prefix_is_not_free=True)
    save(out/'protocol.json',protocol)
    line_rows={(r['seed'],r['family'],*r['location']):r for r in read(line/'rows.json')}
    esc_rows={(r['seed'],r['family'],*r['location']):r for r in read(esc/'rows.json')}
    assert set(line_rows)==set(esc_rows) and len(line_rows)==393
    counts=Counter();rows=[];trajectories=[];ledger=[];files={};guarded=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP/optimizer called in local continuation')
    try:
        for obj,name in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guarded.append((obj,name,getattr(obj,name)));setattr(obj,name,forbid)
        with discovery_box(.12):
            for old in read(base/'rows.json'):
                with np.load(base/old['file'],allow_pickle=False) as z:
                    archived={k:z[k] for k in ['b','h','u','forward_policy','locations','x_observed','v_observed']}
                x,v=archived['x_observed'],archived['v_observed'];locations=archived['locations'];r=len(locations)
                starts={n:[] for n in VARIANTS};horizons={n:[] for n in VARIANTS};ks=[];applicable=[];records=[]
                for i,loc in enumerate(locations):
                    key=(old['seed'],old['family'],*loc.tolist());lr=line_rows[key];er=esc_rows[key]
                    l=read(line/lr['file']);e=read(esc/er['file']);ss,hh,k=entries(l,e)
                    original=np.r_[archived['b'][0,i],archived['h'][0,:,i].ravel(),archived['u'][0,:,i].ravel()]
                    assert np.array_equal(ss['original64'],original)
                    for name in VARIANTS:starts[name].append(ss[name]);horizons[name].append(hh[name])
                    ks.append(k);applicable.append(e['result']['applicable']);records.append((l,e))
                    ledger.append(dict(seed=old['seed'],family=old['family'],location=loc.tolist(),
                                       line_source=lr['file'],escape_source=er['file'],k=k,applicable=e['result']['applicable'],
                                       line_generation_seconds=l['seconds'],escape_seconds=e['incremental_exact_seconds'],
                                       float_exit_seconds=e.get('floating_exit_seconds',0.),
                                       prefix_scope='Historical diagnostic measurements, not matched online timing.'))
                qcodes,_=forward(np.array(starts['corrected64'])[:,:4],x)
                for name in VARIANTS:
                    arrays,meta=rollout(np.array(starts[name]),np.array(horizons[name]),x,v,old['method'])
                    if name=='original64':
                        for key in ['b','h','u']:
                            assert np.array_equal(arrays[key],archived[key]),(old['seed'],old['family'],key)
                            counts['original_individual_state_array_checks']+=65*r
                        assert np.array_equal(arrays['forward_codes'][1:],archived['forward_policy'])
                        counts['original_forward_state_checks']+=64*r
                    filename=f"{old['seed']}_{old['family']}_{name}.npz"
                    np.savez_compressed(out/filename,**arrays,x_observed=x,v_observed=v,locations=locations,
                                        k=np.array(ks),applicable=np.array(applicable,dtype=bool))
                    files[filename]=sha(out/filename)
                    rows.append(dict(seed=old['seed'],family=old['family'],method=old['method'],variant=name,
                                     file=filename,sha256=files[filename],states=r,meta=meta))
                    for i,loc in enumerate(locations):
                        horizon=int(arrays['horizons'][i]);codes=arrays['forward_codes'][:horizon+1,i]
                        first={}
                        for t,code in enumerate(codes):first.setdefault(code.tobytes().hex(),t)
                        prefix=qcodes[i].tobytes().hex() if name in ['escaped64','escaped_matched'] and ks[i] else None
                        modes=set(first)
                        if prefix is not None:modes.add(prefix)
                        trajectories.append(dict(seed=old['seed'],family=old['family'],variant=name,location=loc.tolist(),
                                                 file=filename,index=i,horizon=horizon,k=ks[i],
                                                 trajectory_modes=sorted(first),first_actual_readout_steps=first,
                                                 known_q_prefix_mode=prefix,proposal_modes=sorted(modes)))
                    counts['valid_local_updates']+=meta['valid_local_updates'];counts['trajectories']+=r
                print(dict(seed=old['seed'],family=old['family'],trajectories=counts['trajectories'],
                           local_updates=counts['valid_local_updates'],seconds=time.perf_counter()-begin),flush=True)
    finally:
        for obj,name,value in guarded:setattr(obj,name,value)
    assert counts['trajectories']==1965 and counts['valid_local_updates']==125760
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    for n,digest in protocol['design_sha256'].items():assert sha(docs/n)==digest
    for n,value in [('rows.json',rows),('trajectories.json',trajectories),('prefix_resource_ledger.json',ledger)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    save(out/'before_geometry_manifest.json',dict(files_sha256=files,all_trajectories_sealed=True,
                                                 query_targets_accessed=False,geometry_accessed=False))
    final=dict(passed=True,counts=dict(counts),tasks=64,empty_tasks=protocol['empty_tasks'],
               runtime_no_global_bp_guard_passed=True,query_targets_accessed=False,resources_matched=False,
               independent_task_gain_established=False,seconds=time.perf_counter()-begin,
               manifest_sha256=sha(out/'before_geometry_manifest.json'))
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
