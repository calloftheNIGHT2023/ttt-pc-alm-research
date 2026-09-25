"""319 all-step independent tracer replay, entries, pools and horizon pairs."""
import argparse
from collections import Counter,defaultdict
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def independent_forward(b,x):
    prev=np.broadcast_to(x,b.shape[:-1]+(len(x),)).copy();codes=[]
    for j in range(b.shape[-1]):
        z=prev+b[...,j,None]
        codes.append((z>=0).astype(np.uint8)+(z>=.5).astype(np.uint8)+(z>=1).astype(np.uint8))
        prev=np.maximum(0,1-np.abs(2*z-1))
    return np.concatenate(codes,axis=-1),prev


def run(root,out):
    start=time.perf_counter();source=root/'results/post_escape_continuation/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    manifest=read(source/'before_geometry_manifest.json');assert sha(source/'before_geometry_manifest.json')==summary['manifest_sha256']
    for n,digest in manifest['files_sha256'].items():assert sha(source/n)==digest
    for n,digest in read(source/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    lines=root/'results/invariant_dual_drift/development_v1';esc=root/'results/primal_stasis_escape/development_v1'
    line_rows={(r['seed'],r['family'],*r['location']):r for r in read(lines/'rows.json')}
    esc_rows={(r['seed'],r['family'],*r['location']):r for r in read(esc/'rows.json')}
    trajectories={(r['file'],r['index']):r for r in read(source/'trajectories.json')}
    save(out/'protocol.json',dict(input_summary_sha256=sha(source/'summary.json'),
                                 source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'solver_policy_trace_v1.py']},
                                 query_targets_accessed=False,geometry_accessed=False,resources_matched=False))
    checks=Counter();groups=defaultdict(list)
    for row in read(source/'rows.json'):groups[row['seed'],row['family']].append(row)
    comparisons=[];stats={}
    with discovery_box(.12):
        for (seed,family),group in groups.items():
            loaded={}
            for row in group:
                with np.load(source/row['file'],allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
                loaded[row['variant']]=arrays;x,v=arrays['x_observed'],arrays['v_observed'];horizon=arrays['horizons'];r=len(horizon)
                codes,values=independent_forward(arrays['b'],x)
                assert np.array_equal(codes,arrays['forward_codes']) and np.array_equal(values,arrays['forward_values'])
                for t in range(1,len(arrays['b'])):
                    ids=np.flatnonzero(horizon>=t)
                    result=step(arrays['b'][t-1,ids],arrays['h'][t-1][:,ids],arrays['u'][t-1][:,ids],x,v,row['method'])
                    assert np.array_equal(result['b'],arrays['b'][t,ids])
                    assert np.array_equal(result['h'],arrays['h'][t][:,ids])
                    assert np.array_equal(result['u'],arrays['u'][t][:,ids])
                    checks['independent_update_arrays']+=3*len(ids)
                for i,loc in enumerate(arrays['locations']):
                    key=(seed,family,*loc.tolist());lr=read(lines/line_rows[key]['file']);er=read(esc/esc_rows[key]['file'])
                    original=np.array([float(F(z)) for z in lr['point']]);app=er['result']['applicable']
                    corrected=np.array([float(F(z)) for z in lr['proposal']['q']]) if app else original
                    found=er['result']['status']=='first_primal_exit_found';k=er['result']['first_exit_input_phase']+1 if found else 0
                    exited=np.array(er['floating_exit_output']) if found else corrected
                    variant=row['variant'];expected=original if variant=='original64' else exited if variant.startswith('escaped') else corrected
                    expected_horizon=64+k if variant=='corrected_virtual' else 64-k if variant=='escaped_matched' else 64
                    assert arrays['k'][i]==k and arrays['applicable'][i]==app and horizon[i]==expected_horizon
                    initial=np.r_[arrays['b'][0,i],arrays['h'][0,:,i].ravel(),arrays['u'][0,:,i].ravel()]
                    assert np.array_equal(initial,expected) and np.array_equal(arrays['initial_points'][i],expected)
                    checks['independent_entry_states']+=1
                    valid=codes[:horizon[i]+1,i];first={}
                    for t,code in enumerate(valid):first.setdefault(code.tobytes().hex(),t)
                    proposal=set(first);prefix=None
                    if variant.startswith('escaped') and k:
                        qcode,_=independent_forward(corrected[None,:4],x);prefix=qcode[0].tobytes().hex();proposal.add(prefix)
                    described=trajectories[row['file'],i]
                    assert described['trajectory_modes']==sorted(first) and described['proposal_modes']==sorted(proposal)
                    assert described['first_actual_readout_steps']==first and described['known_q_prefix_mode']==prefix
                    checks['individual_mode_pools']+=1;checks['independent_forward_states']+=horizon[i]+1
                    for t in range(horizon[i]+1,len(codes)):
                        assert np.array_equal(arrays['b'][t,i],arrays['b'][horizon[i],i])
                        assert np.array_equal(arrays['h'][t,:,i],arrays['h'][horizon[i],:,i])
                        assert np.array_equal(arrays['u'][t,:,i],arrays['u'][horizon[i],:,i]);checks['padded_state_arrays']+=3
                assert row['meta']['valid_local_updates']==int(horizon.sum())
                assert row['meta']['batched_local_calls']==sum(set(map(int,horizon)))
            for left,right in [('escaped64','corrected_virtual'),('escaped_matched','corrected64')]:
                ll,rr=loaded[left],loaded[right];name=family+'_'+left+'_vs_'+right
                ss=stats.setdefault(name,dict(cases=0,with_jump=0,exactly_equal_full_trajectories=0,
                                              different_forward_state_count=0,different_final_modes=0,
                                              max_gaps=[0.,0.,0.],different_shifted_mode_pools=0))
                for i,loc in enumerate(ll['locations']):
                    k=int(ll['k'][i]);length=int(ll['horizons'][i])+1;assert int(rr['horizons'][i])==k+length-1
                    gaps=[];same=True
                    for field in ['b','h','u']:
                        aa=ll[field][:length,i] if field=='b' else ll[field][:length,:,i]
                        bb=rr[field][k:k+length,i] if field=='b' else rr[field][k:k+length,:,i]
                        same=bool(same and np.array_equal(aa,bb));gaps.append(float(np.max(abs(aa-bb))))
                    lm=ll['forward_codes'][:length,i];rm=rr['forward_codes'][k:k+length,i]
                    different=np.any(lm!=rm,axis=1);pool_left={q.tobytes().hex() for q in lm};pool_right={q.tobytes().hex() for q in rm}
                    if not k:assert same and not np.any(different)
                    comparison=dict(seed=seed,family=family,location=loc.tolist(),left=left,right=right,k=k,
                                    compared_states=length,bitwise_full_trajectory_equal=same,max_b_h_u_gaps=gaps,
                                    different_forward_states=int(different.sum()),final_mode_different=bool(different[-1]),
                                    left_only_shifted_modes=sorted(pool_left-pool_right),right_only_shifted_modes=sorted(pool_right-pool_left))
                    comparisons.append(comparison);ss['cases']+=1;ss['with_jump']+=k>0
                    ss['exactly_equal_full_trajectories']+=same;ss['different_forward_state_count']+=int(different.sum())
                    ss['different_final_modes']+=bool(different[-1]);ss['different_shifted_mode_pools']+=pool_left!=pool_right
                    ss['max_gaps']=[max(a,b) for a,b in zip(ss['max_gaps'],gaps)]
            print(dict(seed=seed,family=family,updates=checks['independent_update_arrays']//3),flush=True)
    assert checks['independent_update_arrays']==3*125760 and checks['independent_forward_states']==127725
    assert checks['independent_entry_states']==checks['individual_mode_pools']==1965 and len(comparisons)==786
    save(out/'matched_horizon_pairs.json',comparisons);save(out/'matched_horizon_aggregate.json',stats)
    final=dict(passed=True,checks=dict(checks),seconds=time.perf_counter()-start,query_targets_accessed=False,
               resources_matched=False,independent_task_gain_established=False,
               outputs_sha256={n:sha(out/n) for n in ['protocol.json','matched_horizon_pairs.json','matched_horizon_aggregate.json']})
    save(out/'summary.json',final);print(final,flush=True);print(stats,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
