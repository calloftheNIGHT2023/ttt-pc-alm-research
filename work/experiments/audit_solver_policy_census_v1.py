"""313 all-trajectory scalar policy and state replay, independent statistics."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from solver_policy_trace_v1 import step
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def lengths(array):
    groups=[];last=None
    for row in array:
        now=tuple(row)
        if now!=last:groups.append(1)
        else:groups[-1]+=1
        last=now
    return groups


def run(root,out):
    begin=time.perf_counter();source=root/'results/solver_policy_recurrence/development_v1'
    analysis=root/'results/solver_policy_recurrence/analysis_v1';counts=Counter()
    for folder in [source,analysis]:
        summary=read(folder/'summary.json');assert summary['passed']
        for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
        for n,digest in read(folder/'protocol.json')['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    schema=read(source/'policy_schema.json');aggregate=read(analysis/'aggregate.json')
    lens={f:{k:[] for k in ['full','forward']} for f in aggregate}
    pertrajectory={f:{k:[] for k in ['full','forward']} for f in aggregate}
    guard=[]
    def forbid(*args,**kwargs):raise AssertionError('Global BP/global optimizer called during policy audit')
    try:
        for obj,n in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guard.append((obj,n,getattr(obj,n)));setattr(obj,n,forbid)
        with discovery_box(.12):
            for row in read(source/'rows.json'):
                family=row['family']
                with np.load(source/row['file'],allow_pickle=False) as z:
                    full=np.concatenate([z['policy_'+g] for g in schema],axis=2)
                    for i in range(len(z['locations'])):
                        for k,aa in [('full',full[:,i]),('forward',z['forward_policy'][:,i])]:
                            ll=lengths(aa);lens[family][k].extend(ll);pertrajectory[family][k].append(ll)
                        for t in range(64):
                            b=z['b'][t,i:i+1];h=z['h'][t,:,i:i+1];u=z['u'][t,:,i:i+1]
                            r=step(b,h,u,z['x_observed'],z['v_observed'],row['method'])
                            for g,a in r['policies'].items():
                                assert np.array_equal(a[0],z['policy_'+g][t,i]);counts['scalar_policy_group_checks']+=1
                            for k,a in [('b',r['b'][0]),('h',r['h'][:,0]),('u',r['u'][:,0])]:
                                expected=z[k][t+1,i] if k=='b' else z[k][t+1,:,i]
                                assert np.array_equal(a,expected);counts['scalar_output_array_checks']+=1
                            # Independent true forward coding; neither trace.forward_policy nor mode_list used.
                            val=z['x_observed'].copy();codes=[]
                            for bias in r['b'][0]:
                                zz=val+bias;codes.extend(int(q>=0)+int(q>=.5)+int(q>=1) for q in zz)
                                val=np.maximum(0,1-np.abs(2*zz-1))
                            assert np.array_equal(codes,z['forward_policy'][t,i]);counts['independent_forward_codes']+=1
                print(dict(seed=row['seed'],family=family,steps=counts['independent_forward_codes']),flush=True)
    finally:
        for obj,n,v in guard:setattr(obj,n,v)
    assert counts['independent_forward_codes']==25152
    for family,a in aggregate.items():
        for k in ['full','forward']:
            ll=lens[family][k];expected=a[k]
            assert dict(Counter(ll))=={int(n):q for n,q in expected['length_histogram'].items()}
            assert len(ll)==expected['segments'] and sum(ll)==8384==expected['total_steps']
            assert sum(q-1 for q in ll)==expected['redundant_steps_ideal_upper_bound']
            assert sum(q==1 for q in ll)==expected['singleton_segments']
            assert float(np.median(ll))==expected['median_segment_length'] and max(ll)==expected['maximum_segment_length']
            for q,values in expected['thresholds'].items():
                threshold=int(q)
                assert values['segments']==sum(t>=threshold for t in ll)
                assert values['steps_in_segments']==sum(t for t in ll if t>=threshold)
                assert values['trajectories_with_segment']==sum(max(t)>=threshold for t in pertrajectory[family][k])
                counts['threshold_numeric_checks']+=3
            counts['segment_summary_fields']+=7
    final=dict(passed=True,counts=dict(counts),query_targets_accessed=False,resources_matched=False,
        source_sha256=sha(Path(__file__)),input_summary_sha256={str(f.relative_to(root)):sha(f/'summary.json') for f in [source,analysis]},
        runtime_no_global_bp_guard_passed=True,independent_task_gain_established=False,seconds=time.perf_counter()-begin)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
