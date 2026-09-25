"""314 fixed period-two execution-signature census, no candidate generation."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def cycles(a):
    a=np.asarray(a);equal=np.all(a[2:]==a[:-2],axis=1);segments=[];start=None
    for i in range(len(equal)+1):
        if i<len(equal) and equal[i]:
            if start is None:start=i
        elif start is not None:
            low,high=start,i+1  # input rows low..high inclusive
            if high-low+1>=4 and not np.array_equal(a[low],a[low+1]):
                segments.append(dict(start=low+1,end=high+1,length=high-low+1))
            start=None
    return segments


def independent_cycles(a):
    a=[tuple(row) for row in a];found=[]
    for i in range(len(a)-3):
        if a[i]==a[i+1]:continue
        j=i+2
        while j<len(a) and a[j]==a[i+(j-i)%2]:j+=1
        if j-i>=4:found.append((i+1,j))
    maximal=[(a,b) for a,b in found if not any(c<=a and b<=d and (a,b)!=(c,d) for c,d in found)]
    return [dict(start=a,end=b,length=b-a+1) for a,b in maximal]


def selftests():
    count=0
    # Exhaustive all binary sequences up to length ten, including overlap/constant cases.
    for n in range(1,11):
        for q in range(2**n):
            a=np.array([[(q>>j)&1] for j in range(n)])
            assert cycles(a)==independent_cycles(a);count+=1
    return count


def run(root,out):
    begin=time.perf_counter();test_count=selftests()
    folder=root/'results/solver_policy_recurrence/development_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
    census=root/'results/successful_transition_census/development_v2'
    cs=read(census/'summary.json');assert cs['passed']
    for n,digest in cs['outputs_sha256'].items():assert sha(census/n)==digest
    schema=read(folder/'policy_schema.json');families=['alm_keep','alm_reset','nodual']
    events=[e for e in read(census/'state_events.json') if e['family'] in families]
    save(out/'protocol.json',dict(source_sha256=sha(Path(__file__)),support_summary_sha256=sha(folder/'summary.json'),
        old_census_summary_sha256=sha(census/'summary.json'),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/314_two_policy_cycle_protocol.md'),
        period=2,minimum_steps=4,query_targets_accessed=False,new_candidates_generated=False,
        resources_matched=False,synthetic_exhaustive_sequence_checks=test_count))
    trajectories=[];event_records=[];checks=Counter()
    for row in read(folder/'rows.json'):
        seed,family=row['seed'],row['family']
        with np.load(folder/row['file'],allow_pickle=False) as z:
            arrays=dict(full=np.concatenate([z['policy_'+g] for g in schema],axis=2),forward=z['forward_policy'])
            for i,loc in enumerate(z['locations']):
                value=dict(seed=seed,family=family,location=loc.tolist())
                for kind,aa in arrays.items():
                    ss=cycles(aa[:,i]);assert ss==independent_cycles(aa[:,i]);checks['independent_full_trajectory_cycle_lists']+=1
                    covered=set(t for s in ss for t in range(s['start'],s['end']+1))
                    value[kind]=dict(segments=ss,covered_steps=len(covered))
                trajectories.append(value)
                matching=[e for e in events if e['seed']==seed and e['family']==family and e['location']==loc.tolist()]
                for e in matching:
                    entry=dict(**e)
                    for kind,aa in arrays.items():
                        for label,stop in [('strictly_before',e['first_step']-1),('through_arrival',e['first_step'])]:
                            ss=cycles(aa[:stop,i]);assert ss==independent_cycles(aa[:stop,i]);checks['independent_event_prefix_cycle_lists']+=1
                            entry[kind+'_'+label]=dict(segments=ss,longest=max((s['length'] for s in ss),default=0))
                    event_records.append(entry)
    assert len(trajectories)==393 and len(event_records)==24
    aggregate={}
    for f in families:
        rr=[r for r in trajectories if r['family']==f];ee=[e for e in event_records if e['family']==f]
        a={}
        for kind in ['full','forward']:
            lens=[s['length'] for r in rr for s in r[kind]['segments']]
            a[kind]=dict(trajectories_with_cycle=sum(bool(r[kind]['segments']) for r in rr),segments=len(lens),
                covered_steps=sum(r[kind]['covered_steps'] for r in rr),total_steps=8384,
                length_histogram=dict(sorted(Counter(lens).items())),maximum_length=max(lens,default=0))
            for label in ['strictly_before','through_arrival']:
                a[kind]['success_events_'+label]=dict(total=len(ee),
                    with_cycle=sum(e[kind+'_'+label]['longest']>=4 for e in ee),
                    longest_histogram=dict(sorted(Counter(e[kind+'_'+label]['longest'] for e in ee).items())))
        aggregate[f]=a
    files={}
    for n,value in [('trajectories.json',trajectories),('events.json',event_records),('aggregate.json',aggregate)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,trajectories=393,events=24,checks=dict(checks),aggregate=aggregate,
        query_targets_accessed=False,resources_matched=False,new_candidates_generated=False,
        independent_task_gain_established=False,seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final);print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
