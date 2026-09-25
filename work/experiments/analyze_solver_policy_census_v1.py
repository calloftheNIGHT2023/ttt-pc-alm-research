"""313 post-seal segment statistics and fixed old success-event linkage."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha

FAMILIES=['alm_keep','alm_reset','nodual']
THRESHOLDS=[1,2,4,8,16,32,64]


def spans(a):
    result=[];start=0
    for i in range(1,len(a)+1):
        if i==len(a) or not np.array_equal(a[i],a[i-1]):
            result.append(dict(start=start+1,end=i,length=i-start));start=i
    return result


def summarize(rr,key):
    lengths=[s['length'] for r in rr for s in r[key]];hist=Counter(lengths)
    return dict(trajectories=len(rr),segments=len(lengths),length_histogram=dict(sorted(hist.items())),
        singleton_segments=hist[1],singleton_fraction=hist[1]/len(lengths),
        median_segment_length=float(np.median(lengths)),maximum_segment_length=max(lengths),
        redundant_steps_ideal_upper_bound=sum(t-1 for t in lengths),
        total_steps=sum(lengths),thresholds={str(q):dict(
            trajectories_with_segment=sum(any(s['length']>=q for s in r[key]) for r in rr),
            segments=sum(t>=q for t in lengths),steps_in_segments=sum(t for t in lengths if t>=q)) for q in THRESHOLDS})


def run(root,out):
    begin=time.perf_counter();folder=root/'results/solver_policy_recurrence/development_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    assert sha(folder/'before_event_analysis_manifest.json')==summary['manifest_sha256']
    for n,digest in summary['outputs_sha256'].items():assert sha(folder/n)==digest
    protocol=read(folder/'protocol.json')
    for n,digest in protocol['source_sha256'].items():assert sha(root/'work/experiments'/n)==digest
    census=root/'results/successful_transition_census/development_v2'
    cs=read(census/'summary.json');assert cs['passed']
    for n,digest in cs['outputs_sha256'].items():assert sha(census/n)==digest
    save(out/'protocol.json',dict(source_sha256={Path(__file__).name:sha(Path(__file__))},
        support_summary_sha256=sha(folder/'summary.json'),old_census_summary_sha256=sha(census/'summary.json'),
        query_targets_accessed=False,resources_matched=False,thresholds=THRESHOLDS,
        signature_scope='Conservative execution partition, not minimal affine-map identity.'))
    schema=read(folder/'policy_schema.json');stored=read(folder/'trajectories.json')
    lookup={(r['seed'],r['family'],*r['location']):r for r in stored}
    events=[e for e in read(census/'state_events.json') if e['family'] in FAMILIES]
    changes={f:Counter() for f in FAMILIES};field_changes={f:Counter() for f in FAMILIES}
    event_records=[];checks=Counter()
    for row in read(folder/'rows.json'):
        seed,family=row['seed'],row['family'];assert sha(folder/row['file'])==row['sha256']
        with np.load(folder/row['file'],allow_pickle=False) as z:
            arrays={g:z['policy_'+g] for g in schema};full=np.concatenate(list(arrays.values()),axis=2)
            forward=z['forward_policy'];fc=np.any(full[1:]!=full[:-1],axis=2);wc=np.any(forward[1:]!=forward[:-1],axis=2)
            changes[family]['transition_opportunities']+=fc.size
            changes[family]['full_changed']+=int(fc.sum());changes[family]['forward_changed']+=int(wc.sum())
            changes[family]['full_changed_forward_unchanged']+=int(np.sum(fc&~wc))
            changes[family]['full_unchanged_forward_changed']+=int(np.sum(~fc&wc))
            for group,aa in arrays.items():
                offset=0
                for field in schema[group]:
                    width=field['width'];part=aa[:,:,offset:offset+width];offset+=width
                    field_changes[family][group+'.'+field['name']]+=int(np.any(part[1:]!=part[:-1],axis=2).sum())
                assert offset==aa.shape[2]
            for i,loc in enumerate(z['locations']):
                key=(seed,family,*loc.tolist());saved=lookup[key]
                assert spans(full[:,i])==saved['full_segments'];assert spans(forward[:,i])==saved['forward_segments']
                checks['independent_full_and_forward_segments']+=2
                for group,aa in arrays.items():
                    assert spans(aa[:,i])==saved['group_segments'][group];checks['independent_component_segments']+=1
                for event in (e for e in events if (e['seed'],e['family'],*e['location'])==key):
                    k=event['first_step'];assert 1<=k<=64
                    assert bytes(map(int,forward[k-1,i])).hex()==event['mode']
                    current=next(s for s in saved['full_segments'] if s['start']<=k<=s['end'])
                    earlier=[s for s in saved['full_segments'] if s['end']<k]
                    event_records.append(dict(**event,policy_start=current['start'],policy_age_at_arrival=k-current['start']+1,
                        longest_completed_policy_segment_before_arrival=max((s['length'] for s in earlier),default=0),
                        immediately_preceding_completed_segment_length=earlier[-1]['length'] if earlier else 0,
                        uses_future_segment_endpoint=False))
    assert len(event_records)==len(events)==24
    aggregate={}
    for family in FAMILIES:
        rr=[r for r in stored if r['family']==family];assert len(rr)==131
        ag={k:summarize(rr,k+'_segments') for k in ['full','forward']}
        ag['components']={g:summarize([dict(s=r['group_segments'][g]) for r in rr],'s') for g in schema}
        ag['transitions']=dict(changes[family]);ee=[e for e in event_records if e['family']==family]
        ag['success_events']=dict(count=len(ee),policy_age_histogram=dict(Counter(e['policy_age_at_arrival'] for e in ee)),
            prior_max_segment_histogram=dict(Counter(e['longest_completed_policy_segment_before_arrival'] for e in ee)),
            events_with_completed_segment_at_least_4=sum(e['longest_completed_policy_segment_before_arrival']>=4 for e in ee))
        assert ag['full']['total_steps']==ag['forward']['total_steps']==8384
        aggregate[family]=ag
    files={}
    for n,value in [('aggregate.json',aggregate),('field_changes.json',field_changes),('events.json',event_records)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,trajectories=393,steps=25152,events=24,checks=dict(checks),
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final)
    print({f:{k:v for k,v in a.items() if k!='components'} for f,a in aggregate.items()},flush=True)
    print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
