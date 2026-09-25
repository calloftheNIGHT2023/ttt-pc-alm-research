"""315 post-seal all-trajectory map persistence and past-only event linkage."""
import argparse
from collections import Counter
import gzip
import json
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha
from analyze_solver_policy_census_v1 import summarize,spans

FAMILIES=['alm_keep','alm_reset','nodual']


def run(root,out):
    begin=time.perf_counter();source=root/'results/effective_affine_map/development_v1'
    summary=read(source/'summary.json');assert summary['passed']
    for n,digest in summary['outputs_sha256'].items():assert sha(source/n)==digest
    assert sha(source/'before_event_analysis_manifest.json')==summary['manifest_sha256']
    census=root/'results/successful_transition_census/development_v2'
    cs=read(census/'summary.json');assert cs['passed']
    for n,digest in cs['outputs_sha256'].items():assert sha(census/n)==digest
    save(out/'protocol.json',dict(source_sha256={n:sha(root/'work/experiments'/n) for n in [Path(__file__).name,'analyze_solver_policy_census_v1.py']},
        support_summary_sha256=sha(source/'summary.json'),old_census_summary_sha256=sha(census/'summary.json'),
        query_targets_accessed=False,resources_matched=False,formula_identity_only=True))
    trajectories=read(source/'trajectories.json')
    lookup={(t['seed'],t['family'],*t['location']):t for t in trajectories};checks=Counter()
    rows=read(source/'rows.json');resources=[];seen=set()
    for row in rows:
        with np.load(source/row['file'],allow_pickle=False) as z:
            for i,loc in enumerate(z['locations']):
                saved=lookup[(row['seed'],row['family'],*loc.tolist())]
                assert spans(z['map_ids'][:,i,None])==saved['map_segments'];checks['independent_full_map_segments']+=1
                for j,k in enumerate(['b','h','u']):
                    assert spans(z['block_map_ids'][:,i,j,None])==saved['block_segments'][k];checks['independent_block_map_segments']+=1
                assert np.max(z['gaps'][:,i],axis=0).tolist()==saved['max_gaps']
        if row['seed'] not in seen:
            with gzip.open(source/row['map_bank_file'],'rt',encoding='utf-8') as stream:bank=json.load(stream)
            resources.append({k:bank[k] for k in ['seed','unique_effective_maps','full_policy_cache_entries','nonzero_rational_coefficients','seconds','unique_block_maps']})
            seen.add(row['seed'])
    events=[e for e in read(census/'state_events.json') if e['family'] in FAMILIES];event_records=[]
    for e in events:
        t=lookup[(e['seed'],e['family'],*e['location'])];k=e['first_step'];entry=dict(**e)
        for name,ss in [('full',t['map_segments']),('old_full',t['old_full_segments'])]+list(t['block_segments'].items()):
            now=next(s for s in ss if s['start']<=k<=s['end']);before=[s for s in ss if s['end']<k]
            entry[name]=dict(age_at_arrival=k-now['start']+1,start=now['start'],
                longest_completed_before_arrival=max((s['length'] for s in before),default=0),
                previous_completed_length=before[-1]['length'] if before else 0)
        event_records.append(entry)
    assert len(event_records)==24 and len(trajectories)==393
    aggregate={}
    for f in FAMILIES:
        rr=[r for r in trajectories if r['family']==f];ee=[e for e in event_records if e['family']==f]
        result=dict(full=summarize(rr,'map_segments'),old_full=summarize(rr,'old_full_segments'),
                    old_policy_changes_merged=sum(r['old_policy_changes_merged'] for r in rr),
                    blocks={k:summarize([dict(s=r['block_segments'][k]) for r in rr],'s') for k in ['b','h','u']})
        result['success_events']={k:dict(count=len(ee),
            age_at_arrival_histogram=dict(sorted(Counter(e[k]['age_at_arrival'] for e in ee).items())),
            completed_max_histogram=dict(sorted(Counter(e[k]['longest_completed_before_arrival'] for e in ee).items())),
            with_completed_segment_at_least_4=sum(e[k]['longest_completed_before_arrival']>=4 for e in ee))
            for k in ['full','old_full','b','h','u']}
        assert result['full']['segments']+result['old_policy_changes_merged']==result['old_full']['segments']
        aggregate[f]=result
    files={}
    for n,value in [('aggregate.json',aggregate),('events.json',event_records),('resources.json',resources)]:
        save(out/n,value);files[n]=sha(out/n)
    files['protocol.json']=sha(out/'protocol.json')
    final=dict(passed=True,checks=dict(checks),events=24,trajectories=393,
        query_targets_accessed=False,resources_matched=False,independent_task_gain_established=False,
        seconds=time.perf_counter()-begin,outputs_sha256=files)
    save(out/'summary.json',final)
    print({f:dict(full=a['full'],merged=a['old_policy_changes_merged'],events=a['success_events']) for f,a in aggregate.items()},flush=True)
    print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
