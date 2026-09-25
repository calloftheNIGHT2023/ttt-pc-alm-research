"""327 independent aggregation, frozen arrays and isolated worker evidence."""
import argparse
from collections import Counter
import math
from pathlib import Path
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def quantile(values,p):
    values=sorted(values);position=(len(values)-1)*p;lo=int(position);hi=min(lo+1,len(values)-1)
    return values[lo]+(values[hi]-values[lo])*(position-lo)


def run(root,out):
    start=time.perf_counter();folder=root/'results/online_credit_resources/calibration_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for n,d in summary['outputs_sha256'].items():assert sha(folder/n)==d
    p=read(folder/'protocol.json')
    for n,d in p['source_sha256'].items():assert sha(root/'work/experiments'/n)==d
    rows=read(folder/'timings.json');memory=read(folder/'memory.json');configs=p['configs'];names=[c['name'] for c in configs]
    assert len(names)==len(set(names))==51 and len(rows)==816 and len(memory)==102
    expected={(s,n,r) for s in p['seeds'] for n in names for r in range(2)}
    assert {(r['seed'],r['method'],r['repeat']) for r in rows}==expected
    jobs=[(seed,cfg['name'],rep) for seed in p['seeds'] for cfg in configs for rep in range(2)]
    assert [(r['seed'],r['method'],r['repeat']) for r in rows]==[jobs[int(i)] for i in np.random.default_rng(327929).permutation(816)]
    assert {(r['seed'],r['method']) for r in memory}=={(s,n) for s in [5910000,5910063] for n in names}
    files=read(folder/'files.json');counts=Counter();maxgap=0.;failures=0
    for n,d in files.items():assert sha(folder/n)==d;counts['saved_predictors']+=1
    for i,r in enumerate(rows):
        assert r['order']==i and r['seconds']>0 and math.isfinite(r['seconds'])
        assert r['seconds']==r['metadata']['charged_complete_seconds']
        assert not r['metadata']['query_targets_accessed'] and files[r['file']]==r['sha256']
        assert read(folder/'calls'/f'timing_{i:04d}.json')==r
        failures+=r['metadata']['execution_failed'];counts['timing_records']+=1
    for r in memory:
        path=folder/r['record_file'];assert sha(path)==r['record_sha256']
        expected={k:v for k,v in r.items() if k not in ['record_file','record_sha256']};assert read(path)==expected
        assert r['parent_pid']!=r['pid'] and r['source_sha256']==p['source_sha256'] and r['checkpoint_manifest']==p['checkpoint_manifest']
        assert all(e['pid']==r['parent_pid'] for e in r['environment']['other_research_or_git_pack_processes'])
        assert r['process_after']['peak_wset']>=r['process_after']['rss']>0
        assert r['traced_peak_bytes']>=r['traced_current_bytes']>=0
        assert r['instrumented_seconds_not_benchmark']==r['metadata']['charged_complete_seconds']
        assert sha(path.parent/'arrays.npz')==r['arrays_sha256']
        with np.load(path.parent/'arrays.npz',allow_pickle=False) as a,np.load(folder/f'{r["seed"]}_{r["method"]}.npz',allow_pickle=False) as b:
            assert set(a.files)==set(b.files)
            for n in a.files:assert a[n].tobytes()==b[n].tobytes() and a[n].shape==b[n].shape and a[n].dtype==b[n].dtype;counts['worker_bitwise_arrays']+=1
        counts['isolated_memory_records']+=1
    means={}
    for m in read(folder/'methods.json'):
        name=m['method'];rr=[r for r in rows if r['method']==name];mm=[r for r in memory if r['method']==name]
        values=[r['seconds'] for r in rr];means[name]=math.fsum(values)/len(values)
        expected=dict(mean_seconds=means[name],median_seconds=quantile(values,.5),p90_seconds=quantile(values,.9),maximum_seconds=max(values),
            failures=sum(r['metadata']['execution_failed'] for r in rr),maximum_traced_peak_bytes=max(r['traced_peak_bytes'] for r in mm),
            maximum_absolute_lifetime_peak_wset=max(r['process_after']['peak_wset'] for r in mm))
        for k,v in expected.items():
            gap=abs(m[k]-v);assert gap<=1e-12*max(1,abs(v));maxgap=max(maxgap,gap);counts['independent_aggregate_fields']+=1
    assert len(means)==51
    for name,s in read(folder/'selection.json').items():
        cap=means[name]
        assert s['within_budget']==[n for n in names if means[n]<=cap]
        assert s['sensitivity_110_percent']==[n for n in names if means[n]<=1.1*cap]
        assert s['highest_adam_still_within']==(means['probe_then_adam3840_33']<=cap)
        assert s['all_methods_retained'] and not s['query_quality_used'];counts['budget_sets']+=2
    for e in read(folder/'environments.json'):assert not e['other_research_or_git_pack_processes']
    assert failures==summary['failures']
    result=dict(passed=True,counts=dict(counts),maximum_aggregate_gap=maxgap,failures=failures,seconds=time.perf_counter()-start,
        calibration_summary_sha256=sha(folder/'summary.json'),source_sha256={Path(__file__).name:sha(Path(__file__))},
        query_targets_accessed=False,resources_scope='Full-time and memory measured, not a task-quality or exact FLOP equivalence claim',core_research_goal_complete=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
