"""324 independently sum region moments and verify the complete risk panel."""
import argparse
from collections import defaultdict,Counter
from pathlib import Path
import math
import time
import traceback
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    start=time.perf_counter();base=root/'results/support_consistency_trigger';folder=base/'risk_v1';summary=read(folder/'summary.json');assert summary['passed']
    for n,d in summary['outputs_sha256'].items():assert sha(folder/n)==d
    mom=root/'results/confirmation_conditional_risk/moments';index={r['seed']:r for r in read(mom/'tasks.json')}
    records=read(folder/'scores.json');scores={(r['seed'],r['method'],r['grid']):r for r in records}
    counts=Counter();maxgap=0.
    def close(a,b):
        nonlocal maxgap
        gap=abs(a-b);maxgap=max(maxgap,gap);assert gap<3e-13,(a,b,gap);counts['numeric_fields']+=1
    for t in read(folder/'pools.json'):
        seed=t['seed'];idx=index[seed];p=mom/idx['file'];assert sha(p)==idx['sha256']
        with np.load(p,allow_pickle=False) as z:vol,means=z['volumes'],z['means']
        weights=[float(v)/math.fsum(map(float,vol)) for v in vol];keys=idx['keys']
        def compute(ids):
            mass=math.fsum(weights[i] for i in ids)
            mu=np.array([[math.fsum(weights[i]*float(means[i,b,q]) for i in ids)/mass for q in range(257)] for b in range(4)])
            return mass,mu
        _,full=compute(list(range(len(keys))));cache={}
        for name,pool in t['methods'].items():
            chosen=set(pool);mass,mu=compute([i for i,k in enumerate(keys) if k in chosen]);r=mu-full
            for grid,qs in [(257,range(257)),(129,range(0,257,2))]:
                pairs=[math.fsum(float(r[a,q])*float(r[b,q]) for q in qs)/grid for a,b in [(0,1),(2,3)]]
                s=scores[seed,name,grid];close(s['mass'],mass)
                for got,want in zip(pairs,s['excess_pairs']):close(got,want)
                close(math.fsum(pairs)/2,s['conditional_excess']);cache[name,grid]=dict(mass=mass,pairs=pairs,value=math.fsum(pairs)/2)
                counts['risk_rows']+=1
        for name,pool in t['methods'].items():
            for grid in [257,129]:
                s=scores[seed,name,grid];a=cache[name,grid];b=cache[s['reference'],grid]
                close(a['value']-b['value'],s['delta']);close(a['mass']-b['mass'],s['added_mass'])
                for got,want in zip([x-y for x,y in zip(a['pairs'],b['pairs'])],s['delta_pairs']):close(got,want)
                assert sorted(set(pool)-set(t['methods'][s['reference']]))==s['added_modes']
        counts['tasks']+=1
        if counts['tasks']%16==0:print(dict(audited_tasks=counts['tasks'],seconds=time.perf_counter()-start),flush=True)
    groups=defaultdict(list)
    for r in records:
        groups[r['method'],r['grid'],'all'].append(r)
        groups[r['method'],r['grid'],'fitted' if r['fitted_trigger'] else 'fallback'].append(r)
    for a in read(folder/'aggregate.json'):
        rows=groups[a['method'],a['grid'],a['stratum']];n=len(rows);assert n==a['tasks']
        for field in ['conditional_excess','delta','mass','added_mass']:close(math.fsum(r[field] for r in rows)/n,a[field])
        for j in range(2):close(math.fsum(r['delta_pairs'][j] for r in rows)/n,a['delta_pairs'][j])
        assert sum(len(r['added_modes']) for r in rows)==a['added_mode_pairs']
        assert sum(bool(r['added_modes']) for r in rows)==a['tasks_with_new_modes'];counts['aggregates']+=1
    result=dict(passed=True,counts=dict(counts),max_numeric_gap=maxgap,seconds=time.perf_counter()-start,
                input_summary_sha256=sha(folder/'summary.json'),source_sha256=sha(Path(__file__)),query_targets_accessed=False,
                resources_matched=False,independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],a.out)
    except Exception:save(a.out/'failure.json',dict(traceback=traceback.format_exc()));raise
