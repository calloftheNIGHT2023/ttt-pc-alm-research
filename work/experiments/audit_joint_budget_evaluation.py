"""Independent scalar teacher, every risk, task statistics and resource tables."""
import argparse
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def scalar_teacher(q,b):
    values=[]
    for x in q:
        h=float(x)
        for bias in b:h=max(0.,1.-abs(2.*(h+float(bias))-1.))
        values.append(h)
    return np.array(values)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/matched_budget_confirmation';inp=base/'joint_evaluation';out=base/'joint_evaluation_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((inp/'protocol.json').read_text());summary=json.loads((inp/'summary.json').read_text());assert summary['passed'];hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','query_rows','methods','paired','implementation_changes']:assert sha(inp/f'{n}.json')==summary[f'{n}_sha256']
    dump(out/'protocol.json',dict(source_sha256=hashes,evaluation_summary_sha256=sha(inp/'summary.json'),scope='independent scalar composition and summation; all paired intervals and numerical cost dominance'))
    versions={'original':'confirmation','conditioned':'conditioned_confirmation'};queries=json.loads((inp/'query_rows.json').read_text());methods=json.loads((inp/'methods.json').read_text());paired=json.loads((inp/'paired.json').read_text());changes=json.loads((inp/'implementation_changes.json').read_text())
    assert len(queries)==5888 and len({(r['version'],r['seed'],r['method']) for r in queries})==5888
    sources={v:{(r['seed'],r['method']):r for r in json.loads((base/d/'rows.json').read_text())} for v,d in versions.items()};qp=json.loads((base/'confirmation/protocol.json').read_text());seeds=qp['seeds'];counts=Counter();gaps=[];maxgap=0.;truth={s:scalar_teacher(np.linspace(0,1,257),np.random.default_rng(s).uniform(-.12,.12,4)) for s in seeds}
    for row in queries:
        source=sources[row['version']][row['seed'],row['method']];path=base/versions[row['version']]/source['file'];assert sha(path)==source['sha256']
        with np.load(path) as z:
            for field,array in [('mse','prediction'),('point_mse','point_prediction')]:
                risk=math.fsum(float(t)**2 for t in z[array]-truth[row['seed']])/257;gap=abs(risk-row[field]);assert gap<1e-13;maxgap=max(maxgap,gap);counts['independent_risks']+=1
        assert row['seconds']==source['seconds']==source['metadata']['charged_complete_seconds'];assert row['execution_failed']==source['metadata']['execution_failed'];counts['resource_rows']+=1
    lookup={(r['version'],r['seed'],r['method']):r for r in queries};indices=np.random.default_rng(262193).integers(0,64,(20000,64))
    for version in versions:
        for method in methods[version]:
            rr=[lookup[version,s,method['method']] for s in seeds];risks=[r['mse'] for r in rr];times=[r['seconds'] for r in rr];ordered=sorted(risks)
            assert abs(math.fsum(risks)/64-method['mean_mse'])<1e-14;assert (ordered[31]+ordered[32])/2==method['median_mse'];assert max(risks)==method['worst_task_mse']
            assert abs(math.fsum(times)/64-method['mean_seconds'])<1e-14;assert sum(r['execution_failed'] for r in rr)==method['failures'];counts['method_risk_and_time_tables']+=1
            expected=[c['method'] for c in methods[version] if c['method']!=method['method'] and (c['mean_seconds']<=method['mean_seconds'] and c['mean_mse']<=method['mean_mse']) and (c['mean_seconds'],c['mean_mse'])!=(method['mean_seconds'],method['mean_mse'])]
            assert expected==method['numerically_dominated_by'];counts['numerical_dominance_sets']+=1
        for row in paired[version]:
            diff=np.array([lookup[version,s,p['primary']]['mse']-lookup[version,s,row['control']]['mse'] for s in seeds]);assert np.array(row['differences']).tobytes()==diff.tobytes()
            assert abs(math.fsum(diff)/64-row['mean_difference'])<1e-14;assert sum(t< -1e-12 for t in diff)==row['lower'] and sum(t>1e-12 for t in diff)==row['higher'] and sum(abs(t)<=1e-12 for t in diff)==row['same']
            interval=np.percentile(np.average(diff[indices],axis=1),[2.5,97.5]);assert np.max(abs(interval-row['descriptive_bootstrap95']))<1e-14
            assert bool(interval[1]<0)==row['interval_entirely_below_zero'] and bool(interval[0]>0)==row['interval_entirely_above_zero'];counts['paired_intervals']+=1
    changed=[]
    for row in changes:
        observed=[s for s in seeds if lookup['original',s,row['method']]['mse']!=lookup['conditioned',s,row['method']]['mse']];assert observed==row['changed_task_seeds'];assert set(observed)<={5910048}
        delta=math.fsum(lookup['conditioned',s,row['method']]['mse']-lookup['original',s,row['method']]['mse'] for s in seeds)/64;assert abs(delta-row['mean_change'])<1e-14
        if observed:changed.append(row['method'])
        counts['implementation_change_tables']+=1
    ans=dict(passed=True,counts=counts,max_independent_risk_gap=maxgap,methods_changed_by_geometry=len(changed),changed_methods=changed,protocol_sha256=sha(out/'protocol.json'))
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
