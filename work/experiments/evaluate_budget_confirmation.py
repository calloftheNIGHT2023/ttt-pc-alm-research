"""Single evaluation after every64-task prediction and pre-query audit."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/matched_budget_confirmation';inp=base/'confirmation';audit=base/'prediction_audit';out=base/'evaluation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    # The live prediction batch exposed a shared numerical geometry defect.
    # Do not open answers until its observation-only disposition is committed.
    review_path=base/'geometry_disposition.json';review=json.loads(review_path.read_text())
    assert review['may_open_query_answers'] and review['numerical_failures_reviewed'] and not review['query_targets_accessed']
    for relative,expected_sha in review['required_artifacts'].items():assert sha(base/relative)==expected_sha,relative
    a=json.loads((audit/'summary.json').read_text());assert a['passed'] and not a['query_targets_accessed'];hashes=dict(json.loads((audit/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=json.loads((inp/'before_query_manifest.json').read_text());run=json.loads((inp/'summary.json').read_text());assert run['passed'] and sha(inp/'before_query_manifest.json')==run['before_query_manifest_sha256'];assert sha(inp/'rows.json')==before['rows_sha256'];p=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'rows.json').read_text())
    for r in rows:assert sha(inp/r['file'])==r['sha256']==before['prediction_files'][r['file']]
    dump(out/'protocol.json',dict(source_sha256=hashes,prediction_audit_sha256=sha(audit/'summary.json'),before_query_manifest_sha256=sha(inp/'before_query_manifest.json'),
        geometry_disposition_sha256=sha(review_path),tasks=64,primary=p['primary'],bootstrap_seed=262193,bootstrap_samples=20000,scope='single fixed deployment MC; task-level paired descriptive percentile intervals, not multiplicity-adjusted significance'))
    # FIRST access to new-query targets. All methods, failures and files fixed.
    query=[]
    for seed in p['seeds']:
        truth=forward(np.linspace(0,1,257),np.random.default_rng(seed).uniform(-.12,.12,(1,4)))[0]
        for row in [r for r in rows if r['seed']==seed]:
            with np.load(inp/row['file']) as z:
                risk=float(np.mean((z['prediction']-truth)**2));point_risk=float(np.mean((z['point_prediction']-truth)**2))
            query.append(dict(seed=seed,method=row['method'],readout=row['readout'],mse=risk,point_mse=point_risk,seconds=row['seconds'],execution_failed=row['metadata']['execution_failed']))
    dump(out/'query_rows.json',query);lookup={(r['seed'],r['method']):r for r in query};primary=np.array([lookup[s,p['primary']]['mse'] for s in p['seeds']]);methods=[];paired=[];indices=np.random.default_rng(262193).integers(0,64,(20000,64))
    for cfg in p['configs']:
        name=cfg['name'];rr=[lookup[s,name] for s in p['seeds']];risk=np.array([r['mse'] for r in rr]);times=np.array([r['seconds'] for r in rr]);diff=primary-risk
        methods.append(dict(method=name,family=cfg['family'],readout=rr[0]['readout'],mean_mse=float(risk.mean()),median_mse=float(np.median(risk)),worst_task_mse=float(risk.max()),task_mses=risk.tolist(),
            mean_point_mse=float(np.mean([r['point_mse'] for r in rr])),mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),failures=sum(r['execution_failed'] for r in rr)))
        if name!=p['primary']:
            boots=diff[indices].mean(1);interval=np.quantile(boots,[.025,.975]);paired.append(dict(control=name,mean_difference=float(diff.mean()),lower=int(np.sum(diff< -1e-12)),same=int(np.sum(abs(diff)<=1e-12)),higher=int(np.sum(diff>1e-12)),
                differences=diff.tolist(),descriptive_bootstrap95=interval.tolist(),interval_entirely_below_zero=bool(interval[1]<0),interval_entirely_above_zero=bool(interval[0]>0)))
    for row in methods:row['numerically_dominated_by']=[c['method'] for c in methods if c['method']!=row['method'] and c['mean_seconds']<=row['mean_seconds'] and c['mean_mse']<=row['mean_mse'] and (c['mean_seconds']<row['mean_seconds'] or c['mean_mse']<row['mean_mse'])]
    dump(out/'methods.json',methods);dump(out/'paired.json',paired);ans=dict(passed=True,tasks=64,methods=len(methods),query_evaluations=len(query),primary=next(r for r in methods if r['method']==p['primary']),
        query_rows_sha256=sha(out/'query_rows.json'),methods_sha256=sha(out/'methods.json'),paired_sha256=sha(out/'paired.json'),protocol_sha256=sha(out/'protocol.json'),
        scope='independent synthetic batch; all prespecified methods and failures kept; bootstrap intervals descriptive and unadjusted; no claim of universal necessity')
    dump(out/'summary.json',ans);print(json.dumps({**ans,'primary':{k:v for k,v in ans['primary'].items() if k!='task_mses'}}),flush=True)

if __name__=='__main__':main()
