"""One query opening after BOTH original and corrected predictions are sealed."""
import argparse
import json
from pathlib import Path
import numpy as np
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/matched_budget_confirmation';out=base/'joint_evaluation';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    review_path=base/'geometry_disposition.json';review=json.loads(review_path.read_text())
    assert review['may_open_query_answers'] and review['numerical_failures_reviewed'] and not review['query_targets_accessed']
    for relative,h in review['required_artifacts'].items():assert sha(base/relative)==h,relative
    versions={'original':('confirmation','prediction_audit'),'conditioned':('conditioned_confirmation','conditioned_prediction_audit')}
    hashes=dict(review['source_sha256']);inputs={};protocols={};manifests={};memories={}
    for version,(pred,audit) in versions.items():
        directory=base/pred;aa=json.loads((base/audit/'summary.json').read_text());assert aa['passed'] and not aa['query_targets_accessed']
        hashes.update(json.loads((base/audit/'protocol.json').read_text())['source_sha256']);summary=json.loads((directory/'summary.json').read_text());before=json.loads((directory/'before_query_manifest.json').read_text())
        assert summary['passed'] and summary['predictions_complete'] and not summary['query_targets_accessed'];assert sha(directory/'before_query_manifest.json')==summary['before_query_manifest_sha256']
        for n in ['protocol','rows','memory']:assert sha(directory/f'{n}.json')==before[f'{n}_sha256']
        rows=json.loads((directory/'rows.json').read_text())
        for r in rows:assert sha(directory/r['file'])==r['sha256']==before['prediction_files'][r['file']]
        inputs[version]={(r['seed'],r['method']):r for r in rows};protocols[version]=json.loads((directory/'protocol.json').read_text());memories[version]=json.loads((directory/'memory.json').read_text())
        manifests[version]=dict(before_query_manifest_sha256=sha(directory/'before_query_manifest.json'),audit_sha256=sha(base/audit/'summary.json'))
    p=protocols['original'];assert protocols['conditioned']['seeds']==p['seeds'] and protocols['conditioned']['configs']==p['configs']
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,geometry_disposition_sha256=sha(review_path),manifests=manifests,tasks=64,methods=46,primary=p['primary'],bootstrap_seed=262193,bootstrap_samples=20000,
        scope='same64 tasks, not128; both original failure-policy and universal corrected implementation; descriptive paired percentile intervals, no multiplicity-adjusted significance'))
    # FIRST access to new-query targets. Both versions of every method are fixed.
    q=np.linspace(0,1,257);query=[]
    for seed in p['seeds']:
        truth=forward(q,np.random.default_rng(seed).uniform(-.12,.12,(1,4)))[0]
        for version,(pred,_) in versions.items():
            for cfg in p['configs']:
                row=inputs[version][seed,cfg['name']]
                with np.load(base/pred/row['file']) as z:risk=float(np.mean((z['prediction']-truth)**2));point_risk=float(np.mean((z['point_prediction']-truth)**2))
                query.append(dict(version=version,seed=seed,method=cfg['name'],readout=row['readout'],mse=risk,point_mse=point_risk,seconds=row['seconds'],execution_failed=row['metadata']['execution_failed'],geometry_repairs=row['metadata'].get('geometry_repair_count',0)))
    dump(out/'query_rows.json',query);lookup={(r['version'],r['seed'],r['method']):r for r in query};methods={};paired={};indices=np.random.default_rng(262193).integers(0,64,(20000,64))
    for version in versions:
        primary=np.array([lookup[version,s,p['primary']]['mse'] for s in p['seeds']]);methods[version]=[];paired[version]=[]
        for cfg in p['configs']:
            name=cfg['name'];rr=[lookup[version,s,name] for s in p['seeds']];risk=np.array([r['mse'] for r in rr]);times=np.array([r['seconds'] for r in rr]);diff=primary-risk
            mem=[r for r in memories[version] if r['method']==name];basic=[r['traced_peak_bytes'] for r in mem if r['seed'] in [5910000,5910063]];extra=[r['traced_peak_bytes'] for r in mem if r['seed']==5910048]
            methods[version].append(dict(method=name,family=cfg['family'],readout=rr[0]['readout'],mean_mse=float(risk.mean()),median_mse=float(np.median(risk)),worst_task_mse=float(risk.max()),task_mses=risk.tolist(),
                mean_point_mse=float(np.mean([r['point_mse'] for r in rr])),mean_seconds=float(times.mean()),median_seconds=float(np.median(times)),failures=sum(r['execution_failed'] for r in rr),geometry_repairs=sum(r['geometry_repairs'] for r in rr),
                traced_peak_two_prespecified_tasks=max(basic),traced_peak_extra_failure_task=max(extra) if extra else None,memory_scope='Python traced allocations, not native/RSS/deployment memory or a worst-case bound'))
            if name!=p['primary']:
                interval=np.quantile(diff[indices].mean(1),[.025,.975]);paired[version].append(dict(control=name,mean_difference=float(diff.mean()),lower=int(np.sum(diff< -1e-12)),same=int(np.sum(abs(diff)<=1e-12)),higher=int(np.sum(diff>1e-12)),differences=diff.tolist(),
                    descriptive_bootstrap95=interval.tolist(),interval_entirely_below_zero=bool(interval[1]<0),interval_entirely_above_zero=bool(interval[0]>0)))
        for row in methods[version]:row['numerically_dominated_by']=[c['method'] for c in methods[version] if c['method']!=row['method'] and c['mean_seconds']<=row['mean_seconds'] and c['mean_mse']<=row['mean_mse'] and (c['mean_seconds']<row['mean_seconds'] or c['mean_mse']<row['mean_mse'])]
    changes=[]
    for cfg in p['configs']:
        name=cfg['name'];a=np.array([lookup['original',s,name]['mse'] for s in p['seeds']]);b=np.array([lookup['conditioned',s,name]['mse'] for s in p['seeds']]);changes.append(dict(method=name,original_mean=float(a.mean()),conditioned_mean=float(b.mean()),mean_change=float((b-a).mean()),changed_task_seeds=[s for i,s in enumerate(p['seeds']) if b[i]!=a[i]]))
    dump(out/'methods.json',methods);dump(out/'paired.json',paired);dump(out/'implementation_changes.json',changes);primaries={v:next(r for r in methods[v] if r['method']==p['primary']) for v in versions}
    ans=dict(passed=True,tasks=64,methods=46,implementation_versions=2,query_evaluations=len(query),primary=primaries,protocol_sha256=sha(out/'protocol.json'),query_rows_sha256=sha(out/'query_rows.json'),methods_sha256=sha(out/'methods.json'),paired_sha256=sha(out/'paired.json'),implementation_changes_sha256=sha(out/'implementation_changes.json'),
        scope='same64 independent synthetic tasks; all prespecified comparators, failures and costs; no selective implementation choice or universal superiority claim')
    dump(out/'summary.json',ans);print(json.dumps({**ans,'primary':{v:{k:w for k,w in r.items() if k!='task_mses'} for v,r in primaries.items()}}),flush=True)

if __name__=='__main__':main()
