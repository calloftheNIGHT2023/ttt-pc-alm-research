"""Task-level aggregation, fixed-repeat uncertainty, full cold resource ledger."""
import argparse
import json
from pathlib import Path
import numpy as np
import cold_stagnation_switch as cold
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/shared_mode_readout';inp=base/'development';out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    p=json.loads((inp/'protocol.json').read_text());audit=json.loads((base/'audit/summary.json').read_text());resource=json.loads((base/'resources/summary.json').read_text());assert audit['passed'] and resource['passed']
    queries=json.loads((inp/'query_rows.json').read_text());rows=json.loads((inp/'rows.json').read_text());marginal=json.loads((inp/'marginal_evaluation.json').read_text())
    old=json.loads((root/'results/cold_stagnation_switch/analysis/methods.json').read_text());oldq=json.loads((root/'results/cold_stagnation_switch/development/query_rows.json').read_text())
    task={};methods={};contrasts={}
    for method in p['pools']:
        task[method]=np.array([np.mean([r['mse'] for r in queries if r['seed']==s and r['method']==method]) for s in p['seeds']])
        rep=[np.mean([r['mse'] for r in queries if r['method']==method and r['repetition']==i]) for i in range(5)]
        mm=dict(mse=float(task[method].mean()),repeat_mean_range=[float(min(rep)),float(max(rep))],repeat_means=rep,
            empty_pool_tasks=len({r['seed'] for r in rows if r['method']==method and r['fallback']}),
            mean_positive_modes=float(np.mean([len(r['mode_keys']) for r in rows if r['method']==method])))
        if method in resource['methods']:mm.update(resource['methods'][method]);mm['original_point_mse']=old[method]['mean_mse']
        else:
            names=[n for n in p['methods'] if method=='all_union' or n!=p['primary']]
            mm['discovery_seconds_lower_ledger']=sum(resource['methods'][n]['mean_discovery_seconds'] for n in names)
            mm['resource_scope']='capacity diagnostic; sum of discovery components only, not a free method or full endpoint timing'
        methods[method]=mm
    for n in old:
        if n in p['methods']:continue
        task[n]=np.array([next(r['mse'] for r in oldq if r['seed']==s and r['method']==n) for s in p['seeds']])
        methods[n]=dict(mse=float(task[n].mean()),mean_seconds=old[n]['mean_seconds'],source='frozen247 baseline; same supports; no refit/tuning')
    for primary in [p['primary'],'alm16','alm128']:
        contrasts[primary]={}
        for n in task:
            if n==primary:continue
            delta=task[primary]-task[n];contrasts[primary][n]=dict(mean_difference=float(delta.mean()),better=int(np.sum(delta< -1e-12)),same=int(np.sum(abs(delta)<=1e-12)),worse=int(np.sum(delta>1e-12)),differences=delta.tolist())
    provenance=[];key=json.loads((inp/'marginal_rows.json').read_text())[0]['new_keys'][0];seed=p['marginal_seed'];a=np.load(root/'results/cold_stagnation_switch/development'/f'{seed}_{p["primary"]}.npz')
    for t in range(len(a['b'])):
        for i,b in enumerate(a['b'][t]):
            actual=cold.base.pattern(a['x'],b).astype(np.uint8).tobytes().hex()
            if actual==key:provenance.append(dict(seed=seed,restart=i,sweep=t,trigger_after=int(a['first_trigger'][-1,i])))
    assert provenance;first=min(provenance,key=lambda r:(r['sweep'],r['restart']))
    mr=dict(new_volume_fraction=marginal[0]['w'],mean_before=float(np.mean([r['prior_mse'] for r in marginal])),mean_after=float(np.mean([r['full_mse'] for r in marginal])),
        difference_range=[min(r['difference'] for r in marginal),max(r['difference'] for r in marginal)],mean_difference=float(np.mean([r['difference'] for r in marginal])),
        first_actual_source=first,actual_source_restarts=sorted({r['restart'] for r in provenance}),actual_source_events=len(provenance),all5_improved=all(r['difference']<0 for r in marginal))
    dump(out/'methods.json',methods);dump(out/'contrasts.json',contrasts);dump(out/'task_risks.json',{n:a.tolist() for n,a in task.items()});dump(out/'novel_mode_provenance.json',provenance)
    lines=['| Method/readout | Query MSE | Full cold mean seconds | Positive modes/task |','|---|---:|---:|---:|']
    for n,r in methods.items():lines.append(f'| {n} | {r["mse"]:.9f} | {r.get("mean_seconds",float("nan")):.6f} | {r.get("mean_positive_modes",float("nan")):.3f} |')
    (out/'all_methods.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),audit_sha256=sha(base/'audit/summary.json'),resources_sha256=sha(base/'resources/summary.json'),
        marginal=mr,outputs_sha256={n:sha(out/n) for n in ['methods.json','contrasts.json','task_risks.json','novel_mode_provenance.json','all_methods.md']},
        scope='Sixteen old tasks, five Monte Carlo repeats not80 independent tasks; full cold pipeline time on all10 actual methods; no multiplicity-adjusted confirmation')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
