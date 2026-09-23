"""Task-equal risks and paired differences; MC repeats are not extra tasks."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/minimum_dual_query';inp=base/'development';out=base/'analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    p=json.loads((inp/'protocol.json').read_text());queries=json.loads((inp/'query_rows.json').read_text());rows=json.loads((inp/'rows.json').read_text());groups=sorted(set((r['method'],r['readout']) for r in queries));tasks=[];methods=[];paired=[]
    for method,readout in groups:
        values=[];mc=[]
        for seed in p['seeds']:
            rr=[r['mse'] for r in queries if (r['seed'],r['method'],r['readout'])==(seed,method,readout)];value=float(np.mean(rr));values.append(value)
            tasks.append(dict(seed=seed,method=method,readout=readout,mse=value,mc_min=min(rr),mc_max=max(rr),repetitions=len(rr)))
        for rep in range(5 if readout=='mode' else 1):mc.append(float(np.mean([r['mse'] for r in queries if r['method']==method and r['readout']==readout and r['repetition']==rep])))
        methods.append(dict(method=method,readout=readout,mean_mse=float(np.mean(values)),median_task_mse=float(np.median(values)),worst_task_mse=max(values),task_mses=values,replicate_task_mean_mses=mc,
            fallback_tasks=len(set(r['seed'] for r in rows if r['method']==method and r['readout']==readout and r.get('fallback')))))
    lookup={(r['seed'],r['method'],r['readout']):r for r in tasks};primary=p['primary']
    for readout in ['point','mode']:
        for method,control_readout in groups:
            if method==primary:continue
            # Mode candidate vs mode solvers or point-only heads; point vs point.
            if readout=='point' and control_readout!='point':continue
            if readout=='mode' and control_readout=='point' and (method,'mode') in groups:continue
            diff=np.array([lookup[s,primary,readout]['mse']-lookup[s,method,control_readout]['mse'] for s in p['seeds']])
            paired.append(dict(primary_readout=readout,control=method,control_readout=control_readout,mean_difference=float(np.mean(diff)),
                lower=int(np.sum(diff< -1e-12)),same=int(np.sum(abs(diff)<=1e-12)),higher=int(np.sum(diff>1e-12)),
                differences=diff.tolist(),best_task_difference=float(min(diff)),worst_task_difference=float(max(diff))))
    # Specifically inspect both tasks where retaining u restored support feasibility.
    selected_tasks=[r for r in tasks if r['seed'] in [5900000,5900003] and r['method'] in ['minimum_dual_alm64','minimum_reset_alm64','minimum_activity_alm64','dual_alm64','dual_reset_alm64','activity_alm64','alm65','adam240','gn20','probe_all_alm64','cold__alm16']]
    dump(out/'task_risks.json',tasks);dump(out/'methods.json',methods);dump(out/'paired.json',paired);dump(out/'two_support_gain_tasks.json',selected_tasks)
    result=dict(passed=True,task_risk_rows=len(tasks),method_readout_groups=len(methods),paired_comparisons=len(paired),primary_mode_comparisons=[r for r in paired if r['primary_readout']=='mode' and r['control'] in ['minimum_dual_alm64','minimum_reset_alm64','minimum_activity_alm64','dual_alm64','dual_reset_alm64','activity_alm64','alm65','probe_all_alm64','adam240','gn20','cold__alm16','cold__meta_ridge128','cold__prior4096_ridge']],
        query_rows_sha256=sha(inp/'query_rows.json'),source_sha256=sha(Path(__file__)),outputs_sha256={f.name:sha(f) for f in out.glob('*.json')},
        inference_scope='descriptive old-development tasks;5 MC replicates per task, not80 independent tasks; no confirmatory p-value')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
