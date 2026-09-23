"""Full matched-host resource accounting and quality/cost descriptive frontier."""
import argparse
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/dual_jump_query';out=base/'cost_analysis';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    resource=json.loads((base/'resources/summary.json').read_text());assert resource['passed'];timings=json.loads((base/'resources/timings.json').read_text());quality=json.loads((base/'analysis/methods.json').read_text());rows=[]
    for name,r in resource['methods'].items():
        q=next((q for q in quality if q['method']==name and q['readout']=='mode'),None)
        if q is None:q=next(q for q in quality if q['method']==name and q['readout']=='point')
        times=[t for t in timings if t['method']==name];parts={}
        for key in ['preparation_seconds','search_seconds','atomic_seconds','continuation_collection_seconds','discovery_seconds','geometry_seconds','sampling_seconds','read_seconds']:
            if key in times[0]['metadata']:parts[key]=float(np.mean([t['metadata'][key] for t in times]))
        if 'search_seconds' in parts:parts['unassigned_call_seconds']=r['mean_seconds']-sum(parts.values())
        rows.append(dict(method=name,readout=q['readout'],mean_mse=q['mean_mse'],worst_task_mse=q['worst_task_mse'],mean_seconds=r['mean_seconds'],max_traced_peak_bytes=r['max_traced_peak_bytes'],parts=parts))
    for row in rows:
        row['numerically_dominated_by']=[c['method'] for c in rows if c['method']!=row['method'] and c['mean_seconds']<=row['mean_seconds'] and c['mean_mse']<=row['mean_mse'] and (c['mean_seconds']<row['mean_seconds'] or c['mean_mse']<row['mean_mse'])]
    dump(out/'methods.json',rows)
    table=['# 256全部33方法的查询与完整调用费用','','平均风险按16任务等权；模式读出每任务5个MC重复，资源为固定第一个重复的一次完整冷调用。不是确认性显著性结论。','','| 方法 | 读出 | MSE | 最坏任务MSE | 完整秒 | Python跟踪峰字节 |','|---|---|---:|---:|---:|---:|']
    for r in rows:table.append(f'| {r["method"]} | {r["readout"]} | {r["mean_mse"]:.12f} | {r["worst_task_mse"]:.9f} | {r["mean_seconds"]:.6f} | {r["max_traced_peak_bytes"]} |')
    table+=['','峰值是两个固定任务的tracemalloc最大值，五个冻结meta模型共同预加载，不能当原生分配完整值、独立部署峰或最坏情况上界。']
    (out/'all_methods.md').write_text('\n'.join(table)+'\n',encoding='utf-8')
    primary=next(r for r in rows if r['method']=='dual_alm64');cold=next(r for r in rows if r['method']=='cold__alm16')
    result=dict(passed=True,primary=primary,primary_to_cold_alm16_time_ratio=primary['mean_seconds']/cold['mean_seconds'],primary_minus_cold_alm16_risk=primary['mean_mse']-cold['mean_mse'],
        source_sha256=sha(Path(__file__)),resource_summary_sha256=sha(base/'resources/summary.json'),quality_methods_sha256=sha(base/'analysis/methods.json'),
        outputs_sha256={p.name:sha(p) for p in out.glob('*')},frontier_scope='numeric means only; small timing differences are not statistically certified speed differences')
    dump(out/'summary.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':main()
