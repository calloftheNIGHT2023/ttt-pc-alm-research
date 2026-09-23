"""285 audited full-cost table and scientific resource figure, without quality."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_resources';inp=base/'calibration';gate=base/'audit';out=base/'figures';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    aa=json.loads((gate/'summary.json').read_text());assert aa['passed'];gp=json.loads((gate/'protocol.json').read_text());hashes=dict(gp['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert sha(inp/'summary.json')==gp['calibration_summary_sha256'];s=json.loads((inp/'summary.json').read_text())
    for n,h in s['outputs_sha256'].items():assert sha(inp/n)==h,n
    rows=json.loads((inp/'methods.json').read_text());choice=json.loads((inp/'selected_configs.json').read_text());p=json.loads((inp/'protocol.json').read_text());within=set(choice['within_budget'])
    dump(out/'protocol.json',dict(source_sha256=hashes,resource_audit_sha256=sha(gate/'summary.json'),phase_accesses_query_targets=False,scope='Eight old contexts, three repeats, complete CPU costs; memory instrumentation separate, not native worst-case peak'))
    plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False});fig,axes=plt.subplots(1,2,figsize=(13,10),layout='constrained');yy=np.arange(len(rows));colors=['#c7742e' if r['method']==p['primary'] else '#508b89' if r['method'] in within else '#cbd5d7' for r in rows]
    axes[0].barh(yy,[r['mean_seconds'] for r in rows],color=colors);axes[0].set_yticks(yy,[r['method'] for r in rows]);axes[0].set_xscale('log');axes[0].axvline(choice['budget_seconds'],color='#c7742e',linestyle='--',linewidth=1.5);axes[0].set_xlabel('Mean seconds, log scale; dashed = fixed candidate budget');axes[0].set_title('A. Full untraced adaptation + geometry + readout',loc='left')
    axes[1].barh(yy,[r['max_traced_peak_bytes']/2**20 for r in rows],color=colors);axes[1].set_yticks(yy,['']*len(rows));axes[1].set_xlabel('Maximum traced peak across two contexts, MiB');axes[1].set_title('B. Separate instrumented allocation peak',loc='left')
    for ax in axes:ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
    fig.suptitle('Resource matching: fixed candidate, stronger controls selected by time only',fontsize=13,fontweight='bold');fig.supxlabel('600 timed calls; 50 separate memory calls. Loading/imports separate; tracemalloc can omit native allocations. No quality claim.',fontsize=9)
    fig.savefig(out/'285_probe_credit_resources.png',dpi=175);plt.close(fig)
    lines=['# 285 实际完整资源校准','',f"固定主cap={choice['budget_seconds']:.9f}秒；8任务×3重复。查询质量未用于入选。",'', '| 配置 | 完整均值秒 | 中位秒 | p90秒 | 最大秒 | 两例最大tracemalloc MiB | cap内 |','|---|---:|---:|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['method']} | {r['mean_seconds']:.8f} | {r['median_seconds']:.8f} | {r['p90_seconds']:.8f} | {r['maximum_seconds']:.8f} | {r['max_traced_peak_bytes']/2**20:.4f} | {'是' if r['method'] in within else '否'} |")
    lines.extend(['','数组/返回值小计不等于峰值；进程RSS/平台峰包含模型导入及此前运行；tracemalloc可遗漏原生分配。完整按任务重复、修复和失败见calibration/timings.json，内存明细见calibration/memory.json。','','全部cap内：'+', '.join(choice['within_budget']),'','每族最低超预算背景：'+', '.join(choice['over_budget_background']),'','1.10敏感性：'+', '.join(choice['sensitivity_110_percent'])])
    (out/'resource_table.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    ans=dict(passed=True,configs=25,outputs_sha256={n:sha(out/n) for n in ['protocol.json','285_probe_credit_resources.png','resource_table.md']},phase_accesses_query_targets=False)
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
