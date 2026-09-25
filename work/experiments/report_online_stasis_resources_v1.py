"""305 complete audited resource table and static scientific cost figure."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_gradient_flat_split_states_v1 import read,save,sha


def run(root,out):
    source=root/'results/online_stasis_resources/calibration_v1'
    audited=root/'results/online_stasis_resources/audit_v1'
    audit=read(audited/'summary.json');assert audit['passed']
    summary=read(source/'summary.json')
    for n,d in summary['outputs_sha256'].items():assert sha(source/n)==d
    for n,d in audit['outputs_sha256'].items():assert sha(audited/n)==d
    methods=read(source/'methods.json');selection=read(source/'selection.json');protocol=read(source/'protocol.json')
    byname={r['method']:r for r in methods};primary=selection['primary'];cap=selection['budget_seconds']
    ordered=sorted(methods,key=lambda r:r['mean_seconds'],reverse=True)
    fig,ax=plt.subplots(figsize=(13.8,12.6))
    y=np.arange(len(ordered));colors=['#087f8c' if r['method']==primary else '#64748b' for r in ordered]
    means=np.array([r['mean_seconds']*1000 for r in ordered]);p90=np.array([r['p90_seconds']*1000 for r in ordered])
    ax.barh(y,means,color=colors,height=.65,label='Mean full-call time')
    ax.scatter(p90,y,color='#b45309',marker='|',s=60,label='90th percentile',zorder=3)
    ax.axvline(cap*1000,color='#087f8c',linestyle='--',linewidth=1.1,label='Candidate mean budget')
    ax.set_yticks(y,[r['method'] for r in ordered],fontsize=8)
    ax.set_xscale('log');ax.set_xlabel('Complete untraced call (milliseconds; log scale)')
    ax.set_title('38 methods / 8 fixed resource contexts / 3 interleaved repeats\nCost benchmark only: not a new-task quality result',fontsize=13,pad=15)
    ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    ax.legend(loc='lower right',fontsize=9,framealpha=1)
    fig.subplots_adjust(left=.32,right=.97,bottom=.065,top=.925)
    fig.savefig(out/'complete_cost.png',dpi=150)
    plt.close(fig)
    lines=['# 305：真实在线停滞续接的完整调用成本','',
        f'已完成全部 **38方法×8上下文×3重复=912次** 正式计时，另有38次warmup。正式执行失败{summary["failed_timing_calls"]}次；干扰暂停快照{summary["interference_snapshots"]}次。独立审计核对全部304份唯一预测文件、912行配对成本、38行方法统计和历史输出。','',
        f'候选 `{primary}` 平均完整耗时 **{cap:.6f}秒**。计入支持轨迹、事件检测、影子续接、几何及其共同修复、采样、读出、投影和验证。输入/模型载入、磁盘存档、哈希、GC和诊断自检在计时外。','',
        f'固定模型共享状态{protocol["all_preloaded_model_bytes"]:,}字节，加载{protocol["model_loading_seconds"]:.6f}秒；观察数组共享状态{protocol["shared_observed_array_bytes"]:,}字节。这里没有新的逐方法原生峰值内存测量，不能仅据命名数组小计宣称内存优势。','',
        '## 重点对照','',
        '| 方法 | 平均秒 | 方法/候选均值 | 说明 |','|---|---:|---:|---|']
    focus=[(primary,'新在线候选'),('credit_control_probe33','原同probe ALM32'),('counterfactual_mode_change_alm','旧mode-change候选'),
        ('probe_alm40_33','同probe ALM40'),('probe_alm64_33','同probe ALM64'),('probe_then_adam240_33','同probe强BP'),
        ('probe_then_adam480_33','更长BP'),('cold__prior65536_ridge','强先验闭式回归'),('cold__meta_ridge128','预训练固定特征回归'),
        ('cold__meta_shallow64_20','浅层适应头')]
    for name,note in focus:
        r=byname[name];lines.append(f'| {name} | {r["mean_seconds"]:.6f} | {r["mean_over_candidate"]:.4f} | {note} |')
    lines+=['',f'描述性候选均值预算内有{len(selection["within_budget"])}个方法，1.10倍敏感性预算内有{len(selection["sensitivity_110_percent"])}个；所有超预算方法仍保留。此预算不使用查询效果选择。','',
        '## 全部方法','',f'![完整成本]({(out/"complete_cost.png").as_posix()})','',
        '| 方法 | 均值秒 | 中位数秒 | p90秒 | 方法/候选均值 | 配对均差秒（方法−候选） |','|---|---:|---:|---:|---:|---:|']
    for r in methods:
        lines.append(f'| {r["method"]} | {r["mean_seconds"]:.9f} | {r["median_seconds"]:.9f} | {r["p90_seconds"]:.9f} | {r["mean_over_candidate"]:.6f} | {r["paired_mean_difference"]:.9f} |')
    lines+=['','## 能支持什么','',
        f'无trace候选的{audit["counts"]["unique_online_trace_arrays"]}个唯一共同数组与304trace版本逐位相同；全部方法共{audit["counts"]["unique_frozen_arrays"]}个唯一数组与冻结存档相同。每次重复由执行器逐位核对；独立审计重新读取唯一输出和检查日志，不把未另存的重复数组说成独立重算。','',
        '这些证据支持真实接口的成本和行为可复现性，不支持“在同资源下质量一定更优”。8个上下文是既有资源任务，不能作为新的盲测或部署普遍加速结论。尤其强回归往往明显更便宜，后续须用独立查询效果证明增加成本值得。','',
        '## 接着做什么','',
        '306协议在本轮计时完成前已固定：同一在线触发位置的九族真实因果对照先通过轨迹/分支等价门槛，再固定新增任务、完整成本及独立查询评分。303旧理想池优势仍受单任务影响，不将本次工程通过升级为论文级优势。核心目标保持进行中。','',
        f'- [完整计时记录]({(source/"timings.json").as_posix()})',
        f'- [完整方法统计]({(source/"methods.json").as_posix()})',
        f'- [独立审计]({(audited/"summary.json").as_posix()})','']
    with (out/'report.md').open('x',encoding='utf-8') as stream:stream.write('\n'.join(lines))
    # Validate every full-table number against the audited machine-readable data.
    table=[line for line in (out/'report.md').read_text(encoding='utf-8').splitlines() if line.startswith('| ') and len(line.split('|'))==8]
    numeric=[line for line in table if line.split('|')[1].strip() in byname]
    assert len(numeric)==38
    keys=['mean_seconds','median_seconds','p90_seconds','mean_over_candidate','paired_mean_difference']
    for line in numeric:
        cells=[s.strip() for s in line.strip('|').split('|')];row=byname[cells[0]]
        for value,key in zip(cells[1:],keys):assert abs(float(value)-row[key])<=5.01e-7
    save(out/'summary.json',dict(passed=True,full_table_rows=38,numeric_cells_checked=190,
        source_sha256={Path(__file__).name:sha(Path(__file__))},calibration_summary_sha256=sha(source/'summary.json'),
        audit_summary_sha256=sha(audited/'summary.json'),visual_review_pending=True,
        outputs_sha256={n:sha(out/n) for n in ['report.md','complete_cost.png']}))
    print(dict(report=str(out/'report.md'),primary_seconds=cap,methods=38),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/online_stasis_resources/report_v1'
    out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True));raise
