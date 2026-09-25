"""385 audited continuous repeats: every control, task-level uncertainty, no risk claim."""
from collections import defaultdict
from pathlib import Path
import math
import numpy as np
import run_contiguous_regional_repeat_v1 as run
from report_search_radius_development_v1 import table, write


def main():
    root = Path(__file__).resolve().parents[2]; base = root/run.BASE
    development = run.complete(base/'development_v1'); audit = run.complete(base/'audit_v1')
    assert audit['development_summary_sha256'] == run.sha(base/'development_v1/summary.json')
    protocol = run.read(base/'development_v1/protocol.json')
    assert protocol['source_sha256'] == run.hashes(root)
    rows = run.read(base/'development_v1/rows.json'); groups = run.read(base/'audit_v1/task_groups.json')
    comparisons = run.read(base/'audit_v1/comparisons.json')
    lookup = {(g['seed'], g['method'], g['schedule'], g['ordering']): g for g in groups}
    raw = defaultdict(list)
    for row in rows: raw[row['seed'], row['method'], row['schedule'], row['ordering']].append(row)
    assert len(rows) == 960 and len(raw) == len(lookup) == 320 and len(comparisons) == 18
    for key, rr in raw.items():
        g = lookup[key]; times = [r['seconds'] for r in sorted(rr, key=lambda r: r['repetition'])]
        assert g['times'] == times and g['median_seconds'] == float(np.median(times))
        assert g['mean_seconds'] == math.fsum(times)/3 and g['all_resolved'] == all(r['resolved'] for r in rr)
    seeds = sorted({r['seed'] for r in rows}); assert len(seeds) == 16
    # Recompute the prespecified task-paired bootstrap directly from raw timing rows.
    comparison_index = {(c['schedule'], c['ordering'], c['control']): c for c in comparisons}
    comparison_rows = []; main_rows = []; figure_groups = []; summaries = []
    for si, (schedule, order) in enumerate(run.SETTINGS):
        setting_groups = []
        for name in run.NAMES:
            gg = [lookup[s, name, schedule, order] for s in seeds]
            rr = [r for s in seeds for r in raw[s, name, schedule, order]]
            g = dict(method=name, schedule=schedule, ordering=order,
                mean_task_median=math.fsum(x['median_seconds'] for x in gg)/16,
                mean_all_calls=math.fsum(r['seconds'] for r in rr)/48,
                mean_search_seconds=math.fsum(r['search_seconds'] for r in rr)/48,
                mean_readout_seconds=math.fsum(r['readout_seconds'] for r in rr)/48,
                max_call_seconds=max(r['seconds'] for r in rr),
                all_resolved_tasks=sum(x['all_resolved'] for x in gg), resolved_calls=sum(r['resolved'] for r in rr),
                mean_returned_array_bytes=math.fsum(r['returned_array_bytes'] for r in rr)/48)
            setting_groups.append(g); figure_groups.append(g)
            main_rows.append([schedule, order, name, g['all_resolved_tasks'], g['resolved_calls'],
                f"{g['mean_task_median']:.6f}", f"{g['mean_all_calls']:.6f}",
                f"{g['mean_search_seconds']:.6f}", f"{g['mean_readout_seconds']:.6f}",
                f"{g['max_call_seconds']:.6f}", f"{g['mean_returned_array_bytes']/2**20:.4f}"])
        best = min((g for g in setting_groups if g['method'].startswith('pdhg_')), key=lambda g:g['mean_task_median'])
        primary = next(g for g in setting_groups if g['method'] == run.PRIMARY)
        summaries.append(f"{schedule.upper()}/{order}：主active1024的任务中位数均值为{primary['mean_task_median']:.6f}秒，"
            f"本轮最低PDHG为{best['method']}、{best['mean_task_median']:.6f}秒；"
            f"主相对该配置均时变化{100*(primary['mean_task_median']/best['mean_task_median']-1):+.2f}%。")
        for ci, control in enumerate(n for n in run.NAMES if n != run.PRIMARY):
            c = comparison_index[schedule, order, control]
            delta = np.array([np.median([r['seconds'] for r in raw[s, run.PRIMARY, schedule, order]])-
                              np.median([r['seconds'] for r in raw[s, control, schedule, order]]) for s in seeds])
            rng = np.random.default_rng(np.random.SeedSequence([384991, si, ci]))
            ci95 = np.quantile(delta[rng.integers(0, 16, (20000, 16))].mean(1), [.025, .975])
            assert c['mean_paired_difference'] == float(delta.mean()) and c['bootstrap95'] == ci95.tolist()
            assert c['primary_faster_tasks'] == int((delta < 0).sum())
            comparison_rows.append([schedule, order, control, '*' if c['predeclared_primary_comparator'] else '',
                f"{c['mean_paired_difference']:.6f}", f"[{ci95[0]:.6f}, {ci95[1]:.6f}]",
                c['primary_faster_tasks'], c['all_primary_resolved'] and c['all_control_resolved']])
    # All complete readouts should agree, even across method and search schedule.
    prediction_hashes = {}; prediction_checks = 0
    for row in rows:
        if not row['canonical'] or not row['resolved']: continue
        digest = run.read(root/row['directory']/'array_hashes.json')['readout_prediction']
        if row['seed'] in prediction_hashes:
            assert digest == prediction_hashes[row['seed']]; prediction_checks += 1
        else: prediction_hashes[row['seed']] = digest
    selected = [c for c in comparisons if c['predeclared_primary_comparator']]
    supported = all(c['all_primary_resolved'] and c['all_control_resolved'] and c['bootstrap95'][1] < 0 for c in selected)
    conclusion = ('相对两个预列主要PDHG控制，任务配对均值差的95%区间均在零以下，支持旧开发任务上的连续完整链成本优势。' if supported else
                  '相对两个预列主要PDHG控制，未同时获得任务配对95%区间低于零的结果；不能宣称两设置均确认稳定成本优势。')
    out = base/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    for name, value in [('table_data.json', main_rows), ('comparison_data.json', comparison_rows),
                        ('figure_data.json', figure_groups)]: run.save(out/name, value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 7.7))
    for ax, (schedule, order) in zip(axes, run.SETTINGS):
        gg = [g for g in figure_groups if (g['schedule'], g['ordering']) == (schedule, order)]
        values = [g['mean_task_median'] for g in gg]; yy = np.arange(len(gg))
        ax.barh(yy, values, color=['#b76432' if g['method'] == run.PRIMARY else '#527998' for g in gg], height=.64)
        ax.set_yticks(yy, [g['method'].replace('regional_', '').replace('pdhg_', 'PDHG ').replace('_', ' ') for g in gg], fontsize=9.5)
        ax.invert_yaxis(); peak = max(values); ax.set_xlim(0, peak*1.32)
        for y, g in enumerate(gg): ax.text(values[y]+peak*.018, y, f"{values[y]:.3f}s | {g['all_resolved_tasks']}/16", va='center', fontsize=9)
        ax.set_title(f'{schedule.upper()} | {order.replace("_", " ")}', loc='left', fontsize=13, pad=13)
        ax.set_xlabel('Mean of task medians (3 repeats each)', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.subplots_adjust(left=.145, right=.98, top=.85, bottom=.17, wspace=.5)
    fig.suptitle('Continuous support-to-prediction cost: 960 actual calls', fontsize=16, y=.95)
    fig.text(.145, .055, 'Orange: fixed primary active1024. 16 old development tasks; 3 technical repeats are not 48 tasks.\n'
             'Fresh search and common geometry every call. No query targets. Not a whole-pipeline deadline experiment.', fontsize=9.5, linespacing=1.6)
    fig.savefig(out/'contiguous_cost.png', dpi=160); plt.close(fig)
    warmup = run.read(base/'development_v1/warmup.json')
    body = ['# 385｜连续支持到预测：交错重复费用结果', '', *summaries, '', conclusion, '',
        '这是旧16任务上的算法执行成本复验，不是新任务泛化确认，也没有查询答案或查询MSE。主配置active1024由382开发结果选择后，在384执行前冻结。原380/382主active256没有被改写。', '',
        '![连续整链费用](contiguous_cost.png)', '',
        '## 全部十配置与实际费用', '',
        table(['调度', '排序', '方法', '三次均解析/16', '解析调用/48', '任务中位数均值秒', '48调用均值秒', '均搜索秒', '均读出秒', '最大调用秒', '均返回数组MiB'], main_rows), '',
        '主要统计量是16个任务各自三次总时的中位数，再跨任务取均值。组件时间列是48次调用的均值，因此应与48调用总均值比较，不能强行与中位数均值相加。完整三次原值保存在task_groups.json；没有删去慢调用或选最快一次。', '',
        f"单独共同热身：{warmup['calls']}调用、{warmup['seconds']:.6f}秒。在线时间包含搜索、逐候选全局几何、2048粒子读出及包装开销；不含事后摘要、磁盘归档、独立审计和冷启动导入。返回数组字节不是峰值RSS，未涵盖全部原生几何临时状态。", '',
        '## 配对比较与不确定性', '',
        table(['调度', '排序', '控制', '预列主要控制', '主减控制秒', '任务bootstrap95%', '主更快/16', '双方全部解析'], comparison_rows), '',
        '星号为执行前固定的BFS cold1024与DFS box1024。其他八项比较全部显示，避免事后选择弱控制。区间为16任务、20000次配对bootstrap；没有多重比较校正，次要比较是描述性分析。旧开发任务和少量任务使区间不能替代新任务确认。', '',
        '## 数学预测与审计', '',
        '安全的完整搜索保留所有正体积可行区域；共同几何完整分类后，相同先验和随机采样规则应给出相同读出。因此本阶段可能改善的是计算费用，而非同一完整后验的查询误差。费用优势的条件是额外局部求证成本小于省去的候选几何验证成本；不能把候选数量直接当时间。', '',
        f"本报告另核对{prediction_checks}个跨方法/调度的完整预测字节摘要，相同任务的预测一致。独立审计计数：{audit['checks']}。", '',
        '全部960次重新计算。320份规范数组经独立精确证书、枚举覆盖和预测重放；其余640次实际计算后，保存逐数组字节摘要及完整非时间元数据并与规范档案比较。归档去重不是跳过运行。全局LP明确属于共同几何读出，不能宣称整体只有局部PC操作。', '',
        '## 下一关仍须完成', '',
        '搜索只有8秒协作边界和65536展开约束，几何读出不受整链时限约束；当前不是等总时间anytime竞赛。后续必须在新冻结任务上设置明确且计费的共同fallback和全链预算，评估所有任务的独立查询误差。强回归/核、浅头、同参数BP/PC/PC-ALM及匹配任务训练的官方TTT仍是必要控制。没有查询风险与实际任务验证前，不能宣布论文核心结果或完成持续目标。', '',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/384_contiguous_repeat_protocol_v1.md) · [960调用](../development_v1/rows.json) · [三次原值](../audit_v1/task_groups.json) · [独立审计](../audit_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(body))
    entry = 'outputs/ttt-pc-alm-research/385_contiguous_repeat_results_v1.md'
    write(root/entry, '# 385｜连续完整链重复结果\n\n'+'\n\n'.join(summaries)+'\n\n'+conclusion+
        '\n\n[全部控制、图文和区间](../../results/contiguous_regional_repeat/report_v1/report.md)\n\n尚无本轮新query结果，目标保持ACTIVE。\n')
    run.save(out/'manifest.json', dict(source_sha256=run.sha(Path(__file__)), entry_file=entry, entry_sha256=run.sha(root/entry),
        input_summary_sha256={p:run.sha(base/p) for p in ['development_v1/summary.json', 'audit_v1/summary.json']},
        outputs_sha256={f:run.sha(out/f) for f in ['table_data.json', 'comparison_data.json', 'figure_data.json', 'report.md', 'contiguous_cost.png']}))
    report_text = (out/'report.md').read_text(encoding='utf-8')
    for row in main_rows+comparison_rows: assert '| '+' | '.join(map(str, row))+' |' in report_text
    run.save(out/'qa_numeric.json', dict(passed=True, raw_rows_recomputed=960, task_groups_recomputed=320,
        bootstrap_comparisons_recomputed=18, cross_method_prediction_digests=prediction_checks,
        table_cells=len(main_rows)*11+len(comparison_rows)*8,
        both_predeclared_comparator_intervals_below_zero=supported, manifest_sha256=run.sha(out/'manifest.json')))
    print(dict(passed=True, report=str(out/'report.md'), summaries=summaries, conclusion=conclusion), flush=True)


if __name__ == '__main__': main()
