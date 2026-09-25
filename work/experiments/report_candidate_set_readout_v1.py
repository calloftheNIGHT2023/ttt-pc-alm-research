"""383 full candidate validation/readout cost, all settings and no query-risk claim."""
from collections import Counter
from pathlib import Path
import math
import numpy as np
import run_candidate_set_readout_v1 as run
from report_search_radius_development_v1 import table, write


def main():
    root = Path(__file__).resolve().parents[2]; base = root/run.BASE
    development = run.complete(base/'development_v1'); audit = run.complete(base/'audit_v1')
    assert audit['development_summary_sha256'] == run.sha(base/'development_v1/summary.json')
    assert run.read(base/'development_v1/protocol.json')['source_sha256'] == run.hashes(root)
    groups = run.read(base/'audit_v1/groups.json'); rows = run.read(base/'development_v1/rows.json')
    index = {(g['method'], g['schedule'], g['ordering']): g for g in groups}
    names = [c[0] for c in run.previous.model.CONFIGS]; settings = run.previous.model.SETTINGS; primary = run.previous.model.PRIMARY
    assert len(rows) == 544 and len(index) == 34 and all(g['calls'] == 16 for g in groups)
    main_rows = []; resource_rows = []; comparisons = []; summaries = []
    for schedule, order in settings:
        pg = index[primary, schedule, order]
        controls = [index[name, schedule, order] for name in names if name.startswith('pdhg_') and index[name, schedule, order]['resolved'] == 16]
        if controls:
            best = min(controls, key=lambda g: g['mean_staged_total_seconds'])
            summaries.append(f"{schedule.upper()}/{order}：固定主active256完整解析{pg['resolved']}/16、均拼接费用{pg['mean_staged_total_seconds']:.4f}秒；全解析PDHG配置中最低均时为{best['method']}、{best['mean_staged_total_seconds']:.4f}秒。")
            for name in names:
                if not name.startswith('regional_active'): continue
                g = index[name, schedule, order]
                own = {r['seed']: r for r in rows if (r['method'], r['schedule'], r['ordering']) == (name, schedule, order)}
                other = {r['seed']: r for r in rows if (r['method'], r['schedule'], r['ordering']) == (best['method'], schedule, order)}
                paired = [s for s in own if own[s]['full_candidate_set_resolved'] and other[s]['full_candidate_set_resolved']]
                comparisons.append(dict(method=name, schedule=schedule, ordering=order, fixed_primary=name == primary,
                    resolved=g['resolved'], comparator=best['method'], comparator_resolved=best['resolved'],
                    mean_staged_seconds=g['mean_staged_total_seconds'], comparator_mean_staged_seconds=best['mean_staged_total_seconds'],
                    staged_mean_ratio=g['mean_staged_total_seconds']/best['mean_staged_total_seconds'],
                    paired_resolved_tasks=len(paired), faster_tasks=sum(own[s]['staged_total_seconds'] < other[s]['staged_total_seconds'] for s in paired),
                    lower_mean_with_all_tasks_resolved=g['resolved'] == 16 and g['mean_staged_total_seconds'] < best['mean_staged_total_seconds']))
        else:
            summaries.append(f'{schedule.upper()}/{order}没有16/16完整解析的PDHG配置；不作相同完整成果的成本优势结论。')
        for name in names:
            g = index[name, schedule, order]
            rr = [r for r in rows if (r['method'], r['schedule'], r['ordering']) == (name, schedule, order)]
            assert g['completed'] == sum(r['completed'] for r in rr)
            assert g['resolved'] == sum(r['full_candidate_set_resolved'] for r in rr)
            assert g['available'] == sum(r['readout_available'] for r in rr)
            assert g['candidates'] == sum(r['remaining'] for r in rr)
            for key in ['search_seconds', 'readout_seconds', 'staged_total_seconds', 'geometry_seconds', 'sampling_seconds', 'reading_seconds']:
                assert g['mean_'+key] == math.fsum(r[key] for r in rr)/16
            for key in ['geometry_numeric_bytes', 'returned_array_bytes']:
                assert g[key] == sum(r[key] for r in rr)
            main_rows.append([schedule, order, name, g['completed'], g['resolved'], g['available'],
                f"{g['mean_search_seconds']:.4f}", f"{g['mean_readout_seconds']:.4f}", f"{g['mean_staged_total_seconds']:.4f}", g['candidates']])
            resource_rows.append([schedule, order, name, f"{g['mean_geometry_seconds']:.4f}",
                f"{g['mean_sampling_seconds']:.5f}", f"{g['mean_reading_seconds']:.5f}",
                f"{g['geometry_numeric_bytes']/16/2**20:.4f}", f"{g['returned_array_bytes']/16/2**20:.4f}",
                f"{math.fsum(r['search_named_bytes'] for r in rr)/16/2**20:.4f}"])
    by_active = {name: [c for c in comparisons if c['method'] == name] for name in names if name.startswith('regional_active')}
    both = [name for name, cc in by_active.items() if len(cc) == 2 and all(c['lower_mean_with_all_tasks_resolved'] for c in cc)]
    conclusion = ('以下预列active配置在两个设置均完整解析16/16，且本次拼接平均费用低于各设置全部PDHG配置中的最低值：'+', '.join(both)+'。这支持进入连续整链重复计时，不等于已经确认稳定加速或未见查询收益。' if both else
        '没有一个预列active配置在两个设置同时满足全解析且低于最低PDHG拼接均时；本轮不能声称统一完整链成本优势，需要利用分项费用定位下一改进。')
    out = base/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    for name, value in [('table_data.json', main_rows), ('resource_data.json', resource_rows), ('comparisons.json', comparisons), ('figure_data.json', groups)]: run.save(out/name, value)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(15, 10.2))
    for ax, (schedule, order) in zip(axes, settings):
        gg = [index[name, schedule, order] for name in names]; yy = np.arange(len(names))
        search = np.array([g['mean_search_seconds'] for g in gg]); readout = np.array([g['mean_readout_seconds'] for g in gg])
        ax.barh(yy, search, label='Search (stage 380)', color='#527998', height=.67)
        ax.barh(yy, readout, left=search, label='Common validation + readout', color='#cf9c4b', height=.67)
        labels = [name.replace('regional_', '').replace('pdhg_', 'PDHG ').replace('_', ' ')+(' *' if name == primary else '') for name in names]
        ax.set_yticks(yy, labels, fontsize=9); ax.invert_yaxis(); peak = max(search+readout)
        ax.set_xlim(0, peak*1.28)
        for y, g in enumerate(gg): ax.text(g['mean_staged_total_seconds']+peak*.012, y, f"{g['mean_staged_total_seconds']:.2f}s | {g['resolved']}/16", va='center', fontsize=8.5)
        ax.set_title(f'{schedule.upper()} | {order.replace("_", " ")}', loc='left', fontsize=13, pad=12)
        ax.set_xlabel('Mean staged seconds | fully resolved tasks', fontsize=9.5)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.13, .064), ncol=2, frameon=False)
    fig.subplots_adjust(left=.14, right=.98, top=.90, bottom=.18, wspace=.51)
    fig.suptitle('Full candidate validation and readout: all 17 configurations', fontsize=15.5, y=.967)
    fig.text(.14, .045, '* Fixed primary: active256. Staged timing sum, not a repeated contiguous end-to-end run. No query targets.', fontsize=9)
    fig.savefig(out/'staged_readout_cost.png', dpi=160); plt.close(fig)
    classifications = Counter()
    for row in rows: classifications.update(row['classification_counts'])
    body = ['# 383｜共同候选验证与读出后的费用', '',
        '本轮把380全部544份实际搜索输出接入同一几何验证和2048粒子读出。不同方法之间没有免费共享几何缓存、没有从母树补候选；原搜索费用逐行全额加回。研究问题是主动局部求证节约的下游费用能否超过其额外成本。', '',
        '## 结果与全部控制', '', *summaries, '', conclusion, '',
        '![搜索与共同读出费用](staged_readout_cost.png)', '',
        table(['调度', '排序', '方法', '搜索完成/16', '完整解析/16', '可用读出/16', '均搜索秒', '均读出组件秒', '均拼接秒', '候选总数'], main_rows), '',
        '固定主仍为active256。描述性最佳次要配置不能改称原主。拼接费用=380已测搜索费用+本轮独立实测验证/读出包装总费用；不是一次连续计时，未做重复置信区间，也不是严格等时预算竞赛。不完整搜索可能已有条件读出，完整解析要求搜索完成且全部候选分类无未决。可用读出不等于完整或准确。', '',
        '## 可检验的数学机制', '',
        '令S_m为方法m输出的完整候选集合，F为真正正体积可行区域。完整且安全的搜索满足F包含于S_m；共同几何完整分类后只接受F。先验、排序、采样数量和随机数都相同时，应得到相同读出；浮点几何也一致时，同种子预测应逐位相同。本轮独立审计检验这一预测，而不是假定更多局部优化会改善同一完整后验的准确率。', '',
        '总费用分解为C_search(m)+sum_{R in S_m}g(R)+C_read(F)。m相对b占优，当且仅当其搜索额外费用小于省去的逐候选验证费用。S_m不必包含于S_b，每个g(R)也不假定相同；本轮逐一实际求解几何，不以候选数乘统一常数代替测量。更贵的局部求证可能因为少留下难假候选而获益，也可能因收效不足而得不偿失。', '',
        '同完整后验的预测一致性只验证执行/推断等价性，不构成查询风险降低。真正目标是把完整链费用优势用于相同预算下更可靠的适应，并与强回归、浅层头和同参数优化器比较；本轮尚未检验该结论。', '',
        '## 独立审计与状态', '',
        f"全部候选分类计数：{dict(classifications)}。544次读出中，完整解析{development['resolved']}，无可用读出{development['unavailable']}；全部原始行均保留。", '',
        f"审计通过：{audit['checks'].get('exact_geometry_certificates', 0)}条精确几何证明、{audit['checks'].get('posterior_support_checks', 0)}个粒子支持误差检查、{audit['checks'].get('prediction_replays', 0)}次预测重放、{audit['checks'].get('full_candidate_readout_checks', 0)}次完整候选读出核对，以及{audit['checks'].get('common_complete_readout_arrays', 0)}个跨方法/排序共同读出数组逐位比较。", '',
        '严格内点、不可行和零体积结论必须有对应精确证书支持；几何体积和粒子采样仍是数值计算，不称精确贝叶斯后验。全局LP属于共同几何组件，明确计费，不伪称整个算法只有局部PC更新。查询输入固定为257点，查询答案从未进入本轮。空输出没有免费fallback，也没有删除失败任务计算MSE。', '',
        table(['调度', '排序', '方法', '均几何秒', '均采样秒', '均读取秒', '均保留几何MiB', '均读出归档MiB', '均搜索命名MiB'], resource_rows), '',
        '命名数组是可见状态计数，不是RSS峰值，也不包含所有几何临时状态。搜索与读出分别计状态，不把它们简单相加冒充精确峰值。加载封存档案、写盘压缩和独立审计不在在线费用中；应在后续连续完整链执行和进程级内存测量中验证。', '',
        '## 下一关', '',
        '若本轮预列候选形成完整链成本区间，冻结候选和强控制，进行交错重复的连续搜索→几何→读出计时；再进入新冻结任务、同总预算anytime预测和计费fallback。强闭式/固定非线性回归、浅头、同参数BP/普通PC/PC-ALM、官方TTT及实际下游验证仍不能省略。本轮不完成核心研究目标。', '',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/382_common_readout_protocol_v1.md) · [全部544调用](../development_v1/rows.json) · [独立审计](../audit_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(body))
    entry = 'outputs/ttt-pc-alm-research/383_common_readout_results_v1.md'
    write(root/entry, '# 383｜共同验证与读出阶段结果\n\n'+'\n\n'.join(summaries)+'\n\n'+conclusion+
        '\n\n[完整图文与17配置](../../results/candidate_set_readout/report_v1/report.md)\n\n没有新query风险结果，目标保持ACTIVE。\n')
    run.save(out/'manifest.json', dict(source_sha256=run.sha(Path(__file__)), entry_file=entry, entry_sha256=run.sha(root/entry),
        input_summary_sha256={p: run.sha(base/p) for p in ['development_v1/summary.json', 'audit_v1/summary.json']},
        outputs_sha256={f: run.sha(out/f) for f in ['table_data.json', 'resource_data.json', 'comparisons.json', 'figure_data.json', 'report.md', 'staged_readout_cost.png']}))
    text = (out/'report.md').read_text(encoding='utf-8')
    for row in main_rows+resource_rows: assert '| '+' | '.join(map(str, row))+' |' in text
    run.save(out/'qa_numeric.json', dict(passed=True, table_cells=len(main_rows)*10+len(resource_rows)*9, groups=len(groups), raw_rows_recomputed=len(rows),
        active_configs_lower_than_all_pdhg_both_settings=both, manifest_sha256=run.sha(out/'manifest.json')))
    print(dict(passed=True, report=str(out/'report.md'), comparisons=comparisons), flush=True)


if __name__ == '__main__': main()
