"""381 evidence-grounded retired-search cost report; all controls and actual work."""
from collections import Counter
from pathlib import Path
import math
import numpy as np
import run_retired_region_join_v1 as run
from report_search_radius_development_v1 import table, write


def main():
    root = Path(__file__).resolve().parents[2]; base = root/run.BASE
    preflight = run.complete(base/'preflight_v1'); development = run.complete(base/'development_v1'); audit = run.complete(base/'audit_v1')
    assert audit['development_summary_sha256'] == run.sha(base/'development_v1/summary.json')
    assert run.read(base/'development_v1/protocol.json')['source_sha256'] == run.hashes(root)
    groups = run.read(base/'audit_v1/groups.json'); rows = run.read(base/'development_v1/rows.json')
    index = {(g['method'], g['schedule'], g['ordering']): g for g in groups}
    assert len(index) == 34 and len(rows) == 544 and all(g['calls'] == 16 for g in groups)
    names = [name for name, _, _ in run.model.CONFIGS]; table_rows = []; resource_rows = []; comparisons = []; summaries = []
    for schedule, order in run.model.SETTINGS:
        primary = index[run.model.PRIMARY, schedule, order]
        controls = [index[name, schedule, order] for name in names if name.startswith('pdhg_')]
        full_controls = [c for c in controls if c['completed'] == 16]; assert full_controls
        strongest = min(full_controls, key=lambda g: g['mean_seconds'])
        active = min((index[name, schedule, order] for name in names if name.startswith('regional_active') and index[name, schedule, order]['completed'] == 16), key=lambda g: g['mean_seconds'])
        summaries.append(f"{schedule.upper()}/{order}：固定主active256完成{primary['completed']}/16、均{primary['mean_seconds']:.4f}秒；PDHG全完成配置中均时最低为{strongest['method']}、{strongest['mean_seconds']:.4f}秒。")
        primary_rows = {r['seed']: r for r in rows if (r['method'], r['schedule'], r['ordering']) == (run.model.PRIMARY, schedule, order)}
        control_rows = {r['seed']: r for r in rows if (r['method'], r['schedule'], r['ordering']) == (strongest['method'], schedule, order)}
        shared = [seed for seed in primary_rows if primary_rows[seed]['completed'] and control_rows[seed]['completed']]
        comparisons.append(dict(schedule=schedule, ordering=order, primary=run.model.PRIMARY, primary_complete=primary['completed'],
            primary_mean_seconds=primary['mean_seconds'], descriptive_best_pdhg=strongest['method'], pdhg_mean_seconds=strongest['mean_seconds'],
            primary_over_pdhg_mean_ratio=primary['mean_seconds']/strongest['mean_seconds'], matched_completed_tasks=len(shared),
            primary_faster_tasks=sum(primary_rows[s]['seconds'] < control_rows[s]['seconds'] for s in shared),
            descriptive_best_active=active['method'], best_active_mean_seconds=active['mean_seconds'],
            best_active_over_pdhg_mean_ratio=active['mean_seconds']/strongest['mean_seconds']))
        for name in names:
            g = index[name, schedule, order]
            rr = [r for r in rows if (r['method'], r['schedule'], r['ordering']) == (name, schedule, order)]
            for field in ['expanded', 'local_rejected', 'solver_coordinate_steps', 'solver_coordinate_steps_upper_bound']:
                assert g[field] == sum(r[field] for r in rr)
            assert g['completed'] == sum(r['completed'] for r in rr) and g['final_candidates'] == sum(r['remaining'] for r in rr)
            assert g['mean_seconds'] == math.fsum(r['seconds'] for r in rr)/16
            assert g['mean_returned_bytes'] == math.fsum(r['returned_array_bytes'] for r in rr)/16
            assert g['mean_named_bytes'] == math.fsum(r['named_bytes'] for r in rr)/16
            upper = g['solver_coordinate_steps_upper_bound']; actual = g['solver_coordinate_steps']
            saved = f'{100*(1-actual/upper):.2f}%' if upper else 'N/A'
            table_rows.append([schedule, order, name, g['completed'], f"{g['mean_seconds']:.4f}", g['expanded'], g['local_rejected'], g['final_candidates']])
            resource_rows.append([schedule, order, name, actual, upper, saved,
                                  f"{g['mean_named_bytes']/2**20:.3f}", f"{g['mean_returned_bytes']/2**20:.3f}"])
    out = base/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    run.save(out/'table_data.json', table_rows); run.save(out/'resource_data.json', resource_rows)
    run.save(out/'comparisons.json', comparisons); run.save(out/'figure_data.json', groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 10.2))
    for ax, (schedule, order) in zip(axes, run.model.SETTINGS):
        values = [index[name, schedule, order] for name in names]; yy = np.arange(len(names))
        colors = ['#147d72' if name == run.model.PRIMARY else '#4e7396' if name.startswith('pdhg_') else '#bac3cd' for name in names]
        ax.barh(yy, [g['mean_seconds'] for g in values], color=colors, height=.68)
        ax.set_yticks(yy, [name.replace('regional_', '').replace('pdhg_', 'PDHG ').replace('_', ' ') for name in names], fontsize=9)
        ax.invert_yaxis(); peak = max(g['mean_seconds'] for g in values)
        ax.set_xlim(0, peak*1.36)
        for y, g in enumerate(values):
            ax.text(g['mean_seconds']+peak*.015, y, f"{g['mean_seconds']:.3f}s | {g['completed']}/16", va='center', fontsize=8.5)
        ax.set_title(f'{schedule.upper()} | {order.replace("_", " ")}', loc='left', fontsize=13, pad=12)
        ax.set_xlabel('Mean full-search seconds | completed tasks', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
    fig.subplots_adjust(left=.14, right=.98, top=.9, bottom=.15, wspace=.52)
    fig.suptitle('Certified-row retirement: full search, all 17 configurations', fontsize=16, y=.967)
    fig.text(.14, .078, 'Fixed primary: active256 (green). Every solver shares retirement; blue bars are PDHG controls.', fontsize=10)
    fig.text(.14, .045, 'Old development tasks. Single interleaved run. Final geometry, posterior readout and query risk are not included.', fontsize=9)
    fig.savefig(out/'retired_search_cost.png', dpi=160); plt.close(fig)
    primary_wins = all(c['primary_complete'] == 16 and c['primary_over_pdhg_mean_ratio'] < 1 for c in comparisons)
    any_active_competitive = any(c['best_active_over_pdhg_mean_ratio'] < 1 for c in comparisons)
    conclusion = ('固定主在两个设置均全完成且本次平均搜索时间低于全部PDHG配置；这是需要重复计时和读出验证的开发成本信号，不是已确认任务收益。' if primary_wins else
        '固定主尚未在两个设置都低于最快的全完成PDHG控制；不能宣称统一强控制胜出。')
    if any_active_competitive and not primary_wins:
        conclusion += '预列active配置中至少有一个设置出现较好的平均成本区间，需保留其次要/开发选择身份并在后续独立验证。'
    reasons = Counter(r['stop_reason'] or 'complete' for r in rows)
    body = ['# 381｜严格证书行退役后的完整搜索成本', '',
        '本轮检验正向机制：主动乘子反馈能否用更充分的局部求证，减少虚假前缀的后续展开，并在强PDHG控制面前形成完整搜索成本优势。所有方法共用严格证书后行退役；它本身不归为PC独有创新。', '',
        '## 完整结果', '', *summaries, '', conclusion, '',
        '![17配置完整搜索成本与完成数](retired_search_cost.png)', '',
        table(['调度', '排序', '方法', '完成/16', '均搜索秒', '展开总数', '局部排除', '完整候选'], table_rows), '',
        '固定新主regional_active256，原主128保留。表中最优PDHG/active是预列配置之间的描述性最小值，不是免费在线选择器；不把事后最佳次要配置改称原主。若完成率不同，平均时间不能直接解释为相同成果的加速比。候选含未排除的不可行模式，不是正确解释数。', '',
        '## 数学上真正保住了什么', '',
        '每个区域行的状态更新只在本行内跨层和观察耦合，整个批映射因此是各行映射的直积。移除已严格证明不可行的行，不改变任何活行的输入状态和更新算子。归纳可得活行轨迹及首次证书与原实现相同；退役行停在首次证书时刻，不再定义旧cap终态。搜索只消费首次证书，故非时钟截断下的科学输出相同。', '',
        '若第i行首次证书在t_i，未证行记t_i=T，则实际坐标工作为d*n*sum_i(t_i)，旧全行上界为d*n*R*T。独立审计从存下的首次时刻和每检查点的活行数量重建实际费用。该计数不是FLOPs；各方法每步算术不同。拷贝/压缩活动批可能抵消墙钟节约，所以仍报告完整实际时间。', '',
        '证书依据仍是当前前缀盒域上局部残差线性组合的严格正下界。任何合法延伸限制到前缀都会使残差为零，故不能延伸被证不可行区域。向外舍入及独立Fraction核验保证剪枝安全；不保证有限步完备，不证明PC比所有其他求解器更快。', '',
        table(['调度', '排序', '方法', '实际坐标步', '全行坐标步上界', '省去比例', '均命名数组MiB', '均归档MiB'], resource_rows), '',
        '## 审计与信息边界', '',
        f"544调用全部封存后才读取旧可行模式用于审计。结束原因：{dict(reasons)}。预检包括{preflight['checks']['first_proof_and_initial_arrays']}个初始/首次证明数组、{preflight['checks']['permutation_arrays']}个行排列数组、{preflight['checks']['small_search_calls']}个小任务完整搜索及{preflight['checks']['real_search_calls']}个真实n24重放。", '',
        f"正式独立核验：{audit['checks']['exact_wide_domain_cuts']}个精确宽域剪枝证明、{audit['checks']['enumerated_combinations']}次组合展开、{audit['checks']['independent_live_checkpoints']}个活行检查点、{audit['checks']['actual_search_resource_checks']}次实际搜索费用，以及{audit['checks'].get('published_legacy_search_replays', 0)}次与已发布378搜索的逐位重放。可行模式输出或待访问子树覆盖{audit['checks'].get('positive_output_or_pending_coverage', 0)}次。", '',
        '旧审计器只认识全行跑满上界，因此仅对独立内存副本规范化上界，复用其枚举/证书核验；真正活动行费用另行核验，原始档案没有被改写。所有方法只见支持x/v，无查询答案、无LP/BP筛选捷径，几何只在封存后用于覆盖核验。', '',
        '共同65536展开、8秒协作批边界、20000状态阈值；沿用378的BFS/DFS状态计数差异，不宣称两调度严格峰值内存相同。同调度内所有方法规则相同。命名数组不是RSS，未覆盖所有临时复制峰值；全搜索时间包含守卫/语言/收缩/局部筛选/证明拷贝，不含写盘压缩、离线审计、最终几何和读出。', '',
        '## 下一关：把成本信号变成任务收益', '',
        '需按当前预列结果冻结候选，重复交错计时，并为候选及强控制接完全相同的几何和后验读出；未完树保留pending，不把空输出当空后验。在新冻结任务上比较独立查询风险和端到端费用，同时保留强回归、固定非线性/核、浅层头及同参数优化器和官方TTT。只有最终任务收益不能被这些控制解释，才构成目标要求的独立收益。当前goal仍ACTIVE。', '',
        '[执行前协议](../../../outputs/ttt-pc-alm-research/380_certified_retirement_protocol_v1.md) · [全部调用](../development_v1/rows.json) · [独立审计](../audit_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(body))
    entry = 'outputs/ttt-pc-alm-research/381_retired_join_results_v1.md'
    write(root/entry, '# 381｜退役完整搜索阶段结果\n\n'+'\n\n'.join(summaries)+'\n\n'+conclusion+
          '\n\n[完整图文与17配置](../../results/retired_region_join/report_v1/report.md)\n\n未开放新query，目标保持ACTIVE。\n')
    run.save(out/'manifest.json', dict(source_sha256=run.sha(Path(__file__)), entry_file=entry, entry_sha256=run.sha(root/entry),
        input_summary_sha256={p: run.sha(base/p) for p in ['preflight_v1/summary.json', 'development_v1/summary.json', 'audit_v1/summary.json']},
        outputs_sha256={f: run.sha(out/f) for f in ['table_data.json', 'resource_data.json', 'comparisons.json', 'figure_data.json', 'report.md', 'retired_search_cost.png']}))
    content = (out/'report.md').read_text(encoding='utf-8')
    for row in table_rows+resource_rows: assert '| '+' | '.join(map(str, row))+' |' in content
    run.save(out/'qa_numeric.json', dict(passed=True, table_cells=len(table_rows)*8+len(resource_rows)*8,
        groups=len(groups), raw_rows_recomputed=len(rows), primary_wins_both_settings=primary_wins,
        any_prelisted_active_competitive=any_active_competitive, manifest_sha256=run.sha(out/'manifest.json')))
    print(dict(passed=True, report=str(out/'report.md'), comparisons=comparisons), flush=True)


if __name__ == '__main__': main()
