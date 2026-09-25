"""388 all-task deadline risk report. Run only after complete audit/evaluation."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import numpy as np
import deadline_risk_io_v1 as io
import deadline_risk_registry_v1 as registry
from report_search_radius_development_v1 import table, write


def main():
    root = Path(__file__).resolve().parents[2]; base = root/io.BASE; evaluation = base/'evaluation_v1'
    result = io.read(evaluation/'summary.json'); audit = io.read(base/'audit_v1/summary.json')
    assert result['passed'] and audit['passed']
    for p, h in result['outputs_sha256'].items(): assert io.sha(evaluation/p) == h
    before = io.read(evaluation/'before_query.json')
    assert before['prediction_summary_sha256'] == io.sha(base/'development_v1/summary.json')
    assert before['audit_summary_sha256'] == io.sha(base/'audit_v1/summary.json')
    protocol = io.read(base/'development_v1/protocol.json')
    io.verify_hashes(root, protocol['source_sha256']); io.verify_hashes(root, protocol['pretrained_sha256'])
    rr = io.read(evaluation/'risk_rows.json'); groups = io.read(evaluation/'groups.json'); comparisons = io.read(evaluation/'comparisons.json')
    truth = io.load_arrays(evaluation/'query_truth.npz'); truth_index = {int(s):t for s, t in zip(truth['seeds'], truth['targets'])}
    names = [c['name'] for c in registry.configs()]; configs = {c['name']:c for c in registry.configs()}
    by_group = defaultdict(list); prediction_sources = Counter(); row_checks = 0
    for row in rr:
        directory = root/row['directory']; assert io.sha(directory/'outputs.npz') == row['files']['outputs.npz']
        a = io.load_arrays(directory/'outputs.npz'); pred = a['prediction']
        assert row['query_mse'] == float(np.mean((pred-truth_index[row['seed']])**2)); row_checks += 1
        role = row['selected']
        if role == 'fallback': role = 'paid_prior'
        elif role == 'final':
            cfg = configs[row['method']]
            if cfg['kind'] == 'regional':
                m = io.read(directory/'state.json'); role = 'full_posterior' if m['full_candidate_set_resolved'] and m['readout_available'] else 'paid_prior'
            elif cfg['kind'] == 'optimizer':
                m = io.read(directory/'state.json')
                role = 'conditional_union' if m['readout_metadata'] is not None and m['readout_metadata']['readout_available'] else 'optimizer_point'
            else: role = cfg['kind']
        prediction_sources[row['method'], row['budget'], role] += 1
        by_group[row['method'], row['budget']].append(row)
    index = {(g['method'], g['budget']):g for g in groups}; main_rows = []; resource_rows = []; source_rows = []
    for budget in registry.BUDGETS:
        for name in names:
            g = index[name, budget]; rows = by_group[name, budget]
            assert len(rows) == 32 and g['mean_query_mse'] == math.fsum(r['query_mse'] for r in rows)/32
            for field, raw in [('mean_online_seconds','online_seconds'), ('mean_setup_seconds','setup_seconds'),
                               ('mean_archive_seconds','archive_seconds'), ('mean_controller_overrun_seconds','controller_overrun_seconds')]:
                assert g[field] == math.fsum(r[raw] for r in rows)/32
            assert g['mean_cleanup_seconds'] == math.fsum(r['cleanup_seconds']+r['task_cleanup_seconds'] for r in rows)/32
            assert g['max_controller_overrun_seconds'] == max(r['controller_overrun_seconds'] for r in rows)
            assert g['max_worker_sampled_rss'] == max(r['worker_max_sampled_rss'] for r in rows)
            assert g['final'] == sum(r['selected'] == 'final' for r in rows)
            counts = {role:prediction_sources[name, budget, role] for role in ['full_posterior', 'conditional_union', 'optimizer_point', 'regression', 'meta', 'paid_prior', 'constant']}
            assert sum(counts.values()) == 32
            main_rows.append([budget, name, f"{g['mean_query_mse']:.9f}", g['final'], counts['full_posterior'],
                counts['conditional_union'], counts['paid_prior'], counts['constant']])
            resource_rows.append([budget, name, f"{g['mean_online_seconds']:.6f}", f"{g['mean_setup_seconds']:.6f}",
                f"{g['mean_cleanup_seconds']:.6f}", f"{g['mean_archive_seconds']:.6f}",
                f"{g['max_controller_overrun_seconds']:.6f}", f"{g['max_worker_sampled_rss']/2**20:.2f}"])
            source_rows.append(dict(method=name, budget=budget, prediction_sources=counts))
    comparison_rows = []
    lookup = {(r['method'], r['budget'], r['seed']):r for r in rr}
    other_names = [n for n in names if n != registry.PRIMARY]
    cmp_index = {(c['control'], c['budget']):c for c in comparisons}
    for bi, budget in enumerate(registry.BUDGETS):
        for ci, control in enumerate(other_names):
            c = cmp_index[control, budget]
            delta = np.array([lookup[registry.PRIMARY, budget, s]['query_mse']-lookup[control, budget, s]['query_mse'] for s in registry.SEEDS])
            assert c['mean_paired_difference'] == float(delta.mean())
            rng = np.random.default_rng(np.random.SeedSequence([387199, bi, ci]))
            boot = delta[rng.integers(0, 32, (100000, 32))].mean(1)
            assert c['bootstrap95'] == np.quantile(boot, [.025, .975]).tolist()
            assert c['bonferroni_one_sided_upper'] == float(np.quantile(boot, 1.-.05/35))
            comparison_rows.append([budget, control, '*' if control in registry.MAIN_CONTROLS else '',
                f"{c['mean_paired_difference']:.9f}", f"[{c['bootstrap95'][0]:.9f}, {c['bootstrap95'][1]:.9f}]",
                f"{c['bonferroni_one_sided_upper']:.9f}", c['primary_better_tasks'], c['exact_equal_tasks']])
    primary = index[registry.PRIMARY, registry.PRIMARY_BUDGET]
    external = [g for g in groups if g['budget'] == registry.PRIMARY_BUDGET and not g['method'].startswith('regional_active')]
    best = min(external, key=lambda g:g['mean_query_mse'])
    sentence = f"固定主0.5秒预算平均查询MSE为{primary['mean_query_mse']:.9f}；外部控制中最低均值为{best['method']}，MSE={best['mean_query_mse']:.9f}。"
    passed_means = result['primary_budget_lower_mean_all_eight_main_controls']
    passed_bounds = result['primary_budget_adjusted_upper_below_zero_all_eight']
    primary_comparisons = [cmp_index[c, registry.PRIMARY_BUDGET] for c in registry.MAIN_CONTROLS]
    assert len(primary_comparisons) == 8
    assert passed_means == all(c['mean_paired_difference'] < 0 for c in primary_comparisons)
    assert passed_bounds == all(c['bonferroni_one_sided_upper'] < 0 for c in primary_comparisons)
    verdict = f"八个预列主要对照的均值差全部为负：{passed_means}；35比较分配后的单侧上界对八个主要对照均低于零：{passed_bounds}。这仍只是32新开发任务的结果，不是独立确认或完整论文结论。"
    out = base/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    io.save(out/'tables.json', dict(risk=main_rows, resources=resource_rows, comparisons=comparison_rows))
    io.save(out/'prediction_sources.json', source_rows)
    selected = [registry.PRIMARY, *registry.MAIN_CONTROLS]
    figure_data = [index[name, budget] for budget in registry.BUDGETS for name in selected]; io.save(out/'figure_data.json', figure_data)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels = ['Active1024 BFS (primary)', 'Prior65536 ridge', 'Meta-ridge128', 'Meta-shallow20',
        'Adam240 / 256 / union', 'Gauss-Newton40 / 256 / union', 'PDHG cold1024 BFS', 'PDHG box1024 DFS', 'Passive1024 BFS']
    fig, axes = plt.subplots(1, 2, figsize=(14.6, 7.5))
    for ax, budget in zip(axes, registry.BUDGETS):
        values = [index[name, budget]['mean_query_mse'] for name in selected]; yy = np.arange(len(values))
        ax.barh(yy, np.maximum(values, 1e-16), color=['#b76432']+['#527998']*8, height=.64)
        ax.set_xscale('log'); ax.set_yticks(yy, labels, fontsize=9); ax.invert_yaxis()
        ax.set_xlabel('Mean unseen-query MSE (lower is better)', fontsize=10)
        ax.set_title(f'{budget:g}s receipt deadline | 32 tasks', loc='left', fontsize=13)
        ax.spines[['top','right']].set_visible(False); ax.grid(axis='x', alpha=.15); ax.set_axisbelow(True)
        for y, value in enumerate(values): ax.annotate(f'{value:.5g}', (max(value,1e-16), y), xytext=(5,0), textcoords='offset points', va='center', fontsize=9)
        ax.set_xlim(max(min(values)*.3, 1e-17), max(values)*4)
    fig.subplots_adjust(left=.21, right=.97, top=.84, bottom=.19, wspace=.88)
    fig.suptitle('New-task deadline prediction: fixed primary and eight predeclared controls', fontsize=14.5, y=.95)
    fig.text(.21, .07, 'All 32 tasks retained, including timeouts. Paid fallback. Query targets generated after seals and audit.\n'
        'Primary budget: 0.5s. 1s is secondary. All 36 methods and uncertainty intervals are in the report.', fontsize=9.2, linespacing=1.6)
    fig.savefig(out/'deadline_query_risk.png', dpi=160); plt.close(fig)
    body = ['# 388｜新冻结任务：完整链截止时间与查询误差', '', sentence, '', verdict, '',
        '![预列主方法与八个控制](deadline_query_risk.png)', '',
        '## 全部任务与全部36配置', '',
        '32个新任务×36配置×两个独立执行预算=2304次调用。0.5秒是预列主预算，1秒为次要；没有事后删除超时任务或改选主方法。查询目标在所有预测封存并通过原始输出审计后才产生。', '',
        table(['预算秒','方法','均查询MSE','及时最终交付/32','完整后验/32','条件模式/32','实际备用/32','常数/32'], main_rows), '',
        '及时最终交付不等于完整后验：分支方法即使搜索函数结束，也可能因为未完整解析继续采用备用预测。本表从已封存状态另行区分实际来源；这种报告分类不参与模型选择。优化器的条件模式读出不宣称覆盖所有可行模式。', '',
        '## 配对不确定性与预列控制', '',
        table(['预算秒','控制','预列主要','主减控制MSE','任务bootstrap95%','Bonferroni单侧上界','主更好/32','逐任务相等/32'], comparison_rows), '',
        '100000次任务级配对bootstrap，独立样本数32；两个预算不是64独立任务。单侧上界使用1-.05/35分位数，是小样本近似百分位bootstrap，不是有限样本精确保证。星号为执行前指定的八个主要控制，其他全部控制也保留。', '',
        '## 费用与部署边界', '',
        table(['预算秒','方法','均决策秒','均启动/重建秒','均清理秒','均归档传输秒','最大决策超时秒','最大观测工作进程MiB'], resource_rows), '',
        '在线期限从发送支持数据计起，仅采用期限前完整收到的预测；初始化不见支持数据。成功作业复用不变运行时/共享模型，任务状态清除；超时仅终止本实验创建的工作进程并在下一作业重建。这里同时列出启动与清理成本，不能把温运行延迟宣传为冷启动吞吐。块尾正常关闭费用另见sessions.json，不要把groups.json重复引用的method_total_nonforced_session_close_seconds跨两个预算重复相加。', '',
        '5ms资源采样计入在线费用。观测RSS最大值是实际在线峰值下界，Windows lifetime peak_wset包含初始化及先前作业，共享页未去重；没有强制等峰值RAM。完整命名模型/快权重/活动/乘子/搜索状态见逐作业state.json和state.npz。读出几何LP付费，不把整体称为只有局部PC计算。', '',
        '## 机制解释的适用范围', '',
        '共同完整后验近似p和备用r下，预列风险差为E[(I_A-I_C)((r-m)^2-(p-m)^2)]。因此真正证据是相同预算下所有任务的查询损失，而非更多证书或更多可更新参数。数值几何与有限粒子不等于精确Bayes均值；固定先验已知这一任务条件必须保留。', '',
        '本轮保留大固定特征回归、浅层头训练轨迹、64/256初始化强优化器，以及双方可用的相同几何模式读取。各优化器支持带可行性和选点规则相同，但有限步代理目标不逐项一致；不把多模式读出本身归因于信用分配。外层学习的四控制各使用32000训练任务、2048000查询标签，已知生成先验对所有方法可用；没有本轮再训练。', '',
        '匹配当前分布的官方TTT尚未重训，本轮不含该充分对照，也没有真实LLM/VLM任务验证。无论这张表的方向如何，都不能用它单独完成完整论文目标；正向结果需要独立新任务确认，未达预列比较则保留本轮并据实际机制改进，不改写主任务或隐藏强控制。', '',
        f"无查询答案的输出审计计数：{audit['checks']}。本报告重新读取全部2304预测计算MSE，并重算70个配对bootstrap结果。", '',
        '[冻结协议](../../../outputs/ttt-pc-alm-research/387_new_task_deadline_risk_protocol_v1.md) · [全部预测记录](../development_v1/rows.json) · [进程生命周期与额外关闭费](../development_v1/sessions.json) · [独立输出审计](../audit_v1/summary.json) · [查询评估](../evaluation_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(body))
    entry = 'outputs/ttt-pc-alm-research/388_new_task_deadline_risk_results_v1.md'
    write(root/entry, '# 388｜新任务截止时间预测结果\n\n'+sentence+'\n\n'+verdict+
        '\n\n[全部36配置、图文、区间与费用](../../results/new_task_deadline_risk/report_v1/report.md)\n\n核心目标未完成。\n')
    io.save(out/'manifest.json', dict(source_sha256=io.sha(Path(__file__)), entry_file=entry, entry_sha256=io.sha(root/entry),
        input_sha256={p:io.sha(base/p) for p in ['development_v1/summary.json','audit_v1/summary.json','evaluation_v1/summary.json']},
        outputs_sha256={p:io.sha(out/p) for p in ['tables.json','prediction_sources.json','figure_data.json','deadline_query_risk.png','report.md']}))
    report = (out/'report.md').read_text(encoding='utf-8')
    for row in main_rows+resource_rows+comparison_rows: assert '| '+' | '.join(map(str,row))+' |' in report
    io.save(out/'qa_numeric.json', dict(passed=True, raw_prediction_mses_recomputed=row_checks,
        bootstrap_comparisons_recomputed=70, table_cells=(len(main_rows)+len(resource_rows)+len(comparison_rows))*8,
        manifest_sha256=io.sha(out/'manifest.json')))
    print(dict(passed=True, sentence=sentence, verdict=verdict, report=str(out/'report.md')), flush=True)


if __name__ == '__main__': main()
