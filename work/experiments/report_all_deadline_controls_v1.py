"""392 complete 36-control figures; descriptive display, no new selection test."""
from pathlib import Path
import math
import numpy as np
import deadline_risk_io_v1 as io
import deadline_risk_registry_v1 as registry


def main():
    root = Path(__file__).resolve().parents[2]; base = root/io.BASE
    evaluated = io.read(base/'evaluation_v1/summary.json')
    assert evaluated['passed']
    qa = io.read(base/'report_v1/qa_numeric.json')
    assert qa['passed'] and qa['manifest_sha256'] == io.sha(base/'report_v1/manifest.json')
    for name, digest in evaluated['outputs_sha256'].items():
        assert io.sha(base/'evaluation_v1'/name) == digest
    raw = io.read(base/'evaluation_v1/risk_rows.json')
    groups = io.read(base/'evaluation_v1/groups.json')
    index = {(g['method'], g['budget']):g for g in groups}
    methods = [cfg['name'] for cfg in registry.configs()]
    for key, group in index.items():
        rows = [row for row in raw if (row['method'], row['budget']) == key]
        assert len(rows) == 32
        assert group['mean_query_mse'] == math.fsum(row['query_mse'] for row in rows)/32
    out = base/'all_controls_report_v1'; out.mkdir(parents=True, exist_ok=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    image_files = []
    for budget in registry.BUDGETS:
        values = [index[name, budget]['mean_query_mse'] for name in methods]
        fig, ax = plt.subplots(figsize=(12.8, 12.8))
        lower = max(min(values)*.3, 1e-18); upper = max(values)*8
        ax.barh(np.arange(len(values)), np.maximum(values, 1e-17), height=.66,
                color=['#b76432' if name == registry.PRIMARY else '#527998' for name in methods])
        ax.set_xscale('log'); ax.set_xlim(lower, upper)
        ax.set_yticks(np.arange(len(values)), methods, fontsize=8.5)
        ax.invert_yaxis(); ax.spines[['right','top']].set_visible(False)
        ax.set_xlabel('Mean unseen-query MSE, all 32 tasks (lower is better)', fontsize=11)
        ax.grid(axis='x', alpha=.17); ax.set_axisbelow(True)
        for i, value in enumerate(values):
            ax.annotate(f'{value:.6g}', (max(value, 1e-17), i), xytext=(5, 0),
                        textcoords='offset points', va='center', fontsize=8.5)
        scope = 'primary budget' if budget == registry.PRIMARY_BUDGET else 'secondary budget'
        fig.suptitle(f'All 36 frozen configurations | {budget:g}s receipt deadline ({scope})', fontsize=14, y=.975)
        fig.text(.38, .04, 'Orange: original primary. Both 64- and 256-restart optimizers are retained.\n'
                 'Same 32 tasks; late predictions excluded; paid fallback included.\n'
                 'Descriptive full display, not a new primary-method choice or hypothesis test.', fontsize=9.3)
        fig.subplots_adjust(left=.38, right=.98, bottom=.115, top=.94)
        name = f'all_controls_{int(budget*1000)}ms.png'
        fig.savefig(out/name, dpi=150); plt.close(fig); image_files.append(name)
    primary = index[registry.PRIMARY, registry.PRIMARY_BUDGET]
    bp = index['adam240_r64_posterior_union', registry.PRIMARY_BUDGET]
    ridge = index['meta_ridge128', registry.PRIMARY_BUDGET]
    passive = index['regional_passive1024_bfs_observed', registry.PRIMARY_BUDGET]
    summary_sentence = (f"0.5秒原主MSE={primary['mean_query_mse']:.12g}，meta-ridge128={ridge['mean_query_mse']:.12g}，"
        f"被动BFS={passive['mean_query_mse']:.12g}；64初始化Adam联合读出={bp['mean_query_mse']:.12g}。"
        '因此本轮有相对这些回归／被动控制的正向收益，但没有优于同参数强BP的独立优势。')
    body = ['# 392｜新任务截止时间：全部36配置可视化', '', summary_sentence, '',
        '这两张图补充388预列八主对照图；每个原配置均显示，没有更换主方法、删除结果或追加显著性检验。', '',
        '## 原主预算：0.5秒', '', '![全部36配置，0.5秒](all_controls_500ms.png)', '',
        '## 原次要预算：1秒', '', '![全部36配置，1秒](all_controls_1000ms.png)', '',
        '64初始化与256初始化都必须保留：更多重启在本轮固定截止时间下会增加模式几何处理成本，不自动成为更强的实际交付基线。不能只凭256初始化控制超时的结果宣布战胜BP。', '',
        '391机制分解的404个及时完整区域预测逐位相同，576个逐任务差异恒等式重构误差为零。区域方法之间的风险差完全由完整交付时间决定；这证明本轮成本到风险的通道确实存在，不等于完整搜索优于只寻找足够解释的强优化器。', '',
        '[全部数表、配对区间与费用](../report_v1/report.md) · [预查询冻结的机制分解](../mechanism_decomposition_v1/report.md)', '']
    (out/'report.md').write_text('\n'.join(body), encoding='utf-8')
    io.save(out/'figure_data.json', [index[name, budget] for budget in registry.BUDGETS for name in methods])
    entry = root/'outputs/ttt-pc-alm-research/392_all_deadline_controls_results_v1.md'
    entry.write_text('# 392｜完整强对照图\n\n'+summary_sentence+
                    '\n\n[全部36配置图与解释](../../results/new_task_deadline_risk/all_controls_report_v1/report.md)\n', encoding='utf-8')
    manifest = dict(source_sha256=io.sha(Path(__file__)), evaluation_summary_sha256=io.sha(base/'evaluation_v1/summary.json'),
                    parent_numeric_qa_sha256=io.sha(base/'report_v1/qa_numeric.json'),
                    entry_file=entry.relative_to(root).as_posix(), entry_sha256=io.sha(entry),
                    outputs_sha256={name:io.sha(out/name) for name in ['report.md','figure_data.json',*image_files]})
    io.save(out/'manifest.json', manifest)
    io.save(out/'qa_numeric.json', dict(passed=True, means_recomputed=len(index), plotted_values=72,
                                       all_frozen_configurations_in_both_figures=True,
                                       manifest_sha256=io.sha(out/'manifest.json')))
    print(dict(passed=True, report=str(out/'report.md'), plotted_values=72), flush=True)


if __name__ == '__main__':
    main()
