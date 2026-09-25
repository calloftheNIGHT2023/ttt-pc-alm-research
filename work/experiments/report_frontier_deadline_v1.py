"""423 all-control figures and actual changed-coordinate delivery accounting."""
from collections import Counter, defaultdict
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import frontier_deadline_io_v1 as io

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT/io.BASE
OUT = BASE/'report_v1'
ENTRY = 'outputs/ttt-pc-alm-research/423_frontier_deadline_results_v1.md'


def main():
    evaluation = BASE/'evaluation_v1'; checked = io.read(evaluation/'summary.json')
    assert checked['passed'] and checked['calls'] == 1568 and checked['descriptive_only']
    for name, digest in checked['outputs_sha256'].items():
        assert io.sha(evaluation/name) == digest
    protocol = io.read(BASE/'development_v1/protocol.json')
    assert protocol['primary'] == io.PRIMARY and len(protocol['configs']) == 49
    names = [c['name'] for c in protocol['configs']]
    groups = io.read(evaluation/'groups.json'); averages = io.read(evaluation/'prefix_averages.json')
    rows = io.read(evaluation/'risk_rows.json'); comparisons = io.read(evaluation/'comparisons.json')
    with np.load(evaluation/'query_truth.npz', allow_pickle=False) as z:
        truth = dict(zip(map(int, z['seeds']), z['targets']))
    exact_risks = {}; delivery = []; checks = Counter()
    for row in rows:
        folder = ROOT/row['directory']; assert io.sha(folder/'outputs.npz') == row['files']['outputs.npz']
        outputs = io.load_arrays(folder/'outputs.npz'); pred = outputs['prediction']
        risk = float(np.mean(np.square(pred-truth[row['seed']])))
        assert abs(risk-row['query_mse']) < 1e-13
        exact_risks[row['method'], row['n'], row['budget'], row['seed']] = risk
        checks['raw_query_risks_recomputed'] += 1
        if not row['method'].startswith('frontier_'):
            continue
        assert io.sha(folder/'events.json') == row['files']['events.json']
        events = io.read(folder/'events.json')
        accepted = [(i, e) for i, e in enumerate(events) if e['received_seconds'] <= row['budget'] and e['kind'] in ['fallback', 'final']]
        processed = changed = changed_material = 0
        if not accepted:
            category = 'constant'
        else:
            index, event = accepted[-1]
            np.testing.assert_array_equal(pred, outputs[f'event_{index}_prediction'])
            if 'frontier_receipt' in event:
                processed = event['frontier_receipt']['processed']
                category = 'frontier_all_queries' if processed == 257 else 'frontier_partial_queries'
                cheap = outputs['event_1_prediction']
                changed = int(np.count_nonzero(pred != cheap))
                changed_material = int(np.count_nonzero(abs(pred-cheap) > 1e-12))
                assert changed_material <= changed <= processed
            elif event['kind'] == 'final':
                assert io.sha(folder/'state.json') == row['files']['state.json']
                state = io.read(folder/'state.json')
                category = 'completed_posterior' if state['branch'] == 'completed_posterior' else 'cheap'
                assert state['branch'] in ['completed_posterior', 'cheap_only']
            else:
                assert index in [0, 1]
                category = 'prior' if index == 0 else 'cheap'
        delivery.append(dict(method=row['method'], seed=row['seed'], n=row['n'], budget=row['budget'],
            category=category, range_queries_processed=processed, numeric_coordinates_changed_beyond_cheap=changed,
            coordinates_changed_more_than_1e_minus12_beyond_cheap=changed_material,
            processed_does_not_mean_changed=True, changed_does_not_mean_lower_task_risk=True))
        checks['new_delivery_receipt_classifications'] += 1
    for group in groups:
        value = math.fsum(exact_risks[group['method'], group['n'], group['budget'], s] for s in io.SEEDS)/4
        assert abs(value-group['mean_query_mse']) < 1e-13; checks['group_means_recomputed'] += 1
    for group in averages:
        values = [math.fsum(exact_risks[group['method'], n, group['budget'], s] for n in io.STAGES)/4 for s in io.SEEDS]
        assert max(abs(a-b) for a, b in zip(values, group['task_prefix_average_mse'])) < 1e-13
        assert abs(math.fsum(values)/4-group['mean_query_mse']) < 1e-13
        checks['prefix_means_recomputed'] += 1
    lookup = {(r['method'], r['n'], r['budget']): r for r in groups}
    avg = {(r['method'], r['budget']): r for r in averages}
    paired = {(r['control'], r['budget']): r for r in comparisons}
    matrices = {}; figures = []
    for budget in io.BUDGETS:
        values = np.array([[lookup[name, n, budget]['mean_query_mse'] for n in io.STAGES]+[avg[name, budget]['mean_query_mse']] for name in names])
        matrices[str(budget)] = values.tolist()
        fig, ax = plt.subplots(figsize=(14, 18), layout='constrained')
        im = ax.imshow(np.maximum(values, 1e-12), norm=LogNorm(vmin=max(1e-12, float(values.min())), vmax=float(values.max())),
                       cmap='viridis', aspect='auto')
        ax.set_yticks(range(len(names)), labels=names, fontsize=8)
        ax.set_xticks(range(5), labels=['n=4', 'n=8', 'n=16', 'n=24', '4-prefix mean'])
        ax.set_title(f'Continuous frontier readout | {budget:g}s deadline\nAll 49 configurations; four OLD tasks; descriptive development only', fontsize=12)
        for i in range(len(names)):
            for j in range(5):
                color = 'black' if im.norm(max(values[i, j], 1e-12)) > .7 else 'white'
                ax.text(j, i, f'{values[i,j]:.2e}', ha='center', va='center', fontsize=7.5, color=color)
        primary = names.index(io.PRIMARY)
        ax.get_yticklabels()[primary].set_color('#b34f00'); ax.get_yticklabels()[primary].set_fontweight('bold')
        fig.colorbar(im, ax=ax, label='Query MSE (log scale)', fraction=.035, pad=.02)
        file = f'all_methods_{int(budget*1000)}ms.png'; fig.savefig(OUT/file, dpi=160); plt.close(fig); figures.append(file)
    categories = ['constant', 'prior', 'cheap', 'frontier_partial_queries', 'frontier_all_queries', 'completed_posterior']
    labels = ['Constant', 'Initial prior', 'Cheap envelope', 'Partial frontier read', 'All frontier queries', 'Completed posterior']
    colors = ['#bdbdbd', '#7f7f7f', '#decbe4', '#fdb462', '#fb8072', '#80b1d3']
    extra = names[42:]; delivery_groups = []
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), layout='constrained', sharey=True)
    for ax, budget in zip(axes, io.BUDGETS):
        rows_by_name = {name: [r for r in delivery if r['method'] == name and r['budget'] == budget] for name in extra}
        totals = np.zeros(len(extra))
        for category, label, color in zip(categories, labels, colors):
            value = np.array([sum(r['category'] == category for r in rows_by_name[name]) for name in extra])
            ax.barh(range(len(extra)), value, left=totals, label=label, color=color)
            for j, n in enumerate(value):
                if n:
                    ax.text(totals[j]+n/2, j, str(n), ha='center', va='center', fontsize=8)
            totals += value
        assert np.array_equal(totals, np.full(len(extra), 16))
        ax.set_yticks(range(len(extra)), labels=[n.removeprefix('frontier_') for n in extra])
        ax.set_xlim(0, 16); ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_title(f'{budget:g}s: selected prediction source'); ax.set_xlabel('Calls out of 16 (4 tasks x 4 prefixes)')
        for name in extra:
            rr = rows_by_name[name]
            delivery_groups.append(dict(method=name, budget=budget, categories=dict(Counter(r['category'] for r in rr)),
                processed=sum(r['range_queries_processed'] for r in rr),
                changed=sum(r['numeric_coordinates_changed_beyond_cheap'] for r in rr),
                changed_gt_1e_minus12=sum(r['coordinates_changed_more_than_1e_minus12_beyond_cheap'] for r in rr)))
    axes[0].invert_yaxis()
    axes[1].legend(loc='upper center', bbox_to_anchor=(.2, -.18), ncol=3, fontsize=8)
    fig.suptitle('Delivery is not task success; processed query count is not changed prediction count', fontsize=12)
    file = 'new_method_delivery.png'; fig.savefig(OUT/file, dpi=160); plt.close(fig); figures.append(file)
    lines = ['# 423｜连续前沿读出的全控制结果与实际交付', '',
        '419完成1568次新预测、独立审计和封存后查询评估。四个任务均为旧开发任务，不是独立确认；主方法和0.5秒主预算没有更换。全部49配置保留。', '',
        '## 预列主方法与关键对照', '', '| 方法 | .25秒平均MSE | .5秒平均MSE |', '| --- | ---: | ---: |']
    highlighted = [io.PRIMARY, 'regional_active512_dfs_farthest_x', 'adam_gn64_256_anytime', 'adam_gn64_anytime',
                   'cohort_meta_ridge128', 'official_ttt_native_prior256_32_p4', 'cohort_meta_shallow64_20', *extra[1:]]
    for name in highlighted:
        lines.append(f"| {name} | {avg[name,.25]['mean_query_mse']:.12g} | {avg[name,.5]['mean_query_mse']:.12g} |")
    lines += ['', '差值为新主减对照，负值代表本轮平均误差更低；不自动代表显著性或独立贡献。', '',
              '| 预列关键对照 | .25秒配对均差 | .5秒配对均差 |', '| --- | ---: | ---: |']
    for name in highlighted[1:]:
        lines.append(f"| {name} | {paired[name,.25]['mean_paired_difference']:.12g} | {paired[name,.5]['mean_paired_difference']:.12g} |")
    for budget, figure in zip(io.BUDGETS, figures[:2]):
        lines += ['', f'## {budget:g}秒：全部49方法', '', f'![All methods {budget:g}s]({figure})']
    lines += ['', '## 实际收到的预测及非零修正', '', f'![Selected prediction sources]({figures[2]})', '',
        'frontier_all_queries表示读完257查询，不表示搜索已穷尽；completed_posterior也不宣称精确Bayes。所有数量来自截止前实际选中的包，未使用事后补完预测。', '',
        '| 新方法 | 预算 | 范围已处理查询数 | 相对廉价预测严格非零变化 | 变化大于1e-12 |',
        '| --- | ---: | ---: | ---: | ---: |']
    for row in delivery_groups:
        lines.append(f"| {row['method']} | {row['budget']} | {row['processed']} | {row['changed']} | {row['changed_gt_1e_minus12']} |")
    lines += ['', '处理数量不是实际改变数量，实际改变也不是任务风险下降证明；效果必须看上方独立查询MSE。417／418中的128/257仅指处理数，不应宣传为128个值都改变。', '',
        '## 费用和研究边界', '',
        '冷启动、归档传输、清理／重建单列；表中预算为暖在线截止。RSS采样不是严格等峰值RAM。新收据哈希和通信收费，完整几何分支仍使用全局LP。有限步同目标控制及独立新任务确认尚未完成。', '',
        '422分段编译未进入419；不能把之后的组件速度归给本表。没有NLP／CV／Graph真实模型训练结果，389三域六数据集要求继续保留。', '',
        '[全部原始风险表](../evaluation_v1/risk_rows.json) · [完整均值及资源](../evaluation_v1/groups.json) · [独立审计](../audit_v1/summary.json) · [冻结方案](../../../outputs/ttt-pc-alm-research/419_frontier_deadline_protocol_v1.md)']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    io.save(OUT/'delivery_rows.json', delivery); io.save(OUT/'delivery_groups.json', delivery_groups)
    io.save(OUT/'figure_data.json', dict(methods=names, matrices=matrices, delivery_groups=delivery_groups))
    (ROOT/ENTRY).write_text('# 423｜连续前沿读出全控制结果\n\n四旧任务、49配置、1568次新调用，描述性开发结果。\n\n[完整图文、配对差与交付记录](../../results/frontier_deadline/report_v1/report.md)\n', encoding='utf-8')
    io.save(OUT/'manifest.json', dict(source_sha256=io.sha(Path(__file__)),
        evaluation_summary_sha256=io.sha(evaluation/'summary.json'), entry_file=ENTRY, entry_sha256=io.sha(ROOT/ENTRY),
        outputs_sha256={f: io.sha(OUT/f) for f in ['report.md', 'figure_data.json', 'delivery_rows.json', 'delivery_groups.json', *figures]}))
    io.save(OUT/'qa_numeric.json', dict(passed=True, checks=dict(checks), figure_values=490,
        all_configurations_retained=True, manifest_sha256=io.sha(OUT/'manifest.json')))
    print(io.read(OUT/'qa_numeric.json'), flush=True)


if __name__ == '__main__':
    assert io.read(BASE/'evaluation_v1/summary.json')['passed']
    OUT.mkdir(parents=True, exist_ok=False)
    main()
