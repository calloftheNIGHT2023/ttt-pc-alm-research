"""427 query labels remain evaluator-only; all 56 configurations reported."""
from collections import defaultdict
import math
from pathlib import Path
import numpy as np
import compiled_frontier_deadline_io_v1 as io


def evaluate(root, out):
    source = root/io.BASE/'development_v1'; audit = root/io.BASE/'audit_v1'
    ss, au = io.read(source/'summary.json'), io.read(audit/'summary.json')
    assert ss['passed'] and au['passed'] and au['prediction_summary_sha256'] == io.sha(source/'summary.json')
    for folder, summary in [(source, ss), (audit, au)]:
        for f, digest in summary['outputs_sha256'].items():
            assert io.sha(folder/f) == digest
    protocol = io.read(source/'protocol.json')
    io.old.verify_hashes(root, protocol['source_sha256']); io.old.verify_hashes(root, protocol['pretrained_sha256'])
    rows = io.read(source/'rows.json'); assert len(rows) == 1792 and protocol['stage'] == 'run'
    io.save(out/'before_query.json', dict(prediction_summary_sha256=io.sha(source/'summary.json'),
        audit_summary_sha256=io.sha(audit/'summary.json'), protocol_sha256=io.sha(source/'protocol.json'),
        old_development_tasks=True, evaluator_sha256=io.sha(Path(__file__))))
    import streaming_branch_projection as base
    q = np.linspace(0., 1., 257)
    truth = {s: base.forward(q, np.random.default_rng(s).uniform(-.12, .12, 4)) for s in io.SEEDS}
    np.savez_compressed(out/'query_truth.npz', q=q, seeds=np.array(io.SEEDS), targets=np.stack([truth[s] for s in io.SEEDS]))
    risk = []; groups = defaultdict(list); task_curves = defaultdict(list)
    for row in rows:
        folder = root/row['directory']; assert io.sha(folder/'outputs.npz') == row['files']['outputs.npz']
        prediction = io.load_arrays(folder/'outputs.npz')['prediction']
        mse = math.fsum(float(z)**2 for z in prediction-truth[row['seed']])/len(q)
        record = dict(**row, query_mse=mse)
        risk.append(record); groups[row['method'], row['n'], row['budget']].append(record)
        task_curves[row['method'], row['budget'], row['seed']].append(record)
    grouped = []
    for (method, n, budget), rr in groups.items():
        assert sorted(r['seed'] for r in rr) == io.SEEDS
        grouped.append(dict(method=method, n=n, budget=budget, old_tasks=4,
            mean_query_mse=math.fsum(r['query_mse'] for r in rr)/4,
            final=sum(r['selected'] == 'final' for r in rr), fallback=sum(r['selected'] == 'fallback' for r in rr),
            constant=sum(r['selected'] == 'constant' for r in rr),
            mean_online_seconds=math.fsum(r['online_seconds'] for r in rr)/4,
            mean_setup_seconds=math.fsum(r['setup_seconds'] for r in rr)/4,
            mean_cleanup_seconds=math.fsum(r['cleanup_seconds']+r['task_cleanup_seconds'] for r in rr)/4,
            mean_archive_seconds=math.fsum(r['archive_seconds'] for r in rr)/4,
            max_controller_overrun_seconds=max(r['controller_overrun_seconds'] for r in rr),
            max_worker_sampled_rss=max(r['worker_max_sampled_rss'] for r in rr)))
    curves = {}
    for key, rr in task_curves.items():
        assert sorted(r['n'] for r in rr) == io.STAGES
        curves[key] = math.fsum(r['query_mse'] for r in rr)/4
    averaged = []; comparisons = []
    for cfg in protocol['configs']:
        for budget in io.BUDGETS:
            values = [curves[cfg['name'], budget, s] for s in io.SEEDS]
            averaged.append(dict(method=cfg['name'], budget=budget, task_prefix_average_mse=values,
                                  mean_query_mse=math.fsum(values)/4, old_tasks=4))
            if cfg['name'] != io.PRIMARY:
                delta = [curves[io.PRIMARY, budget, s]-curves[cfg['name'], budget, s] for s in io.SEEDS]
                comparisons.append(dict(primary=io.PRIMARY, control=cfg['name'], budget=budget,
                    primary_budget=budget == io.PRIMARY_BUDGET, per_task_difference=delta,
                    mean_paired_difference=math.fsum(delta)/4, descriptive_only=True, significance_test_performed=False))
    assert len(grouped) == 448 and len(averaged) == 112 and len(comparisons) == 110
    for name, value in [('risk_rows.json', risk), ('groups.json', grouped),
                        ('prefix_averages.json', averaged), ('comparisons.json', comparisons)]:
        io.save(out/name, value)
    lookup = {(r['method'], r['n'], r['budget']): r for r in grouped}
    avg = {(r['method'], r['budget']): r for r in averaged}
    lines = ['# 427｜共享编译前沿读出：56配置完整开发结果', '',
        '四旧任务、四支持前缀、两个在线预算、1792次新调用。全部预测先封存并通过独立审计，再生成本轮查询答案。是旧任务开发，不是新任务确认或统计显著性证据。', '',
        '预列主方法compiled_frontier_regional_active512，主预算仍0.5秒；原49配置全部保留。新七族共用现场编译范围与收费收据。', '']
    for budget in io.BUDGETS:
        lines += [f'## {budget:g}秒', '', '| Method | n4 MSE | n8 MSE | n16 MSE | n24 MSE | Prefix average | Final / 16 |',
                  '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for cfg in protocol['configs']:
            name = cfg['name']; values = [lookup[name, n, budget]['mean_query_mse'] for n in io.STAGES]
            values.append(avg[name, budget]['mean_query_mse'])
            completed = sum(lookup[name, n, budget]['final'] for n in io.STAGES)
            lines.append('| '+name+' | '+' | '.join(f'{v:.12g}' for v in values)+f' | {completed} |')
    lines += ['', '## 费用与证据边界', '',
        '所有调用的setup、在线截止接收、归档传输、清理／重建、RSS均保留。暖在线预算不等于同冷启动总费用；采样RSS不是严格同峰值RAM。含收费全局几何LP，有限步目标尚未严格统一。', '',
        '带收据的中间包在worker被杀后仍由原展开切点和数值哈希重建、独立证明覆盖并重放预测；审计补算不替代原截止预测。fallback标签可以表示已完成部分查询的有效向量，并非总是初始备用。', '',
        '本表不改变405或419原结果，不提前证明PC-ALM独立优势，不替代409同目标消融或389三域真实模型要求。', '',
        '[逐任务风险](risk_rows.json) · [全部配对](comparisons.json) · [全部资源组表](groups.json) · [独立审计](../audit_v1/summary.json) · [冻结配置](../development_v1/protocol.json)']
    # A generated result artifact, not a source/protocol edit.
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    summary = dict(passed=True, calls=len(rows), old_tasks=4, groups=len(grouped), prefix_averages=len(averaged),
        comparisons=len(comparisons), descriptive_only=True, independent_confirmation=False, core_research_goal_complete=False,
        outputs_sha256={f: io.sha(out/f) for f in ['before_query.json', 'query_truth.npz', 'risk_rows.json',
            'groups.json', 'prefix_averages.json', 'comparisons.json', 'report.md']})
    io.save(out/'summary.json', summary); print(summary, flush=True)
