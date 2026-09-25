"""328 presentation only: refuse quality access until prediction, scoring and audit finish.

This file is intentionally outside the frozen algorithm SOURCE lists. It never
imports a candidate, changes a method, scores a partial run, or marks the goal
complete. Run only after the timed prediction process has ended.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

BASE = 'results/online_credit_fresh_pilot'
CANDIDATES = ['online_first_fit_dual', 'online_uniform_state_dual_plus_residual']
METRICS = ['mse257', 'mse129', 'point_mse257', 'point_mse129']
REGRESSION = [
    'cold__linear_ls', 'cold__residual_linear_ls', 'cold__residual_linear_ridge',
    'cold__residual_linear_rls', 'cold__rbf_loocv', 'cold__prior4096_ridge',
    'cold__prior16384_ridge', 'cold__prior65536_ridge', 'cold__meta_ridge64',
    'cold__meta_ridge128',
]
SHALLOW = ['cold__meta_shallow64_5', 'cold__meta_shallow64_20']
COMMON_FIGURE_CONTROLS = ['credit_control_probe33', 'probe_then_adam1920_33',
    'probe_then_adam3840_33', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
PANEL = CANDIDATES + ['credit_control_probe33', 'probe_then_pc33', 'probe_pc256_33',
    'probe_nodual256_33', 'probe_then_adam960_33', 'probe_then_adam1920_33',
    'probe_then_adam3840_33'] + REGRESSION + SHALLOW
LABELS = {
    CANDIDATES[0]: 'First-fit / dual', CANDIDATES[1]: 'Uniform / dual + residual',
    'credit_control_probe33': 'Original probe-ALM',
    'probe_then_adam1920_33': 'Same-prefix Adam 1920',
    'probe_then_adam3840_33': 'Same-prefix Adam 3840',
    'cold__meta_ridge128': 'Meta-ridge 128',
    'cold__meta_shallow64_20': 'Meta-shallow 64 / 20',
    'cold__prior65536_ridge': 'Prior-feature ridge 65536',
}


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, value):
    content = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(content + '\n')


def write_text(path, value):
    with path.open('x', encoding='utf-8', newline='\n') as f:
        f.write(value)


def fmt(value):
    if value is None:
        return '—'
    assert math.isfinite(float(value))
    return f'{value:.9g}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
        '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
        ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows]) + '\n'


def same_trigger(candidate):
    prefix = 'online_first_fit_' if candidate == CANDIDATES[0] else 'online_uniform_state_'
    return [prefix + c for c in ['residual', 'bp', 'random_sign', 'zero']]


def comparison_groups(configs, candidate):
    # Membership is derived from frozen method types, never from query quality.
    return {
        '同触发的四种非乘子信用': same_trigger(candidate),
        '线性、核、固定特征及元训练岭回归': REGRESSION,
        '元训练浅层头': SHALLOW,
        '同前缀全部Adam步数': [c['name'] for c in configs if c['group'] == 'probe_adam'],
        '同前缀普通PC': [c['name'] for c in configs if c['group'] == 'probe_pc'],
        '同前缀无乘子局部更新': [c['name'] for c in configs if c['group'] == 'probe_nodual'],
    }


def gate(root, stage):
    base = root / BASE
    folders = {s: base / f'{stage}_{s}_v2' for s in ['predictions', 'evaluation', 'audit']}
    # Do not inspect methods, comparisons, NPZ, partial task commits or truth here.
    for folder in folders.values():
        if not (folder / 'summary.json').is_file() or (folder / 'failure.json').exists():
            raise RuntimeError(f'Complete successful seal required: {folder.name}')
    summaries = {key: read(folder / 'summary.json') for key, folder in folders.items()}
    for key, value in summaries.items():
        assert value['passed'], key
        if key != 'predictions':
            assert value['stage'] == stage, key
    ps, es, audit = (summaries[k] for k in ['predictions', 'evaluation', 'audit'])
    n = 512 if stage == 'pilot' else 3
    assert ps['tasks'] == es['tasks'] == n
    assert ps['predictors'] == es['predictors'] == audit['counts']['predictors'] == n * 51
    assert es['comparisons'] == audit['counts']['comparisons'] == 400
    assert es['main_family'] == 100 and not audit['core_research_goal_complete']
    assert audit['prediction_summary_sha256'] == sha(folders['predictions'] / 'summary.json')
    assert audit['evaluation_summary_sha256'] == sha(folders['evaluation'] / 'summary.json')
    for key in ['predictions', 'evaluation']:
        for name, digest in summaries[key]['outputs_sha256'].items():
            assert sha(folders[key] / name) == digest, (key, name)
    p = read(folders['predictions'] / 'protocol.json')
    ep = read(folders['evaluation'] / 'protocol.json')
    assert p['stage'] == ep['stage'] == stage
    assert p['methods'] == ep['methods'] and p['candidates'] == ep['candidates'] == CANDIDATES
    assert len(p['methods']) == len(set(p['methods'])) == 51
    expected = list(range(328000000, 328000512)) if stage == 'pilot' else [5910000, 5910001, 5910063]
    assert p['seeds'] == ep['seeds'] == expected
    assert not p['query_targets_accessed']
    assert not read(folders['predictions'] / 'before_query_manifest.json')['query_targets_accessed']
    for name, digest in p['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest, name
    for name, digest in audit['source_sha256'].items():
        assert sha(root / 'work/experiments' / name) == digest, name
    assert p['source_sha256'] == ep['source_sha256']
    for key, name in [('design_sha256', '328_fresh_online_credit_protocol_v1.md'),
                      ('serialization_revision_sha256', '328_serialization_correction_v2.md')]:
        assert p[key] == ep[key] == sha(root / 'outputs/ttt-pc-alm-research' / name)
    cal = root / 'results/online_credit_resources/calibration_v1'
    ca = read(root / 'results/online_credit_resources/audit_v1/summary.json')
    assert ca['passed'] and ca['calibration_summary_sha256'] == sha(cal / 'summary.json')
    assert p['resource_audit_sha256'] == sha(root / 'results/online_credit_resources/audit_v1/summary.json')
    cs = read(cal / 'summary.json')
    for name, digest in cs['outputs_sha256'].items():
        assert sha(cal / name) == digest, name
    assert p['frozen_resource_selection'] == read(cal / 'selection.json')
    return folders, summaries, p, ep, cal


def budget_label(name, selection):
    if name in selection['within_budget']:
        return '≤1.00'
    if name in selection['sensitivity_110_percent']:
        return '1.00–1.10'
    return '>1.10'


def category(cfg):
    name = cfg['name']
    if name in CANDIDATES:
        return 'Designated candidates'
    if cfg['family'] == 'online_credit':
        return 'Credit controls / variants'
    if name in REGRESSION:
        return 'Regression / kernel'
    if name in SHALLOW:
        return 'Shallow heads'
    if cfg['group'] in ['probe_adam', 'cold_adam']:
        return 'Adam'
    return 'Other local / branch controls'


def figures(out, methods, cfgs, comparisons, stage):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    byname = {r['method']: r for r in methods}
    cmp = {(r['candidate'], r['control']): r for r in comparisons if r['metric'] == 'mse257'}
    colors = {'Designated candidates': '#b42246', 'Credit controls / variants': '#b17cc9',
        'Regression / kernel': '#287c8e', 'Shallow heads': '#e0a32c', 'Adam': '#2872b8',
        'Other local / branch controls': '#838a91'}
    points = [dict(method=c['name'], category=category(c),
        seconds=byname[c['name']]['mean_seconds'], mse=byname[c['name']]['metrics']['mse257']) for c in cfgs]
    fig, ax = plt.subplots(figsize=(11.5, 6.6), layout='constrained')
    for group, color in colors.items():
        group_points = [p for p in points if p['category'] == group]
        ax.scatter([p['seconds'] for p in group_points], [p['mse'] for p in group_points],
            color=color, label=f'{group} (n={len(group_points)})', s=88 if group == 'Designated candidates' else 38,
            marker='*' if group == 'Designated candidates' else 'o', edgecolors='white', linewidths=.5, alpha=.85)
    labels = CANDIDATES + ['probe_then_adam1920_33', 'probe_then_adam3840_33',
        'cold__meta_ridge128', 'cold__meta_shallow64_20', 'cold__prior65536_ridge']
    offsets = [(16, 27), (16, -30), (-145, 34), (16, -16), (10, 15), (10, -20), (-140, 13)]
    for name, offset in zip(labels, offsets):
        p = byname[name]
        ax.annotate(LABELS[name], (p['mean_seconds'], p['metrics']['mse257']), xytext=offset,
            textcoords='offset points', fontsize=8, arrowprops=dict(arrowstyle='-', color='#777', lw=.6))
    ax.set_xscale('log'); ax.set_ylim(bottom=0)
    ax.set_xlabel('Complete fit + readout time per task (seconds; log scale)')
    ax.set_ylabel('Task-equal unseen-query MSE (lower is better)')
    ax.grid(alpha=.18); ax.legend(loc='upper right', fontsize=8)
    ax.set_title('All 51 frozen methods' + (' — OLD-TASK PREFLIGHT, NOT A RESEARCH RESULT' if stage == 'preflight' else ' — 512 fresh tasks'))
    fig.savefig(out / 'risk_vs_time.png', dpi=180); plt.close(fig)
    forest = []
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5), layout='constrained')
    for ax, candidate in zip(axes, CANDIDATES):
        controls = same_trigger(candidate) + COMMON_FIGURE_CONTROLS
        rows = [cmp[candidate, name] for name in controls]
        for i, row in enumerate(rows):
            mean = row['mean_difference']; lo, hi = row['descriptive95']; upper = row['bonferroni_upper']
            color = '#287c8e' if row['adjusted_negative'] else '#838a91'
            ax.hlines(i, lo, hi, color=color, lw=2)
            ax.scatter([mean], [i], color=color, s=28, zorder=3)
            ax.scatter([upper], [i], marker='>', color='#b42246', s=34, zorder=3)
            forest.append(dict(candidate=candidate, control=row['control'], mean=mean,
                descriptive95=[lo, hi], bonferroni_upper=upper, adjusted_negative=row['adjusted_negative']))
        short = [LABELS.get(n, n.removeprefix('online_first_fit_').removeprefix('online_uniform_state_')) for n in controls]
        ax.set_yticks(range(len(rows)), short, fontsize=8); ax.invert_yaxis()
        ax.axvline(0, color='#444', lw=1); ax.grid(axis='x', alpha=.18)
        ax.ticklabel_format(axis='x', style='sci', scilimits=(-2, 2))
        ax.set_xlabel('Candidate − control query MSE'); ax.set_title(LABELS[candidate], fontsize=11)
    fig.suptitle('Fixed comparison panel: dot = mean; bar = descriptive 95%; triangle = family-adjusted upper bound'
        + ('\nPREFLIGHT ONLY: 3 old tasks, no superiority inference' if stage == 'preflight' else ''), fontsize=11)
    fig.savefig(out / 'paired_comparisons.png', dpi=180); plt.close(fig)
    write_json(out / 'figure_data.json', dict(points=points, forest=forest,
        panel_selected_without_query_quality=True, omitted_from_full_tables=[]))


def build(root, out, stage, include_figures=True):
    live_predictions = root / BASE / 'pilot_predictions_v2'
    if live_predictions.exists() and not (live_predictions / 'summary.json').is_file():
        raise RuntimeError('Pilot prediction timing is incomplete; postpone presentation work.')
    folders, summaries, protocol, ep, cal = gate(root, stage)
    # Gate returns before any method-quality material is opened.
    ev = folders['evaluation']; methods = read(ev / 'methods.json')
    comparisons = read(ev / 'comparisons.json'); groups = read(ev / 'mechanism_groups.json')['groups']
    actual = read(ev / 'actual_costs.json'); calibrated = read(cal / 'methods.json')
    names = protocol['methods']; byname = {r['method']: r for r in methods}; cb = {r['method']: r for r in calibrated}
    cmp = {(r['candidate'], r['control'], r['metric']): r for r in comparisons}
    assert list(byname) == names and len(cmp) == 400 and set(cb) == set(names)
    assert all(name in names for name in PANEL) and len(set(PANEL)) == len(PANEL)
    out.mkdir(parents=True, exist_ok=False)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), stage=stage,
        report_source_sha256=sha(Path(__file__)), presentation_only=True,
        query_quality_used_to_select_methods=False, core_research_goal_complete=False,
        inputs_sha256={str(path.relative_to(root)).replace('\\', '/'): sha(path)
            for folder in folders.values() for path in [folder / 'summary.json']},
        inherited_evaluation_outputs_sha256=summaries['evaluation']['outputs_sha256'],
        fixed_display_panel=PANEL, fixed_regression_group=REGRESSION, fixed_shallow_group=SHALLOW,
        visual_qa_status='pending human/model inspection; not asserted by generation')
    write_json(out / 'provenance.json', provenance)
    all_rows = []
    for row in methods:
        name = row['method']
        all_rows.append([name] + [fmt(row['metrics'][m]) for m in METRICS] +
            [fmt(row[n]) for n in ['mean_seconds', 'median_seconds', 'p90_seconds', 'maximum_seconds']] +
            [row['failures'], cb[name]['maximum_traced_peak_bytes'], cb[name]['maximum_absolute_lifetime_peak_wset']])
    full = '# 全部51方法：固定顺序、不按查询质量筛选\n\n'
    if stage == 'preflight':
        full += '这是3个旧任务的报告管线预检，不能作研究效果结论。\n\n'
    full += table(['方法'] + METRICS + ['均时秒', '中位秒', 'P90秒', '最大秒', '失败',
        '327跟踪峰值字节', '327进程生命周期峰值字节'], all_rows)
    full += '\n327内存来自旧校准任务的独立进程，不是512新任务的峰值；进程峰值含共同加载及框架，跟踪峰值不覆盖全部native内存。\n'
    write_text(out / 'all_methods.md', full)
    comp_text = '# 全部100个主指标配对比较\n\n差值=候选−对照；负值有利于候选。95%为描述性区间；单侧上界采用100比较的预定Bonferroni分位数。\n\n'
    budget_text = '# 校准与实际运行预算：全部方法始终保留\n\n1.00和1.10均时分类不是相同FLOPs、相同状态或相同训练成本的证明。\n\n'
    summaries_by_group = []
    for candidate in CANDIDATES:
        comp_rows = []
        for control in names:
            if candidate == control:
                continue
            r = cmp[candidate, control, 'mse257']
            comp_rows.append([control, fmt(r['mean_difference']), fmt(r['descriptive95'][0]),
                fmt(r['descriptive95'][1]), fmt(r['bonferroni_upper']), str(r['adjusted_negative']).lower(),
                r['improved'], r['equal'], r['worse'], fmt(r['largest_absolute_share']),
                r['largest_absolute_task_seed'], fmt(r['leave_largest_absolute_out_mean']), fmt(r['worst_leave_one_out_mean'])])
        comp_text += '## ' + candidate + '\n\n' + table(['对照', '均值差', '95%下界', '95%上界',
            '校正单侧上界', '上界小于0', '改善', '相同', '变差', '最大绝对贡献占比', '对应任务', '剔除该任务均差', '最坏LOO均差'], comp_rows) + '\n'
        selection = protocol['frozen_resource_selection'][candidate]
        current = actual['candidates'][candidate]
        rows = [[name, fmt(cb[name]['mean_seconds'] / cb[candidate]['mean_seconds']), budget_label(name, selection),
            fmt(byname[name]['mean_seconds'] / byname[candidate]['mean_seconds']), budget_label(name, current)] for name in names]
        budget_text += '## ' + candidate + '\n\n' + table(['方法', '校准时间比', '校准分组', '本轮时间比', '本轮分组'], rows) + '\n'
        for group_name, controls in comparison_groups(protocol['configs'], candidate).items():
            rr = [cmp[candidate, c, 'mse257'] for c in controls]
            summaries_by_group.append(dict(candidate=candidate, group=group_name, controls=controls,
                adjusted_negative=sum(r['adjusted_negative'] for r in rr), total=len(rr),
                negative_mean=sum(r['mean_difference'] < 0 for r in rr),
                worst_adjusted_upper=max(r['bonferroni_upper'] for r in rr),
                all_worst_leave_one_out_negative=all(r['worst_leave_one_out_mean'] < 0 for r in rr)))
    write_text(out / 'all_primary_comparisons.md', comp_text)
    write_text(out / 'all_budgets.md', budget_text)
    write_json(out / 'comparison_groups.json', summaries_by_group)
    group_text = '# 支持信息定义的机制分组（仅描述，不替代总体）\n\n所有方法使用主候选定义的同一批分组任务，不基于查询质量选组，不作分组显著性声明。\n\n'
    for group in groups:
        rows = [[r['method']] + [fmt(r['means'][m]) for m in METRICS] for r in group['metrics']]
        group_text += f"## {group['group']}：{group['tasks']}任务\n\n" + table(['方法'] + METRICS, rows) + '\n'
    group_text += '[各组全部400比较、任务种子及改善/相同/变差计数](../' + stage + '_evaluation_v2/mechanism_groups.json)\n'
    write_text(out / 'mechanism_groups.md', group_text)
    if include_figures:
        figures(out, methods, protocol['configs'], comparisons, stage)
    ps, es, audit = (summaries[k] for k in ['predictions', 'evaluation', 'audit'])
    heading = '512新任务：在线乘子分支搜索的阶段结果' if stage == 'pilot' else '仅供报告管线预检：3个旧任务，不是研究结论'
    report = f'# {heading}\n\n'
    report += (f"完整预测、封存后评分及独立审计均已结束。{ps['tasks']}个任务、51方法、{ps['predictors']}份预测，数值执行回退{es['numerical_failures']}次。"
        '方法与评价协议在新任务运行前冻结；查询答案只进入封存后的评分。\n\n')
    if stage == 'preflight':
        report += '本文件所有效果数字来自旧任务，仅验证报告代码。即使出现负的校正上界，也不能作为新任务收益证据。\n\n'
    report += '## 两个候选的直接结果\n\n' + table(['候选', '查询MSE', '完整平均秒数', '失败次数'],
        [[c, fmt(byname[c]['metrics']['mse257']), fmt(byname[c]['mean_seconds']), byname[c]['failures']] for c in CANDIDATES])
    report += '\n下表分别回答信用信号是否有独立贡献、以及是否优于各类基线。“校正通过”仅指预定近似配对bootstrap单侧上界小于0，不是普遍支配定理。分组数量依据完整冻结名单，不按结果增删。\n\n'
    report += table(['候选', '对照类别', '负均差/总数', '校正通过/总数', '该组最差校正上界'],
        [[r['candidate'], r['group'], f"{r['negative_mean']}/{r['total']}",
            f"{r['adjusted_negative']}/{r['total']}", fmt(r['worst_adjusted_upper'])] for r in summaries_by_group])
    report += '\n“对回归好”与“乘子本身有独立收益”是不同命题；不能用前者替代后者。是否形成竞争性机制还需综合强优化器、全部信用对照、贡献集中度及资源，本文不自动把研究目标标记完成。\n\n'
    report += '## 事先固定的代表面板\n\n' + table(['方法', '257查询MSE', '完整平均秒数', '相对主候选时间'],
        [[n, fmt(byname[n]['metrics']['mse257']), fmt(byname[n]['mean_seconds']),
            fmt(byname[n]['mean_seconds'] / byname[CANDIDATES[0]]['mean_seconds'])] for n in PANEL])
    report += '\n[全部51方法、四种指标与时间尾部](all_methods.md)；[全部100个主比较及单任务剔除](all_primary_comparisons.md)；[全部校准/实际预算分类](all_budgets.md)。超预算方法也全部保留，尤其不能依靠Adam3840接近1.10的边界制造胜利。\n\n'
    if include_figures:
        report += '![全部方法的真实查询误差与完整用时](risk_vs_time.png)\n\n横轴为完整fit及读出的实测时间（对数轴），纵轴为任务等权查询MSE；所有51方法都在图中，图内只标注固定代表项。\n\n'
        report += '![预定信用及强基线对照的配对结果](paired_comparisons.png)\n\n圆点是均值差，横线是描述性95%区间，三角形是100主比较校正后的单侧上界。负值有利于候选；图只是固定面板，完整对照见表。\n\n'
    report += '## 机制与数学含义\n\n'
    report += ('目标瓶颈不是“回归不会反传”，而是有限预算下只找到部分支持一致的非线性参数区域：支持损失已经落入噪声带时，普通BP信用可归零，但动态乘子仍可保存不同的约束信用并改变有限候选的探索次序。旧开发任务已有可复核的正向排序实例；此轮直接检验它能否改善真实有限粒子预测。\n\n'
        '理想独立后验粒子读出满足 $R_M(T\\mid S)=R_{Bayes}(S)+\\|\\mu_T-\\mu_S\\|_Q^2+V_T/M$。新区域带来的均值偏差下降，必须超过增加的读出方差除以粒子数，才保证条件期望风险下降。它不创造额外支持信息，也不是对任意闭式解或任意浅层网络的无条件优势。\n\n'
        '严格假设、混合区域风险变化公式及精确有理数单元检验见[有限读出推导](../../../outputs/ttt-pc-alm-research/328_finite_readout_risk_addendum_v1.md)。数值LP、体积与采样是有容限的实现，不能把理想独立采样恒等式直接称为浮点程序的逐任务保证。\n\n')
    report += '支持分组任务数：' + '；'.join(f"{g['group']}={g['tasks']}" for g in groups) + '。分组只用支持信息。[完整描述性分组表](mechanism_groups.md)，不以其中某组替代全体结果。\n\n'
    report += '## 资源、复核与适用范围\n\n'
    report += ('全部调用计入共同准备、局部活动/乘子更新、搜索、全局LP、粒子与查询读出；读取模型及输入索引、文件保存等框架成本另列，不混入fit时间。非BP候选未借用BP初始化或BP对照状态。LP仍是全局几何计算，不宣称整个方法只有局部操作。\n\n'
        '内存沿用已审计327独立进程校准；生命周期峰值含共享库、预加载先验和框架，tracemalloc不覆盖全部native。新任务日志的named bytes不是总峰值。元训练岭回归/浅层头的checkpoint及外层训练先验单独披露，不当作免费标签。\n\n'
        '这是四层tent非线性、四偏置、有界噪声、4个支持点的固定分布新开发实验；不是官方TTT/GELU-MLP、LLM或VLM验证，也不是已经完成的论文确认。原官方TTT和更大任务验证仍属后续工作。\n\n')
    report += f"独立复核：{audit['counts']['scalar_risks']}个标量风险、{audit['counts']['independent_particle_predictors']}个新候选粒子读出、{audit['all_bootstrap_means_checked']}个bootstrap均值。\n\n"
    report += '\n'.join(f'- [{label}](../{stage}_{key}_v2/summary.json)' for key, label in
        [('predictions', '完整预测摘要'), ('evaluation', '评分摘要'), ('audit', '独立审计摘要')]) + '\n\n'
    report += f"评分摘要SHA256：{sha(ev / 'summary.json')}。\n\n报告为封存数据的展示层；[来源记录](provenance.json)。数值表须再通过独立报告审计，图须实际打开核验后交付。\n"
    write_text(out / 'report.md', report)
    generated = [p for p in out.iterdir() if p.is_file()]
    write_json(out / 'generation_summary.json', dict(passed=True, stage=stage, scientific_audit_passed=True,
        report_numeric_audit_pending=True, visual_qa_pending=include_figures, methods=51, main_comparisons=100,
        core_research_goal_complete=False, outputs_sha256={p.name: sha(p) for p in generated}))
    print(json.dumps(dict(report=str(out / 'report.md'), generated=True,
        independent_report_audit_pending=True, visual_qa_pending=include_figures), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', choices=['preflight', 'pilot'], required=True)
    ap.add_argument('--no-figures', action='store_true')
    args = ap.parse_args()
    project = Path(__file__).resolve().parents[2]
    build(project, project / BASE / f'{args.stage}_report_v1', args.stage, not args.no_figures)
