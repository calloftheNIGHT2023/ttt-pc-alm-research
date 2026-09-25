"""Independent 329 presentation audit; does not import the report generator.

Checks every displayed table value against sealed data, recomputes all method
means from task arrays, validates figure input numbers, links and provenance.
It does not claim that an unviewed image has passed visual QA.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def tables(path):
    section = ''
    result = []
    current = None
    for line in path.read_text(encoding='utf-8').splitlines():
        if line.startswith('## '):
            section = line[3:]
        if line.startswith('| ') and line.endswith(' |'):
            cells = [s.strip() for s in line[1:-1].split('|')]
            if all(re.fullmatch(r':?-+:?', s) for s in cells):
                continue
            if current is None:
                current = dict(section=section, header=cells, rows=[])
                result.append(current)
            else:
                assert len(cells) == len(current['header'])
                current['rows'].append(cells)
        else:
            current = None
    return result


def compare(cell, expected, counts):
    if expected is None:
        assert cell == '—'
    elif isinstance(expected, bool):
        assert cell == str(expected).lower()
    elif isinstance(expected, int):
        assert cell == str(expected), (cell, expected)
    elif isinstance(expected, float):
        assert cell == format(expected, '.9g'), (cell, expected)
    else:
        assert cell == expected, (cell, expected)
    counts['table_cells'] += 1


def check_rows(actual, expected, counts):
    assert len(actual) == len(expected)
    for got, want in zip(actual, expected):
        assert len(got) == len(want)
        for cell, value in zip(got, want):
            compare(cell, value, counts)


def label(name, selection):
    return '≤1.00' if name in selection['within_budget'] else '1.00–1.10' if name in selection['sensitivity_110_percent'] else '>1.10'


def run(root, stage):
    base = root / 'results/online_credit_fresh_pilot'; report = base / f'{stage}_report_v2'
    ev = base / f'{stage}_evaluation_v2'; pred = base / f'{stage}_predictions_v2'
    audit = read(base / f'{stage}_audit_v2/summary.json')
    assert audit['passed'] and audit['stage'] == stage
    assert audit['prediction_summary_sha256'] == sha(pred / 'summary.json')
    assert audit['evaluation_summary_sha256'] == sha(ev / 'summary.json')
    generated = read(report / 'generation_summary.json'); assert generated['passed'] and generated['stage'] == stage
    for name, digest in generated['outputs_sha256'].items():
        assert sha(report / name) == digest, name
    provenance = read(report / 'provenance.json')
    assert provenance['stage'] == stage and provenance['presentation_only']
    assert not provenance['core_research_goal_complete'] and not provenance['query_quality_used_to_select_methods']
    assert provenance['report_source_sha256'] == sha(root / 'work/experiments/report_online_credit_fresh_v2.py')
    for name, digest in provenance['inputs_sha256'].items():
        assert sha(root / name) == digest
    es = read(ev / 'summary.json')
    assert provenance['inherited_evaluation_outputs_sha256'] == es['outputs_sha256']
    for name, digest in es['outputs_sha256'].items():
        assert sha(ev / name) == digest
    p = read(pred / 'protocol.json'); names = p['methods']; candidates = p['candidates']; configs = p['configs']
    methods = read(ev / 'methods.json'); byname = {r['method']: r for r in methods}
    comparisons = read(ev / 'comparisons.json'); cmp = {(r['candidate'], r['control'], r['metric']): r for r in comparisons}
    metrics = ['mse257', 'mse129', 'point_mse257', 'point_mse129']
    cal = root / 'results/online_credit_resources/calibration_v1'
    calibrated = {r['method']: r for r in read(cal / 'methods.json')}
    ca = read(root / 'results/online_credit_resources/audit_v1/summary.json')
    assert ca['passed'] and ca['calibration_summary_sha256'] == sha(cal / 'summary.json')
    cs = read(cal / 'summary.json')
    assert sha(cal / 'methods.json') == cs['outputs_sha256']['methods.json']
    assert sha(cal / 'selection.json') == cs['outputs_sha256']['selection.json']
    assert p['frozen_resource_selection'] == read(cal / 'selection.json')
    counts = Counter()
    import numpy as np
    with np.load(ev / 'task_metrics.npz', allow_pickle=False) as z:
        risk = z['risk']; seconds = z['seconds']; failed = z['failures']
        assert z['methods'].tolist() == names and z['metric_names'].tolist() == metrics
        assert risk.shape == (len(p['seeds']), 51, 4)
        for j, row in enumerate(methods):
            for k, metric in enumerate(metrics):
                value = statistics.fmean(float(x) for x in risk[:, j, k])
                assert math.isclose(value, row['metrics'][metric], rel_tol=2e-12, abs_tol=2e-12)
                counts['independent_method_means'] += 1
            assert math.isclose(statistics.fmean(seconds[:, j]), row['mean_seconds'], rel_tol=2e-12)
            assert int(failed[:, j].sum()) == row['failures']
    t = tables(report / 'all_methods.md'); assert len(t) == 1
    expected = []
    for r in methods:
        name = r['method']; expected.append([name] + [r['metrics'][m] for m in metrics] +
            [r[m] for m in ['mean_seconds', 'median_seconds', 'p90_seconds', 'maximum_seconds']] +
            [r['failures'], calibrated[name]['maximum_traced_peak_bytes'], calibrated[name]['maximum_absolute_lifetime_peak_wset']])
    check_rows(t[0]['rows'], expected, counts)
    t = tables(report / 'all_primary_comparisons.md'); assert len(t) == 2
    for table, candidate in zip(t, candidates):
        assert table['section'] == candidate; expected = []
        for control in names:
            if control == candidate:
                continue
            r = cmp[candidate, control, 'mse257']
            expected.append([control, r['mean_difference'], *r['descriptive95'], r['bonferroni_upper'], r['adjusted_negative'],
                r['improved'], r['equal'], r['worse'], r['largest_absolute_share'], r['largest_absolute_task_seed'],
                r['leave_largest_absolute_out_mean'], r['worst_leave_one_out_mean']])
        check_rows(table['rows'], expected, counts)
    actual = read(ev / 'actual_costs.json'); t = tables(report / 'all_budgets.md'); assert len(t) == 2
    for tab, candidate in zip(t, candidates):
        assert tab['section'] == candidate
        expected = [[name, calibrated[name]['mean_seconds'] / calibrated[candidate]['mean_seconds'],
            label(name, p['frozen_resource_selection'][candidate]), byname[name]['mean_seconds'] / byname[candidate]['mean_seconds'],
            label(name, actual['candidates'][candidate])] for name in names]
        check_rows(tab['rows'], expected, counts)
    grouped = read(ev / 'mechanism_groups.json')['groups']; t = tables(report / 'mechanism_groups.md'); assert len(t) == 3
    for tab, group in zip(t, grouped):
        assert tab['section'] == f"{group['group']}：{group['tasks']}任务"
        check_rows(tab['rows'], [[r['method']] + [r['means'][m] for m in metrics] for r in group['metrics']], counts)
    regression = [c['name'] for c in configs if c['name'].startswith('cold__') and
        (c['name'].endswith(('_ls', '_ridge', '_rls', '_loocv')) or c['name'].startswith('cold__meta_ridge'))]
    assert set(regression) == set(provenance['fixed_regression_group']) and len(regression) == 10
    shallow = [c['name'] for c in configs if c['name'].startswith('cold__meta_shallow')]
    assert set(shallow) == set(provenance['fixed_shallow_group']) and len(shallow) == 2
    sg = read(report / 'comparison_groups.json'); assert len(sg) == 12
    for ci, candidate in enumerate(candidates):
        prefix = 'online_first_fit_' if ci == 0 else 'online_uniform_state_'
        expected_controls = [[prefix + c for c in ['residual', 'bp', 'random_sign', 'zero']],
            regression, shallow] + [[c['name'] for c in configs if c['group'] == g] for g in ['probe_adam', 'probe_pc', 'probe_nodual']]
        for i, controls in enumerate(expected_controls):
            r = sg[ci * 6 + i]; assert r['candidate'] == candidate and set(r['controls']) == set(controls)
            rr = [cmp[candidate, c, 'mse257'] for c in controls]
            assert r['total'] == len(rr) and r['adjusted_negative'] == sum(v['adjusted_negative'] for v in rr)
            assert r['negative_mean'] == sum(v['mean_difference'] < 0 for v in rr)
            assert r['worst_adjusted_upper'] == max(v['bonferroni_upper'] for v in rr)
            assert r['all_worst_leave_one_out_negative'] == all(v['worst_leave_one_out_mean'] < 0 for v in rr)
            counts['comparison_group_records'] += 1
    t = tables(report / 'report.md'); assert len(t) == 3
    check_rows(t[0]['rows'], [[c, byname[c]['metrics']['mse257'], byname[c]['mean_seconds'], byname[c]['failures']] for c in candidates], counts)
    check_rows(t[1]['rows'], [[r['candidate'], r['group'], f"{r['negative_mean']}/{r['total']}",
        f"{r['adjusted_negative']}/{r['total']}", r['worst_adjusted_upper']] for r in sg], counts)
    panel = candidates + ['credit_control_probe33', 'probe_then_pc33', 'probe_pc256_33', 'probe_nodual256_33',
        'probe_then_adam960_33', 'probe_then_adam1920_33', 'probe_then_adam3840_33'] + provenance['fixed_regression_group'] + provenance['fixed_shallow_group']
    assert panel == provenance['fixed_display_panel']
    check_rows(t[2]['rows'], [[n, byname[n]['metrics']['mse257'], byname[n]['mean_seconds'],
        byname[n]['mean_seconds'] / byname[candidates[0]]['mean_seconds']] for n in panel], counts)
    if (report / 'figure_data.json').exists():
        fig = read(report / 'figure_data.json')
        assert fig['panel_selected_without_query_quality'] and fig['omitted_from_full_tables'] == []
        assert [r['method'] for r in fig['points']] == names
        for r in fig['points']:
            assert r['mse'] == byname[r['method']]['metrics']['mse257'] and r['seconds'] == byname[r['method']]['mean_seconds']
            counts['figure_scalar_values'] += 2
        assert len(fig['forest']) == 18
        assert fig['display_revision_only'] is True
        for ci, candidate in enumerate(candidates):
            prefix = 'online_first_fit_' if ci == 0 else 'online_uniform_state_'
            controls = [prefix + c for c in ['residual', 'bp', 'random_sign', 'zero']] + ['credit_control_probe33',
                'probe_then_adam1920_33', 'probe_then_adam3840_33', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
            assert fig['zoom_controls'][candidate] == controls[:7]
            for i, control in enumerate(controls):
                r = fig['forest'][ci * 9 + i]; c = cmp[candidate, control, 'mse257']
                assert r['candidate'] == candidate and r['control'] == control
                assert r['mean'] == c['mean_difference'] and r['descriptive95'] == c['descriptive95']
                assert r['bonferroni_upper'] == c['bonferroni_upper'] and r['adjusted_negative'] == c['adjusted_negative']
                counts['figure_scalar_values'] += 4
    for path in report.glob('*.md'):
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)', path.read_text(encoding='utf-8')):
            assert '://' not in target and (path.parent / target).is_file(), (path.name, target)
            counts['local_links'] += 1
    text = (report / 'report.md').read_text(encoding='utf-8')
    if stage == 'preflight':
        assert '不是研究结论' in text and '不能作为新任务收益证据' in text
    for phrase in ['全部51方法', '全部100个主比较', '不是官方TTT', 'LP仍是全局几何计算', '不自动把研究目标标记完成']:
        assert phrase in text, phrase
    result = dict(passed=True, stage=stage, counts=dict(counts), visual_qa_passed=False,
        visual_qa_status='Separate actual image inspection is required', core_research_goal_complete=False,
        generation_summary_sha256=sha(report / 'generation_summary.json'),
        scientific_audit_summary_sha256=sha(base / f'{stage}_audit_v2/summary.json'),
        audit_source_sha256=sha(Path(__file__)))
    with (report / 'numeric_qa.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2, allow_nan=False)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('--stage', choices=['preflight', 'pilot'], required=True)
    args = ap.parse_args(); run(Path(__file__).resolve().parents[2], args.stage)

