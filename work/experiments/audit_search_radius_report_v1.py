"""341 independently check every rendered table cell, plot datum and local link."""
import hashlib
import json
from pathlib import Path
import re
from PIL import Image

BASE = 'results/search_radius_development'
PRIMARY = 'radius_first_fit_dual__k8__sall'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def f(v): return '—' if v is None else f'{v:.10f}'
def save(p, value):
    with p.open('x', encoding='utf-8') as fp: json.dump(value, fp, indent=2, ensure_ascii=False, allow_nan=False)


def main():
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'report_v1'
    manifest = read(out/'manifest.json')
    for path, digest in manifest['input_summary_sha256'].items():
        assert sha(root/path) == digest and read(root/path)['passed']
    for name, digest in manifest['outputs_sha256'].items(): assert sha(out/name) == digest
    assert sha(root/manifest['entry_file']) == manifest['entry_sha256']
    assert manifest['tasks'] == 32 and manifest['methods'] == 104 and manifest['primary'] == PRIMARY
    folder = root/BASE/'development_evaluation_v1'
    methods = read(folder/'methods.json'); by = {r['method']: r for r in methods}; assert len(by) == 104
    comparisons = {(r['candidate'], r['control'], r['metric']): r for r in read(folder/'comparisons.json')}
    channels = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
    controls = [f'radius_first_fit_{c}__k8__sall' for c in channels if c != 'dual'] + [
        'online_first_fit_dual', 'online_first_fit_dual__k32', 'online_uniform_state_dual_plus_residual__k64',
        'probe_all_alm64', 'probe_then_adam15360_33', 'probe_pc1024_33', 'probe_nodual1024_33',
        'plain_alm1024_33', 'probe_alm512_33', 'cold__prior16384_ridge', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
    key = []
    for name in controls:
        c = comparisons[PRIMARY, name, 'mse257']
        key.append([name, f(by[name]['metrics']['mse257']), f(c['mean_difference']),
                    f"{c['improved']}/{c['equal']}/{c['worse']}", f(c['worst_leave_one_out_mean'])])
    full = [[r['method']]+[f(r['metrics'][m]) for m in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+
            [f(r['mean_current_seconds']), str(r['failures'])] for r in methods]
    cal = root/'results/budget_reinvestment/calibration_v1'
    cm = {r['method']: r for r in read(cal/'methods.json')}
    resource_names = ['reference_fraction_first_fit_dual', 'online_first_fit_dual__k32',
        'online_uniform_state_dual_plus_residual__k64', 'probe_all_alm64', 'probe_then_adam3840_33',
        'probe_pc256_33', 'probe_nodual256_33', 'plain_alm256_33']
    resources = [[n, f(cm[n]['mean_seconds']), str(cm[n]['maximum_traced_peak_bytes']),
                  str(cm[n]['maximum_absolute_lifetime_peak_wset'])] for n in resource_names]
    tables = read(out/'table_data.json')
    assert tables == dict(key_comparisons=key, full_methods=full, resources=resources)
    text = (out/'report.md').read_text(encoding='utf-8'); blocks = []; active = []
    for line in text.splitlines()+['']:
        if line.startswith('| '): active.append([cell.strip() for cell in line.strip().strip('|').split('|')])
        elif active: blocks.append(active); active = []
    assert len(blocks) == 3
    assert [block[2:] for block in blocks] == [key, resources, full]
    assert f"{by[PRIMARY]['metrics']['mse257']:.10f}" in text
    assert '不是新盲集' in text and '不是论文级确认' in text and '不声称普遍胜过所有闭式解' in text
    for row in full:
        if by[row[0]]['old_runtime_not_a_current_benchmark']: assert row[-2] == '—'
    data = read(out/'figure_data.json'); assert data['input_summary_sha256'] == manifest['input_summary_sha256']
    union = next(r for r in read(root/'results/search_radius_reachability/audit_v1/methods.json') if r['method'] == 'strong_union')
    assert data['reachability'] == dict(missing_pairs=union['certified_positive_mode_pairs'],
        outside_pairs=union['certified_categories']['above_window'], inside_pairs=union['certified_categories']['within_window'],
        outside_tasks=union['tasks_with_certified_above_window'], fit_outside_tasks=union['support_fit_tasks_with_certified_above_window'])
    assert len(data['curves']) == 24
    for i, channel in enumerate(channels):
        for j, k in enumerate([1, 2, 4, 8]):
            name = f'radius_first_fit_{channel}__k{k}__sall'
            assert data['curves'][4*i+j] == dict(channel=channel, k=k, method=name, mse257=by[name]['metrics']['mse257'])
    reference_names = ['online_first_fit_dual', 'online_first_fit_dual__k32', 'probe_all_alm64']
    assert data['references'] == [dict(method=n, mse257=by[n]['metrics']['mse257']) for n in reference_names]
    images = []
    for name in ['radius_mechanism.png', 'radius_query_curves.png']:
        with Image.open(out/name) as im:
            assert im.width >= 1800 and im.height >= 800
            images.append(dict(file=name, width=im.width, height=im.height, sha256=sha(out/name)))
    links = []
    for location in [out/'report.md', root/manifest['entry_file']]:
        for target in re.findall(r'\]\(([^)]+)\)', location.read_text(encoding='utf-8')):
            assert not target.startswith(('http:', 'https:', 'file:'))
            resolved = (location.parent/target).resolve(); assert resolved.is_file(), target
            links.append(dict(source=str(location.relative_to(root)), target=target, sha256=sha(resolved)))
    save(out/'qa_numeric.json', dict(passed=True, methods=104, key_comparison_rows=len(key),
        resource_rows=len(resources), full_method_rows=len(full), checked_table_cells=sum(len(r) for r in key+resources+full),
        checked_figure_values=32, images=images, local_links=links, actual_visual_inspection_still_required=True,
        manifest_sha256=sha(out/'manifest.json'), auditor_source_sha256=sha(Path(__file__))))
    print(dict(passed=True, table_cells=sum(len(r) for r in key+resources+full), figure_values=32,
               visual_inspection_still_required=True), flush=True)


if __name__ == '__main__': main()
