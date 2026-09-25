"""354 independent all-cell/all-figure comparison to final experiment data."""
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save


def main():
    root = Path(__file__).resolve().parents[2]; base = root/'results/unvisited_online'; out = base/'report_v1'
    manifest = read(out/'manifest.json')
    for p, digest in manifest['outputs_sha256'].items(): assert sha(out/p) == digest
    for p, digest in manifest['input_summary_sha256'].items(): assert sha(root/p) == digest
    assert sha(root/manifest['entry_file']) == manifest['entry_sha256']
    methods = {m['method']: m for m in read(base/'development_evaluation_v1/methods.json')}
    comps = {(r['candidate'], r['control'], r['metric']): r for r in read(base/'development_evaluation_v1/comparisons.json')}
    mechanisms = read(base/'development_audit_v1/mechanisms.json'); figures = read(out/'figure_data.json'); tables = read(out/'table_data.json')
    fmt = lambda x: '—' if x is None else format(x, '.10f'); cells = 0; values = 2
    assert figures['native_seconds'] == methods['unvisited_native_alm64']['mean_current_seconds']
    assert figures['adam_seconds'] == methods['unvisited_native_adam240']['mean_current_seconds']
    for r, row in zip(figures['credits'], tables['main']):
        name, old = r['method'], r['old_method']; c = comps[name, old, 'mse257']
        assert r['mse'] == methods[name]['metrics']['mse257'] and r['old_mse'] == methods[old]['metrics']['mse257']
        assert r['delta_old'] == c['mean_difference'] and r['seconds'] == methods[name]['mean_current_seconds']
        assert [r[k] for k in ['improved', 'equal', 'worse']] == [c[k] for k in ['improved', 'equal', 'worse']]
        assert r['extra_positive'] == sum(len(m['additional_positive_vs_old']) for m in mechanisms if m['method'] == name)
        assert row == [name, fmt(r['old_mse']), fmt(r['mse']), fmt(r['delta_old']), f"{r['improved']}/{r['equal']}/{r['worse']}", str(r['extra_positive']), fmt(r['seconds'])]
        cells += len(row); values += 8
    for row in tables['comparisons']:
        name = row[0]; c = comps['unvisited_dual', name, 'mse257']
        assert row == [name, fmt(methods[name]['metrics']['mse257']), fmt(c['mean_difference']), f"{c['improved']}/{c['equal']}/{c['worse']}", fmt(c['worst_leave_one_out_mean'])]; cells += len(row)
    assert len(tables['full']) == len(methods) == 137
    for row in tables['full']:
        m = methods[row[0]]
        assert row == [m['method']]+[fmt(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+[fmt(m['mean_current_seconds']), str(m['failures'])]; cells += len(row)
    text = (out/'report.md').read_text(encoding='utf-8')
    for panel in tables.values():
        for row in panel: assert '| '+' | '.join(row)+' |' in text
    result = dict(passed=True, table_cells=cells, figure_values=values, manifest_sha256=sha(out/'manifest.json'),
                  source_sha256=sha(Path(__file__)), visual_inspection_done_by_this_script=False)
    save(out/'qa_numeric.json', result); print(result, flush=True)


if __name__ == '__main__': main()
