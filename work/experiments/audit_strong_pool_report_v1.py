"""349 compare every plotted value and rendered table cell against audited results."""
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save


def main():
    root = Path(__file__).resolve().parents[2]; base = root/'results/strong_pool_online'; out = base/'report_v1'
    manifest = read(out/'manifest.json')
    for n, digest in manifest['outputs_sha256'].items(): assert sha(out/n) == digest
    for p, digest in manifest['input_summary_sha256'].items(): assert sha(root/p) == digest
    assert sha(root/manifest['entry_file']) == manifest['entry_sha256']
    methods = {m['method']: m for m in read(base/'development_evaluation_v1/methods.json')}
    comps = {(r['candidate'], r['control'], r['metric']): r for r in read(base/'development_evaluation_v1/comparisons.json')}
    mechanisms = read(base/'development_audit_v1/mechanisms.json'); data = read(out/'figure_data.json'); tables = read(out/'table_data.json')
    fmt = lambda v: '—' if v is None else format(v, '.10f'); cells = values = 0
    assert data['native_mse257'] == methods['strong_native_alm64']['metrics']['mse257']
    assert data['native_seconds'] == methods['strong_native_alm64']['mean_current_seconds']
    assert data['watch_seconds'] == methods['strong_watch_alm64']['mean_current_seconds']; values += 3
    for r, row in zip(data['credits'], tables['main']):
        name = r['method']; m = methods[name]; c = comps[name, 'strong_native_alm64', 'mse257']; mm = [p for p in mechanisms if p['method'] == name]
        assert r['mse257'] == m['metrics']['mse257'] and r['delta_to_native'] == c['mean_difference']
        assert [r[k] for k in ['improved', 'equal', 'worse']] == [c[k] for k in ['improved', 'equal', 'worse']]
        assert r['new_positive_pairs'] == sum(len(p['new_positive_modes']) for p in mm)
        assert r['tasks_with_new_positive'] == sum(bool(p['new_positive_modes']) for p in mm)
        assert r['mean_seconds'] == m['mean_current_seconds']; values += 8
        assert row == [name, fmt(r['mse257']), fmt(r['delta_to_native']), f"{r['improved']}/{r['equal']}/{r['worse']}", str(r['new_positive_pairs']), str(r['tasks_with_new_positive']), fmt(r['mean_seconds'])]; cells += len(row)
    for row in tables['comparisons']:
        name = row[0]; c = comps['strong_language_dual', name, 'mse257']
        assert row == [name, fmt(methods[name]['metrics']['mse257']), fmt(c['mean_difference']), f"{c['improved']}/{c['equal']}/{c['worse']}", fmt(c['worst_leave_one_out_mean'])]; cells += len(row)
    assert len(tables['full']) == len(methods) == 129
    for row in tables['full']:
        m = methods[row[0]]
        assert row == [m['method']]+[fmt(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+[fmt(m['mean_current_seconds']), str(m['failures'])]; cells += len(row)
    text = (out/'report.md').read_text(encoding='utf-8')
    for panel in tables.values():
        for row in panel: assert '| '+' | '.join(row)+' |' in text
    result = dict(passed=True, table_cells=cells, figure_values=values,
        manifest_sha256=sha(out/'manifest.json'), source_sha256=sha(Path(__file__)), visual_inspection_done_by_this_script=False)
    save(out/'qa_numeric.json', result); print(result, flush=True)


if __name__ == '__main__': main()
