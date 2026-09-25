"""346 independent display-table/figure data check, not a visual inspection."""
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save


def main():
    root = Path(__file__).resolve().parents[2]; base = root/'results/support_language_online'; out = base/'report_v1'
    manifest = read(out/'manifest.json')
    for n, h in manifest['outputs_sha256'].items(): assert sha(out/n) == h
    for p, h in manifest['input_summary_sha256'].items(): assert sha(root/p) == h
    assert sha(root/manifest['entry_file']) == manifest['entry_sha256']
    methods = {m['method']: m for m in read(base/'development_evaluation_v1/methods.json')}
    comparisons = {(r['control'], r['metric']): r for r in read(base/'development_evaluation_v1/comparisons.json') if r['candidate'] == 'language_first_fit_dual'}
    mechanisms = read(base/'development_audit_v1/mechanisms.json')
    figures = read(out/'figure_data.json'); tables = read(out/'table_data.json'); cells = values = 0
    fmt = lambda v: '—' if v is None else format(v, '.10f')
    assert figures['strong_alm'] == methods['probe_all_alm64']['metrics']['mse257']
    assert figures['strong_adam'] == methods['probe_then_adam15360_33']['metrics']['mse257']; values += 2
    for row, table_row in zip(figures['credits'], tables['main']):
        name = row['method']; ch = row['channel']; source = methods[name]; rr = [r for r in mechanisms if r['method'] == name]
        assert row['new_mse257'] == source['metrics']['mse257']
        assert row['old_mse257'] == methods[f'radius_first_fit_{ch}__k8__sall']['metrics']['mse257']
        assert row['added_positive_pairs'] == sum(len(r['new_vs_old_credit_pool']) for r in rr)
        assert row['lost_positive_pairs'] == sum(len(r['old_credit_pool_lost']) for r in rr)
        assert row['mean_current_seconds'] == source['mean_current_seconds']; values += 5
        assert table_row == [name, fmt(row['new_mse257']), fmt(row['old_mse257']), str(row['added_positive_pairs']), str(row['lost_positive_pairs']), fmt(row['mean_current_seconds'])]; cells += len(table_row)
    for row in tables['comparisons']:
        name = row[0]; c = comparisons[name, 'mse257']
        assert row == [name, fmt(methods[name]['metrics']['mse257']), fmt(c['mean_difference']), f"{c['improved']}/{c['equal']}/{c['worse']}", fmt(c['worst_leave_one_out_mean'])]; cells += len(row)
    assert len(tables['full']) == len(methods) == 110
    for row in tables['full']:
        m = methods[row[0]]
        assert row == [m['method']]+[fmt(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+[fmt(m['mean_current_seconds']), str(m['failures'])]; cells += len(row)
    report = (out/'report.md').read_text(encoding='utf-8')
    for panel in tables.values():
        for row in panel: assert '| '+' | '.join(row)+' |' in report
    result = dict(passed=True, table_cells=cells, figure_values=values,
        manifest_sha256=sha(out/'manifest.json'), source_sha256=sha(Path(__file__)), visual_inspection_done_by_this_script=False)
    save(out/'qa_numeric.json', result); print(result, flush=True)


if __name__ == '__main__': main()
