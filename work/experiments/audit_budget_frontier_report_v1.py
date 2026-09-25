"""Independent numeric table/figure QA for the report, before visual inspection."""
from collections import defaultdict
from pathlib import Path
import budget_reinvestment_suite_v1 as io


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; base = root/'results/budget_frontier'; report = base/'report_v1'
    manifest = io.read(report/'manifest.json'); totals = io.complete(base/'development_v1')['totals']
    io.complete(base/'audit_v1')
    for name, digest in manifest['outputs_sha256'].items():
        assert io.sha(report/name) == digest
    assert io.sha(root/manifest['entry_file']) == manifest['entry_sha256']
    aggregate = defaultdict(lambda: [0,0,0,0])
    for r in io.read(base/'development_v1/frontiers.json'):
        if r['budget'] == 8:
            for j, value in enumerate([r['cuts_total'],r['cuts_prefix'],r['cuts_tail'],len(r['selected_positive'])]):
                aggregate[r['method']][j] += value
    lines = (report/'report.md').read_text(encoding='utf-8').splitlines(); cells = 0
    for name, t in totals.items():
        lines_with_name = [s for s in lines if s.startswith('| '+name+' |')]
        work = next(s for s in lines_with_name if len(s.split('|')) == 9)
        values = [x.strip() for x in work.split('|')[2:-1]]
        expected = [str(t['full_response_pairs']),str(t['response_pairs']),f"{1-t['response_pairs']/t['full_response_pairs']:.2%}",
                    str(t['processed']),str(t['batches']),f"{t['seconds']:.6f}"]
        assert values == expected; cells += len(values)
    for name, expected in aggregate.items():
        line = next(s for s in lines if s.startswith('| '+name+' |') and len(s.split('|')) == 7)
        assert [int(x.strip()) for x in line.split('|')[2:-1]] == expected; cells += len(expected)
    examples = sorted([r for r in io.read(base/'development_v1/positive_frontiers.json') if r['method']=='dual_reuse' and not r['selected']],
                      key=lambda r:(r['deficit'],r['seed'],r['index']))[:6]
    for r in examples:
        expected = [r['seed'],r['rank'],r['required_prefix_cuts'],r['actual_prefix_cuts'],r['deficit']]
        assert '| '+' | '.join(map(str,expected))+' |' in lines; cells += len(expected)
    data = io.read(report/'figure_data.json'); checked = 0
    for r in data['work']:
        t = totals[r['method']]
        assert r['full'] == t['full_response_pairs'] and r['lazy'] == t['response_pairs']; checked += 2
    for r in data['coverage']:
        assert r['positive'] == aggregate[r['method']][3] and r['ideal'] == 47; checked += 2
    assert '56.52%' in '\n'.join(lines) and f"{1-totals['dual_independent']['response_pairs']/totals['dual_independent']['full_response_pairs']:.2%}" == '56.52%'
    assert f"{1-totals['zero_independent']['response_pairs']/totals['zero_independent']['full_response_pairs']:.2%}" == '55.53%'
    result = dict(passed=True, numeric_table_cells=cells, figure_values=checked, manifest_sha256=io.sha(report/'manifest.json'),
                  source_sha256=io.sha(Path(__file__)), visual_inspection_still_required=True)
    io.save(report/'qa_numeric.json', result); print(result, flush=True)
