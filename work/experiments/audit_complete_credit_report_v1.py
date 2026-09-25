"""Independent text-table audit of the already generated stage 308 report."""
import json
import hashlib
from pathlib import Path


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def run():
    root = Path(__file__).resolve().parents[2]
    base = root / 'results/complete_credit_amplitude_events'
    report = base / 'report_v1'
    agg = json.loads((base / 'geometry_v1/aggregate.json').read_text(encoding='utf-8'))
    paths = json.loads((report / 'path_counts.json').read_text(encoding='utf-8'))
    text = (report / 'report.md').read_text(encoding='utf-8')
    rows = [line.strip('|').split('|') for line in text.splitlines() if line.startswith('|')]
    actual = {r[0]: r[1:] for r in rows if r[0] in agg}
    assert set(actual) == set(agg) and len(actual) == 35
    numbers = 0
    for name, r in actual.items():
        a = agg[name]
        assert int(r[0]) == a['tasks_with_new_positive']
        assert int(r[1]) == a['task_positive_pairs']
        assert r[2] == format(a['known_new_positive_numeric_volume_sum'], '.12g')
        assert int(r[3]) == a['task_unknown_mode_pairs']
        numbers += 4
    for r in rows:
        if r[0] not in paths:
            continue
        p = paths[r[0]]
        expected = [str(p['directions']), str(p['cells']), str(p['extra_open_mode_directions']),
                    format(p['search_seconds'], '.6f'), format(p['median_search_seconds'], '.6f')]
        assert r[1:] == expected
        numbers += 5
    summary = json.loads((report / 'summary.json').read_text(encoding='utf-8'))
    for name, digest in summary['outputs_sha256'].items():
        assert sha(report / name) == digest
    for name, digest in summary['input_summaries_sha256'].items():
        assert sha(base / name / 'summary.json') == digest
    result = dict(passed=True, method_rows=35, path_rows=3, table_numbers_checked=numbers,
                  report_summary_sha256=sha(report / 'summary.json'),
                  source_sha256=sha(Path(__file__)),
                  scope='Exact published table formatting and all manifest hashes; visual review recorded separately.')
    with (report / 'table_audit_v1.json').open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps(result))


if __name__ == '__main__':
    run()
