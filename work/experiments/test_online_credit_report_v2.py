"""Small presentation tests; run after live pilot prediction timing has ended."""
import ast
from pathlib import Path
import tempfile
from unittest.mock import patch
import report_online_credit_fresh_v2 as report
import audit_online_credit_report_v2 as audit


def selftest():
    counts = {}
    root = Path(__file__).resolve().parents[2]
    for name in ['report_online_credit_fresh_v2.py', 'audit_online_credit_report_v2.py', 'test_online_credit_report_v2.py']:
        ast.parse((root / 'work/experiments' / name).read_text(encoding='utf-8'))
    counts['source_parses'] = 3
    assert report.fmt(None) == '—' and report.fmt(1.25) == '1.25'
    for invalid in [float('inf'), float('-inf'), float('nan')]:
        try:
            report.fmt(invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError('Nonfinite display value accepted')
    counts['format_cases'] = 5
    assert report.budget_label('a', dict(within_budget=['a'], sensitivity_110_percent=['a', 'b'])) == '≤1.00'
    assert report.budget_label('b', dict(within_budget=['a'], sensitivity_110_percent=['a', 'b'])) == '1.00–1.10'
    assert report.budget_label('c', dict(within_budget=['a'], sensitivity_110_percent=['a', 'b'])) == '>1.10'
    counts['budget_cases'] = 3
    for candidate in report.CANDIDATES:
        controls = report.same_trigger(candidate)
        assert len(controls) == len(set(controls)) == 4 and all('_dual' not in c for c in controls)
    counts['same_trigger_panels'] = 2
    with tempfile.TemporaryDirectory(prefix='ttt-report-tests-') as td:
        fixture = Path(td)
        # Any call to JSON reading before all summaries exist is a leak.
        with patch.object(report, 'read', side_effect=AssertionError('Premature quality/source read')):
            try:
                report.gate(fixture, 'pilot')
            except RuntimeError:
                pass
            else:
                raise AssertionError('Missing seal was accepted')
        pred = fixture / report.BASE / 'pilot_predictions_v2'; pred.mkdir(parents=True)
        with patch.object(report, 'gate', side_effect=AssertionError('Live prediction guard bypassed')):
            try:
                report.build(fixture, fixture / 'should-not-exist', 'preflight', False)
            except RuntimeError:
                pass
            else:
                raise AssertionError('Presentation allowed during live prediction')
        assert not (fixture / 'should-not-exist').exists()
        counts['early_access_guards'] = 2
        path = fixture / 'table.md'
        report.write_text(path, '# Test\n\n## panel\n\n' + report.table(['name', 'value'], [['x', '1.25'], ['y', '—']]))
        parsed = audit.tables(path)
        assert parsed == [dict(section='panel', header=['name', 'value'], rows=[['x', '1.25'], ['y', '—']])]
        counts['independent_table_parser'] = 1
        try:
            report.write_text(path, 'overwrite')
        except FileExistsError:
            pass
        else:
            raise AssertionError('Existing report overwritten')
        counts['exclusive_write'] = 1
    return dict(passed=True, counts=counts, no_research_quality_accessed=True)


if __name__ == '__main__':
    import json
    print(json.dumps(selftest()), flush=True)

