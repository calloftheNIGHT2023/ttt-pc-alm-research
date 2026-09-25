"""Gated compact 324-330 source/statistics snapshot; never include live pilot files.

Prepare this script during prediction, but do not run it or Git packing until
the original timed job has ended. A successful build is not a method-win claim.
"""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import snapshot as baseline
import online_stasis_snapshot_20260925 as previous

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release/online-credit-20260925.json'
STAGING = ROOT / 'release/stage-online-credit-20260925.tmp'
PARENTS = previous.PARENTS + ['release/online-stasis-20260925.json']
BASE = 'results/online_credit_fresh_pilot'
REPORT = BASE + '/pilot_report_v2'
SUPPORT = 'results/support_consistency_trigger'
COMMON = ['summary.json', 'protocol.json']
STATIC = {'scripts/online_credit_snapshot_20260925.py', 'release/ONLINE-CREDIT-20260925.md'}
DOCS = [
    '321_branch_image_chain_design_v1.md', '324_support_consistency_trigger_protocol_v1.md',
    '325_credit_exclusive_value_protocol_v1.md', '326_readout_audit_protocol_v1.md',
    '328_fresh_online_credit_protocol_v1.md', '328_serialization_correction_v2.md',
    '328_finite_readout_risk_addendum_v1.md', '330_fresh_pool_mechanism_protocol_v1.md',
    '330_fresh_pool_mechanism_freeze_v1.json',
    '329_visual_revision_v2.md', '332_fresh_pilot_results_v1.md',
]
for name in DOCS:
    STATIC.add('outputs/ttt-pc-alm-research/' + name)


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(test, message):
    baseline.require(test, message)


def safe(name):
    require('\\' not in name, 'Use portable relative paths: ' + name)
    path = (ROOT / name).resolve()
    allowed = name in STATIC or (name.startswith('work/experiments/') and path.suffix in {'.py', '.ps1'}) or (
        name.startswith(('results/support_consistency_trigger/', 'results/online_credit_branch_search/',
                         'results/online_credit_resources/', BASE + '/')) and path.suffix in {'.json', '.md', '.png', '.npz'})
    require(allowed and path.is_relative_to(ROOT) and path.is_file(), 'Missing or unsafe snapshot path: ' + name)
    return path


def completed(folder):
    path = ROOT / folder
    require((path / 'summary.json').is_file() and not (path / 'failure.json').exists(), 'Incomplete stage: ' + folder)
    result = read(path / 'summary.json')
    require(result['passed'], 'Stage did not pass: ' + folder)
    return result


def source_closure(roots):
    """Local flat-module AST closure, without importing or executing research."""
    pending = list(roots); found = set()
    while pending:
        name = pending.pop()
        if name in found:
            continue
        path = ROOT / 'work/experiments' / name
        require(path.is_file() and path.parent == ROOT / 'work/experiments', 'Missing flat research source: ' + name)
        found.add(name)
        if path.suffix != '.py':
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=name)
        except SyntaxError as error:
            # The inherited manifest deliberately preserves the failed 257 v1.
            # It is archival evidence, not an executable dependency. Accept only
            # the parent's exact named error and exact immutable file bytes.
            relative = 'work/experiments/' + name
            archived = read(ROOT / 'release/snapshot.json')['files']
            require(relative in baseline.ARCHIVED_SYNTAX_FAILURES and
                    error.lineno == baseline.ARCHIVED_SYNTAX_FAILURES[relative],
                    'Unexpected dependency syntax error: ' + relative)
            require(relative in archived and baseline.sha(path) == archived[relative]['sha256'],
                    'Changed archived failure: ' + relative)
            continue
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module else [])
            for module in modules:
                local = module.split('.')[0] + '.py'
                if (ROOT / 'work/experiments' / local).is_file() and local not in found:
                    pending.append(local)
    return {'work/experiments/' + name for name in found}


def gate():
    # Do not run parent verification or inventory scans while predictions run.
    ps = completed(BASE + '/pilot_predictions_v2')
    es = completed(BASE + '/pilot_evaluation_v2')
    au = completed(BASE + '/pilot_audit_v2')
    po = completed(BASE + '/pilot_pool_audit_v1')
    require(ps['tasks'] == es['tasks'] == po['counts']['tasks'] == 512, 'All 512 tasks required')
    require(ps['predictors'] == es['predictors'] == au['counts']['predictors'] == 26112, 'All 51 methods required')
    require(es['comparisons'] == 400 and es['main_family'] == 100, 'Changed statistics panel')
    require(au['prediction_summary_sha256'] == baseline.sha(ROOT / BASE / 'pilot_predictions_v2/summary.json'), 'Prediction audit mismatch')
    require(au['evaluation_summary_sha256'] == baseline.sha(ROOT / BASE / 'pilot_evaluation_v2/summary.json'), 'Evaluation audit mismatch')
    pool_protocol = read(ROOT / BASE / 'pilot_pool_audit_v1/protocol.json')
    require(pool_protocol['independent_main_audit_sha256'] == baseline.sha(ROOT / BASE / 'pilot_audit_v2/summary.json'), 'Pool audit source mismatch')
    for folder, summary in [(BASE + '/pilot_predictions_v2', ps), (BASE + '/pilot_evaluation_v2', es), (BASE + '/pilot_pool_audit_v1', po)]:
        for name, digest in summary['outputs_sha256'].items():
            require(baseline.sha(ROOT / folder / name) == digest, 'Changed sealed artifact: ' + folder + '/' + name)
    report = ROOT / REPORT
    gen = read(report / 'generation_summary.json'); qa = read(report / 'numeric_qa.json'); visual = read(report / 'visual_qa.json')
    require(gen['passed'] and qa['passed'] and visual['passed'], 'Actual numerical and visual report QA required')
    require(gen['stage'] == qa['stage'] == visual['stage'] == 'pilot', 'Report stage mismatch')
    require(qa['generation_summary_sha256'] == baseline.sha(report / 'generation_summary.json'), 'Numeric QA generation mismatch')
    require(qa['scientific_audit_summary_sha256'] == baseline.sha(ROOT / BASE / 'pilot_audit_v2/summary.json'), 'Numeric QA scientific source mismatch')
    provenance = read(report / 'provenance.json')
    require(provenance['report_source_sha256'] == baseline.sha(ROOT / 'work/experiments/report_online_credit_fresh_v2.py'), 'Report v2 source changed')
    require(qa['audit_source_sha256'] == baseline.sha(ROOT / 'work/experiments/audit_online_credit_report_v2.py'), 'Report v2 audit source changed')
    require(visual['numeric_qa_sha256'] == baseline.sha(report / 'numeric_qa.json'), 'Visual QA numeric source mismatch')
    require(set(visual['images']) == {'risk_vs_time.png', 'paired_comparisons.png'}, 'Both actual images must be reviewed')
    for name, digest in visual['images'].items():
        require(baseline.sha(report / name) == digest, 'Reviewed image changed: ' + name)
    for name, digest in gen['outputs_sha256'].items():
        require(baseline.sha(report / name) == digest, 'Report changed: ' + name)
    p = read(ROOT / BASE / 'pilot_predictions_v2/protocol.json')
    for name, digest in p['source_sha256'].items():
        require(baseline.sha(ROOT / 'work/experiments' / name) == digest, 'Scientific source changed: ' + name)
    frozen = read(ROOT / 'outputs/ttt-pc-alm-research/330_fresh_pool_mechanism_freeze_v1.json')
    for name, digest in frozen['source_sha256'].items():
        require(baseline.sha(ROOT / 'work/experiments' / name) == digest, 'Supplementary source changed: ' + name)
    require(not ps['query_targets_accessed'] and not po['query_targets_accessed'], 'Prediction/mechanism truth leak')
    require(not es['core_research_goal_complete'] and not po['core_research_goal_complete'], 'Snapshot cannot declare the goal achieved')
    return p


def inventory(p):
    existing = set().union(*(read(ROOT / name)['files'] for name in PARENTS))
    selected = set(STATIC)
    for folder in ['tests_v1', 'states_v1', 'states_audit_v1', 'development_v1', 'search_audit_v1',
                   'geometry_v1', 'geometry_audit_v1', 'risk_v1', 'risk_audit_v1',
                   'exclusive_value_v1', 'exclusive_audit_v1', 'report_audit_v1']:
        for name in COMMON + ['aggregate.json']:
            path = ROOT / SUPPORT / folder / name
            if path.is_file():
                selected.add(path.relative_to(ROOT).as_posix())
    for name in ['report.md', 'credit_case.png', 'manifest.json', 'visual_review.md']:
        selected.add(SUPPORT + '/report_v1/' + name)
    for name in ['scores.json', 'witnesses.json']:
        selected.add(SUPPORT + '/exclusive_value_v1/' + name)
    for base, folders in [('results/online_credit_branch_search', ['tests_v1', 'functional_v1', 'functional_audit_v1']),
                          ('results/online_credit_resources', ['calibration_v1', 'audit_v1'])]:
        for folder in folders:
            for name in COMMON:
                path = ROOT / base / folder / name
                if path.is_file():
                    selected.add(path.relative_to(ROOT).as_posix())
        for name in ['report.md', 'qa.json']:
            selected.add(base + '/report_v1/' + name)
    for name in ['methods.json', 'selection.json', 'environments.json']:
        selected.add('results/online_credit_resources/calibration_v1/' + name)
    for version in [1, 2]:
        for name in COMMON:
            selected.add(BASE + f'/tests_v{version}/' + name)
    selected.add(BASE + '/preflight_predictions_v1/failure.json')
    for stage in ['preflight', 'pilot']:
        for name in COMMON + ['before_query_manifest.json', 'costs.json', 'files.json', 'environments.json']:
            selected.add(BASE + f'/{stage}_predictions_v2/' + name)
        for name in COMMON + ['methods.json', 'comparisons.json', 'actual_costs.json', 'mechanism_groups.json', 'task_metrics.npz']:
            selected.add(BASE + f'/{stage}_evaluation_v2/' + name)
        selected.add(BASE + f'/{stage}_audit_v2/summary.json')
        for name in COMMON + ['methods.json', 'credit_exclusivity.json']:
            selected.add(BASE + f'/{stage}_pool_audit_v1/' + name)
    # The final report only; old preflight display numbers cannot be mistaken
    # for a new-task result. Detailed historical paths remain in local archives.
    # Preserve the visually rejected v1 as well as the accepted v2. Neither
    # directory is overwritten; only v2 is the final delivery entry point.
    for report_folder in [BASE + '/pilot_report_v1', REPORT]:
        for path in (ROOT / report_folder).iterdir():
            if path.is_file() and path.suffix in {'.json', '.md', '.png'}:
                selected.add(path.relative_to(ROOT).as_posix())
    roots = set(p['source_sha256']) | {
        'audit_online_credit_fresh_pools_v1.py', 'report_online_credit_fresh_v1.py',
        'audit_online_credit_report_v1.py', 'test_online_credit_report_v1.py',
        'report_online_credit_fresh_v2.py', 'audit_online_credit_report_v2.py',
        'test_online_credit_report_v2.py',
        'continue_online_credit_pool_audit_v1.ps1', 'continue_online_credit_reporting_v1.ps1',
    }
    selected.update(source_closure(roots))
    return sorted(selected - existing)


def validate(files):
    previous.verify()
    python_count = 0
    for name in files:
        path = safe(name)
        require(path.stat().st_size < 90*1024*1024, 'File exceeds compact Git limit: ' + name)
        if path.suffix in {'.py', '.ps1', '.json', '.md'}:
            text = path.read_text(encoding='utf-8-sig')
            require(not baseline.SECRET.search(text), 'Credential-like text: ' + name)
            if path.suffix == '.py':
                ast.parse(text, filename=name); python_count += 1
    return dict(passed=True, python_syntax_files=python_count,
        inherited_archived_syntax_failures=baseline.ARCHIVED_SYNTAX_FAILURES, fresh_tasks=512, methods=51,
        predictions=26112, primary_comparisons=100, all_comparisons=400,
        full_raw_prediction_or_posterior_archive_included=False, core_research_goal_complete=False,
        algorithm_executed=False, credential_scan='heuristic passed')


def build():
    require(not MANIFEST.exists() and not STAGING.exists(), 'Preserve existing snapshot manifest')
    p = gate(); files = inventory(p); validation = validate(files)
    entries = {name: dict(bytes=safe(name).stat().st_size, sha256=baseline.sha(safe(name))) for name in files}
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifests_sha256={name: baseline.sha(ROOT / name) for name in PARENTS}, files=entries,
        validation=validation, scope='Completed 324-330 compact source/statistics/figures; all pilot predictions scored and audited',
        exclusions='Raw per-call predictors/logs, complete posterior references, query truth, bootstrap arrays and large intermediate geometry; not clean-clone end-to-end reproduction')
    with MANIFEST.open('x', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2); f.write('\n')
    with STAGING.open('xb') as f:
        f.write(('\0'.join(files + [MANIFEST.relative_to(ROOT).as_posix()]) + '\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files), bytes=sum(r['bytes'] for r in entries.values()), validation=validation)))


def verify():
    data = read(MANIFEST)
    require(data['parent_manifests_sha256'] == {name: baseline.sha(ROOT / name) for name in PARENTS}, 'Parent manifest changed')
    for name, record in data['files'].items():
        path = safe(name)
        require(path.stat().st_size == record['bytes'] and baseline.sha(path) == record['sha256'], 'Snapshot changed: ' + name)
    print(json.dumps(dict(files=len(data['files']), validation=validate(data['files']))))


def verify_index():
    verify(); expected = set(read(MANIFEST)['files']) | {MANIFEST.relative_to(ROOT).as_posix()}
    actual = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT).decode('utf-8').split('\0')) - {''}
    require(actual == expected, 'Staged file set differs from the compact allowlist')
    for name in sorted(expected):
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        require(hashlib.sha256(blob).hexdigest() == baseline.sha(ROOT / name), 'Staged bytes differ: ' + name)
    print(json.dumps(dict(staged_hashes_passed=True, files=len(expected))))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('action', choices=['build', 'verify', 'verify-index'])
    {'build': build, 'verify': verify, 'verify-index': verify_index}[ap.parse_args().action]()
