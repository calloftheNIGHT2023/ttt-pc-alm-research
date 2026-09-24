"""Immutable geometry supplement, with explicit terminal-stage allowlists."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import snapshot as baseline
import verification_snapshot_20260924 as previous


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release/geometry-20260924.json'
STAGING = ROOT / 'release/stage-geometry-20260924.tmp'
PARENTS = ['release/snapshot.json', 'release/verification-20260924.json']
SEALS = ['results/round_287_preflight_audit_v4.json', 'results/round_287_preflight_audit_v5.json']
COMPLETED_TREES = [
    'simplex_interval_selftests_v1', 'simplex_interval_independent_selftests_v1',
    'simplex_interval_selftests_v2', 'simplex_interval_selftests_v3',
    'deep_simplex_envelope_selftests_v1', 'query_cover_selftests_v1',
    'adaptive_query_cover_selftests_v1', 'geometry_budget_diagnosis_v1',
    'geometry_transport_diagnosis_v1', 'geometry_transport_lp_diagnosis_v1',
    'geometry_transport_independent_audit_v1', 'geometry_affine_readout_diagnosis_v1',
    'first_task_geometry_census_v2', 'task_5500005_geometry_census_v1',
    'collapsed_vertex_diagnosis_v1', 'collapsed_vertex_independent_audit_v1',
    'pool_geometry_gate_selftests', 'pool_geometry_gate_selftests_v3',
    'path_audit_preflight_v4', 'path_audit_preflight_v5',
    'evaluation_preflight_v4', 'evaluation_preflight_v5',
    'evaluation_audit_preflight_v4', 'evaluation_audit_preflight_v5',
    'figures_preflight_v4', 'figures_preflight_v5', 'figures_preflight_v6',
]
# Intentionally partial records: the release note enumerates the omissions.
METADATA_ONLY = {
    'prediction_audit_v2': ['summary.json', 'protocol.json', 'files.json'],
    'simplex_readout_integral_diagnosis_v1': ['summary.json', 'protocol.json', 'inputs.json', 'files.json'],
    'simplex_readout_integral_diagnosis_v2': ['failure.json', 'protocol.json', 'inputs.json'],
    'simplex_readout_integral_diagnosis_v3': ['summary.json', 'protocol.json', 'inputs.json', 'files.json'],
    'geometry_integral_hardpoint_v1': ['summary.json', 'protocol.json', 'files.json', 'initial_cover.json', 'final_cover.json', 'points.json', 'inherited.json'],
    'first_task_geometry_census_v1': ['failure.json', 'protocol.json'],
    'pool_geometry_gate_selftests_failed_v2': ['failure.json'],
    'pipeline_execution_v3': ['protocol.json', 'failure.json', 'paths_exit.json', 'paths_start.json', 'all_prediction_complete.json'],
    'pipeline_execution_v4': ['protocol.json', 'failure.json', 'paths_exit.json', 'paths_start.json', 'all_prediction_complete.json'],
}
EXTRA_SOURCES = [
    'diagnose_probe_confirmation_geometry_budget.py', 'analyze_probe_geometry_transport_bound.py',
    'analyze_probe_geometry_transport_lp.py', 'audit_probe_geometry_transport_plans.py',
    'diagnose_probe_geometry_affine_readout.py', 'probe_simplex_readout_intervals.py',
    'test_probe_simplex_readout_intervals.py', 'audit_probe_simplex_readout_intervals.py',
    'diagnose_probe_geometry_integral_bounds.py', 'probe_simplex_readout_intervals_v2.py',
    'audit_probe_simplex_readout_intervals_v2.py', 'diagnose_probe_geometry_integral_bounds_v2.py',
    'probe_simplex_readout_intervals_v3.py', 'audit_probe_simplex_readout_intervals_v3.py',
    'diagnose_probe_geometry_integral_bounds_v3.py', 'probe_query_interval_cover.py',
    'test_probe_deep_simplex_envelopes.py', 'refine_probe_geometry_query_cover.py',
    'census_probe_first_task_geometry.py', 'census_probe_first_task_geometry_v2.py',
    'census_probe_task_5500005_geometry.py', 'diagnose_probe_collapsed_vertices.py',
    'audit_probe_collapsed_vertices.py', 'test_probe_pool_geometry_gate_v2.py',
    'continue_probe_confirmation_pipeline_v4.py', 'continue_probe_confirmation_pipeline_v5.py',
]
FORBIDDEN = [
    'results/probe_credit_confirmation/' + name + '/'
    for name in ['predictions', 'path_audit_v5', 'pipeline_execution_v5',
                 'evaluation_v5', 'evaluation_audit_v5', 'figures_v6']
]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe_file(name):
    path = ROOT / name
    baseline.require(path.is_file() and path.resolve().is_relative_to(ROOT), 'Missing/unsafe: ' + name)
    baseline.require(not any(name.startswith(prefix) for prefix in FORBIDDEN), 'Forbidden live/full output: ' + name)
    return path


def inventory():
    existing = set().union(*(read(ROOT / parent)['files'] for parent in PARENTS))
    selected = set()

    def add(name):
        safe_file(name)
        if name not in existing:
            selected.add(name)

    sources = set(EXTRA_SOURCES)
    for name in SEALS:
        seal = read(ROOT / name)
        baseline.require(seal['passed'] and seal['functional_preflight_only'], name)
        sources.update(seal['source_sha256'])
        for report in seal['report_sha256']:
            add('outputs/ttt-pc-alm-research/' + report)
        add(name)
    for source in sources:
        add('work/experiments/' + source)
    add('scripts/geometry_snapshot_20260924.py')
    add('release/GEOMETRY-20260924.md')
    base = ROOT / 'results/probe_credit_confirmation'
    for name in COMPLETED_TREES:
        folder = base / name
        baseline.require((folder / 'summary.json').is_file() and not (folder / 'RUNNING.lock').exists(), 'Incomplete: ' + name)
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in {'.json', '.npz', '.png', '.md'}:
                add(path.relative_to(ROOT).as_posix())
    for folder, files in METADATA_ONLY.items():
        baseline.require(not (base / folder / 'RUNNING.lock').exists(), 'Locked: ' + folder)
        for name in files:
            add('results/probe_credit_confirmation/' + folder + '/' + name)
    return sorted(selected)


def validate(files):
    previous.verify()  # Also verifies the base snapshot and the official gitlinks.
    syntax = 0
    for name in files:
        path = safe_file(name)
        baseline.require(path.stat().st_size < 25 * 1024 * 1024, 'Large file: ' + name)
        if path.suffix in {'.py', '.md', '.json'}:
            content = path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content), 'Credential-like text: ' + name)
            if path.suffix == '.py':
                ast.parse(content, filename=name)
                syntax += 1
    for name in SEALS:
        seal = read(ROOT / name)
        for source, digest in seal['source_sha256'].items():
            baseline.require(baseline.sha(ROOT / 'work/experiments' / source) == digest, 'Frozen source changed: ' + source)
        for report, digest in seal['report_sha256'].items():
            baseline.require(baseline.sha(ROOT / 'outputs/ttt-pc-alm-research' / report) == digest, 'Frozen report changed: ' + report)
    return dict(new_python_syntax_files=syntax, source_counts={name: read(ROOT / name)['source_count'] for name in SEALS},
                credential_scan='passed heuristic, not a security guarantee', algorithm_executed=False)


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(), 'Preserve existing supplement')
    files = inventory()
    checks = validate(files)
    entries = {name: dict(bytes=(ROOT / name).stat().st_size, sha256=baseline.sha(ROOT / name)) for name in files}
    value = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                 parent_manifests_sha256={name: baseline.sha(ROOT / name) for name in PARENTS},
                 scope='Completed geometry proofs, diagnostic fixtures, numerical audit metadata and old-task preflight; NOT new query quality',
                 files=entries, validation=checks, partial_metadata_directories=METADATA_ONLY,
                 exclusions='No live path/pipeline/evaluation directories, full 8192-task inputs/predictions or query truths; see GEOMETRY-20260924.md')
    with MANIFEST.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    with STAGING.open('xb') as stream:
        stream.write(('\0'.join(files + [MANIFEST.relative_to(ROOT).as_posix()]) + '\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files), bytes=sum(entry['bytes'] for entry in entries.values()), validation=checks)))


def verify():
    value = read(MANIFEST)
    baseline.require(value['parent_manifests_sha256'] == {name: baseline.sha(ROOT / name) for name in PARENTS}, 'Parents changed')
    for name, entry in value['files'].items():
        path = safe_file(name)
        baseline.require(path.stat().st_size == entry['bytes'] and baseline.sha(path) == entry['sha256'], 'Supplement changed: ' + name)
    checks = validate(value['files'])
    print(json.dumps(dict(passed=True, files=len(value['files']), validation=checks, scientific_goal_complete=False)))


def verify_index():
    verify()
    expected = set(read(MANIFEST)['files']) | {MANIFEST.relative_to(ROOT).as_posix()}
    actual = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=ROOT).decode('utf-8').split('\0')) - {''}
    baseline.require(actual == expected, 'Staged file census differs')
    for name in sorted(expected):
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        baseline.require(hashlib.sha256(blob).hexdigest() == baseline.sha(ROOT / name), 'Staged bytes differ: ' + name)
    print(json.dumps(dict(staged_byte_hashes_passed=True, files=len(expected))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'verify-index'])
    action = parser.parse_args().action
    {'build': build, 'verify': verify, 'verify-index': verify_index}[action]()
