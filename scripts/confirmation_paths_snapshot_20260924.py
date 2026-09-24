"""Back up the completed fixed64 path gate, without walking any live stage."""
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import snapshot as baseline
import barycentric_snapshot_20260924 as previous


ROOT = Path(__file__).resolve().parents[1]
REL = 'results/probe_credit_confirmation/path_audit_v6'
BASE = ROOT / REL
MANIFEST = ROOT / 'release/confirmation-paths-20260924.json'
STAGING = ROOT / 'release/stage-confirmation-paths-20260924.tmp'
PARENTS = previous.PARENTS + ['release/barycentric-20260924.json']
EXTRAS = {'scripts/confirmation_paths_snapshot_20260924.py',
          'release/CONFIRMATION-PATHS-20260924.md'}
EXPECTED_SUMMARY = 'bb827ab21f3a29222288dd9b0ae5cf9b7e59428694dc64db6c705d45cf819a98'


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def safe(name):
    baseline.require('\\' not in name, 'Use canonical manifest paths')
    baseline.require(name in EXTRAS or name.startswith(REL + '/'),
                     'Outside completed-stage allowlist: ' + name)
    path = (ROOT / name).resolve()
    baseline.require(path.is_relative_to(ROOT) and path.is_file(),
                     'Missing or escaping file: ' + name)
    return path


def stage_inventory():
    baseline.require(not (BASE / 'RUNNING.lock').exists(), 'Path stage still locked')
    summary = read(BASE / 'summary.json')
    baseline.require(baseline.sha(BASE / 'summary.json') == EXPECTED_SUMMARY,
                     'Unexpected completed path summary')
    baseline.require(summary['passed'] and summary['stage'] == 'confirmation'
                     and not summary['query_targets_accessed']
                     and summary['may_evaluate_confirmation'], 'Incomplete path gate')
    selected = {REL + '/summary.json'}

    def add_checked(name, digest):
        canonical = REL + '/' + name.replace('\\', '/')
        baseline.require(baseline.sha(safe(canonical)) == digest,
                         'Path evidence hash differs: ' + canonical)
        selected.add(canonical)

    for name, digest in summary['outputs_sha256'].items():
        add_checked(name, digest)
    entries = read(BASE / 'files.json')
    baseline.require(len(entries) == 64, 'Fixed64 task census differs')
    totals = Counter()
    for seed, (name, digest) in zip(range(5500000, 5500064), entries.items()):
        canonical = name.replace('\\', '/')
        baseline.require(canonical == f'tasks/{seed}/commit.json', 'Task order differs')
        add_checked(name, digest)
        commit = read(safe(REL + '/' + canonical))
        baseline.require(commit['seed'] == seed and not commit['query_targets_accessed'],
                         'Task identity or query boundary differs')
        baseline.require(commit['protocol_sha256'] == summary['outputs_sha256']['protocol.json'],
                         'Task protocol link differs')
        totals.update(commit['counts'])
        for child, child_hash in commit['files'].items():
            baseline.require(child.replace('\\', '/').startswith(f'tasks/{seed}/'),
                             'Task dependency crosses task directory')
            add_checked(child, child_hash)
    baseline.require(dict(totals) == summary['counts'], 'Aggregate path counts differ')
    disk_files = {p.relative_to(ROOT).as_posix() for p in BASE.rglob('*') if p.is_file()}
    baseline.require(disk_files == selected, 'Undeclared or missing completed-stage file')
    protocol = read(BASE / 'protocol.json')
    published = set().union(*(read(ROOT / name)['files'] for name in PARENTS))
    for name, digest in protocol['source_sha256'].items():
        source = 'work/experiments/' + name
        baseline.require(source in published and baseline.sha(ROOT / source) == digest,
                         'Missing or altered published path source: ' + name)
    return selected


def validate(files):
    previous.verify()
    baseline.require(set(files) == stage_inventory() | EXTRAS, 'Supplement census differs')
    for name in files:
        path = safe(name)
        baseline.require(path.stat().st_size < 25 * 1024 * 1024, 'Oversized file: ' + name)
        if path.suffix in {'.json', '.md', '.py'}:
            content = path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content), 'Credential-like text: ' + name)
            if path.suffix == '.py':
                ast.parse(content, filename=name)
    return dict(completed_tasks=64, predictors=1728, exact_positive_geometries=1197,
                query_targets_accessed=False, algorithm_executed=False,
                credential_scan='passed heuristic, not a security guarantee')


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(), 'Preserve existing supplement')
    files = sorted(stage_inventory() | EXTRAS)
    checks = validate(files)
    entries = {name: dict(bytes=safe(name).stat().st_size, sha256=baseline.sha(safe(name)))
               for name in files}
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                parent_manifests_sha256={name: baseline.sha(ROOT / name) for name in PARENTS},
                scope='Completed fixed64 path evidence; no confirmation query-quality results',
                files=entries, validation=checks,
                exclusions='No full8192inputs/predictions, live pipeline/evaluation/statistics/figures or query truths; see CONFIRMATION-PATHS-20260924.md')
    with MANIFEST.open('x', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False)
        stream.write('\n')
    with STAGING.open('xb') as stream:
        stream.write(('\0'.join(files + [MANIFEST.relative_to(ROOT).as_posix()]) + '\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files), bytes=sum(r['bytes'] for r in entries.values()), validation=checks)))


def verify():
    data = read(MANIFEST)
    baseline.require(data['parent_manifests_sha256'] == {name: baseline.sha(ROOT / name) for name in PARENTS},
                     'Parent manifest changed')
    for name, row in data['files'].items():
        path = safe(name)
        baseline.require(path.stat().st_size == row['bytes'] and baseline.sha(path) == row['sha256'],
                         'Supplement changed: ' + name)
    print(json.dumps(dict(passed=True, files=len(data['files']),
                          validation=validate(data['files']), scientific_goal_complete=False)))


def verify_index():
    verify()
    expected = set(read(MANIFEST)['files']) | {MANIFEST.relative_to(ROOT).as_posix()}
    actual = set(subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'],
                                        cwd=ROOT).decode('utf-8').split('\0')) - {''}
    baseline.require(actual == expected, 'Staged census differs')
    for name in sorted(expected):
        blob = subprocess.check_output(['git', 'show', ':' + name], cwd=ROOT)
        baseline.require(hashlib.sha256(blob).hexdigest() == baseline.sha(ROOT / name),
                         'Staged bytes differ: ' + name)
    print(json.dumps(dict(staged_byte_hashes_passed=True, files=len(expected))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify', 'verify-index'])
    {'build': build, 'verify': verify, 'verify-index': verify_index}[parser.parse_args().action]()
