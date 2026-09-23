"""Build/verify an explicit publication inventory without importing experiments."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / 'release/snapshot.json'
UPSTREAM = {
    'work/third_party/ttt-lm-pytorch': 'cd831db10c8c9a0f6340f02da5613316a8a92b67',
    'work/third_party/pc-alm': '660747f61a8a7e547c0ecd2c48c8883380a7d1f6',
}
FROZEN_PROTOCOLS = [
    'results/round_246_audit.json',
    'results/round_286_audit.json',
    'results/probe_credit_confirmation/predictions/protocol.json',
]
DRAFTS = [
    'probe_confirmation_independent_heads.py',
    'audit_probe_confirmation_predictions.py',
    'probe_confirmation_path_reference.py',
    'audit_probe_confirmation_paths.py',
    'probe_confirmation_statistics.py',
    'evaluate_probe_credit_confirmation.py',
    'audit_probe_credit_confirmation_evaluation.py',
    'plot_probe_credit_confirmation.py',
]
# Round257 deliberately archived both the failed v1 and successful v2.
# Do not rewrite an immutable failed source to make the archive look cleaner.
ARCHIVED_SYNTAX_FAILURES = {
    'work/experiments/analyze_dual_jump_query.py': 17,
}
SECRET = re.compile(
    r'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}'
    r'|sk-[A-Za-z0-9_-]{24,}|AKIA[A-Z0-9]{16}'
    r'|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'
    r'|https://[^/ :]+:[^/ @]+@'
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inventory():
    files = set()

    def add(path):
        require(path.is_file(), f'Missing selected file: {path}')
        files.add(path.relative_to(ROOT).as_posix())

    def tree(directory, suffixes):
        for path in (ROOT / directory).rglob('*'):
            if path.is_file() and path.suffix in suffixes:
                add(path)

    for name in ['README.md', '.gitignore', '.gitattributes', '.gitmodules',
                 'requirements-research.txt']:
        add(ROOT / name)
    for path in (ROOT / 'scripts').glob('*.py'):
        add(path)
    for path in (ROOT / 'release').glob('*.md'):
        add(path)
    sources = set(DRAFTS)
    for protocol in FROZEN_PROTOCOLS:
        sources.update(json.loads((ROOT / protocol).read_text(encoding='utf-8'))['source_sha256'])
    for name in sorted(sources):
        add(ROOT / 'work/experiments' / name)
    for path in (ROOT / 'outputs/ttt-pc-alm-research').glob('*.md'):
        if path.name != 'RESEARCH_STATE.md':
            add(path)
    tree('results/multiplier_fixed_point', {'.json', '.npz', '.png', '.md'})
    for family in ['probe_credit_budget', 'probe_credit_resources']:
        tree(f'results/{family}/figures', {'.json', '.png', '.md'})
        for name in ['protocol.json', 'summary.json']:
            add(ROOT / f'results/{family}/audit/{name}')
    for name in ['files.json', 'methods.json', 'paired.json', 'protocol.json', 'summary.json']:
        add(ROOT / f'results/probe_credit_budget/evaluation/{name}')
    tree('results/probe_credit_budget/evaluation_audit', {'.json'})
    for name in ['methods.json', 'selected_configs.json', 'protocol.json', 'summary.json']:
        add(ROOT / f'results/probe_credit_resources/calibration/{name}')
    tree('results/probe_credit_confirmation/planning', {'.json'})
    tree('results/probe_credit_confirmation/head_preflight', {'.json', '.npz'})
    for directory in ['head_audit', 'runner_audit_v2']:
        tree(f'results/probe_credit_confirmation/{directory}', {'.json'})
    # Frozen protocol only: NEVER enumerate live prediction tasks or teachers.
    add(ROOT / 'results/probe_credit_confirmation/predictions/protocol.json')
    for number in [246, 286]:
        add(ROOT / f'results/round_{number}_audit.json')
    return sorted(files)


def checks(files):
    syntax_count = 0
    syntax_failures = {}
    for name in files:
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), f'Unsafe path: {name}')
        require(path.stat().st_size < 25 * 1024 * 1024, f'Large file: {name}')
        if path.suffix in {'.py', '.md', '.json', '.txt'}:
            content = path.read_text(encoding='utf-8-sig')
            require(not SECRET.search(content), f'Credential-like content in {name}')
            if path.suffix == '.py':
                try:
                    ast.parse(content, filename=name)
                    syntax_count += 1
                except SyntaxError as error:
                    require(name in ARCHIVED_SYNTAX_FAILURES and
                            error.lineno == ARCHIVED_SYNTAX_FAILURES[name],
                            f'Unexpected syntax error: {name}:{error.lineno}')
                    syntax_failures[name] = error.lineno
    require(syntax_failures == ARCHIVED_SYNTAX_FAILURES, 'Archived failure census differs')
    frozen = {}
    for name in FROZEN_PROTOCOLS:
        source = json.loads((ROOT / name).read_text(encoding='utf-8'))
        hashes = source['source_sha256']
        for filename, digest in hashes.items():
            path = ROOT / 'work/experiments' / filename
            require(sha(path) == digest, f'Frozen source differs: {filename}')
        frozen[name] = len(hashes)
    return dict(python_syntax_passed_files=syntax_count,
                known_archived_syntax_failures=syntax_failures, frozen_source_checks=frozen,
                credential_pattern_scan='passed; heuristic, not a security guarantee')


def build():
    files = inventory()
    result = checks(files)
    for path, commit in UPSTREAM.items():
        actual = subprocess.check_output(['git', '-C', str(ROOT / path),
                                          'rev-parse', 'HEAD'], text=True).strip()
        require(actual == commit, f'Upstream differs: {path}')
    entries = {name: dict(bytes=(ROOT / name).stat().st_size,
                         sha256=sha(ROOT / name)) for name in files}
    data = dict(created_utc=datetime.now(timezone.utc).isoformat(),
                scope='phase snapshot; no live predictions or new query outcomes',
                files=entries, upstream_gitlinks=UPSTREAM, validation=result,
                exclusions='See release/SCOPE.md')
    MANIFEST.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    paths = files + ['release/snapshot.json'] + list(UPSTREAM)
    (ROOT / 'release/stage-paths.txt').write_bytes(('\0'.join(paths) + '\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files), bytes=sum(e['bytes'] for e in entries.values()),
                          validation=result), ensure_ascii=False))


def verify():
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    for name, entry in data['files'].items():
        path = ROOT / name
        require(path.resolve().is_relative_to(ROOT), f'Unsafe path: {name}')
        require(path.stat().st_size == entry['bytes'] and sha(path) == entry['sha256'],
                f'Snapshot differs: {name}')
    result = checks(data['files'])
    require(data['upstream_gitlinks'] == UPSTREAM, 'Upstream manifest differs')
    if (ROOT / '.git').exists():
        for path, commit in UPSTREAM.items():
            item = subprocess.check_output(['git', '-C', str(ROOT), 'ls-files',
                                            '--stage', '--', path], text=True).strip()
            require(item.startswith(f'160000 {commit} 0\t'), f'Gitlink differs: {path}')
    print(json.dumps(dict(passed=True, files=len(data['files']), validation=result,
                          algorithm_executed=False), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build', 'verify'])
    args = parser.parse_args()
    build() if args.action == 'build' else verify()
