"""Immutable explicitly scoped supplement; never stages unrelated work."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST = 'release/dyadic-20260925.json'
BASE = 'results/online_credit_fresh_pilot'
SOURCES = ['diagnose_fresh_cost_readout_v1.py', 'branch_image_chain_dyadic_v1.py',
           'test_dyadic_branch_search_v1.py', 'support_consistency_trigger_dyadic_v1.py',
           'online_credit_dyadic_v1.py', 'test_dyadic_online_v1.py',
           'audit_dyadic_diagnostics_v1.py', 'report_dyadic_diagnostics_v1.py', 'report_dyadic_diagnostics_v2.py']
DOCS = ['333_fresh_cost_readout_diagnostic_v1.md', '334_dyadic_branch_search_protocol_v1.md',
        '335_dyadic_online_equivalence_protocol_v1.md', '336_dyadic_compute_results_v1.md', '336_dyadic_compute_results_v2.md']
FOLDERS = ['cost_readout_diagnostic_v1', 'dyadic_branch_kernel_v1', 'dyadic_online_v1',
           'dyadic_diagnostics_audit_v1', 'dyadic_report_v1', 'dyadic_report_v2']


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def paths(root):
    selected = {'scripts/dyadic_snapshot_20260925.py', 'release/DYADIC-20260925.md'}
    selected.update('work/experiments/'+n for n in SOURCES)
    selected.update('outputs/ttt-pc-alm-research/'+n for n in DOCS)
    for folder in FOLDERS:
        directory = root/BASE/folder
        summary = read(directory/'summary.json'); assert summary['passed']
        assert not (directory/'failure.json').exists()
        for name, digest in summary.get('outputs_sha256', {}).items(): assert sha(directory/name) == digest
        selected.update(p.relative_to(root).as_posix() for p in directory.rglob('*.json'))
    selected.add(BASE+'/dyadic_report_v1/online_equivalent_runtime.png')
    report = root/BASE/'dyadic_report_v2'
    visual = read(report/'visual_qa.json'); assert visual['passed'] and visual['actually_viewed']
    assert visual['image_sha256'] == sha(report/'online_equivalent_runtime.png')
    assert sha(root/'outputs/ttt-pc-alm-research/336_dyadic_compute_results_v2.md') == read(report/'summary.json')['document_sha256']
    selected.add((report/'online_equivalent_runtime.png').relative_to(root).as_posix())
    return sorted(selected)


def build(root):
    files = paths(root); entries = []
    for name in files:
        p = root/name; assert p.is_file() and p.resolve().is_relative_to(root.resolve())
        if name.endswith('.py'): ast.parse(p.read_text(encoding='utf-8'), filename=name)
        entries.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    value = dict(version=1, parent_commit='90c3675d7b8d6693ee727c318bf2aa7ac5dccce4',
        compact_archive=True, original_predictor_npz_included=False, files=entries,
        file_count=len(entries), bytes=sum(e['bytes'] for e in entries))
    with (root/MANIFEST).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
    return verify(root)


def verify(root):
    m = read(root/MANIFEST); assert m['file_count'] == len(m['files'])
    assert m['bytes'] == sum(e['bytes'] for e in m['files'])
    assert [e['path'] for e in m['files']] == paths(root)
    for entry in m['files']:
        p = root/entry['path']; assert p.stat().st_size == entry['bytes'] and sha(p) == entry['sha256'], entry['path']
    return dict(passed=True, files=m['file_count'], bytes=m['bytes'])


def stage(root):
    result = verify(root)
    existing = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root)
    assert not existing, 'Existing staged changes belong to the user; do not mix them.'
    files = [e['path'] for e in read(root/MANIFEST)['files']] + [MANIFEST]
    # NUL-delimited stdin pathspec, not shell expansion or a blanket add.
    subprocess.run(['git', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=root,
                   input=('\0'.join(files)+'\0').encode('utf-8'), check=True)
    actual = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root).decode('utf-8').strip('\0').split('\0')
    assert sorted(actual) == sorted(files)
    for name in files:
        blob = subprocess.check_output(['git', 'show', ':'+name], cwd=root)
        assert hashlib.sha256(blob).hexdigest() == sha(root/name), name
    return dict(**result, staged_exact_files=len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['build', 'verify', 'stage'])
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    print(dict(action=args.action, **globals()[args.action](root)), flush=True)
