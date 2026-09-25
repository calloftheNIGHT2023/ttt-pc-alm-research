"""Compact 347--349 release, exclusive manifest, exact staged Git bytes."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST = 'release/strong-pool-20260925.json'
SOURCES = ['strong_pool_credit_bridge_v1.py', 'test_strong_pool_credit_bridge_v1.py',
    'strong_pool_online_suite_v1.py', 'run_strong_pool_online_v1.py', 'evaluate_strong_pool_online_v1.py',
    'audit_strong_pool_online_v1.py', 'continue_strong_pool_online_v1.ps1',
    'report_strong_pool_online_v1.py', 'audit_strong_pool_report_v1.py']
DOCS = ['347_strong_pool_credit_bridge_protocol_v1.md', '348_strong_pool_online_protocol_v1.md', '349_strong_pool_results_v1.md']
PANELS = {
    'results/strong_pool_credit_bridge/preflight_v1': ['summary.json', 'protocol.json', 'rows.json', 'files.json'],
    'results/strong_pool_online/preflight_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/strong_pool_online/development_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/strong_pool_online/development_evaluation_v1': ['summary.json', 'protocol.json', 'methods.json', 'comparisons.json', 'groups.json', 'task_metrics.npz'],
    'results/strong_pool_online/development_audit_v1': ['summary.json', 'mechanisms.json']}
REPORT = 'results/strong_pool_online/report_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))


def paths(root, *, full_local):
    files = {'scripts/strong_pool_snapshot_20260925.py', 'release/STRONG-POOL-20260925.md'}
    files.update('work/experiments/'+n for n in SOURCES)
    files.update('outputs/ttt-pc-alm-research/'+n for n in DOCS)
    for folder, names in PANELS.items():
        p = root/folder; summary = read(p/'summary.json')
        assert summary['passed'] and not (p/'failure.json').exists()
        if full_local:
            for name, digest in summary.get('outputs_sha256', {}).items(): assert sha(p/name) == digest
        files.update(folder+'/'+n for n in names)
    p = root/REPORT; m = read(p/'manifest.json'); q = read(p/'qa_numeric.json'); v = read(p/'visual_qa.json')
    assert q['passed'] and q['manifest_sha256'] == sha(p/'manifest.json')
    assert v['passed'] and v['actually_viewed'] and v['numeric_qa_sha256'] == sha(p/'qa_numeric.json')
    assert {r['file'] for r in v['images']} == {'strong_pool_mechanism.png', 'incremental_risk_and_cost.png'}
    for row in v['images']: assert sha(p/row['file']) == row['sha256']
    for name, digest in m['outputs_sha256'].items(): assert sha(p/name) == digest
    assert sha(root/m['entry_file']) == m['entry_sha256']
    files.update(REPORT+'/'+n for n in ['report.md', 'figure_data.json', 'table_data.json', 'manifest.json',
        'qa_numeric.json', 'visual_qa.json', 'strong_pool_mechanism.png', 'incremental_risk_and_cost.png'])
    return sorted(files)


def build(root):
    entries = []
    for name in paths(root, full_local=True):
        p = root/name; assert p.is_file() and p.resolve().is_relative_to(root.resolve()) and p.stat().st_size < 90*2**20
        if name.endswith('.py'): ast.parse(p.read_text(encoding='utf-8'), filename=name)
        entries.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    m = dict(version=1, parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        compact_archive=True, complete_particle_predictors_included=False, complete_call_geometry_logs_included=False,
        clean_clone_complete_scientific_reproduction=False, task_risk_matrix_included=True,
        file_count=len(entries), bytes=sum(e['bytes'] for e in entries), files=entries)
    with (root/MANIFEST).open('x', encoding='utf-8') as f: json.dump(m, f, indent=2, ensure_ascii=False, allow_nan=False)
    return verify(root)


def verify(root):
    m = read(root/MANIFEST); assert [e['path'] for e in m['files']] == paths(root, full_local=False)
    assert len(m['files']) == m['file_count'] and sum(e['bytes'] for e in m['files']) == m['bytes']
    for e in m['files']: assert sha(root/e['path']) == e['sha256'] and (root/e['path']).stat().st_size == e['bytes']
    return dict(passed=True, files=m['file_count'], bytes=m['bytes'])


def stage(root):
    result = verify(root); m = read(root/MANIFEST)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == m['parent_commit']
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root)
    names = [e['path'] for e in m['files']]+[MANIFEST]
    subprocess.run(['git', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=root,
        input=('\0'.join(names)+'\0').encode('utf-8'), check=True)
    staged = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root).decode().strip('\0').split('\0')
    assert sorted(staged) == sorted(names)
    for name in names: assert hashlib.sha256(subprocess.check_output(['git', 'show', ':'+name], cwd=root)).hexdigest() == sha(root/name)
    return dict(**result, staged_exact_files=len(names))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['build', 'verify', 'stage']); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; print(dict(action=args.action, **globals()[args.action](root)), flush=True)
