"""Explicit compact 343--346 release; source/result SHA and exact staging."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST = 'release/support-language-20260925.json'
SOURCES = ['audit_radius_candidate_census_v1.py', 'support_language_chain_v1.py', 'test_support_language_chain_v1.py',
    'support_language_online_suite_v1.py', 'run_support_language_online_v1.py', 'evaluate_support_language_online_v1.py',
    'audit_support_language_online_v1.py', 'continue_support_language_online_v1.ps1',
    'report_support_language_online_v1.py', 'audit_support_language_report_v1.py']
DOCS = ['343_radius_candidate_census_protocol_v1.md', '344_support_language_primitive_protocol_v1.md',
        '345_support_language_online_protocol_v1.md', '346_support_language_results_v1.md']
PANELS = {
    'results/radius_candidate_census/audit_v1': ['summary.json', 'protocol.json', 'channels.json', 'missing.json', 'files.json', 'inputs.json'],
    'results/support_language_primitive/preflight_v1': ['summary.json', 'protocol.json', 'exhaustive.json', 'calls.json'],
    'results/support_language_online/preflight_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/support_language_online/development_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/support_language_online/development_evaluation_v1': ['summary.json', 'protocol.json', 'methods.json', 'comparisons.json', 'groups.json', 'task_metrics.npz'],
    'results/support_language_online/development_audit_v1': ['summary.json', 'mechanisms.json']}
REPORT = 'results/support_language_online/report_v1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path): return json.loads(path.read_text(encoding='utf-8'))


def paths(root, *, full_local):
    files = {'scripts/support_language_snapshot_20260925.py', 'release/SUPPORT-LANGUAGE-20260925.md'}
    files.update('work/experiments/'+n for n in SOURCES)
    files.update('outputs/ttt-pc-alm-research/'+n for n in DOCS)
    for folder, names in PANELS.items():
        p = root/folder; s = read(p/'summary.json')
        assert s['passed'] and not (p/'failure.json').exists()
        if full_local:
            for n, digest in s.get('outputs_sha256', {}).items(): assert sha(p/n) == digest
        files.update(folder+'/'+n for n in names)
    census = 'results/radius_candidate_census/audit_v1'
    for name, digest in read(root/census/'files.json').items():
        assert sha(root/census/name) == digest; files.add(census+'/'+name)
    p = root/REPORT; m = read(p/'manifest.json'); q = read(p/'qa_numeric.json'); v = read(p/'visual_qa.json')
    assert q['passed'] and q['manifest_sha256'] == sha(p/'manifest.json')
    assert v['passed'] and v['actually_viewed'] and v['numeric_qa_sha256'] == sha(p/'qa_numeric.json')
    assert {r['file'] for r in v['images']} == {'query_comparison.png', 'memory_and_cost.png'}
    for r in v['images']: assert sha(p/r['file']) == r['sha256']
    for n, digest in m['outputs_sha256'].items(): assert sha(p/n) == digest
    assert sha(root/m['entry_file']) == m['entry_sha256']
    files.update(REPORT+'/'+n for n in ['report.md', 'figure_data.json', 'table_data.json', 'manifest.json',
                                      'qa_numeric.json', 'visual_qa.json', 'query_comparison.png', 'memory_and_cost.png'])
    return sorted(files)


def build(root):
    entries = []
    for name in paths(root, full_local=True):
        p = root/name; assert p.is_file() and p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size < 90*2**20
        if name.endswith('.py'): ast.parse(p.read_text(encoding='utf-8'), filename=name)
        entries.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    manifest = dict(version=1, parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        compact_archive=True, complete_particle_predictors_included=False, complete_call_geometry_logs_included=False,
        clean_clone_complete_scientific_reproduction=False, task_risks_and_census_included=True,
        file_count=len(entries), bytes=sum(e['bytes'] for e in entries), files=entries)
    with (root/MANIFEST).open('x', encoding='utf-8') as f: json.dump(manifest, f, indent=2, ensure_ascii=False, allow_nan=False)
    return verify(root)


def verify(root):
    m = read(root/MANIFEST); assert [e['path'] for e in m['files']] == paths(root, full_local=False)
    assert len(m['files']) == m['file_count'] and sum(e['bytes'] for e in m['files']) == m['bytes']
    for e in m['files']:
        assert sha(root/e['path']) == e['sha256'] and (root/e['path']).stat().st_size == e['bytes']
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
    for name in names:
        assert hashlib.sha256(subprocess.check_output(['git', 'show', ':'+name], cwd=root)).hexdigest() == sha(root/name)
    return dict(**result, staged_exact_files=len(names))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['build', 'verify', 'stage']); args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]; print(dict(action=args.action, **globals()[args.action](root)), flush=True)
