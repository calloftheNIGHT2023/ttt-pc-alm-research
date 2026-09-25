"""Explicit compact 337--341 release, with no blanket Git staging or auto-push."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

MANIFEST = 'release/radius-development-20260925.json'
SOURCES = ['budget_reinvestment_suite_v1.py', 'test_budget_reinvestment_v1.py',
    'calibrate_budget_reinvestment_v1.py', 'audit_budget_reinvestment_v1.py', 'continue_budget_reinvestment_v1.ps1',
    'audit_search_radius_reachability_v1.py', 'branch_image_chain_radius_v1.py', 'test_search_radius_primitive_v1.py',
    'search_radius_development_suite_v1.py', 'run_search_radius_development_v1.py',
    'evaluate_search_radius_development_v1.py', 'audit_search_radius_development_v1.py',
    'continue_search_radius_development_v1.ps1', 'report_search_radius_development_v1.py', 'audit_search_radius_report_v1.py']
DOCS = ['337_budget_reinvestment_calibration_protocol_v1.md', '338_search_radius_reachability_protocol_v1.md',
    '339_search_radius_primitive_protocol_v1.md', '340_radius_development_protocol_v1.md',
    '341_radius_development_results_v1.md', '342_radius_stage_conclusions_v1.md']
PANELS = {
    'results/budget_reinvestment/preflight_v1': ['summary.json', 'protocol.json', 'exhaustive.json', 'calls.json', 'files.json'],
    'results/budget_reinvestment/calibration_v1': ['summary.json', 'protocol.json', 'selection.json', 'methods.json', 'files.json', 'environments.json'],
    'results/budget_reinvestment/audit_v1': ['summary.json'],
    'results/search_radius_reachability/audit_v1': ['summary.json', 'protocol.json', 'tasks.json', 'methods.json', 'files.json'],
    'results/search_radius_primitive/preflight_v1': ['summary.json', 'protocol.json', 'exhaustive.json', 'calls.json', 'sources.json'],
    'results/search_radius_development/preflight_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/search_radius_development/development_predictions_v1': ['summary.json', 'protocol.json', 'rows.json', 'before_query_manifest.json'],
    'results/search_radius_development/development_evaluation_v1': ['summary.json', 'protocol.json', 'methods.json', 'comparisons.json', 'groups.json', 'task_metrics.npz'],
    'results/search_radius_development/development_audit_v1': ['summary.json'],
}
REPORT = 'results/search_radius_development/report_v1'


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8'))


def paths(root, *, full_local_validation):
    selected = {'scripts/radius_development_snapshot_20260925.py', 'release/RADIUS-DEVELOPMENT-20260925.md'}
    selected.update('work/experiments/'+n for n in SOURCES)
    selected.update('outputs/ttt-pc-alm-research/'+n for n in DOCS)
    for directory, names in PANELS.items():
        folder = root/directory; summary = read(folder/'summary.json')
        assert summary['passed'] and not (folder/'failure.json').exists(), directory
        if full_local_validation:
            for name, digest in summary.get('outputs_sha256', {}).items(): assert sha(folder/name) == digest
        selected.update(directory+'/'+n for n in names)
    report = root/REPORT; manifest = read(report/'manifest.json'); numeric = read(report/'qa_numeric.json'); visual = read(report/'visual_qa.json')
    assert numeric['passed'] and numeric['manifest_sha256'] == sha(report/'manifest.json')
    assert visual['passed'] and visual['actually_viewed'] and visual['numeric_qa_sha256'] == sha(report/'qa_numeric.json')
    assert {r['file'] for r in visual['images']} == {'radius_mechanism.png', 'radius_query_curves.png'}
    for row in visual['images']: assert row['sha256'] == sha(report/row['file'])
    for name, digest in manifest['outputs_sha256'].items(): assert sha(report/name) == digest
    assert manifest['entry_sha256'] == sha(root/manifest['entry_file'])
    selected.update(REPORT+'/'+n for n in ['report.md', 'manifest.json', 'figure_data.json', 'table_data.json',
        'qa_numeric.json', 'visual_qa.json', 'radius_mechanism.png', 'radius_query_curves.png'])
    return sorted(selected)


def build(root):
    files = paths(root, full_local_validation=True); entries = []
    for name in files:
        p = root/name; assert p.is_file() and p.resolve().is_relative_to(root.resolve())
        assert p.stat().st_size < 90*2**20, 'Do not silently add a large Git blob'
        if name.endswith('.py'): ast.parse(p.read_text(encoding='utf-8'), filename=name)
        entries.append(dict(path=name, sha256=sha(p), bytes=p.stat().st_size))
    value = dict(version=1, parent_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip(),
        preceding_scientific_release='70f62686f7c7813b3511db5beb92edbfc8f57a81', compact_archive=True,
        original_and_new_particle_predictor_npz_included=False, detailed_call_and_geometry_certificates_included=False,
        complete_clean_clone_scientific_reproduction=False, packaged_task_risk_matrix_included=True,
        files=entries, file_count=len(entries), bytes=sum(e['bytes'] for e in entries))
    with (root/MANIFEST).open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
    return verify(root)


def verify(root):
    m = read(root/MANIFEST); assert m['file_count'] == len(m['files'])
    assert m['bytes'] == sum(e['bytes'] for e in m['files'])
    assert [e['path'] for e in m['files']] == paths(root, full_local_validation=False)
    for e in m['files']:
        p = root/e['path']; assert p.stat().st_size == e['bytes'] and sha(p) == e['sha256'], e['path']
    return dict(passed=True, files=m['file_count'], bytes=m['bytes'])


def stage(root):
    result = verify(root); manifest = read(root/MANIFEST)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root).decode().strip() == manifest['parent_commit']
    assert not subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root), 'Preserve existing staged user changes'
    files = [e['path'] for e in manifest['files']]+[MANIFEST]
    subprocess.run(['git', 'add', '-f', '--pathspec-from-file=-', '--pathspec-file-nul'], cwd=root,
                   input=('\0'.join(files)+'\0').encode('utf-8'), check=True)
    actual = subprocess.check_output(['git', 'diff', '--cached', '--name-only', '-z'], cwd=root).decode('utf-8').strip('\0').split('\0')
    assert sorted(actual) == sorted(files)
    for name in files:
        content = subprocess.check_output(['git', 'show', ':'+name], cwd=root)
        assert hashlib.sha256(content).hexdigest() == sha(root/name), name
    return dict(**result, staged_exact_files=len(files))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=['build', 'verify', 'stage'])
    args = parser.parse_args(); root = Path(__file__).resolve().parents[1]
    print(dict(action=args.action, **globals()[args.action](root)), flush=True)
