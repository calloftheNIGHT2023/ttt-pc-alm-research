"""Explicit completed-verification supplement; never enumerate live outputs."""
import argparse
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import snapshot as baseline


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'release/verification-20260924.json'
STAGING=ROOT/'release/stage-verification-20260924.tmp'
BASE=ROOT/'release/snapshot.json'
SEALS=['results/round_287_preflight_audit_v2.json','results/round_287_preflight_audit_v3.json']
COMPLETED_TREES=[
    'boundary_diagnosis_preflight','boundary_error_budget_preflight','boundary_diagnostic_selftests',
    'exact_polytope_certificate_preflight','exact_polytope_certificate_independent_audit','exact_geometry_gate_selftests',
    'prediction_audit_preflight','prediction_audit_preflight_v2',
    'path_audit_preflight_v2','path_audit_preflight_v3',
    'evaluation_preflight_v3','evaluation_audit_preflight_v3','figures_preflight_v3',
    'precise_head_selftests_failed_v1','precise_head_selftests_v2',
]
EXTRA_SOURCES=[
    'diagnose_probe_confirmation_boundary.py','analyze_probe_boundary_error_budget.py','test_probe_boundary_diagnosis.py',
    'diagnose_probe_confirmation_head.py','probe_confirmation_precise_heads.py','test_probe_confirmation_precise_heads.py',
    'continue_probe_confirmation_pipeline_v2.py','continue_probe_confirmation_pipeline_v3.py','audit_round_287.py',
]


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def inventory():
    existing=set(read(BASE)['files']); selected=set()
    def add(relative):
        path=ROOT/relative
        baseline.require(path.is_file() and path.resolve().is_relative_to(ROOT), 'Missing/unsafe selected file: '+relative)
        if relative not in existing: selected.add(relative)
    sources=set(EXTRA_SOURCES)
    for name in SEALS:
        seal=read(ROOT/name);baseline.require(seal['passed'] and seal['functional_preflight_only'],name)
        sources.update(seal['source_sha256'])
        for report in seal['report_sha256']:add('outputs/ttt-pc-alm-research/'+report)
        add(name)
    for name in sources:add('work/experiments/'+name)
    for name in ['287_boundary_preflight_diagnosis.md','287_head_reference_diagnosis.md','287_exact_geometry_certificate.md']:
        add('outputs/ttt-pc-alm-research/'+name)
    add('scripts/verification_snapshot_20260924.py');add('release/VERIFICATION-20260924.md')
    base=ROOT/'results/probe_credit_confirmation'
    for directory in COMPLETED_TREES:
        folder=base/directory
        baseline.require((folder/'summary.json').is_file() and not (folder/'RUNNING.lock').exists(), 'Incomplete selected stage: '+directory)
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in {'.json','.npz','.png','.md'}:add(path.relative_to(ROOT).as_posix())
    # New-task reference diagnostics: metadata ONLY, no prediction/input arrays.
    for directory,files in {
        'head_discrepancy_diagnosis':['summary.json','heads.json'],
        'meta_ridge_arithmetic_diagnosis':['summary.json','diagnostics.json'],
        'meta_ridge_reference_census':['summary.json','protocol.json','files.json'],
        'path_audit_preflight':['protocol.json'],
        'pipeline_execution_v2':['protocol.json','failure.json'],
    }.items():
        for name in files:add(f'results/probe_credit_confirmation/{directory}/{name}')
    return sorted(selected)


def validate(files):
    baseline.verify()
    count=0
    for name in files:
        path=ROOT/name
        baseline.require(path.resolve().is_relative_to(ROOT) and path.is_file(),name)
        baseline.require(path.stat().st_size<25*1024*1024,'Large selected file: '+name)
        if path.suffix in {'.py','.md','.json'}:
            content=path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content),'Credential-like text: '+name)
            if path.suffix=='.py':ast.parse(content,filename=name);count+=1
    for name in SEALS:
        seal=read(ROOT/name)
        for source,digest in seal['source_sha256'].items():
            baseline.require(baseline.sha(ROOT/'work/experiments'/source)==digest,'Frozen source changed: '+source)
        for report,digest in seal['report_sha256'].items():
            baseline.require(baseline.sha(ROOT/'outputs/ttt-pc-alm-research'/report)==digest,'Frozen report changed: '+report)
    return dict(new_python_syntax_files=count,source_counts={name:read(ROOT/name)['source_count'] for name in SEALS},
        credential_pattern_scan='passed heuristic; not a security guarantee',algorithm_executed=False)


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(),'Preserve existing supplement')
    files=inventory();checks=validate(files)
    entries={name:dict(bytes=(ROOT/name).stat().st_size,sha256=baseline.sha(ROOT/name)) for name in files}
    value=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_snapshot_sha256=baseline.sha(BASE),
        scope='Completed numerical verification and old-task functional preflight; NOT new task-quality results',
        files=entries,validation=checks,
        exclusions='No live stage directories, new-task prediction/input arrays, query truths, full reference-census task records, or full historical checkpoints; see VERIFICATION-20260924.md')
    with MANIFEST.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,ensure_ascii=False);stream.write('\n')
    with STAGING.open('xb') as stream:stream.write(('\0'.join(files+[MANIFEST.relative_to(ROOT).as_posix()])+'\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files),bytes=sum(row['bytes'] for row in entries.values()),validation=checks)))


def verify():
    value=read(MANIFEST);baseline.require(value['parent_snapshot_sha256']==baseline.sha(BASE),'Base snapshot changed')
    for name,entry in value['files'].items():
        path=ROOT/name
        baseline.require(path.resolve().is_relative_to(ROOT) and path.stat().st_size==entry['bytes'] and baseline.sha(path)==entry['sha256'], 'Supplement file changed: '+name)
    checks=validate(value['files'])
    print(json.dumps(dict(passed=True,files=len(value['files']),validation=checks,scientific_goal_complete=False)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['build','verify'])
    args=parser.parse_args();build() if args.action=='build' else verify()
