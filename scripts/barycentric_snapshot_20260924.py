"""Completed mean-coupling supplement; never traverse the live v6 pipeline."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import snapshot as baseline
import geometry_snapshot_20260924 as previous


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'release/barycentric-20260924.json'
STAGING=ROOT/'release/stage-barycentric-20260924.tmp'
PARENTS=['release/snapshot.json','release/verification-20260924.json','release/geometry-20260924.json']
SEAL='results/round_287_preflight_audit_v6.json'
TREES=['task_5500037_geometry_census_v1','barycentric_position_diagnosis_v1',
       'pool_geometry_gate_selftests_v4','path_audit_preflight_v6','evaluation_preflight_v6',
       'evaluation_audit_preflight_v6','figures_preflight_v7']
FORBIDDEN=['results/probe_credit_confirmation/'+name+'/' for name in
           ['predictions','path_audit_v6','pipeline_execution_v6','evaluation_v6','evaluation_audit_v6','figures_v7']]


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def safe(name):
    path=ROOT/name
    baseline.require(path.is_file() and path.resolve().is_relative_to(ROOT),'Missing/unsafe: '+name)
    baseline.require(not any(name.startswith(prefix) for prefix in FORBIDDEN),'Live/full output selected: '+name)
    return path


def inventory():
    existing=set().union(*(read(ROOT/name)['files'] for name in PARENTS))
    selected=set()
    def add(name):
        safe(name)
        if name not in existing:selected.add(name)
    seal=read(ROOT/SEAL)
    baseline.require(seal['passed'] and seal['functional_preflight_only'] and seal['tasks']==2,'Old preflight required')
    add(SEAL)
    for name in seal['source_sha256']:add('work/experiments/'+name)
    add('work/experiments/continue_probe_confirmation_pipeline_v6.py')
    for name in seal['report_sha256']:add('outputs/ttt-pc-alm-research/'+name)
    add('scripts/barycentric_snapshot_20260924.py');add('release/BARYCENTRIC-20260924.md')
    base=ROOT/'results/probe_credit_confirmation'
    for name in TREES:
        folder=base/name
        baseline.require((folder/'summary.json').is_file() and not (folder/'RUNNING.lock').exists(),'Incomplete: '+name)
        for path in folder.rglob('*'):
            if path.is_file() and path.suffix in {'.json','.npz','.png','.md'}:add(path.relative_to(ROOT).as_posix())
    failed=base/'pipeline_execution_v5'
    baseline.require(read(failed/'paths_exit.json')['returncode']==1 and (failed/'failure.json').is_file(),'Terminal failure required')
    for name in ['protocol.json','failure.json','paths_start.json','paths_exit.json','all_prediction_complete.json']:
        add('results/probe_credit_confirmation/pipeline_execution_v5/'+name)
    return sorted(selected)


def validate(files):
    previous.verify()
    count=0
    for name in files:
        path=safe(name)
        baseline.require(path.stat().st_size<25*1024*1024,'Large file: '+name)
        if path.suffix in {'.py','.json','.md'}:
            content=path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content),'Credential-like text: '+name)
            if path.suffix=='.py':ast.parse(content,filename=name);count+=1
    seal=read(ROOT/SEAL)
    for name,h in seal['source_sha256'].items():baseline.require(baseline.sha(ROOT/'work/experiments'/name)==h,'Frozen source changed: '+name)
    for name,h in seal['report_sha256'].items():baseline.require(baseline.sha(ROOT/'outputs/ttt-pc-alm-research'/name)==h,'Frozen report changed: '+name)
    return dict(new_python_syntax_files=count,frozen_source_count=seal['source_count'],
                credential_scan='passed heuristic, not a security guarantee',algorithm_executed=False)


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(),'Preserve existing supplement')
    files=inventory();checks=validate(files)
    entries={name:dict(bytes=(ROOT/name).stat().st_size,sha256=baseline.sha(ROOT/name)) for name in files}
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),parent_manifests_sha256={name:baseline.sha(ROOT/name) for name in PARENTS},
              scope='Mean-barycentric proof, completed diagnostic fixtures and old-task preflight; NOT new query-quality results',
              files=entries,validation=checks,exclusions='No live v6 stages, full8192predictions/inputs or query truths; see BARYCENTRIC-20260924.md')
    with MANIFEST.open('x',encoding='utf-8') as stream:json.dump(data,stream,indent=2,ensure_ascii=False);stream.write('\n')
    with STAGING.open('xb') as stream:stream.write(('\0'.join(files+[MANIFEST.relative_to(ROOT).as_posix()])+'\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files),bytes=sum(row['bytes'] for row in entries.values()),validation=checks)))


def verify():
    data=read(MANIFEST)
    baseline.require(data['parent_manifests_sha256']=={name:baseline.sha(ROOT/name) for name in PARENTS},'Parent manifest changed')
    for name,row in data['files'].items():
        path=safe(name)
        baseline.require(path.stat().st_size==row['bytes'] and baseline.sha(path)==row['sha256'],'Supplement changed: '+name)
    print(json.dumps(dict(passed=True,files=len(data['files']),validation=validate(data['files']),scientific_goal_complete=False)))


def verify_index():
    verify()
    expected=set(read(MANIFEST)['files'])|{MANIFEST.relative_to(ROOT).as_posix()}
    actual=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0'))-{''}
    baseline.require(actual==expected,'Staged census differs')
    for name in sorted(expected):
        blob=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
        baseline.require(hashlib.sha256(blob).hexdigest()==baseline.sha(ROOT/name),'Staged bytes differ: '+name)
    print(json.dumps(dict(staged_byte_hashes_passed=True,files=len(expected))))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['build','verify','verify-index'])
    {'build':build,'verify':verify,'verify-index':verify_index}[ap.parse_args().action]()
