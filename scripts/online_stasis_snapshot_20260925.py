"""Immutable compact stage300-307 evidence supplement; no raw-data completeness claim."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import snapshot as baseline
import branch_research_snapshot_20260924 as previous

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'release/online-stasis-20260925.json'
STAGING=ROOT/'release/stage-online-stasis-20260925.tmp'
PARENTS=previous.PARENTS+['release/branch-research-20260924.json']
TREES=[
 "results/stagnant_trigger_exploration/development_v1",
 "results/stagnant_trigger_exploration/evaluation_v2",
 "results/stagnant_trigger_exploration/report_v1",
 "results/projected_stationarity/development_v1",
 "results/projected_stationarity/audit_v1",
 "results/support_boundary_events/development_v1",
 "results/observable_trap_continuation/development_v1",
 "results/observable_trap_continuation/report_v1",
 "results/observable_trap_continuation/report_v2",
 "results/observable_trap_continuation/report_v3",
 "results/online_stasis_memory/functional_v1",
 "results/online_stasis_memory/functional_v2",
 "results/online_stasis_resources/calibration_v1",
 "results/online_stasis_resources/audit_v1",
 "results/online_stasis_resources/report_v1",
 "results/online_stasis_resources/report_v2",
 "results/online_stasis_controls/functional_v1",
 "results/online_stasis_controls/functional_v2",
 "results/online_stasis_controls/audit_v1",
 "results/online_stasis_controls/reduction_audit_v1",
 "results/online_stasis_fresh_pilot/tests_v1",
 "results/online_stasis_fresh_pilot/preflight_evaluation_v1",
 "results/online_stasis_fresh_pilot/pilot_evaluation_v1",
 "results/online_stasis_fresh_pilot/audit_v1",
 "results/online_stasis_fresh_pilot/diagnosis_v1",
 "results/online_stasis_fresh_pilot/report_v1",
 "results/online_stasis_fresh_pilot/drift_lemma_tests_v1"
]
EXTRAS={'scripts/online_stasis_snapshot_20260925.py','release/ONLINE-STASIS-20260925.md'}
for stage in ['preflight','pilot']:
    for name in ['summary.json','protocol.json','before_query_manifest.json','costs.json','files.json','environments.json']:
        EXTRAS.add(f'results/online_stasis_fresh_pilot/{stage}_predictions_v1/{name}')
    EXTRAS.add(f'results/online_stasis_fresh_pilot/{stage}_evaluation_v1/task_metrics.npz')

SOURCE_PATTERN=re.compile(r'(stagnant_trigger|stationarity|support_boundary_events|observable_trap|online_stasis|residual_memory_drift)')
DOC_PATTERN=re.compile(r'^30[0-7]_')


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def allowed(name):
    return (name in EXTRAS or
      any(name.startswith(folder+'/') and Path(name).suffix in {'.json','.md','.png'} for folder in TREES) or
      (name.startswith('work/experiments/') and Path(name).suffix in {'.py','.ps1'} and SOURCE_PATTERN.search(Path(name).name)) or
      (name.startswith('outputs/ttt-pc-alm-research/') and Path(name).suffix=='.md' and DOC_PATTERN.match(Path(name).name) and '_handoff_' not in name))


def safe(name):
    baseline.require('\\' not in name and bool(allowed(name)),'Outside compact allowlist: '+name)
    path=(ROOT/name).resolve()
    baseline.require(path.is_relative_to(ROOT) and path.is_file(),'Missing/unsafe: '+name)
    return path


def inventory():
    existing=set().union(*(read(ROOT/name)['files'] for name in PARENTS))
    selected=set(EXTRAS)
    for folder in TREES:
        base=ROOT/folder
        baseline.require(base.is_dir(),'Missing stage: '+folder)
        selected.update(p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file() and p.suffix in {'.json','.md','.png'})
    for base,suffixes in [('work/experiments',{'.py','.ps1'}),('outputs/ttt-pc-alm-research',{'.md'})]:
        selected.update(p.relative_to(ROOT).as_posix() for p in (ROOT/base).iterdir()
                        if p.is_file() and p.suffix in suffixes and allowed(p.relative_to(ROOT).as_posix()))
    return sorted(selected-existing)


def validate(files):
    previous.verify()
    # The manifest fixes the published inventory; future ignored experiments
    # must not invalidate verification of an earlier immutable supplement.
    baseline.require(all(allowed(n) for n in files),'Supplement contains an unapproved path')
    required={
       "results/online_stasis_fresh_pilot/pilot_predictions_v1/summary.json": "996e6b6819e88a9bb7c3ee86282fb8dbd9f372e8b2a12240d1ae51e4e354d725",
       "results/online_stasis_fresh_pilot/pilot_evaluation_v1/summary.json": "a3b158d330209a143965a198725ce7bf40534ebbee6e2bb75bd692113a9b75dd",
       "results/online_stasis_fresh_pilot/audit_v1/summary.json": "a3bb5ea471a024bbe9d5edb1181450d71d2d1ed47649c4e38e78c58d4de94dd2",
       "results/online_stasis_fresh_pilot/diagnosis_v1/summary.json": "e340e294a317080d5ce00f9a67cd6b7f36c0cb4e3f6ba8994adcda12bd838af7",
       "results/online_stasis_fresh_pilot/report_v1/summary.json": "33a1c9d2c63ef3460c6a7842e717934c270bf26f08c984728938f2f0c9f7801a",
       "results/online_stasis_fresh_pilot/report_v1/visual_qa_v1.json": "2d0d3cb445d1ebed283c82160e51176fd9480101c73f00da18cd8f93cdb5872b",
       "results/online_stasis_fresh_pilot/drift_lemma_tests_v1/summary.json": "86d126c380c21fbd958fb9e0eb324758a5730cefc05dcee603ba8fd9f41befa4"
   }
    for name,digest in required.items():
        baseline.require(baseline.sha(ROOT/name)==digest,'Result seal differs: '+name)
    qa=read(ROOT/'results/online_stasis_fresh_pilot/report_v1/visual_qa_v1.json')
    baseline.require(qa['passed'] and qa['checks']['numeric_fields_checked']==1447,'Actual QA missing')
    for name,digest in qa['images'].items():
        baseline.require(baseline.sha(ROOT/'results/online_stasis_fresh_pilot/report_v1'/name)==digest,'QA image changed')
    python_count=0
    for name in files:
        path=safe(name)
        baseline.require(path.stat().st_size<90*1024*1024,'Oversized: '+name)
        if path.suffix in {'.py','.ps1','.json','.md'}:
            content=path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content),'Credential-like text: '+name)
            if path.suffix=='.py':
                ast.parse(content,filename=name)
                python_count+=1
    return dict(passed=True,python_syntax_files=python_count,new_tasks=256,methods=46,predictions=11776,
        main_comparisons=45,all_comparisons=180,old_development_tasks=64,forward_stasis_states=131,
        full_raw_prediction_or_posterior_archive_included=False,
        core_research_goal_complete=False,algorithm_executed=False,credential_scan='heuristic passed')


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(),'Preserve existing manifest')
    files=inventory()
    validation=validate(files)
    entries={n:dict(bytes=safe(n).stat().st_size,sha256=baseline.sha(safe(n))) for n in files}
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifests_sha256={n:baseline.sha(ROOT/n) for n in PARENTS},files=entries,validation=validation,
        scope='Completed stages300-307 compact source/statistics/figures; ongoing308 excluded',
        exclusions='Most raw NPZ and prediction metadata, full posterior reference, query truth, large bootstrap arrays, ongoing308; not clean-clone end-to-end reproduction')
    with MANIFEST.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    with STAGING.open('xb') as stream:
        stream.write(('\0'.join(files+[MANIFEST.relative_to(ROOT).as_posix()])+'\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files),bytes=sum(v['bytes'] for v in entries.values()),validation=validation)))


def verify():
    data=read(MANIFEST)
    baseline.require(data['parent_manifests_sha256']=={n:baseline.sha(ROOT/n) for n in PARENTS},'Parent changed')
    for name,r in data['files'].items():
        path=safe(name)
        baseline.require(path.stat().st_size==r['bytes'] and baseline.sha(path)==r['sha256'],'Changed: '+name)
    print(json.dumps(dict(files=len(data['files']),validation=validate(data['files']))))


def verify_index():
    verify()
    expected=set(read(MANIFEST)['files'])|{MANIFEST.relative_to(ROOT).as_posix()}
    actual=set(subprocess.check_output(['git','diff','--cached','--name-only','-z'],cwd=ROOT).decode('utf-8').split('\0'))-{''}
    baseline.require(actual==expected,'Staged census mismatch')
    for name in sorted(expected):
        blob=subprocess.check_output(['git','show',':'+name],cwd=ROOT)
        baseline.require(hashlib.sha256(blob).hexdigest()==baseline.sha(ROOT/name),'Staged bytes differ: '+name)
    print(json.dumps(dict(staged_hashes_passed=True,files=len(expected))))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action',choices=['build','verify','verify-index'])
    {'build':build,'verify':verify,'verify-index':verify_index}[ap.parse_args().action]()
