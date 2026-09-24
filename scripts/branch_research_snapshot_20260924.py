"""Immutable compact stage290-299 evidence supplement; no raw-data completeness claim."""
import argparse
import ast
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import snapshot as baseline
import confirmation_results_snapshot_20260924 as previous

ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'release/branch-research-20260924.json'
STAGING=ROOT/'release/stage-branch-research-20260924.tmp'
PARENTS=previous.PARENTS+['release/confirmation-results-20260924.json']
TREES=[
 'results/state_matched_dual_intervention/development_v1',
 'results/state_matched_dual_intervention/development_v2',
 'results/state_matched_dual_intervention/figures_v1',
 'results/counterfactual_branch_proposals/development_v1',
 'results/counterfactual_credit_branching/preflight_v1',
 'results/counterfactual_credit_branching/full_support_v2',
 'results/counterfactual_credit_branching/first_global_support_v1',
 'results/counterfactual_resources/preflight_v1',
 'results/counterfactual_resources/calibration_v1',
 'results/counterfactual_resources/audit_v1',
 'results/counterfactual_fresh_pilot/tests_v1',
 'results/counterfactual_fresh_pilot/preflight_evaluation_v1',
 'results/counterfactual_fresh_pilot/pilot_evaluation_v1',
 'results/counterfactual_fresh_pilot/preflight_diagnosis_v1',
 'results/counterfactual_fresh_pilot/pilot_diagnosis_v1',
 'results/counterfactual_fresh_pilot/preflight_report_v1',
 'results/counterfactual_fresh_pilot/pilot_report_v1',
 'results/counterfactual_fresh_pilot/preflight_report_v2',
 'results/counterfactual_fresh_pilot/pilot_report_v2',
 'results/counterfactual_conditional_risk/development_v1',
 'results/counterfactual_conditional_risk/development_v2',
 'results/counterfactual_conditional_risk/report_v1',
 'results/novel_branch_continuation/development_v1',
 'results/novel_branch_continuation/report_v1',
 'results/dual_amplitude_paths/development_v1',
 'results/dual_amplitude_paths/report_v1',
 'results/dual_amplitude_paths/report_v2',
 'results/stagnant_local_states/development_v1',
 'results/stagnant_local_states/development_v2',
]
EXTRAS={'scripts/branch_research_snapshot_20260924.py','release/BRANCH-RESEARCH-20260924.md'}
for stage in ['preflight','pilot']:
    for name in ['summary.json','protocol.json','before_query_manifest.json']:
        EXTRAS.add(f'results/counterfactual_fresh_pilot/{stage}_predictions_v1/{name}')
    EXTRAS.add(f'results/counterfactual_fresh_pilot/{stage}_evaluation_v1/task_metrics.npz')
EXTRAS.add('results/stagnant_local_states/development_v2/metrics.npz')
SOURCE_PATTERN=re.compile(r'(counterfactual|state_matched_dual|dual_amplitude|novel_branch|stagnant_local)')
DOC_PATTERN=re.compile(r'^(29[0-9]|300)_')


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
      'results/counterfactual_fresh_pilot/pilot_evaluation_v1/summary.json':'90f6555991bfcab0902ab6192705b533c9e8c4e29dbe2f9f3d732378e67167eb',
      'results/counterfactual_conditional_risk/development_v2/summary.json':'95852bf236596935f7ac0dd3c6fe389118b6976b5b2f368e09cc8ac37325a3ec',
      'results/counterfactual_credit_branching/first_global_support_v1/summary.json':'4f9ced52ec8c61548e00ebe2824998732c51febf3838fdbfb7016da99b5b247d',
      'results/novel_branch_continuation/development_v1/summary.json':'e56044455259e98b68ed5d2e7b0c7ad4c2860036e4f967d3755ce9325f21ee81',
      'results/dual_amplitude_paths/development_v1/summary.json':'1a90bc14baebe225c2501f8886e1e5e59077031fa894a0080a5f3a27d95a972c'}
    for name,digest in required.items():
        baseline.require(baseline.sha(ROOT/name)==digest,'Result seal differs: '+name)
        baseline.require(read(ROOT/name)['passed'],'Failed required stage')
    for folder in ['results/counterfactual_fresh_pilot/pilot_report_v2',
                   'results/counterfactual_conditional_risk/report_v1',
                   'results/novel_branch_continuation/report_v1',
                   'results/dual_amplitude_paths/report_v2']:
        baseline.require(read(ROOT/folder/'visual_review.json')['passed'],'Actual figure QA required: '+folder)
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
    return dict(passed=True,python_syntax_files=python_count,original128_predictions=4736,
        main_comparisons=36,old_development_tasks=64,latest_configs=46,support_states_censused=69632,
        full_raw_prediction_or_posterior_archive_included=False,
        core_research_goal_complete=False,algorithm_executed=False,credential_scan='heuristic passed')


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(),'Preserve existing manifest')
    files=inventory()
    validation=validate(files)
    entries={n:dict(bytes=safe(n).stat().st_size,sha256=baseline.sha(safe(n))) for n in files}
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        parent_manifests_sha256={n:baseline.sha(ROOT/n) for n in PARENTS},files=entries,validation=validation,
        scope='Completed stages290-299 compact source/statistics/figures; stage300 protocol NOT executed',
        exclusions='Most raw NPZ, full posterior reference, query truth, large bootstrap arrays, full288 supplement; not clean-clone end-to-end reproduction')
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
