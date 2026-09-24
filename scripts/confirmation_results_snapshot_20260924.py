"""Archive completed287 results/289 diagnostics/288preflight; exclude live stages."""
import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import snapshot as baseline
import confirmation_paths_snapshot_20260924 as previous


ROOT=Path(__file__).resolve().parents[1]
MANIFEST=ROOT/'release/confirmation-results-20260924.json'
STAGING=ROOT/'release/stage-confirmation-results-20260924.tmp'
PARENTS=previous.PARENTS+['release/confirmation-paths-20260924.json']
SEAL='results/round_287_audit_v6.json'
SEAL_HASH='4a0d5757631f62cf812de36681d573aadd4cf23d00ed8f09762a4234085c7af3'
TREES=[
    'results/probe_credit_confirmation/evaluation_v6',
    'results/probe_credit_confirmation/evaluation_audit_v6',
    'results/probe_credit_confirmation/figures_v7',
    'results/probe_credit_confirmation/pipeline_execution_v6',
    'results/gradient_flat_split_states/development_v1',
    'results/gradient_flat_split_states/branch_mass_v1',
    'results/gradient_flat_split_states/figures_v1',
    'results/known_range_projection/preflight_v1',
    'results/known_range_projection/predictions_preflight_v1',
    'results/known_range_projection/evaluation_preflight_v1',
    'results/known_range_projection/audit_preflight_v1',
]
EXTRAS={SEAL,'scripts/confirmation_results_snapshot_20260924.py','release/CONFIRMATION-RESULTS-20260924.md'}
EXTRAS.update('work/experiments/'+name for name in [
    'diagnose_gradient_flat_split_states_v1.py','diagnose_probe_branch_mass_v1.py',
    'plot_gradient_flat_branch_mass_v1.py','verify_known_range_projection_v1.py',
    'known_range_projection_pipeline_v1.py','audit_known_range_projection_v1.py','continue_known_range_projection_v1.py'])
EXTRAS.update('outputs/ttt-pc-alm-research/'+name for name in [
    '288_bounded_output_baseline_audit_plan.md','288_full_projection_protocol.md',
    '289_gradient_flat_split_state_protocol.md','289_branch_mass_followup_protocol.md','287_289_stage_report.md'])
EXTRAS.update('results/probe_credit_confirmation/prediction_audit_v2/'+name for name in ['summary.json','protocol.json','files.json'])
EXTRAS.update('results/probe_credit_confirmation/predictions/'+name for name in ['summary.json','before_query_manifest.json'])


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def safe(name):
    baseline.require('\\' not in name,'Canonical paths required')
    baseline.require(name in EXTRAS or any(name.startswith(folder+'/') for folder in TREES),'Outside completed allowlist: '+name)
    path=(ROOT/name).resolve()
    baseline.require(path.is_relative_to(ROOT) and path.is_file(),'Missing or escaping: '+name)
    return path


def inventory():
    existing=set().union(*(read(ROOT/name)['files'] for name in PARENTS))
    selected=set(EXTRAS)
    for folder in TREES:
        base=ROOT/folder
        baseline.require((base/'summary.json').is_file() and not (base/'failure.json').exists() and not (base/'RUNNING.lock').exists(),'Incomplete stage: '+folder)
        summary=read(base/'summary.json')
        baseline.require(summary['passed'],'Failed stage: '+folder)
        for name,digest in summary.get('outputs_sha256',{}).items():
            baseline.require(baseline.sha(safe(folder+'/'+name.replace('\\','/')))==digest,'Dependency differs: '+name)
        if 'protocol_sha256' in summary:
            baseline.require(baseline.sha(base/'protocol.json')==summary['protocol_sha256'],'Protocol differs: '+folder)
        if 'rows_sha256' in summary:
            baseline.require(baseline.sha(base/'rows.json')==summary['rows_sha256'],'Rows differ: '+folder)
        selected.update(p.relative_to(ROOT).as_posix() for p in base.rglob('*') if p.is_file())
    return sorted(selected-existing)


def validate(files):
    previous.verify()
    baseline.require(set(files)==set(inventory()),'Completed supplement census differs')
    baseline.require(baseline.sha(ROOT/SEAL)==SEAL_HASH,'Final original seal differs')
    seal=read(ROOT/SEAL)
    baseline.require(seal['passed'] and seal['tasks']==8192 and seal['main_adjusted_negative_upper_bounds']==21 and not seal['core_research_goal_complete'],'Unexpected scientific scope')
    for name,digest in seal['source_sha256'].items():
        baseline.require(baseline.sha(ROOT/'work/experiments'/name)==digest,'Frozen source differs: '+name)
    for name,digest in seal['report_sha256'].items():
        baseline.require(baseline.sha(ROOT/'outputs/ttt-pc-alm-research'/name)==digest,'Frozen report differs: '+name)
    fig=ROOT/'results/probe_credit_confirmation/figures_v7'
    review=read(fig/'visual_review.json')
    baseline.require(review['passed'] and review['stage']=='confirmation','Actual confirmation QA required')
    for name,digest in review['reviewed_files'].items():
        baseline.require(baseline.sha(fig/name)==digest,'Reviewed artifact differs: '+name)
    for name in files:
        path=safe(name)
        baseline.require(path.stat().st_size<90*1024*1024,'Oversized file: '+name)
        if path.suffix in {'.py','.json','.md'}:
            content=path.read_text(encoding='utf-8-sig')
            baseline.require(not baseline.SECRET.search(content),'Credential-like content: '+name)
            if path.suffix=='.py': ast.parse(content,filename=name)
    return dict(original_tasks=8192,original_predictors=221184,comparisons=104,main_thresholds_passed=21,
                old_mechanism_tasks=64,supplementary_preflight_tasks=2,
                raw_full_predictions_included=False,live_supplementary_results_included=False,
                credential_scan='Passed heuristic only',algorithm_executed=False,scientific_goal_complete=False)


def build():
    baseline.require(not MANIFEST.exists() and not STAGING.exists(),'Preserve existing manifest')
    files=inventory()
    checks=validate(files)
    entries={name:dict(bytes=safe(name).stat().st_size,sha256=baseline.sha(safe(name))) for name in files}
    data=dict(created_utc=datetime.now(timezone.utc).isoformat(),
              parent_manifests_sha256={name:baseline.sha(ROOT/name) for name in PARENTS},
              scope='Completed287 statistics/figures/seal, old289 diagnostics and288preflight, not live full288',
              files=entries,validation=checks,
              exclusions='No full original predictions, query truths, full old geometry prerequisites or live supplementary outputs')
    with MANIFEST.open('x',encoding='utf-8') as stream:
        json.dump(data,stream,ensure_ascii=False,indent=2)
        stream.write('\n')
    with STAGING.open('xb') as stream:
        stream.write(('\0'.join(files+[MANIFEST.relative_to(ROOT).as_posix()])+'\0').encode('utf-8'))
    print(json.dumps(dict(files=len(files),bytes=sum(r['bytes'] for r in entries.values()),validation=checks)))


def verify():
    data=read(MANIFEST)
    baseline.require(data['parent_manifests_sha256']=={name:baseline.sha(ROOT/name) for name in PARENTS},'Parent manifest changed')
    for name,record in data['files'].items():
        path=safe(name)
        baseline.require(path.stat().st_size==record['bytes'] and baseline.sha(path)==record['sha256'],'Supplement changed: '+name)
    print(json.dumps(dict(passed=True,files=len(data['files']),validation=validate(data['files']))))


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
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['build','verify','verify-index'])
    {'build':build,'verify':verify,'verify-index':verify_index}[parser.parse_args().action]()
