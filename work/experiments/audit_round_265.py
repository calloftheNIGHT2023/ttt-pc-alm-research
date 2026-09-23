"""Freeze the budget-matched independent batch, numerical repair and all results."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;out=root/'results/round_265_audit.json';assert not out.exists()
    parent=root/'results/round_261_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);checks=Counter();summaries={};base=root/'results/matched_budget_confirmation'
    directories=['primitive','calibration','calibration_audit','confirmation','prediction_audit','geometry_diagnosis','geometry_determinant_audit','geometry_preflight','conditioned_confirmation','conditioned_prediction_audit','mode_identity','joint_evaluation','joint_evaluation_audit','figures','figures_v2']
    for name in directories:
        directory=base/name;data=json.loads((directory/'summary.json').read_text());assert data['passed'];summaries[name]=sha(directory/'summary.json');p=json.loads((directory/'protocol.json').read_text())
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        if 'protocol_sha256' in data:assert sha(directory/'protocol.json')==data['protocol_sha256']
        for n,h in data.get('outputs_sha256',{}).items():assert sha(directory/n)==h;checks['figure_and_table_artifacts']+=1
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for name,total in [('primitive',25),('calibration',400)]:
        directory=base/name;filename='rows.json' if name=='primitive' else 'timings.json';rows=json.loads((directory/filename).read_text());assert len(rows)==total
        for row in rows:assert sha(directory/row['file'])==row['sha256'];checks['calibration_primitive_files']+=1
    for name,memory_count in [('confirmation',92),('conditioned_confirmation',138)]:
        directory=base/name;summary=json.loads((directory/'summary.json').read_text());before=json.loads((directory/'before_query_manifest.json').read_text());assert sha(directory/'before_query_manifest.json')==summary['before_query_manifest_sha256']
        for n in ['protocol','rows','memory']:assert sha(directory/f'{n}.json')==before[f'{n}_sha256']
        rows=json.loads((directory/'rows.json').read_text());assert len(rows)==len(before['prediction_files'])==2944 and len({(r['seed'],r['method']) for r in rows})==2944
        assert len(json.loads((directory/'memory.json').read_text()))==memory_count;checks['memory_replays']+=memory_count
        for row in rows:
            assert sha(directory/row['file'])==row['sha256']==before['prediction_files'][row['file']];assert row['seconds']==row['metadata']['charged_complete_seconds']>0;checks['prediction_files_and_costs']+=1
            if name=='conditioned_confirmation':
                assert not row['metadata']['execution_failed']
                for event in row['metadata']['geometry_repair_log']:assert event['completed'] and event['repair_note']['boundary_relative_residual']<1e-10;checks['charged_geometry_repairs']+=1
    original=json.loads((base/'confirmation/summary.json').read_text());corrected=json.loads((base/'conditioned_confirmation/summary.json').read_text());assert original['failures']==32 and corrected['failures']==0 and corrected['unchanged_successes']==2912 and corrected['repaired_original_failures']==32
    for row in json.loads((base/'geometry_preflight/rows.json').read_text()):assert sha(base/'geometry_preflight'/row['file'])==row['sha256'];checks['repair_preflight_files']+=1
    for row in json.loads((base/'conditioned_prediction_audit/geometries.json').read_text()).values():assert sha(base/'conditioned_prediction_audit'/row['file'])==row['sha256'];checks['independent_repair_geometries']+=1
    gate=json.loads((base/'geometry_disposition.json').read_text());assert gate['may_open_query_answers'] and not gate['query_targets_accessed']
    for n,h in gate['required_artifacts'].items():assert sha(base/n)==h;checks['prequery_gate_artifacts']+=1
    inp=base/'joint_evaluation';ss=json.loads((inp/'summary.json').read_text());assert ss['query_evaluations']==5888 and ss['tasks']==64 and ss['methods']==46 and ss['implementation_versions']==2
    for n in ['protocol','query_rows','methods','paired','implementation_changes']:assert sha(inp/f'{n}.json')==ss[f'{n}_sha256']
    independent=json.loads((base/'joint_evaluation_audit/summary.json').read_text());assert independent['counts']['independent_risks']==11776 and independent['counts']['paired_intervals']==90
    checks['independent_risks']=11776;checks['paired_intervals']=90;checks['method_tables']=92
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['263_budget_calibration_results.md','264_common_geometry_prequery_protocol.md','265_matched_budget_confirmation_results.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_links']+=1
    assert reports['264_common_geometry_prequery_protocol.md']==json.loads((base/'geometry_preflight/protocol.json').read_text())['design_sha256']
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        geometry_disposition_sha256=sha(base/'geometry_disposition.json'),unused_single_version_evaluator_sha256=sha(src/'evaluate_budget_confirmation.py'),
        scope='64 tasks in two implementations, not128 independent tasks; original and common-corrected results both preserved',next='research goal remains active; choose the next mechanism from frozen task evidence, not by changing this confirmation')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)

if __name__=='__main__':main()
