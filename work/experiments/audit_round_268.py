"""Seal the complete observation-only posterior and risk decomposition."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent;out=root/'results/round_268_audit.json'
    assert not out.exists();parent=root/'results/round_265_audit.json';old=json.loads(parent.read_text());assert old['passed']
    hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);counts=Counter();summaries={}
    base=root/'results/confirmation_conditional_risk'
    for name in ['reference','boundary_certificates','reference_audit','moments','moments_audit','decomposition','decomposition_audit','figures']:
        directory=base/name;s=json.loads((directory/'summary.json').read_text());assert s['passed']
        p=json.loads((directory/'protocol.json').read_text());summaries[name]=sha(directory/'summary.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        if 'protocol_sha256' in s:assert sha(directory/'protocol.json')==s['protocol_sha256']
        for n,h in s.get('outputs_sha256',{}).items():assert sha(directory/n)==h;counts['output_manifests_checked']+=1
        for n in ['rows','tasks','files','coverage','gaps','certificates']:
            if f'{n}_sha256' in s:assert sha(directory/f'{n}.json')==s[f'{n}_sha256'];counts['named_manifest_checks']+=1
        if 'phase_accesses_query_targets' in s:assert not s['phase_accesses_query_targets']
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for r in json.loads((base/'reference/coverage.json').read_text()):
        assert sha(base/'reference'/r['file'])==r['sha256'];reference=json.loads((base/'reference'/r['file']).read_text())
        for g in reference['geometry'].values():assert sha(base/'reference'/g['file'])==g['sha256'];counts['reference_geometries']+=1
        counts['reference_tasks']+=1
    for directory,expected in [('moments',6440),('decomposition',64)]:
        ff=json.loads((base/directory/'files.json').read_text());assert len(ff)==expected
        for n,h in ff.items():assert sha(base/directory/n)==h;counts[f'{directory}_arrays']+=1
    ra=json.loads((base/'reference_audit/summary.json').read_text());assert ra['complete_up_to_certified_zero_volume']==64
    ma=json.loads((base/'moments_audit/summary.json').read_text());assert ma['counts']['support_and_membership_particles']==13058048
    da=json.loads((base/'decomposition_audit/summary.json').read_text());assert da['counts']['frozen_predictors']==2944 and da['counts']['paired_intervals']==270
    counts['moment_particles']=13058048;counts['risk_pair_metrics']=da['counts']['pair_metric_replays'];counts['paired_intervals']=270
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['266_confirmation_conditional_risk_protocol.md','267_conditional_risk_accounting_addendum.md','268_conditional_risk_results.md','269_complementary_mode_accounting_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;counts['local_report_links']+=1
    for n,h in json.loads((base/'decomposition/protocol.json').read_text())['design_sha256'].items():assert reports[n]==h
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,
        report_sha256=reports,summary_sha256=summaries,checks=counts,
        scope='All64 old confirmation tasks, no new predictors or query labels; numerical posterior diagnostic, not a new blind confirmation',
        next='269 fixed-pool complementary versus discarded-mode accounting; research goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)


if __name__=='__main__':main()
