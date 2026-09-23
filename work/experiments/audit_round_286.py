"""Seal actual285 resources, joint77 quality and fixed287 confirmation design."""
import argparse,json,re
from collections import Counter
from pathlib import Path
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_286_audit.json';assert not out.exists();parent=root/'results/round_284_audit.json';old=json.loads(parent.read_text());assert old['passed'];hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);counts=Counter();summaries={}
    stages=[('probe_credit_resources',n) for n in ['primitive','calibration','audit','figures']]+[('probe_credit_budget',n) for n in ['development','audit','evaluation','evaluation_audit','figures']]+[('probe_credit_budget_sensitivity','development'),('probe_credit_confirmation','planning')]
    for family,stage in stages:
        d=root/'results'/family/stage;s=json.loads((d/'summary.json').read_text());assert s['passed'];p=json.loads((d/'protocol.json').read_text());summaries[family+'/'+stage]=sha(d/'summary.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        for n,h in s.get('outputs_sha256',{}).items():assert sha(d/n)==h;counts['output_manifests']+=1
        for key in ['protocol','rows','geometries','coverage','comparisons']:
            if key+'_sha256' in s:assert sha(d/(key+'.json'))==s[key+'_sha256']
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for family in ['probe_credit_budget','probe_credit_budget_sensitivity']:
        d=root/'results'/family/'development';before=json.loads((d/'before_evaluation_manifest.json').read_text())
        for n,h in before['prediction_files'].items():assert sha(d/n)==h;counts['new_frozen_predictions']+=1
    assert counts['new_frozen_predictions']==512
    for family,stage,field in [('probe_credit_resources','calibration','resource_predictors'),('probe_credit_budget','evaluation','evaluation_curves')]:
        d=root/'results'/family/stage
        for n,h in json.loads((d/'files.json').read_text()).items():assert sha(d/n)==h;counts[field]+=1
    d=root/'results/probe_credit_budget/audit'
    for row in json.loads((d/'geometries.json').read_text()):assert sha(d/row['file'])==row['sha256'];counts['geometry_arrays']+=1
    counts.update(json.loads((root/'results/probe_credit_budget/evaluation_audit/summary.json').read_text())['counts']);assert counts['paired_intervals']==608
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['285_probe_credit_budget_sensitivity.md','286_probe_credit_budget_results.md','287_probe_credit_confirmation_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;counts['local_report_links']+=1
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=counts,
        scope='77-method development plus complete runtime selection; conditional signal but no independent-query confirmation yet',next='287 fixed8192-task27-method confirmation; check added heads/resources before any fresh contexts; goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)


if __name__=='__main__':main()
