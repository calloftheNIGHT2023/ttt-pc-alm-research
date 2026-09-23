"""Seal all fixed-restart development evidence and next prespecified design."""
import argparse,json,re
from pathlib import Path
from collections import Counter
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_274_audit.json';assert not out.exists();parent=root/'results/round_272_audit.json';old=json.loads(parent.read_text());assert old['passed']
    hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);checks=Counter();summaries={};base=root/'results/anchor_preserving_fork'
    for name in ['primitive','development','audit','evaluation','evaluation_audit','figures']:
        directory=base/name;s=json.loads((directory/'summary.json').read_text());assert s['passed'];p=json.loads((directory/'protocol.json').read_text());summaries[name]=sha(directory/'summary.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        for n,h in s.get('outputs_sha256',{}).items():assert sha(directory/n)==h;checks['output_manifests']+=1
        for key in ['protocol','rows','geometries','coverage']:
            if key+'_sha256' in s:assert sha(directory/(key+'.json'))==s[key+'_sha256']
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    before=json.loads((base/'development/before_evaluation_manifest.json').read_text())
    for n,h in before['prediction_files'].items():assert sha(base/'development'/n)==h;checks['frozen_predictions']+=1
    assert checks['frozen_predictions']==256
    for g in json.loads((base/'audit/geometries.json').read_text()):assert sha(base/'audit'/g['file'])==g['sha256'];checks['geometry_arrays']+=1
    for n,h in json.loads((base/'evaluation/files.json').read_text()).items():assert sha(base/'evaluation'/n)==h;checks['evaluation_curve_files']+=1
    ra=json.loads((base/'evaluation_audit/summary.json').read_text());assert ra['counts']['frozen_predictors']==3456 and ra['counts']['paired_intervals']==424
    checks.update(ra['counts']);docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['274_anchor_preserving_fork_results.md','275_distinct_prior_credit_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_report_links']+=1
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        scope='Development, all54 methods; protected-anchor trajectories and54-method risks audited; no matched-wall-time claim',next='275 distinct33-prior action assignment, fixed1088 restart-sweeps; goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)


if __name__=='__main__':main()
