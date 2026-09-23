"""Seal exact279 response-path primitive and281 attribution design."""
import argparse,json,re
from pathlib import Path
from collections import Counter
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_280_audit.json';assert not out.exists();parent=root/'results/round_278_audit.json';old=json.loads(parent.read_text());assert old['passed']
    hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);checks=Counter();summaries={};base=root/'results/actual_credit_response'
    for name in ['primitive','primitive_audit','figures']:
        directory=base/name;s=json.loads((directory/'summary.json').read_text());assert s['passed'];p=json.loads((directory/'protocol.json').read_text());summaries[name]=sha(directory/'summary.json')
        for n,h in p['source_sha256'].items():
            if n in hashes:assert hashes[n]==h,n
            hashes[n]=h
        for n,h in s.get('outputs_sha256',{}).items():assert sha(directory/n)==h;checks['output_manifests']+=1
        if 'protocol_sha256' in s:assert sha(directory/'protocol.json')==s['protocol_sha256']
    hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n,h in json.loads((base/'primitive/files.json').read_text()).items():assert sha(base/'primitive'/n)==h;checks['exact_curve_files']+=1
    ra=json.loads((base/'primitive_audit/summary.json').read_text());assert ra['checks']['states']==132;checks.update(ra['checks'])
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['280_actual_credit_response_results.md','281_certificate_activity_attribution_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_report_links']+=1
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        scope='Exact open-cell mechanism primitive, no independent response-shape gain over simple chord; no query claim',
        next='281 certificate-selected activity attribution versus six full33-start controls; goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)


if __name__=='__main__':main()
