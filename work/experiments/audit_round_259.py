"""Freeze258-259 mechanism and engineering evidence plus260 protocol."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;out=root/'results/round_259_audit.json';assert not out.exists()
    parent=root/'results/round_257_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);base=root/'results/minimum_sufficient_dual';summaries={};checks=Counter()
    for folder in ['short_primitive','short_resources','threshold','threshold_audit','transfer','transfer_audit','analysis','figures','figures_v2']:
        directory=base/folder;summary=json.loads((directory/'summary.json').read_text());assert summary['passed'];summaries[folder]=sha(directory/'summary.json');protocol=directory/'protocol.json'
        if protocol.exists():
            data=json.loads(protocol.read_text());hashes.update(data['source_sha256']);assert sha(protocol)==summary['protocol_sha256']
        for name,h in summary.get('output_sha256',{}).items():assert sha(directory/name)==h;checks['analysis_artifacts']+=1
        for name in ['rows','files','support_rows','geometry_files','exact_bias_gaps','credit','timings','memory']:
            key=name+'_sha256'
            if key in summary:assert sha(directory/f'{name}.json')==summary[key];checks['summary_artifacts']+=1
    hashes.update({n:sha(src/n) for n in ['plot_minimum_dual.py','plot_minimum_dual_v2.py',Path(__file__).name]})
    for n,h in hashes.items():assert sha(src/n)==h,n
    for folder in ['threshold','transfer']:
        directory=base/folder
        for manifest in ['files.json','support_rows.json','geometry_files.json']:
            if not (directory/manifest).exists():continue
            for item in json.loads((directory/manifest).read_text()):assert sha(directory/item['file'])==item['sha256'];checks[folder+'_files']+=1
    resource=json.loads((base/'short_resources/summary.json').read_text());assert resource['bytewise_predictor_replays']==162;timings=json.loads((base/'short_resources/timings.json').read_text());memory=json.loads((base/'short_resources/memory.json').read_text())
    assert len(timings)==144 and len(memory)==18
    for key,data in resource['methods'].items():
        # Benchmark records identify each backend in the full config name.
        subset=[r for r in timings if r['method']==key];assert len(subset)==16
        assert float(np.mean([r['seconds'] for r in subset]))==data['mean_seconds'];checks['resource_groups']+=1
    rows=json.loads((base/'threshold/rows.json').read_text());assert len(rows)==272;assert sum(r['selected'] is not None for r in rows)==260
    assert all(0<r['norm_ratio']<1 for r in rows if r['selected'] is not None);assert sum(r['fallbacks'] for r in rows)==2
    a=json.loads((base/'transfer_audit/summary.json').read_text());assert a['counts']['exact_bias_blocks']==2176 and a['counts']['interior_credit_identities']==982
    for folder,name in [('figures','259_minimum_dual.png'),('figures_v2','259_minimum_dual_v2.png')]:assert sha(base/folder/name)==json.loads((base/folder/'summary.json').read_text())['figure_sha256']
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for name in ['259_minimum_sufficient_dual_results.md','260_minimum_dual_continuation_query_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_links']+=1
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        main_figure='minimum_sufficient_dual/figures_v2/259_minimum_dual_v2.png',query_targets_accessed=False,
        next='260 fixed continuation and query protocol; not yet run when this audit is written; goal active')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)

if __name__=='__main__':main()
