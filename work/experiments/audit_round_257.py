"""Freeze256-257 query/resource evidence and258's unexecuted mechanism design."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/round_257_audit.json';assert not out.exists();parent=root/'results/round_255_audit.json';old=json.loads(parent.read_text());base=root/'results/dual_jump_query'
    hashes=dict(old['source_sha256']);summary_hashes={}
    for folder in ['development','audit','live_primitive','resources','analysis','cost_analysis','figures','figures_v2']:
        data=json.loads((base/folder/'summary.json').read_text());assert data['passed'];summary_hashes[folder]=sha(base/folder/'summary.json')
        protocol=base/folder/'protocol.json'
        if protocol.exists():
            p=json.loads(protocol.read_text());hashes.update(p.get('source_sha256',{}))
            if 'protocol_sha256' in data:assert sha(protocol)==data['protocol_sha256']
    extras=['analyze_dual_jump_query.py','analyze_dual_jump_query_v2.py','summarize_dual_jump_resources.py','plot_dual_jump_query.py','plot_dual_jump_query_v2.py',Path(__file__).name]
    hashes.update({n:sha(src/n) for n in extras})
    for name,h in hashes.items():assert sha(src/name)==h,name
    docs=root/'outputs/ttt-pc-alm-research';reports=dict(old['report_sha256'])
    for name,h in reports.items():assert sha(docs/name)==h,name
    links=0
    for name in ['257_local_dual_jump_query_results.md','258_minimum_sufficient_dual_design.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target:continue
            assert (path.parent/target).resolve().exists(),(name,target);links+=1
    inp=base/'development';before=json.loads((inp/'before_query_manifest.json').read_text())
    for name in ['protocol','rows','reference_rows','representative_files','geometry_files']:assert sha(inp/f'{name}.json')==before[f'{name}_sha256']
    for group in ['arrays','geometry']:
        for name,h in before[group].items():assert sha(inp/name)==h,name
    references=json.loads((inp/'reference_rows.json').read_text())
    for row in references:assert sha(root/row['file'])==row['sha256']
    queries=json.loads((inp/'query_rows.json').read_text());query={(r['seed'],r['method'],r['readout'],r['repetition']):r['mse'] for r in queries};assert len(query)==len(queries)==2368
    methods=json.loads((base/'analysis/methods.json').read_text());checks=Counter()
    for r in methods:
        bytask=[]
        for seed in range(5900000,5900016):bytask.append(float(np.mean([v for (s,m,k,rep),v in query.items() if (s,m,k)==(seed,r['method'],r['readout'])])))
        assert bytask==r['task_mses'];assert float(np.mean(bytask))==r['mean_mse'];assert max(bytask)==r['worst_task_mse'];checks['risk_groups']+=1
    rr=json.loads((base/'resources/summary.json').read_text());assert rr['timing_runs']==528 and rr['memory_runs']==66 and rr['bytewise_predictor_replays']==594
    for name in ['protocol','timings','memory']:assert sha(base/'resources'/f'{name}.json')==rr[f'{name}_sha256']
    timings=json.loads((base/'resources/timings.json').read_text());memory=json.loads((base/'resources/memory.json').read_text())
    assert len({(r['seed'],r['method']) for r in timings})==528;assert len({(r['seed'],r['method']) for r in memory})==66
    for name,row in rr['methods'].items():
        assert float(np.mean([r['seconds'] for r in timings if r['method']==name]))==row['mean_seconds']
        assert max(r['traced_peak_bytes'] for r in memory if r['method']==name)==row['max_traced_peak_bytes'];checks['resource_groups']+=1
    for r in timings:
        if r['family']=='new':
            assert r['metadata']['lp_calls']==len(r['metadata']['visited_modes']);assert r['metadata']['preparation_seconds']>0
            expected=17 if r['method'] in ['dual_alm64','activity_alm64','reorder_alm64','dual_reset_alm64'] else 0
            assert r['metadata']['scans']['scanned_states']==expected;assert r['metadata']['scans']['tau_trials']==expected*12*7;checks['fresh_search_accounting']+=1
    for folder,file in [('figures','257_query_and_cost.png'),('figures_v2','257_query_and_cost_v2.png')]:assert sha(base/folder/file)==json.loads((base/folder/'summary.json').read_text())['figure_sha256']
    result=dict(passed=True,source_sha256=hashes,source_count=len(hashes),report_sha256=reports,parent_audit_sha256=sha(parent),summary_sha256=summary_hashes,
        checks=checks,local_links=links,new_predictions=1248,unchanged_reference_predictions=1120,full_timing_runs=528,memory_runs=66,
        main_figure='dual_jump_query/figures_v2/257_query_and_cost_v2.png',
        preserved_attempt='analysis original missing closing parenthesis;v2 syntax-only fix; original source retained',
        next='258 minimum-sufficient dual design frozen but not implemented; goal active, no independent task advantage established')
    dump(out,result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)

if __name__=='__main__':main()
