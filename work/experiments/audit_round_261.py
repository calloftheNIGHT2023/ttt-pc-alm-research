"""Freeze260-261 trajectories, query risks, all36 real costs and262 plan."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;out=root/'results/round_261_audit.json';assert not out.exists()
    parent=root/'results/round_259_audit.json';old=json.loads(parent.read_text());hashes=dict(old['source_sha256']);reports=dict(old['report_sha256']);summaries={};checks=Counter();base=root/'results/minimum_dual_query'
    for directory in [root/'results/minimum_sufficient_dual'/s for s in ['continuation','continuation_audit','mode_audit']]+[base/s for s in ['development','audit','live_primitive','resources','analysis','cost_analysis','figures','figures_v2']]:
        data=json.loads((directory/'summary.json').read_text());assert data['passed'];summaries[str(directory.relative_to(root))]=sha(directory/'summary.json');protocol=directory/'protocol.json'
        if protocol.exists():
            p=json.loads(protocol.read_text())
            for n,h in p['source_sha256'].items():
                if n in hashes:assert hashes[n]==h,n
                hashes[n]=h
            if 'protocol_sha256' in data:assert sha(protocol)==data['protocol_sha256']
        for n,h in data.get('outputs_sha256',{}).items():assert sha(directory/n)==h;checks['analysis_artifacts']+=1
    for n in ['audit_minimum_dual_modes.py','analyze_minimum_dual_query.py','summarize_minimum_dual_resources.py','plot_minimum_dual_query.py','plot_minimum_dual_query_v2.py',Path(__file__).name]:hashes[n]=sha(src/n)
    for n,h in hashes.items():assert sha(src/n)==h,n
    cont=root/'results/minimum_sufficient_dual/continuation';cp=json.loads((cont/'summary.json').read_text());assert sha(cont/'support_rows.json')==cp['support_rows_sha256']
    for n,h in cp['geometry_sha256'].items():assert sha(cont/n)==h
    for row in json.loads((cont/'support_rows.json').read_text()):assert sha(cont/row['file'])==row['sha256'];checks['continuation_files']+=1
    inp=base/'development';run=json.loads((inp/'summary.json').read_text());assert run['complete'] and run['new_predictions']==288 and run['reference_predictions']==2368;before=json.loads((inp/'before_query_manifest.json').read_text());assert sha(inp/'before_query_manifest.json')==run['before_query_manifest_sha256']
    for name in ['protocol','rows','reference_rows','representative_files','geometry_files']:assert sha(inp/f'{name}.json')==before[f'{name}_sha256']
    for group in ['arrays','geometry']:
        for n,h in before[group].items():assert sha(inp/n)==h;checks['prediction_geometry_files']+=1
    for row in json.loads((inp/'reference_rows.json').read_text()):assert sha(root/row['file'])==row['sha256'];checks['reference_files']+=1
    queryrows=json.loads((inp/'query_rows.json').read_text());assert sha(inp/'query_rows.json')==run['query_rows_sha256'];query={(r['seed'],r['method'],r['readout'],r['repetition']):r['mse'] for r in queryrows};assert len(query)==len(queryrows)==2656
    methods=json.loads((base/'analysis/methods.json').read_text())
    for r in methods:
        tasks=[float(np.mean([v for (s,m,k,rep),v in query.items() if (s,m,k)==(seed,r['method'],r['readout'])])) for seed in range(5900000,5900016)]
        assert tasks==r['task_mses'] and float(np.mean(tasks))==r['mean_mse'];checks['risk_groups']+=1
    resource=base/'resources';rs=json.loads((resource/'summary.json').read_text());assert rs['timing_runs']==576 and rs['memory_runs']==72 and rs['bytewise_predictor_replays']==648
    for name in ['protocol','timings','memory']:assert sha(resource/f'{name}.json')==rs[f'{name}_sha256']
    timings=json.loads((resource/'timings.json').read_text());memory=json.loads((resource/'memory.json').read_text());assert len({(r['seed'],r['method']) for r in timings})==576;assert len({(r['seed'],r['method']) for r in memory})==72
    for name,r in rs['methods'].items():
        assert float(np.mean([t['seconds'] for t in timings if t['method']==name]))==r['mean_seconds'];assert max(t['traced_peak_bytes'] for t in memory if t['method']==name)==r['max_traced_peak_bytes'];checks['resource_groups']+=1
    for row in timings:
        if row['family'] in ['minimum','new']:
            m=row['metadata'];expected=17 if row['family']=='minimum' or row['method'] in ['dual_alm64','activity_alm64','reorder_alm64','dual_reset_alm64'] else 0
            assert m['scans']['scanned_states']==expected and m['lp_calls']==len(m['visited_modes']) and m['preparation_seconds']>0;checks['cold_prefix_and_search_accounting']+=1
            if expected:assert m['scans']['exact_tau']==m['scans']['accepted_trials'];checks['accepted_exact_checks']+=m['scans']['exact_tau']
            if row['family']=='minimum':
                assert m['first_crossing_seconds']>0 and m['search_seconds']>=m['first_crossing_seconds'];checks['first_crossing_events']+=m['scans']['first_crossing_events'];checks['rounding_fallbacks']+=m['scans']['first_crossing_fallbacks']
    assert checks['first_crossing_events']==780 and checks['rounding_fallbacks']==6
    for folder,name in [('figures','261_minimum_dual_query.png'),('figures_v2','261_minimum_dual_query_v2.png')]:assert sha(base/folder/name)==json.loads((base/folder/'summary.json').read_text())['figure_sha256']
    docs=root/'outputs/ttt-pc-alm-research'
    for n,h in reports.items():assert sha(docs/n)==h,n
    for n in ['261_minimum_dual_query_results.md','262_matched_budget_confirmation_protocol.md']:
        path=docs/n;reports[n]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' not in target:assert (path.parent/target).resolve().exists(),target;checks['local_links']+=1
    ans=dict(passed=True,parent_audit_sha256=sha(parent),source_count=len(hashes),source_sha256=hashes,report_sha256=reports,summary_sha256=summaries,checks=checks,
        main_figure='minimum_dual_query/figures_v2/261_minimum_dual_query_v2.png',next='262 resource-matched baseline reinforcement then64 new tasks; not implemented or run; goal active; no independent confirmation yet')
    dump(out,ans);print(json.dumps({k:v for k,v in ans.items() if k not in ['source_sha256','report_sha256','summary_sha256']}),flush=True)

if __name__=='__main__':main()
