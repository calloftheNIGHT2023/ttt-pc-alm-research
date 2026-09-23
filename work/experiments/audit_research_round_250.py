"""Freeze cold-start, shared-mode and complete-resource evidence and reports."""
import argparse
import json
from pathlib import Path
import re
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump

def read(p):return json.loads(p.read_text())
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent;docs=root/'outputs/ttt-pc-alm-research'
    target=root/'results/round_250_audit.json';assert not target.exists();c=root/'results/cold_stagnation_switch';s=root/'results/shared_mode_readout'
    cp=read(c/'development/protocol.json');sp=read(s/'development/protocol.json');rp=read(s/'resources/protocol.json');sources=dict(rp['source_sha256'])
    for protocol in [read(c/'primitive/summary.json'),cp,sp,rp]:
        for n,h in protocol['source_sha256'].items():assert sha(src/n)==h,n
    for n,h in read(root/'results/round_246_audit.json')['source_sha256'].items():assert sha(src/n)==h,n
    summaries={}
    for b,paths in [(c,['primitive/summary.json','development/summary.json','audit/summary.json','resources/summary.json','analysis/summary.json','analysis/figure_audit.json','analysis/figure_audit_v2.json']),
                    (s,['development/summary.json','audit/summary.json','resources/summary.json','analysis/summary.json','analysis/figure_audit.json'])]:
        for name in paths:
            obj=read(b/name);assert obj['passed'];summaries[str((b/name).relative_to(root))]=sha(b/name)
        manifest=read(b/'development/before_query_manifest.json');assert manifest['protocol_sha256']==sha(b/'development/protocol.json')
        for group in ['arrays','geometry']:
            for n,h in manifest[group].items():assert sha(b/'development'/n)==h,n
    ca=read(c/'audit/summary.json');cr=read(c/'resources/summary.json');sa=read(s/'audit/summary.json');sr=read(s/'resources/summary.json')
    assert ca['checks']['method_replays']==ca['checks']['risk_replays']==320 and ca['checks']['trigger_steps']==2048 and ca['checks']['causal_multiplier_updates']==2048
    assert ca['checks']['independent_regressions']==96 and ca['checks']['frozen_meta_replays']==64 and ca['checks']['unique_branch_lp_replays']==2141
    assert cr['timing_runs']==960 and cr['memory_runs']==40 and cr['bytewise_predictions']==1000
    assert sa['checks']['positive_volume_rational_cubes']==116 and sa['checks']['exact_corners']==1856 and sa['checks']['support_particles']==1843200
    assert sa['checks']['predictions_and_risks']==960 and sa['checks']['unchanged_fallbacks']==60 and sa['checks']['marginal_risk_identities']==5
    assert sr['timing_runs']==160 and sr['memory_runs']==20 and sr['bytewise_predictor_replays']==180
    for b,result,key in [(c,cr,'output_sha256'),(s,sr,'outputs_sha256')]:
        for n,h in result[key].items():assert sha(b/'resources'/n)==h
    for b,key in [(c,'output_sha256'),(s,'outputs_sha256')]:
        a=read(b/'analysis/summary.json')
        for n,h in a[key].items():assert sha(b/'analysis'/n)==h
        assert a['audit_sha256']==sha(b/'audit/summary.json') and a['resources_sha256']==sha(b/'resources/summary.json')
    figures=[(c,'figure_audit_v2.json','cold_stagnation_switch_v2.png','plot_cold_stagnation_switch_v2.py'),(s,'figure_audit.json','shared_mode_readout.png','plot_shared_mode_readout.py')]
    for b,a,n,script in figures:
        f=read(b/'analysis'/a);assert f['source_sha256']==sha(src/script) and f['figure_sha256']==sha(b/'analysis'/n) and f['analysis_sha256']==sha(b/'analysis/summary.json')
    # Independently reaggregate all task means (sampling repeats are not tasks).
    q=read(s/'development/query_rows.json');methods=read(s/'analysis/methods.json')
    for name in sp['pools']:
        values=[np.mean([r['mse'] for r in q if r['seed']==seed and r['method']==name]) for seed in sp['seeds']]
        assert float(np.mean(values))==methods[name]['mse']
    assert sum(methods[n]['empty_pool_tasks'] for n in sp['pools'])*5==60
    new=read(s/'analysis/summary.json')['marginal'];assert new['first_actual_source']==dict(seed=5900007,restart=4,sweep=128,trigger_after=106)
    assert new['actual_source_restarts']==[4] and new['actual_source_events']==1 and new['all5_improved']
    extra=['plot_cold_stagnation_switch.py','plot_cold_stagnation_switch_v2.py','analyze_shared_mode_readout.py','plot_shared_mode_readout.py',Path(__file__).name]
    sources.update({n:sha(src/n) for n in extra});reports={};links=0
    for name in ['247_cold_start_stagnation_switch_protocol.md','248_cold_start_stagnation_switch_results.md','249_shared_mode_readout_protocol.md','250_shared_mode_readout_results.md','251_local_dual_jump_design.md']:
        path=docs/name;reports[name]=sha(path)
        for ref in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in ref or ref.startswith('#'):continue
            assert (path.parent/ref).resolve().exists(),ref;links+=1
    assert reports['247_cold_start_stagnation_switch_protocol.md']==cp['design_sha256'] and reports['249_shared_mode_readout_protocol.md']==sp['design_sha256']
    assert reports['247_cold_start_stagnation_switch_protocol.md']==read(root/'results/round_246_audit.json')['report_sha256']['247_cold_start_stagnation_switch_protocol.md']
    result=dict(passed=True,source_sha256=sources,frozen_sources=len(sources),report_sha256=reports,summaries_sha256=summaries,links=links,figures=2,
        cold_predictions=320,shared_readout_predictions=960,cold_resource_runs=1000,live_geometry_resource_runs=180,positive_volume_certificates=116,
        scope='Frozen old-development cold-start and shared readout evidence, not a primary success or final independent online result')
    dump(target,result);print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','report_sha256','summaries_sha256']}),flush=True)

if __name__=='__main__':main()
