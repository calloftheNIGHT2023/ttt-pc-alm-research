"""Freeze the complete online experiment, reports, resources and figure."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/band_conditioned_online';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research';p=read(inp/'protocol.json');run=read(inp/'run_audit.json')
    a=read(base/'audit/summary.json');s=read(base/'analysis/summary.json');prim=read(base/'primitive_v2/summary.json');f=read(base/'analysis/figure_audit.json')
    assert run['execution_complete'] and all(r['passed'] for r in [a,s,prim,f]);assert run['episodes']==1248 and run['stages']==4992 and run['failures']==0
    for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    assert len(p['source_sha256'])==246 and p['primitive_sha256']==sha(base/'primitive_v2/summary.json') and p['design_sha256']==prim['design_sha256']==sha(docs/'243_band_conditioned_online_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_242_audit.json') and p['checkpoint_manifest']==prim['checkpoint_manifest']
    for name,h in prim['output_sha256'].items():assert sha(base/'primitive_v2'/name)==h
    for record in [run,a]:
        for name,h in record['input_sha256'].items():assert sha(inp/name)==h
    assert a['source_sha256']==sha(src/'audit_band_conditioned_online.py') and s['source_sha256']==sha(src/'analyze_band_conditioned_online.py')
    assert a['generation_sha256']==sha(base/'audit/generations.json')
    for name,h in s['output_sha256'].items():assert sha(base/'analysis'/name)==h
    assert s['audit_sha256']==sha(base/'audit/summary.json') and s['protocol_sha256']==sha(inp/'protocol.json')
    expected=dict(states=4992,observations_exact=4992,posterior_replays=3584,independent_regression_replays=768,frozen_meta_state_replays=640,
        risk_replays=4992,unchanged_old_control_states=1536,shared_start_replays=864,details=896,gated_c5_states_bytewise=768,credit_repetitions_exact=96,episodes_accounted=1248)
    for key,value in expected.items():assert a['checks'][key]==value,(key,a['checks'][key],value)
    assert a['checks']['fraction_certificates']==a['checks']['parameter_infeasible_lp']==a['checks']['independent_relaxed_lp']
    resources={}
    names=['band_inverse_alm_native_gate','band_inverse_alm_c5','band_inverse_adam60_c5','band_inverse_adam240_c5','band_inverse_gn20_c5',
        'band_inverse_pc_c5','band_inverse_nodual_c5','band_inverse_direct64_c5','prior4096_ridge','meta_ridge128','meta_shallow64_20']
    for seed in [5920000,5920015]:
        for name in names:
            file=base/'resources'/f'{seed}_{name}.json';r=read(file);assert r['passed'] and r['complete'] and r['stage_replays']==4
            assert r['source_sha256']==sha(src/'audit_band_conditioned_resources.py') and r['protocol_sha256']==sha(inp/'protocol.json');resources[file.name]=sha(file)
    assert f['source_sha256']==sha(src/'plot_band_conditioned_online.py') and f['analysis_sha256']==sha(base/'analysis/summary.json')
    assert f['contrast_sha256']==sha(base/'analysis/primary_contrasts.json') and f['figure_sha256']==sha(base/'analysis/band_conditioned_online.png')
    reports={};links=figures=0
    for name in ['243_band_conditioned_online_protocol.md','244_band_conditioned_online_results.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures>=1
    extra=['audit_band_conditioned_online.py','analyze_band_conditioned_online.py','audit_band_conditioned_resources.py','plot_band_conditioned_online.py',Path(__file__).name]
    result=dict(passed=True,frozen_sources=246,episodes=1248,states=4992,checks=a['checks'],resources_sha256=resources,links=links,figures=figures,
        report_sha256=reports,extra_source_sha256={n:sha(src/n) for n in extra},source_sha256=sha(Path(__file__)),
        scope='Frozen complete old-development online comparison; not independent confirmation or final project completion')
    target=root/'results/round_244_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
