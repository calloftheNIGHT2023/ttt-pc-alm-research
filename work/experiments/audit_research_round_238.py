"""Freeze complete-history evidence, resource scopes and the next protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/baseline_history';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research';src=Path(__file__).parent
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(src/name)==value,name
    assert len(p['source_sha256'])==218 and p['primitive_sha256']==sha(base/'primitive/summary.json')
    assert p['design_sha256']==sha(docs/'237_full_baseline_history_protocol.md')==prim['design_sha256']
    assert prim['parent_audit_sha256']==sha(root/'results/round_236_audit.json') and prim['records_sha256']==sha(base/'primitive/records.json')
    for seed,r in p['input_hashes'].items():
        assert r['pool']==sha(root/f'results/common_pool_credit/development/pool_{seed}.npz')
        for item in r['banks'].values():assert item['sha256']==sha(root/item['file'])
    for name,value in s['input_sha256'].items():assert value==sha(inp/name)
    for name,value in a['input_sha256'].items():assert value==sha(inp/name)
    for name,value in a['output_sha256'].items():assert value==sha(base/'audit'/name)
    for rec in read(inp/'banks.json'):assert rec['rows_sha256']==sha(inp/rec['rows_file'])
    for rec in read(inp/'captures.json'):assert rec['arrays_sha256']==sha(inp/rec['arrays_file'])
    assert s['source_sha256']==sha(src/'run_baseline_history.py') and a['source_sha256']==sha(src/'audit_baseline_history.py')
    assert a['checks']==dict(capture_arrays=480,trajectory_events=6624,evaluation_calls=976,old_arrays=192,old_direction_inclusions=24554,captures=48,
        reduced_lp_checks=2800,new_exact_positive=2148,regions=2800,exact_nonpositive=527,exact_primal_directions=5037820,
        slow_fraction_cases=120,slow_fraction_directions=1256614,banks=64,raw_unknown=125,unique_parameter_infeasible=539)
    assert a['comparisons']==s['comparisons'] and a['old_strict_case_transitions']==dict(positive=8,nonpositive=8)
    ca=read(base/'analysis/case_analysis.json');assert ca['passed'] and ca['cases']==16 and ca['old_witnesses_still_valid']==8
    assert ca['source_sha256']==sha(src/'analyze_baseline_history_cases.py') and ca['audit_sha256']==sha(base/'audit/summary.json') and ca['cases_sha256']==sha(base/'analysis/strict_cases.json')
    g=read(base/'analysis/geometry_audit.json');assert g['passed'] and g['cases']==8 and g['seeds']==[5920012]
    assert g['source_sha256']==sha(src/'analyze_full_history_geometry.py') and g['cases_sha256']==ca['cases_sha256'] and g['geometry_sha256']==sha(base/'analysis/retained_geometry.json')
    f=read(base/'analysis/figure_audit.json');assert f['passed'] and f['source_sha256']==sha(src/'plot_baseline_history.py')
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json') and f['cases_sha256']==ca['cases_sha256'] and f['figure_sha256']==sha(base/'analysis/baseline_history.png')
    resources=list((base/'resources').glob('*.json'));assert len(resources)==14;rc=rs=0
    for path in resources:
        r=read(path);assert r['passed'] and r['source_sha256']==sha(src/'audit_baseline_history_resources.py') and r['protocol_sha256']==sha(inp/'protocol.json')
        if r['phase']=='capture':rc+=r['checked_arrays']
        else:assert r['exact_output_reproduced'];rs+=1
    assert (rc,rs)==(60,8)
    reports={};links=figures=0
    for name in ['237_full_baseline_history_protocol.md','238_full_baseline_history_results.md','239_forward_credit_envelope_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    extra=['audit_baseline_history.py','audit_baseline_history_resources.py','analyze_baseline_history_cases.py','analyze_full_history_geometry.py','plot_baseline_history.py',Path(__file__).name]
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],resources=14,resource_capture_arrays=rc,resource_solver_cases=rs,
        retained_strict_cases=8,retained_seeds=[5920012],links=links,figures=figures,report_sha256=reports,extra_source_sha256={name:sha(src/name) for name in extra},
        source_sha256=sha(Path(__file__)),scope='Fixed complete-trajectory capacity separation, not universal BP impossibility or independent online task victory')
    target=root/'results/round_238_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
