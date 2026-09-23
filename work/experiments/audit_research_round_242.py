"""Freeze shared-parameter realization and the next online protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/joint_forward_realization';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(src/name)==value,name
    assert len(p['source_sha256'])==233 and p['primitive_sha256']==sha(base/'primitive/summary.json')
    assert p['design_sha256']==prim['design_sha256']==sha(docs/'241_joint_forward_realization_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_240_audit.json')
    for name,value in prim['output_sha256'].items():assert value==sha(base/'primitive'/name)
    for seed,r in p['input_hashes'].items():
        assert r['pool']==sha(root/f'results/common_pool_credit/development/pool_{seed}.npz')
        for item in r['banks'].values():assert item['sha256']==sha(root/item['file'])
    for name,value in p['full_history_input_sha256'].items():assert value==sha(root/'results/baseline_history/development'/name)
    assert p['alm_banks_sha256']==sha(root/'results/exact_credit_hull/development/banks.json')
    assert s['source_sha256']==sha(src/'run_joint_forward_realization.py') and a['source_sha256']==sha(src/'audit_joint_forward_realization.py')
    for data in [s,a]:
        for name,value in data['input_sha256'].items():assert value==sha(inp/name)
    for name,value in a['output_sha256'].items():assert value==sha(base/'audit'/name)
    for r in read(inp/'generations.json'):
        for f,h in [('arrays_file','arrays_sha256'),('generation_file','generation_sha256')]:assert sha(inp/r[f])==r[h]
    for r in read(inp/'banks.json'):assert sha(inp/r['rows_file'])==r['rows_sha256']
    assert a['checks']==dict(replayed_arrays=304,exact_source_chains=1996,original_shared_forwards=1996,original_bp_credit_arrays=16,generations=16,
        raw_nonpositive=489,fraction_primal_directions=2600062,regions=1400,raw_positive=897,exact_rounding=897,banks=32,raw_unknown=14,unique_parameter_infeasible=539)
    assert a['comparisons']==s['comparisons']
    assert [r['counts'] for r in s['summaries']]==[dict(positive=358,nonpositive=328,unknown=14),dict(positive=539,nonpositive=161)]
    case=read(base/'analysis/case_analysis.json');assert case['passed'] and case['cases']==case['cases_with_actual_witness_violation']==8
    assert case['source_sha256']==sha(src/'analyze_joint_forward_cases.py') and case['audit_sha256']==sha(base/'audit/summary.json') and case['cases_sha256']==sha(base/'analysis/old_cases.json')
    resources={}
    for seed in [5920000,5920015]:
        for method in ['generation','constructed_joint','old_plus_constructed_joint']:
            name=f'{seed}_{method}.json';r=read(base/'resources'/name);assert r['passed'] and r['seed']==seed and r['method']==method
            assert r['source_sha256']==sha(src/'audit_joint_forward_resources.py') and r['protocol_sha256']==sha(inp/'protocol.json');resources[name]=sha(base/'resources'/name)
    f=read(base/'analysis/figure_audit.json');assert f['passed'] and f['source_sha256']==sha(src/'plot_joint_forward_realization.py')
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json') and f['figure_sha256']==sha(base/'analysis/joint_forward_realization.png') and f['cases_sha256']==case['cases_sha256']
    reports={};links=figures=0
    for name in ['241_joint_forward_realization_protocol.md','242_joint_forward_realization_results.md','243_band_conditioned_online_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    extra=['audit_joint_forward_realization.py','audit_joint_forward_resources.py','analyze_joint_forward_cases.py','plot_joint_forward_realization.py',Path(__file__).name]
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],all_eight_realized_positive=True,resources_sha256=resources,links=links,figures=figures,
        report_sha256=reports,extra_source_sha256={name:sha(src/name) for name in extra},source_sha256=sha(Path(__file__)),
        scope='Realized shared-parameter credit capacity; online query advantage remains unproved')
    target=root/'results/round_242_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
