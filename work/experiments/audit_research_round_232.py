"""Final frozen-history result chain, including all exact expansions."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_history';inp=base/'development'
    docs=root/'outputs/ttt-pc-alm-research';p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert p['primitive_sha256']==sha(base/'primitive/summary.json') and p['design_sha256']==sha(docs/'231_credit_history_attribution_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_230_audit.json')
    assert s['source_sha256']==sha(Path(__file__).with_name('run_credit_history.py'))
    for name in ['protocol','banks','captures','paired']:assert s[name+'_sha256']==sha(inp/(name+'.json'))
    assert s['banks']==64 and s['tasks']==16
    parent=root/'results/common_pool_credit/development'
    for cap in read(inp/'captures.json'):
        assert sha(inp/cap['history_file'])==cap['history_sha256'] and sha(parent/cap['pool_file'])==cap['pool_sha256']
        assert cap['baseline']['trajectory_sha256']==cap['captured']['trajectory_sha256'] and cap['captured']['shadow_checks']==16
    banks={(r['seed'],r['method']):r for r in read(inp/'banks.json')}
    for r in banks.values():assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
    assert a['checks']==dict(tasks=16,trajectory_events=800,shadow_steps=256,banks=64,replayed_regions=2420,replayed_arrays=960,native_prior_arrays=240,
        old_exact=380,new_exact=622,convex_exact=622,rounding_exact=622,method_exclusions=1002,unique_lp_infeasible=320,
        expanded=263,expanded_positive=263,expanded_lower_positive=263)
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_credit_history.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    for name,value in a['output_sha256'].items():assert sha(base/'audit'/name)==value
    assert a['comparisons']==s['comparisons']
    assert a['six_example_coverage']==dict(history_full=2,history_uniform12=0,history_recent12=3)
    ex=read(base/'audit/expansions.json');assert len(ex)==263 and all(r['positive'] and r['lower_positive'] and r['ideal_positive'] for r in ex)
    assert min(r['normalized_lower'] for r in ex)==a['min_expansion_normalized_lower']
    assert sum(r['full_history_value']>.001001 for r in ex)==a['expanded_above_delta']==117
    examples=read(base/'audit/six_examples.json');remaining=sum(not any(r['covered'].values()) for r in examples);assert remaining==3
    f=read(base/'analysis/figure_audit.json');assert f['passed'] and f['source_sha256']==sha(Path(__file__).with_name('plot_credit_history.py'))
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    assert f['expansions_sha256']==sha(base/'audit/expansions.json') and f['figure_sha256']==sha(base/'analysis/credit_history.png')
    resources=[]
    for seed in [5920000,5920015]:
        for method in p['methods']:
            r=read(base/f'resources/{seed}_{method}.json');gold=banks[seed,method];assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_credit_history_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            assert r['total']==gold['total_positive'] and r['new']==gold['positive'] and r['joint_regions']==gold['regions'];resources.append(r)
    totals={k:sum(r[k] for r in resources) for k in ['pool_regions','joint_regions','replayed_arrays','old','new','total']}
    assert totals==dict(pool_regions=352,joint_regions=199,replayed_arrays=120,old=153,new=34,total=187)
    links=0;figures=0;reports={}
    for name in ['231_credit_history_attribution_protocol.md','232_credit_history_results.md','233_credit_wall_budget_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],six_example_coverage=a['six_example_coverage'],remaining_six_examples=remaining,
        fresh_processes=8,resource_totals=totals,links=links,figures=figures,report_sha256=reports,source_sha256=sha(Path(__file__)),
        scope='Positive-history information equivalence and finite-path representation effects verified; matched-time and independent task gains not established')
    target=root/'results/round_232_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
