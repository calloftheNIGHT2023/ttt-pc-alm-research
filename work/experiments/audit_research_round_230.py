"""Frozen chain, all common-pool evidence and report asset checks."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/common_pool_credit';inp=base/'development'
    docs=root/'outputs/ttt-pc-alm-research';p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert p['primitive_sha256']==sha(base/'primitive/summary.json') and p['design_sha256']==sha(docs/'229_common_pool_credit_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_228_audit.json')
    assert s['source_sha256']==sha(Path(__file__).with_name('run_common_pool_credit.py'))
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['banks_sha256']==sha(inp/'banks.json')
    assert s['pools_sha256']==sha(inp/'pools.json') and s['paired_sha256']==sha(inp/'paired.json')
    assert s['banks']==96 and s['tasks']==16 and s['unique_task_regions']==700 and s['method_region_pairs']==4200
    for r in read(inp/'pools.json'):assert sha(inp/r['arrays_file'])==r['arrays_sha256']
    banks={(r['seed'],r['method']):r for r in read(inp/'banks.json')}
    for r in banks.values():
        assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
        assert sha(root/'results/light_h2_credit/full_bank_ceiling'/r['source_file'])==r['source_sha256']
    assert a['checks']==dict(tasks=16,banks=96,common_regions=700,replayed_regions=3418,replayed_arrays=1440,old_exact=782,new_exact=473,
        convex_exact=473,rounding_exact=473,method_exclusions=1255,unique_parameter_lp_infeasible=421,paired_sets=240)
    assert a['grouped']==dict(alm=263,nonalm_union=395,alm_only_vs_nonalm=15,nonalm_only=147,alm_only_vs_all_other=6)
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_common_pool_credit.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    for key,name in [('proofs_sha256','proofs.json'),('tasks_sha256','tasks.json'),('grouped_sets_sha256','grouped_sets.json')]:assert a[key]==sha(base/'audit'/name)
    assert a['comparisons']==s['comparisons']
    f=read(base/'analysis/figure_audit.json');assert f['passed']
    assert f['source_sha256']==sha(Path(__file__).with_name('plot_common_pool_credit.py'))
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    assert f['figure_sha256']==sha(base/'analysis/common_pool_credit.png')
    resources=[]
    for seed in [5920000,5920015]:
        for method in p['methods']:
            r=read(base/f'resources/{seed}_{method}.json');ref=banks[seed,method];assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_common_pool_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            assert r['pool_regions']==ref['pool_regions'] and r['replayed_joint_regions']==ref['regions']
            assert r['total']==ref['total_positive'] and r['new']==ref['positive'];resources.append(r)
    totals={key:sum(r[key] for r in resources) for key in ['pool_regions','replayed_joint_regions','replayed_arrays','old','new','total']}
    assert totals==dict(pool_regions=528,replayed_joint_regions=329,replayed_arrays=180,old=199,new=28,total=227)
    links=0;figures=0;report_hashes={}
    for name in ['229_common_pool_credit_protocol.md','230_common_pool_credit_results.md','231_credit_history_attribution_protocol.md']:
        path=docs/name;report_hashes[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],grouped=a['grouped'],fresh_processes=12,resource_totals=totals,
        links=links,figures=figures,report_sha256=report_hashes,source_sha256=sha(Path(__file__)),
        scope='Verified finite common-pool complementarity; no claim of hull impossibility, matched-deadline or independent query superiority')
    target=root/'results/round_230_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
