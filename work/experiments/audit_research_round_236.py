"""Freeze the all-region exact hull results, analysis, and next protocol."""
import argparse
import hashlib
import json
from pathlib import Path
import re


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/exact_credit_hull';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert len(p['source_sha256'])==210 and p['primitive_sha256']==sha(base/'primitive/summary.json')
    assert p['design_sha256']==sha(docs/'235_exact_credit_hull_separation_protocol.md') and prim['design_sha256']==p['design_sha256']
    assert prim['parent_audit_sha256']==sha(root/'results/round_234_audit.json') and prim['records_sha256']==sha(base/'primitive/records.json')
    for seed,rec in p['input_hashes'].items():
        assert rec['pool']==sha(root/f'results/common_pool_credit/development/pool_{seed}.npz')
        for item in rec['banks'].values():assert item['sha256']==sha(root/item['file'])
    assert s['source_sha256']==sha(Path(__file__).with_name('run_exact_credit_hull.py'))
    for name in ['protocol','banks','paired']:assert s[name+'_sha256']==sha(inp/(name+'.json'))
    for rec in read(inp/'banks.json'):assert rec['rows_sha256']==sha(inp/rec['rows_file'])
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_exact_credit_hull.py'))
    for name,value in a['input_sha256'].items():assert value==sha(inp/name)
    assert a['checks_sha256']==sha(base/'audit/checks.json')
    assert a['checks']==dict(regions=7000,positive=4833,exact_rounding=4833,nonpositive=2142,exact_primal_directions=1232219,
        banks=160,unknown=25,unique_parameter_infeasible=539,timed_positive_compatible=2861,timed_positive_still_unknown=0,union_positive_constituent_still_unknown=0)
    assert a['comparisons']==s['comparisons']
    ca=read(base/'analysis/case_analysis.json');cases=read(base/'analysis/strict_cases.json')
    assert ca['passed'] and ca['strict_method_cases']==len(cases)==16 and ca['unique_task_regions']==12
    assert ca['all_baseline_upper_exact_zero'] and ca['all_same_alm_residual_positive'] and ca['seeds']==[5920000,5920011,5920012]
    assert ca['source_sha256']==sha(Path(__file__).with_name('analyze_exact_credit_separations.py'))
    assert ca['audit_sha256']==sha(base/'audit/summary.json') and ca['cases_sha256']==sha(base/'analysis/strict_cases.json')
    strict={r['comparator']:r['status_pairs'].get('positive__nonpositive',0) for r in s['comparisons']}
    assert strict==dict(alm_residual=0,adam60_native=10,adam60_residual=539,pc_native=4,nodual_native=2,history_full=0,history_uniform12=0,history_recent12=0,strong_union=0)
    assert all(r['status_pairs'].get('nonpositive__positive',0)==0 for r in s['comparisons'])
    for audit,plot,fig in [('figure_audit.json','plot_exact_credit_hull.py','exact_credit_hull.png'),('figure_v2_audit.json','plot_exact_credit_hull_v2.py','exact_credit_hull_v2.png')]:
        f=read(base/'analysis'/audit);assert f['passed'] and f['source_sha256']==sha(Path(__file__).with_name(plot))
        assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json') and f['figure_sha256']==sha(base/'analysis'/fig)
    links=figures=0;reports={}
    for name in ['235_exact_credit_hull_separation_protocol.md','236_exact_credit_hull_results.md','237_full_baseline_history_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],strict_counts=strict,strict_method_cases=16,strict_unique_task_regions=12,
        links=links,figures=figures,report_sha256=reports,source_sha256=sha(Path(__file__)),
        scope='Exact mathematical separations against fixed single-learner credit banks; no necessity against strong union or full-history/online-task victory')
    target=root/'results/round_236_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
