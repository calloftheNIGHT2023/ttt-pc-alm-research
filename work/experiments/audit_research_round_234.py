"""Close the actual-clock experiment and freeze its evidence chain."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/credit_wall_budget';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json');rows=read(inp/'rows.json')
    assert s['execution_complete'] and a['passed'] and prim['passed'] and len(rows)==s['runs']==1152
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert p['primitive_sha256']==sha(base/'primitive/summary.json') and p['design_sha256']==sha(docs/'233_credit_wall_budget_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_232_audit.json')
    assert s['source_sha256']==sha(Path(__file__).with_name('run_credit_wall_budget.py'))
    assert s['protocol_sha256']==sha(inp/'protocol.json') and s['rows_sha256']==sha(inp/'rows.json')
    for seed,rec in p['input_hashes'].items():
        assert rec['pool']==sha(root/f'results/common_pool_credit/development/pool_{seed}.npz')
        for item in rec['banks'].values():assert sha(root/item['file'])==item['sha256']
    for r in rows:
        assert sha(inp/r['arrays_file'])==r['arrays_sha256'] and sha(inp/r['meta_file'])==r['meta_sha256']
        assert r['stop_reason']=='deadline' and r['nonfinite_failures']==0
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_credit_wall_budget.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    for name in ['unique_proofs','run_checks']:assert a[name+'_sha256']==sha(base/'audit'/(name+'.json'))
    assert a['checks']==dict(runs=1152,replayed_arrays=18432,old_certificate_instances=8280,unique_old_certificates=1035,
        new_certificate_instances=7276,unique_new_certificates=1826,exact_convex_unique=1826,exact_rounding_unique=1826,
        unique_parameter_infeasible=486,late_old=0,late_new=31,clock_events=1000571,cooperative_stops=1152,step_caps=0)
    for block in s['summaries']:
        for rec in block['summaries']:
            rr=[r for r in rows if r['method']==rec['method'] and r['budget_seconds']==block['budget_seconds']]
            assert max(r['response_batches'] for r in rr)==rec['max_response_batches']
    assert max(r['response_batches'] for r in rows)==845
    resources=[]
    for seed in [5920000,5920015]:
        for method in p['methods']:
            r=read(base/f'resources/{seed}_{method}.json');gold=next(t for t in rows if (t['seed'],t['method'],t['budget_seconds'],t['repeat'])==(seed,method,.4,0))
            assert r['passed'] and r['source_sha256']==sha(Path(__file__).with_name('audit_credit_wall_resources.py'))
            assert r['protocol_sha256']==sha(inp/'protocol.json')
            for k,g in [('old','old_count'),('new','new_count'),('total','total_positive'),('response_batches','response_batches'),('oracle_pairs','oracle_pairs')]:assert r[k]==gold[g]
            with np.load(inp/gold['arrays_file']) as z:assert r['output_array_bytes']==sum(z[k].nbytes for k in z.files)
            resources.append(r)
    totals={k:sum(r[k] for r in resources) for k in ['pool_regions','replayed_arrays','old','new','total']}
    assert totals==dict(pool_regions=792,replayed_arrays=288,old=307,new=109,total=416)
    f=read(base/'analysis/figure_audit.json');assert f['passed'] and f['source_sha256']==sha(Path(__file__).with_name('plot_credit_wall_budget.py'))
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json') and f['figure_sha256']==sha(base/'analysis/credit_wall_budget.png')
    links=figures=0;reports={}
    for name in ['233_credit_wall_budget_protocol.md','234_credit_wall_budget_results.md','235_exact_credit_hull_separation_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],fresh_processes=18,resource_totals=totals,
        max_internal_overrun_seconds=max(r['overrun_seconds'] for r in rows),max_external_overrun_seconds=max(r['external_guard_seconds']-r['budget_seconds'] for r in rows),
        links=links,figures=figures,report_sha256=reports,source_sha256=sha(Path(__file__)),
        scope='Finite-time credit representation advantage over full history; no core task or universal optimizer-superiority claim')
    target=root/'results/round_234_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
