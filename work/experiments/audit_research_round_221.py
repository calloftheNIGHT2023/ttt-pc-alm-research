"""Complete chain audit for joint-credit information and margin diagnostics."""
import argparse
import hashlib
import json
import re
from pathlib import Path
import numpy as np


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/joint_credit_minimax';inp=base/'development';p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text())
    assert s['execution_complete'] and s['counts']==dict(regions=3998,old_proofs=732,lp_calls=3266,lp_failures=0,new_exact_proofs=1802,new_from_old_gate_skip=1218)
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert s['source_sha256']==sha(Path(__file__).with_name('run_joint_credit_minimax.py')) and s['protocol_sha256']==sha(inp/'protocol.json')
    assert s['banks_sha256']==sha(inp/'banks.json') and p['primitive_sha256']==sha(base/'primitive/summary.json')
    banks=json.loads((inp/'banks.json').read_text());assert len(banks)==96
    for bank in banks:
        for fn,h in [('rows_file','rows_sha256'),('arrays_file','arrays_sha256')]:assert sha(inp/bank[fn])==bank[h]
        assert sha(root/'results/light_h2_credit/full_bank_ceiling'/bank['source_data_file'])==bank['source_data_sha256']
    a=json.loads((base/'audit/summary.json').read_text());assert a['passed'] and a['checks']['exact_convex_positive']==1802
    assert a['source_sha256']==sha(Path(__file__).with_name('audit_joint_credit_minimax.py')) and a['proofs_sha256']==sha(base/'audit/proofs.json')
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    assert all(not r['native_only'] and not r['residual_only'] for r in a['alm_same_trajectory_comparison'])
    f=json.loads((base/'analysis/figure_audit.json').read_text());assert f['passed']
    assert f['source_sha256']==sha(Path(__file__).with_name('plot_joint_credit_minimax.py')) and f['figure_sha256']==sha(base/'analysis/joint_credit_minimax.png')
    assert f['summary_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json')
    m=json.loads((base/'analysis/margin_summary.json').read_text());assert m['passed']
    assert m['source_sha256']==sha(Path(__file__).with_name('analyze_joint_credit_margins.py')) and m['rows_sha256']==sha(base/'analysis/margin_rows.json')
    assert m['audit_sha256']==sha(base/'audit/summary.json') and m['protocol_sha256']==sha(inp/'protocol.json')
    mr=json.loads((base/'analysis/margin_rows.json').read_text());assert len(mr)==405
    bmap={(r['seed'],r['method']):r for r in banks};margin_checks=0
    for row in mr:
        aa=bmap[row['seed'],'alm_native'];bb=bmap[row['seed'],'alm_residual']
        ratio=np.log(aa['directions'])/np.log(bb['directions'])*(row['residual_gamma']/row['native_gamma'])**2
        assert abs(ratio-row['ideal_bound_ratio'])<1e-12;margin_checks+=1
    assert sum(r['ideal_bound_ratio']<1 for r in mr)==m['native_smaller_sufficient_bound']==278
    resource_count=0;replays=0;resource_proofs=0
    for seed in [5920000,5920015]:
        for name in p['methods']:
            r=json.loads((base/f'resources/{seed}_{name}.json').read_text());assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_joint_credit_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            resource_count+=1;replays+=r['lp_replays'];resource_proofs+=r['positive_proofs']
    links=0;figures=[];docs=root/'outputs/ttt-pc-alm-research'
    for name in ['220_joint_credit_minimax_design.md','221_joint_credit_minimax_results.md','222_finite_credit_game_design.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists(),target;links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert len(figures)==1 and resource_count==12
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),counts=s['counts'],independent_checks=a['checks'],
        margin_checks=margin_checks,fresh_processes=resource_count,resource_lp_replays=replays,resource_positive_proofs=resource_proofs,
        links=links,figures=figures,source_sha256=sha(Path(__file__)),scope='Verified joint-credit information diagnostic, not independent task superiority or measured finite-step gains')
    out=root/'results/round_221_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
