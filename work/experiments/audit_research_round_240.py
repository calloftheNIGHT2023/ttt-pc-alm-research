"""Freeze forward-envelope results without claiming joint realizability."""
import argparse
import hashlib
import json
from pathlib import Path
import re

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/forward_credit_envelope';inp=base/'development';docs=root/'outputs/ttt-pc-alm-research'
    p=read(inp/'protocol.json');s=read(inp/'summary.json');a=read(base/'audit/summary.json');prim=read(base/'primitive/summary.json')
    assert s['execution_complete'] and a['passed'] and prim['passed']
    for name,value in p['source_sha256'].items():assert sha(src/name)==value,name
    assert len(p['source_sha256'])==227 and p['primitive_sha256']==sha(base/'primitive/summary.json')
    assert p['design_sha256']==prim['design_sha256']==sha(docs/'239_forward_credit_envelope_protocol.md')
    assert prim['parent_audit_sha256']==sha(root/'results/round_238_audit.json') and prim['records_sha256']==sha(base/'primitive/records.json')
    for seed,r in p['input_hashes'].items():
        assert r['pool']==sha(root/f'results/common_pool_credit/development/pool_{seed}.npz')
        for item in r['banks'].values():assert item['sha256']==sha(root/item['file'])
    assert s['source_sha256']==sha(src/'run_forward_credit_envelope.py') and a['source_sha256']==sha(src/'audit_forward_credit_envelope.py')
    for name,value in s['input_sha256'].items():assert value==sha(inp/name)
    for name,value in a['input_sha256'].items():assert value==sha(inp/name)
    for name,value in a['output_sha256'].items():assert value==sha(base/'audit'/name)
    for r in read(inp/'banks.json'):
        for f,h in [('bank_file','bank_file_sha256'),('intervals_file','intervals_sha256'),('rows_file','rows_sha256')]:assert sha(inp/r[f])==r[h]
    assert a['checks']==dict(single_observation_patterns=32768,lp_calls=65536,both_empty=29770,nonempty_agree=2998,bank_arrays=48,raw_positive=1617,
        exact_rounding=1617,regions=2100,raw_nonpositive=483,fraction_primal_directions=39566,banks=48,unique_parameter_infeasible=539)
    assert a['comparisons']==s['comparisons']
    assert all(r['counts']==dict(positive=539,nonpositive=161) for r in s['summaries'])
    assert len(a['old_strict_cases'])==8 and all(all(v=='positive' for v in r['statuses'].values()) for r in a['old_strict_cases'])
    f=read(base/'analysis/figure_audit.json');assert f['passed'] and f['source_sha256']==sha(src/'plot_forward_credit_envelope.py')
    assert f['main_sha256']==sha(inp/'summary.json') and f['audit_sha256']==sha(base/'audit/summary.json') and f['figure_sha256']==sha(base/'analysis/forward_credit_envelope.png')
    reports={};links=figures=0
    for name in ['239_forward_credit_envelope_protocol.md','240_forward_credit_envelope_results.md','241_joint_forward_realization_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            resolved=(path.parent/target).resolve();assert resolved.exists(),resolved;links+=1;figures+=int(resolved.suffix=='.png')
    assert figures==1
    extra=['audit_forward_credit_envelope.py','plot_forward_credit_envelope.py',Path(__file__).name]
    result=dict(passed=True,frozen_sources=len(p['source_sha256']),checks=a['checks'],all_old_eight_positive_in_outer_cones=True,links=links,figures=figures,
        report_sha256=reports,extra_source_sha256={name:sha(src/name) for name in extra},source_sha256=sha(Path(__file__)),
        scope='Forward-cone relaxation capacity; no realized joint BP credit proof or online task gain')
    target=root/'results/round_240_audit.json';assert not target.exists();target.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
