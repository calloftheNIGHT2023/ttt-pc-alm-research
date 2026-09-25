"""Final chain audit for cold-block experiment and exact reachability result."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/context_block_scaling';inp=base/'development';p=json.loads((inp/'protocol.json').read_text())
    run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete'] and run['adaptations']==2304
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());assert len(rows)==run['adaptations']
    states=0;details=0
    for r in rows:
        if r['complete']:assert sha(inp/r['state_file'])==r['state_sha256'];states+=1
        if 'detail_file' in r:assert sha(inp/r['detail_file'])==r['detail_sha256'];details+=1
    a=json.loads((base/'analysis/summary.json').read_text());assert a['passed'] and a['checks']['states']==states==2108
    assert a['source_sha256']==sha(Path(__file__).with_name('analyze_context_block_scaling.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    cert=json.loads((base/'certificate_audit/summary.json').read_text());assert cert['passed'] and cert['checks']['exact_certificates']==1457
    assert cert['source_sha256']==sha(Path(__file__).with_name('audit_context_block_certificates.py'))
    assert cert['protocol_sha256']==sha(inp/'protocol.json')
    resources=0;blocks=0;failures=0
    for seed in [5920001,5920015]:
        for cfg in p['configs']:
            r=json.loads((base/f'resources/{seed}_{cfg["name"]}.json').read_text());assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_context_block_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            resources+=1;blocks+=len(r['blocks']);failures+=sum(not b['complete'] for b in r['blocks'])
    reach=base/'reachability';rp=json.loads((reach/'protocol.json').read_text());rs=json.loads((reach/'summary.json').read_text())
    assert rs['passed'] and rs['source_sha256']==sha(Path(__file__).with_name('audit_cold_pattern_reachability.py'))
    for name,value in rp['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value
    assert rp['rows_sha256']==sha(inp/'rows.json') and rp['parent_protocol_sha256']==sha(inp/'protocol.json')
    exact=json.loads((reach/'integer_audit.json').read_text());assert exact['passed'] and exact['checks']['strict_two_edit_certificates']==1
    assert exact['source_sha256']==sha(Path(__file__).with_name('verify_cold_reachability_integer.py'))
    assert exact['modes_sha256']==sha(reach/'modes.json') and exact['protocol_sha256']==sha(reach/'protocol.json')
    figure=json.loads((reach/'figure_audit.json').read_text());assert figure['passed']
    assert figure['source_sha256']==sha(Path(__file__).with_name('plot_cold_reachability_result.py'))
    assert figure['input_sha256']==sha(reach/'integer_audit.json') and figure['figure_sha256']==sha(reach/'finite_search_witness.png')
    docs=root/'outputs/ttt-pc-alm-research';links=0;figures=[]
    for name in ['209_context_block_scaling_protocol.md','210_context_block_scaling_results.md','211_cold_pattern_reachability_protocol.md','212_finite_search_witness_results.md','213_primal_upper_gate_design.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists(),target;links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert len(figures)==2
    result=dict(passed=True,online_sources=len(p['source_sha256']),reachability_sources=len(rp['source_sha256']),
        adaptations=run['adaptations'],states=states,failures=run['failures'],details=details,
        independent_checks=a['checks'],certificate_checks=cert['checks'],fresh_processes=resources,resource_blocks=blocks,resource_failed_blocks=failures,
        exact_reachability_checks=exact['checks'],links=links,figures=figures,source_sha256=sha(Path(__file__)),
        scope='Audited development result and finite-family search witness; no independent PC-ALM dominance')
    (root/'results/round_210_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
