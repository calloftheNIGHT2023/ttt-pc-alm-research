"""Full chain audit for common-recovery online result and linked report."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_online_comparison/development';p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    assert run['execution_complete'] and run['episodes']==len(p['configs'])*len(p['seeds'])*p['repetitions']==576
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text());failures=json.loads((inp/'failures.json').read_text())
    assert len(rows)==run['stages']==sum(e['stages'] for e in episodes) and len(failures)==sum(not e['complete'] for e in episodes)==run['failures']
    for r in rows:
        assert sha(inp/r['state_file'])==r['state_sha256']
        if 'detail_file' in r:assert sha(inp/r['detail_file'])==r['detail_sha256']
    a=json.loads((root/'results/recovered_online_comparison/analysis/summary.json').read_text())
    assert a['complete'] and a['checks']['risk_replays']==len(rows) and a['checks']['recovery_events']==run['recovery_events']
    assert a['source_sha256']==sha(Path(__file__).with_name('analyze_recovered_online_comparison.py'))
    old=json.loads((root/'results/round_201_audit.json').read_text());assert old['passed']
    assert old['source_sha256']==sha(Path(__file__).with_name('audit_research_round_201.py'))
    oldcert=json.loads((root/'results/online_endpoint_h2/certificate_audit/summary.json').read_text());assert oldcert['passed']
    assert oldcert['source_sha256']==sha(Path(__file__).with_name('audit_online_endpoint_certificates.py'))
    diagnostic=json.loads((root/'results/common_prior_recovery/development/independent_audit.json').read_text());assert diagnostic['passed']
    assert diagnostic['source_sha256']==sha(Path(__file__).with_name('audit_common_prior_recovery.py'))
    resources=0;resource_stages=0;resource_recoveries=0
    for seed in [5920001,5920015]:
        for cfg in p['configs']:
            r=json.loads((root/f'results/recovered_online_comparison/resources/{seed}_{cfg["name"]}.json').read_text())
            assert r['passed'] and r['protocol_sha256']==sha(inp/'protocol.json')
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_recovered_online_resources.py'))
            resources+=1;resource_stages+=r['stage_replays'];resource_recoveries+=r['recovery_stages']
    docs=root/'outputs/ttt-pc-alm-research';links=0;figures=[]
    for name in ['202_common_prior_recovery_protocol.md','203_recovered_online_protocol.md','204_recovered_online_results.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists();links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert figures
    result=dict(passed=True,source_hashes=len(p['source_sha256']),episodes=len(episodes),states=len(rows),failures=len(failures),
        independent_checks=a['checks'],fresh_resources=resources,resource_state_replays=resource_stages,resource_recovery_events=resource_recoveries,
        prior_certificate_audit_sha256=sha(root/'results/online_endpoint_h2/certificate_audit/summary.json'),
        links=links,figures=figures,source_sha256=sha(Path(__file__)),scope='Audited complete development comparison, not core goal completion')
    (root/'results/round_204_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
