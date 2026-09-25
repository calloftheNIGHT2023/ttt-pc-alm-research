"""Final frozen-source, state, proof, resource and report audit for round200."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/online_endpoint_h2/development';p=json.loads((inp/'protocol.json').read_text())
    run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete'] and run['episodes']==1120
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text());failures=json.loads((inp/'failures.json').read_text())
    assert len(rows)==run['stages']==sum(e['stages'] for e in episodes)
    assert len(failures)==run['failures']==sum(not e['complete'] for e in episodes)
    details=0
    for row in rows:
        assert sha(inp/row['state_file'])==row['state_sha256']
        if 'detail_file' in row:assert sha(inp/row['detail_file'])==row['detail_sha256'];details+=1
    analysis=json.loads((root/'results/online_endpoint_h2/analysis/summary.json').read_text())
    assert analysis['complete'] and analysis['checks']['prediction_replays']==len(rows)
    assert analysis['source_sha256']==sha(Path(__file__).with_name('analyze_online_endpoint_h2.py'))
    cert=json.loads((root/'results/online_endpoint_h2/certificate_audit/summary.json').read_text());assert cert['passed']
    assert cert['source_sha256']==sha(Path(__file__).with_name('audit_online_endpoint_certificates.py'))
    assert cert['checks']['exact_certificates']==cert['checks']['independent_full_region_lp']==cert['checks']['independent_relaxed_lp']
    resources=0
    for cfg in p['configs']:
        r=json.loads((root/f'results/online_endpoint_h2/resources/{cfg["name"]}.json').read_text())
        assert r['passed'] and r['stage_replays']==4 and r['protocol_sha256']==sha(inp/'protocol.json')
        assert r['source_sha256']==sha(Path(__file__).with_name('audit_online_endpoint_resources.py'));resources+=1
    links=0;figures=[];docs=root/'outputs/ttt-pc-alm-research'
    for name in ['200_online_endpoint_h2_protocol.md','201_online_endpoint_h2_results.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists();links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert figures
    result=dict(passed=True,source_hashes=len(p['source_sha256']),episodes=len(episodes),states=len(rows),details=details,
        failed_episodes=len(failures),original_states_bitwise=run['checks']['original191_states_bitwise'],
        independent_checks=analysis['checks'],certificate_checks=cert['checks'],fresh_resources=resources,
        resource_state_replays=resources*4,links=links,figures=figures,source_sha256=sha(Path(__file__)),
        scope='Complete finite development result; not official TTT or downstream validation or core research goal completion')
    (root/'results/round_201_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
