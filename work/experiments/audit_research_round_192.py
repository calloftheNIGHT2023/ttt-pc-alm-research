"""Final source, artifact, proof and resource accounting for rounds191-192."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/light_h2_credit/development';p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    assert run['execution_complete'] and run['episodes']==864
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text());failures=json.loads((inp/'failures.json').read_text())
    assert len(rows)==run['stages'] and len(failures)==run['failures'] and sum(e['stages'] for e in episodes)==len(rows)
    detail_count=0
    for row in rows:
        assert sha(inp/row['state_file'])==row['state_sha256']
        if 'detail_file' in row:assert sha(inp/row['detail_file'])==row['detail_sha256'];detail_count+=1
    a=json.loads((root/'results/light_h2_credit/analysis/summary.json').read_text());assert a['complete'] and a['checks']['prediction_replays']==len(rows)
    c=json.loads((root/'results/light_h2_credit/certificate_audit/summary.json').read_text());assert c['passed']
    assert c['checks']['exact_certificates']==c['checks']['full_region_lp_infeasible']==c['checks']['independent_relaxed_lp']
    assert sha(Path(__file__).with_name('analyze_light_h2_credit.py'))==a['source_sha256']
    assert sha(Path(__file__).with_name('audit_light_h2_certificates_v2.py'))==c['source_sha256']
    resource_count=0
    for cfg in p['configs']:
        r=json.loads((root/f'results/light_h2_credit/resources/{cfg["name"]}.json').read_text())
        assert r['passed'] and r['stage_replays']==4 and r['protocol_sha256']==sha(inp/'protocol.json')
        assert r['source_sha256']==sha(Path(__file__).with_name('audit_light_h2_resources.py'));resource_count+=1
    docs=root/'outputs/ttt-pc-alm-research';links=0;figures=[]
    for name in ['191_light_h2_credit_protocol.md','192_light_h2_credit_results.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            linked=(path.parent/target).resolve();assert linked.exists(),str(linked);links+=1
            if linked.suffix=='.png':figures.append(dict(file=str(linked),sha256=sha(linked)))
    assert figures
    result=dict(passed=True,source_hashes=len(p['source_sha256']),stages=len(rows),failed_episodes_preserved=len(failures),details=detail_count,
        independent_prediction_replays=a['checks']['prediction_replays'],same_frontier_input_hashes=a['checks']['same_frontier_input_hashes'],
        exact_certificates=c['checks']['exact_certificates'],independent_region_lp=c['checks']['full_region_lp_infeasible'],
        independent_relaxation_lp=c['checks']['independent_relaxed_lp'],cross_bank_exact_directions=c['checks']['cross_bank_exact_directions'],
        fresh_resources=resource_count,resource_stage_replays=4*resource_count,links=links,figures=figures,
        scope='light-H2 stage verified; independent full research objective assessed separately, not marked achieved by this audit')
    (root/'results/round_192_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
