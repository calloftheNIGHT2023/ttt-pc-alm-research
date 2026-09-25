"""Final provenance, state/resource counts, and linked report audit for 189-190."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text())
    assert run['execution_complete'] and run['stages']+4*run['failures']==2432
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());state_checks=0;detail_checks=0
    for row in rows:
        assert sha(inp/row['state_file'])==row['state_sha256'];state_checks+=1
        if 'detail_file' in row:assert sha(inp/row['detail_file'])==row['detail_sha256'];detail_checks+=1
    analysis=root/'results/independent_hybrid/analysis/summary.json';a=json.loads(analysis.read_text())
    assert a['complete'] and a['checks']['state_hashes']==run['stages']
    assert a['checks']['own_state_links']==3*(448-run['failures'])
    assert sha(Path(__file__).with_name('analyze_independent_hybrid_memory.py'))==a['source_sha256']
    cost=root/'results/independent_hybrid/cost_model/summary.json';c=json.loads(cost.read_text())
    assert c['passed'] and c['records']==64
    assert sha(Path(__file__).with_name('audit_independent_hybrid_cost_model.py'))==c['source_sha256']
    failure_path=root/'results/independent_hybrid/failure_audit/summary.json';f=json.loads(failure_path.read_text())
    assert f['passed'] and f['failures']==run['failures'] and f['partial_state_replays']==10
    assert all(r['all_raw_cells_lp_infeasible'] and r['failed_n']==8 for r in f['records'])
    assert sha(Path(__file__).with_name('audit_independent_hybrid_failures.py'))==f['source_sha256']
    resource_checks=0
    for cfg in p['configs']:
        path=root/f'results/independent_hybrid/resources/{cfg["name"]}.json';r=json.loads(path.read_text())
        assert r['passed'] and r['stage_states_and_predictions_bitwise']==4 and r['protocol_sha256']==sha(inp/'protocol.json')
        assert r['source_sha256']==sha(Path(__file__).with_name('audit_independent_hybrid_resources.py'));resource_checks+=1
    docs=root/'outputs/ttt-pc-alm-research';links=0;figures=[]
    for name in ['189_independent_hybrid_protocol.md','190_independent_hybrid_results.md']:
        path=docs/name;text=path.read_text(encoding='utf-8')
        for target in re.findall(r'\]\(([^)]+)\)',text):
            if '://' in target or target.startswith('#'):continue
            linked=(path.parent/target).resolve();assert linked.exists(),str(linked);links+=1
            if linked.suffix=='.png':figures.append(dict(file=str(linked),sha256=sha(linked)))
    assert figures
    result=dict(passed=True,source_hash_checks=len(p['source_sha256']),state_files=state_checks,
        detail_files=detail_checks,independent_prediction_replays=run['stages'],failed_trials_preserved=run['failures'],resource_processes=resource_checks,
        resource_stage_replays=resource_checks*4,cost_prediction_pairs=64,cost_primitive=c['primitive'],
        partial_failed_trial_state_replays=f['partial_state_replays'],failure_lp_checks=sum(r['raw_candidate_cells'] for r in f['records']),
        local_links=links,figures=figures,analysis_sha256=sha(analysis),cost_summary_sha256=sha(cost),
        scope='189-190 complete stage result; core independent-PC research objective remains separately assessed')
    (root/'results/round_190_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
