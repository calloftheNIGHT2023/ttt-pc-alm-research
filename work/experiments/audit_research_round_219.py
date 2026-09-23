"""Frozen inputs, exact skips, all-state arithmetic, resources and reports."""
import argparse
import hashlib
import json
import re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/online_primal_gate';inp=base/'development_v2';p=json.loads((inp/'protocol.json').read_text())
    run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete'] and run['episodes']==768 and run['stages']==3072 and run['failures']==0
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert p['failed_attempt_protocol_sha256']==sha(base/'development/protocol.json')
    assert not (base/'development/run_audit.json').exists()
    rows=json.loads((inp/'rows.json').read_text());assert len(rows)==3072
    for row in rows:
        assert sha(inp/row['state_file'])==row['state_sha256']
        if 'detail_file' in row:assert sha(inp/row['detail_file'])==row['detail_sha256']
    a=json.loads((base/'analysis/summary.json').read_text());assert a['complete'] and a['all_methods_completed']
    assert a['checks']['states']==a['checks']['risk_replays']==3072 and a['checks']['recovery_events']==run['recovery_events']
    assert a['source_sha256']==sha(Path(__file__).with_name('analyze_online_primal_gate.py'))
    for name,value in a['input_sha256'].items():assert sha(inp/name)==value
    s=json.loads((base/'skip_audit/summary.json').read_text());assert s['passed'] and s['checks']['banks']==96
    assert s['source_sha256']==sha(Path(__file__).with_name('audit_online_primal_skips.py'))
    for name,value in s['input_sha256'].items():assert sha(inp/name)==value
    croot=root/'results/bounded_error_primal_gate/component';c=json.loads((croot/'integer_audit.json').read_text())
    assert c['passed'] and c['checks']['exact_directions']==1889166 and c['checks']['skipped_directions']==840143
    assert c['source_sha256']==sha(Path(__file__).with_name('audit_primal_gate_integer.py'))
    for field,name in [('protocol_sha256','protocol.json'),('audits_sha256','audits.json'),('summary_sha256','summary.json')]:assert c[field]==sha(croot/name)
    f=json.loads((croot/'figure_audit.json').read_text());assert f['passed'] and f['source_sha256']==sha(Path(__file__).with_name('plot_primal_gate_components.py'))
    for name,value in f['input_sha256'].items():assert sha(root/name)==value
    assert f['figure_sha256']==sha(croot/'primal_gate_components.png')
    cf=json.loads((base/'analysis/contrast_figure_audit.json').read_text());assert cf['passed']
    assert cf['source_sha256']==sha(Path(__file__).with_name('plot_online_primal_contrasts.py'))
    assert cf['summary_sha256']==sha(base/'analysis/summary.json') and cf['figure_sha256']==sha(base/'analysis/primal_gate_paired_contrasts.png')
    prior=json.loads((root/'results/round_204_audit.json').read_text());assert prior['passed']
    assert prior['source_sha256']==sha(Path(__file__).with_name('audit_research_round_204.py'))
    resources=0;stages=0;recoveries=0
    for seed in [5920001,5920015]:
        for cfg in p['configs']:
            r=json.loads((base/f'resources/{seed}_{cfg["name"]}.json').read_text());assert r['passed']
            assert r['source_sha256']==sha(Path(__file__).with_name('audit_online_primal_resources.py')) and r['protocol_sha256']==sha(inp/'protocol.json')
            resources+=1;stages+=r['stage_replays'];recoveries+=r['recovery_stages']
    assert resources==48 and stages==192
    links=0;figures=[];docs=root/'outputs/ttt-pc-alm-research'
    for name in ['213_primal_upper_gate_design.md','214_directional_upper_gate_protocol.md','215_upper_gate_factorization_protocol.md',
        '216_bounded_roundoff_upper_protocol.md','217_primal_upper_gate_results.md','218_online_primal_gate_protocol.md','218_execution_note.md','219_online_primal_gate_results.md']:
        path=docs/name
        for target in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in target or target.startswith('#'):continue
            target=(path.parent/target).resolve();assert target.exists(),target;links+=1
            if target.suffix=='.png':figures.append(dict(path=str(target),sha256=sha(target)))
    assert len(figures)>=2
    result=dict(passed=True,source_hashes=len(p['source_sha256']),episodes=run['episodes'],states=len(rows),failures=0,
        independent_checks=a['checks'],integer_component_checks=c['checks'],integer_online_checks=s['checks'],
        fresh_processes=resources,resource_stages=stages,resource_recoveries=recoveries,links=links,figures=figures,
        source_sha256=sha(Path(__file__)),scope='Development complete-equivalence gate and cost result; not independent PC-ALM supremacy')
    out=root/'results/round_219_audit.json';assert not out.exists();out.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
