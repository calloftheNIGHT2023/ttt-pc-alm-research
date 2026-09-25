"""Integrity audit of light routing, real online run and priority diagnostic."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research';results=root/'results';protocols=[]
    for folder in ['light_tied_routing/diagnostic','typed_routing_memory/development','dual_priority/diagnostic','deferred_router/diagnostic']:
        path=results/folder/'protocol.json';p=json.loads(path.read_text())
        for name,value in p['source_sha256'].items():assert sha(code/name)==value,(folder,name)
        protocols.append(dict(folder=folder,source_hashes=len(p['source_sha256']),protocol_sha256=sha(path)))
    typed=results/'typed_routing_memory';priority=results/'dual_priority';light=results/'light_tied_routing'
    p=json.loads((typed/'development/protocol.json').read_text());assert sha(code/'typed_routing_development.json')==p['config_sha256']
    rr=json.loads((typed/'development/run_audit.json').read_text());assert rr['complete'] and rr['rows']==1280 and rr['old_reference_states_bitwise']==384
    a=json.loads((typed/'analysis/summary.json').read_text());assert sha(code/'analyze_typed_routing_memory.py')==a['analysis_source_sha256']
    assert a['audit']['states_and_queries']==1280 and a['audit']['exact_proofs']==1948 and a['audit']['independent_lp_modes']==361
    resources=[]
    for cfg in p['configs']:
        path=typed/'development'/f'resources_{cfg["name"]}.json';r=json.loads(path.read_text())
        assert r['method']==cfg['name'] and r['source_sha256']==p['source_sha256'] and len(r['stages'])==4
        assert sha(code/'audit_typed_routing_resources.py')==r['audit_source_sha256']
        assert all(z['original_outputs_exact'] for z in r['stages']);resources.append(dict(file=path.name,sha256=sha(path),peak_bytes=r['tracked_peak_bytes']))
    lp=json.loads((light/'diagnostic/summary.json').read_text());assert lp['original_states']==112 and lp['old_records']==21666
    figures=[];fa=json.loads((light/'diagnostic/figure_audit.json').read_text())
    assert sha(code/'plot_light_tied_routing.py')==fa['source_sha256']
    assert sha(light/'diagnostic/summary.json')==fa['summary_sha256'] and sha(light/'diagnostic/costs.json')==fa['costs_sha256']
    assert sha(light/'complexity.json')==fa['complexity_sha256'];assert sha(light/'diagnostic/light_routing_admission.png')==fa['figure_sha256'];figures.append(fa)
    figures.append(dict(file='typed_routing_memory/analysis/typed_routing_online.png',sha256=sha(typed/'analysis/typed_routing_online.png'),source_sha256=sha(code/'analyze_typed_routing_memory.py')))
    pp=json.loads((priority/'diagnostic/protocol.json').read_text());assert pp['parent_protocol_sha256']==sha(typed/'development/protocol.json')
    pr=json.loads((priority/'diagnostic/run_audit.json').read_text());assert pr['complete'] and pr['original_records']==2331 and pr['actual_first_write_readouts']==784
    pa=json.loads((priority/'analysis/summary.json').read_text());assert sha(code/'analyze_dual_priority.py')==pa['analysis_source_sha256']
    for name,value in pa['input_sha256'].items():assert sha(priority/'diagnostic'/name)==value
    assert pa['audit']['states_and_queries']==784 and pa['audit']['score_orders']==112 and pa['audit']['full_states_equal']==96
    fa=json.loads((priority/'analysis/figure_audit.json').read_text());assert sha(code/'analyze_dual_priority.py')==fa['source_sha256']
    assert sha(priority/'analysis/summary.json')==fa['summary_sha256'] and sha(priority/'analysis/dual_priority.png')==fa['figure_sha256'];figures.append(fa)
    late=results/'deferred_router/diagnostic';lp=json.loads((late/'protocol.json').read_text());ls=json.loads((late/'summary.json').read_text())
    assert lp['parent_protocol_sha256']==sha(typed/'development/protocol.json') and ls['complete']
    assert ls['original_inputs']==2331 and ls['proposal_visits']==169370 and ls['exact_online_states']==16
    assert ls['old_c20_calls']==938 and ls['new_c20_calls']==22 and ls['c20_rows']==3797
    lr=json.loads((late/'rows.json').read_text());assert len(lr)==16 and all(r['state_bitwise'] for r in lr)
    documents=[];links=0
    for number in range(163,171):
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1,(number,matches);doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(file=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocols=protocols,total_source_hashes=sum(p['source_hashes'] for p in protocols),
        documents=documents,local_links=links,figures=figures,resources=resources,typed_analysis_sha256=sha(typed/'analysis/summary.json'),
        priority_analysis_sha256=sha(priority/'analysis/summary.json'),deferred_summary_sha256=sha(late/'summary.json'),scope='artifact and audit integrity; not independent confirmation or task-superiority claim')
    (results/'round_168_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
