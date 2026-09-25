"""Source, output, independent resource and report integrity for round 171-172."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research';group=root/'results/deferred_typed_memory';inp=group/'development';out=group/'analysis'
    protocol=json.loads((inp/'protocol.json').read_text())
    for name,value in protocol['source_sha256'].items():assert sha(code/name)==value,name
    assert sha(code/'deferred_typed_development.json')==protocol['config_sha256'] and sha(code/protocol['base_config'])==protocol['base_config_sha256']
    assert protocol['parent_protocol_sha256']==sha(root/'results/deferred_router/diagnostic/protocol.json')
    assert protocol['parent_summary_sha256']==sha(root/'results/deferred_router/diagnostic/summary.json')
    run=json.loads((inp/'run_audit.json').read_text());assert run['complete'] and run['rows']==4352 and run['extra_queues_and_origins_exact']==896
    analysis=json.loads((out/'summary.json').read_text());assert sha(code/'analyze_deferred_typed_memory.py')==analysis['analysis_source_sha256']
    for name,value in analysis['input_sha256'].items():assert sha(inp/name)==value,name
    assert analysis['conditional_reference_sha256']==sha(root/'results/typed_routing_memory/analysis/summary.json')
    assert analysis['audit']['states_bitwise']==4352 and analysis['audit']['distinct_method_state_query_replays']==1280 and analysis['audit']['query_rows_checked']==4352
    resources=[]
    for cfg in protocol['configs']:
        path=inp/f'resources_{cfg["name"]}_{cfg["routing_implementation"]}.json';r=json.loads(path.read_text())
        assert r['method']==cfg['name'] and r['implementation']==cfg['routing_implementation'] and r['source_sha256']==protocol['source_sha256']
        assert sha(code/'audit_deferred_typed_resources.py')==r['audit_source_sha256']
        assert len(r['stages'])==4 and all(s['original_outputs_exact'] for s in r['stages'])
        resources.append(dict(method=r['method'],implementation=r['implementation'],file=path.name,sha256=sha(path),peak_bytes=r['tracked_peak_bytes']))
    fa=json.loads((out/'figure_audit.json').read_text());assert sha(code/'analyze_deferred_typed_memory.py')==fa['source_sha256'] and sha(out/'summary.json')==fa['summary_sha256']
    for name,value in fa['figures'].items():assert sha(out/name)==value,name
    documents=[];links=0
    for number in [171,172]:
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1;doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(file=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),frozen_source_hashes=len(protocol['source_sha256']),protocol_sha256=sha(inp/'protocol.json'),
        analysis_sha256=sha(out/'summary.json'),run_audit=run,resources=resources,figures=fa,documents=documents,local_links=links,
        scope='full frozen experiment and artifact integrity; no novel quality improvement or independent superiority implied')
    (root/'results/round_172_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
