"""Integrity and independent primitive rerun for rounds 175-176."""
import argparse,hashlib,json,re
from pathlib import Path
import verify_conditional_fiber as primitive


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research';inp=root/'results/readout_hierarchy/diagnostic';w=root/'results/conditional_fiber/primitive'
    p=json.loads((inp/'protocol.json').read_text());wp=json.loads((w/'protocol.json').read_text())
    for protocol in [p,wp]:
        for name,value in protocol['source_sha256'].items():assert sha(code/name)==value,name
    parent=root/'results/sampling_averaged_risk/diagnostic'
    for name,key in [('protocol.json','parent_protocol_sha256'),('summary.json','parent_summary_sha256'),('rows.json','parent_rows_sha256')]:assert sha(parent/name)==p[key],name
    rows=json.loads((inp/'rows.json').read_text());s=json.loads((inp/'summary.json').read_text());contexts=json.loads((inp/'contexts.json').read_text())
    assert s['complete'] and len(rows)==s['rows']==s['sum_checks']==320
    for row in rows:assert abs(row['within']+row['between']-row['prior_sampling'])<1e-15
    assert len(contexts)==16 and sum(c['last_layer_bound_applicable'] for c in contexts)==s['last_layer_bound_contexts']==16
    rerun=primitive.verify();assert rerun==json.loads((w/'result.json').read_text())
    f=json.loads((inp/'figure_audit.json').read_text());assert sha(code/'plot_readout_hierarchy.py')==f['source_sha256']
    assert sha(inp/'summary.json')==f['hierarchy_summary_sha256'] and sha(w/'result.json')==f['witness_result_sha256']
    for name,value in f['figures'].items():assert sha(root/name)==value,name
    links=0;documents=[]
    for number in [175,176]:
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1;doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('https:','http:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(name=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),hierarchy_frozen_sources=len(p['source_sha256']),
                primitive_frozen_sources=len(wp['source_sha256']),sum_identities=len(rows),contexts_with_last_layer_bound=16,
                independent_primitive_rerun=rerun,figures=f,documents=documents,local_links=links,
                scope='risk attribution, mathematical witness and primitive; no online or PC-specific success implied')
    (root/'results/round_176_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
