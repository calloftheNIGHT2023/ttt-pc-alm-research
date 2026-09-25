"""Audit frozen sources, replay counts, links and figures for rounds 157-161."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research';results=root/'results';group=results/'tied_local_block'
    protocols=[]
    for folder in ['physical_cut_memory/development','tied_local_block/diagnostic','tied_local_block/pool_audit']:
        path=results/folder/'protocol.json';p=json.loads(path.read_text())
        for name,value in p['source_sha256'].items():assert sha(code/name)==value,(folder,name)
        protocols.append(dict(folder=folder,source_hashes=len(p['source_sha256']),protocol_sha256=sha(path)))
    admission=json.loads((group/'diagnostic/summary.json').read_text());pool=json.loads((group/'pool_audit/summary.json').read_text())
    assert admission['original_state_replays']==112 and admission['old_records_replayed']==21666 and admission['prior_scalar_task_replays']==64
    assert len(json.loads((group/'diagnostic/rows.json').read_text()))==336
    assert pool['passed'] and pool['original_states']==16 and pool['old_records']==2331 and pool['parent_selection_replays']==6993
    assert len(json.loads((group/'pool_audit/rows.json').read_text()))==96
    p=json.loads((group/'pool_audit/protocol.json').read_text());assert sha(group/'diagnostic/protocol.json')==p['parent_protocol_sha256']
    assert sha(group/'diagnostic/summary.json')==p['parent_summary_sha256']
    figures=[]
    a=json.loads((group/'witness_figure_audit.json').read_text());assert sha(code/'plot_tied_block_witness.py')==a['source_sha256']
    assert sha(group/'constructive_witness.json')==a['input_sha256'];assert sha(group/'tied_block_witness.png')==a['figure_sha256'];figures.append(a)
    a=json.loads((group/'diagnostic/figure_audit.json').read_text());assert sha(code/'plot_tied_block_admission.py')==a['source_sha256']
    assert sha(group/'diagnostic/summary.json')==a['summary_sha256'];assert sha(group/'diagnostic/costs.json')==a['costs_sha256']
    assert sha(group/'diagnostic/tied_block_admission.png')==a['figure_sha256'];figures.append(a)
    a=json.loads((group/'pool_audit/figure_audit.json').read_text());assert sha(code/'plot_tied_pool_routing.py')==a['source_sha256']
    assert sha(group/'pool_audit/analysis.json')==a['input_sha256'];assert sha(group/'pool_audit/tied_pool_routing.png')==a['figure_sha256'];figures.append(a)
    analysis=json.loads((group/'pool_audit/analysis.json').read_text());assert sha(code/'analyze_tied_candidate_pool.py')==analysis['source_sha256']
    for name,value in analysis['input_sha256'].items():assert sha(group/'pool_audit'/name)==value
    assert analysis['independent_positive_regions']==5 and analysis['exact_new_examples_replayed']==6
    witness=json.loads((group/'constructive_witness.json').read_text());assert witness['passed']
    for field,name in [('source_sha256','tied_local_block_witness.py'),('runtime_sha256','tied_local_block_repair.py')]:
        assert sha(code/name)==witness[field],field
    documents=[];links=0
    for number in range(157,163):
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1,(number,matches);doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(file=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),protocols=protocols,total_source_hashes=sum(p['source_hashes'] for p in protocols),
        documents=documents,local_links=links,figures=figures,admission_summary_sha256=sha(group/'diagnostic/summary.json'),pool_summary_sha256=sha(group/'pool_audit/summary.json'),
        scope='source/result/link integrity; not proof of independent task superiority; prior rounds retain their separate audits')
    (results/'round_161_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
