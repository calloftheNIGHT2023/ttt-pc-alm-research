"""Hash/link/count audit for the causal feedback and physical-cut round."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project.resolve();code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research'
    groups=[('causal_conflict/diagnostic',None),('conflict_feedback_memory/development','conflict_feedback_development.json'),
            ('feedback_repair_paths/diagnostic',None),('physical_residual_cut/diagnostic',None),('physical_cut_memory/development','physical_cut_development.json')]
    protocols=[]
    for folder,config in groups:
        path=root/'results'/folder/'protocol.json';data=json.loads(path.read_text())
        for name,value in data['source_sha256'].items():assert sha(code/name)==value,(folder,name)
        if config:assert sha(code/config)==data['config_sha256']
        protocols.append(dict(folder=folder,source_hashes=len(data['source_sha256']),protocol_sha256=sha(path)))
    links=0;documents=[]
    for number in range(145,157):
        found=list(docs.glob(f'{number}_*.md'));assert len(found)==1,(number,found);doc=found[0];documents.append(dict(file=doc.name,sha256=sha(doc)))
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
    counts={'conflict_feedback_memory/development/episodes.json':1920,'feedback_repair_paths/diagnostic/rows.json':64,
            'physical_residual_cut/diagnostic/rows.json':128,'physical_cut_memory/development/episodes.json':896}
    for path,count in counts.items():assert len(json.loads((root/'results'/path).read_text()))==count,path
    inputs=root/'results';independent=json.loads((inputs/'causal_conflict/diagnostic/independent_coverage.json').read_text())
    assert sha(code/'analyze_causal_conflict.py')==independent['source_sha256'];assert sha(inputs/'causal_conflict/diagnostic/chronology.json')==independent['input_sha256']
    for folder,name in [('conflict_feedback_memory','analyze_conflict_feedback_memory.py'),('physical_cut_memory','analyze_physical_cut_memory.py')]:
        report=json.loads((inputs/folder/'analysis/summary.json').read_text());assert sha(code/name)==report['analysis_source_sha256']
    figure=inputs/'physical_residual_cut/diagnostic';audit=json.loads((figure/'figure_audit.json').read_text());assert sha(code/'plot_physical_residual_cut.py')==audit['source_sha256']
    assert sha(figure/'summary.json')==audit['input_sha256'];assert sha(figure/'physical_cut_admission.png')==audit['figure_sha256']
    primitive=json.loads((inputs/'physical_residual_cut/quadratic_verification.json').read_text());assert primitive['passed'];assert sha(code/'verify_physical_cut_quadratics.py')==primitive['verification_source_sha256']
    assert sha(code/'credit_residual_cut.py')==primitive['runtime_source_sha256']
    tied=json.loads((inputs/'tied_local_block/derivation_verification.json').read_text());assert tied['passed']
    assert sha(code/'verify_tied_block_derivation.py')==tied['source_sha256'];assert sha(code/'credit_residual_cut.py')==tied['scalar_source_sha256']
    resources=[]
    for folder in ['conflict_feedback_memory','physical_cut_memory']:
        resource_paths=sorted((inputs/folder/'development').glob('resources_*.json'))
        assert len(resource_paths)==(15 if folder=='conflict_feedback_memory' else 14)
        audit_source='audit_conflict_feedback_resources.py' if folder=='conflict_feedback_memory' else 'audit_physical_cut_resources.py'
        for path in resource_paths:
            data=json.loads(path.read_text());assert all(s['original_outputs_exact'] for s in data['stages']);assert len(data['stages'])==4
            assert sha(code/audit_source)==data['audit_source_sha256']
            resources.append(dict(folder=folder,method=data['method'],source_sha256=data['audit_source_sha256'],file_sha256=sha(path)))
    result=dict(passed=True,protocols=protocols,total_source_hashes=sum(r['source_hashes'] for r in protocols),documents=documents,local_links=links,row_counts=counts,
                resource_files_checked=resources,source_sha256=sha(Path(__file__)),
                figures={str(path.relative_to(inputs)):sha(path) for path in [inputs/'causal_conflict/diagnostic/causal_conflict_results.png',inputs/'conflict_feedback_memory/analysis/conflict_feedback_results.png',figure/'physical_cut_admission.png',inputs/'physical_cut_memory/analysis/physical_cut_online.png']},
                scope='consistency audit; mathematical and empirical claims require their separate result audits; 156 is algebra-tested derivation, not an implemented adaptation rule')
    (inputs/'round_156_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
