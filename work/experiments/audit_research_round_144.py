"""Read-only source/link consistency checks and generated audit artifact."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project.resolve();docs=root/'outputs/ttt-pc-alm-research';code=root/'work/experiments'
    paths=['retained_credit/diagnostic','retained_credit/cross_bank','retained_credit/fused_verification','retained_credit_memory/development',
        'credit_primal_reconstruction/diagnostic','credit_primal_reconstruction/diagnostic_v2','credit_conflict/diagnostic','matched_conflict_parents/diagnostic']
    protocols=[]
    for path in paths:
        pp=root/'results'/path/'protocol.json';data=json.loads(pp.read_text())
        for name,h in data['source_sha256'].items():assert sha(code/name)==h,(path,name)
        if 'config_sha256' in data:assert sha(code/'retained_credit_development.json')==data['config_sha256']
        protocols.append(dict(path=path,source_hashes=len(data['source_sha256']),protocol_sha256=sha(pp)))
    links=0;files=[]
    for number in range(132,145):
        found=list(docs.glob(f'{number}_*.md'));assert len(found)==1,(number,found);doc=found[0];files.append(doc.name)
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            target=target.split('#')[0].strip('<>');assert (doc.parent/target).exists(),(doc.name,target);links+=1
    checks={'retained_credit_memory/development/episodes.json':2304,'credit_primal_reconstruction/diagnostic_v2/rows.json':576,'credit_conflict/diagnostic/rows.json':144,'matched_conflict_parents/diagnostic/rows.json':144}
    for path,count in checks.items():assert len(json.loads((root/'results'/path).read_text()))==count,path
    figure=root/'results/matched_conflict_parents/diagnostic';audit=json.loads((figure/'figure_audit.json').read_text())
    assert sha(code/'plot_matched_conflict_parents.py')==audit['source_sha256'];assert sha(figure/'summary.json')==audit['input_sha256'];assert sha(figure/'matched_parent_results.png')==audit['figure_sha256']
    result=dict(passed=True,protocols=protocols,total_source_checks=sum(r['source_hashes'] for r in protocols),documents=files,local_links_checked=links,row_counts=checks,figure_hashes_verified=True,
        source_sha256=sha(Path(__file__)),scope='source hashes, document links and counts; empirical/numerical audits remain in their own result files; v1 partial failure intentionally preserved')
    (root/'results/round_144_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
