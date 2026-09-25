"""Integrity and algebraic checks for expected-particle-risk attribution."""
import argparse,hashlib,json,re
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project.resolve()
    code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research';inp=root/'results/sampling_averaged_risk/diagnostic'
    p=json.loads((inp/'protocol.json').read_text());s=json.loads((inp/'summary.json').read_text());rows=json.loads((inp/'rows.json').read_text())
    for name,value in p['source_sha256'].items():assert sha(code/name)==value,name
    assert s['complete'] and s['context_methods']==s['original_truncation_replays']==len(rows)==320
    assert p['conditional_analysis_sha256']==sha(root/'results/typed_routing_memory/analysis/summary.json')
    assert p['conditional_rows_sha256']==sha(root/'results/typed_routing_memory/analysis/conditional_risks.json')
    for row in rows:
        assert abs(row['ideal_truncation']+row['expected_sampling']-row['expected_total_excess'])<1e-15
        assert abs(row['fixed_state_excess']-row['expected_total_excess']-row['fixed_minus_sampling_expectation'])<1e-15
    curve=root/'results/posterior_state_reuse/conditional_risk'
    for audit in json.loads((inp/'curve_audits.json').read_text()):assert sha(curve/audit['curve_file'])==audit['curve_sha256']
    fa=json.loads((inp/'figure_audit.json').read_text());assert sha(code/'plot_sampling_averaged_risk.py')==fa['source_sha256']
    assert sha(inp/'summary.json')==fa['summary_sha256'] and sha(inp/'sampling_averaged_risk.png')==fa['figure_sha256']
    documents=[];links=0
    for number in [173,174]:
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1;doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(file=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),source_hashes=len(p['source_sha256']),rows=len(rows),
        protocol_sha256=sha(inp/'protocol.json'),summary_sha256=sha(inp/'summary.json'),rows_sha256=sha(inp/'rows.json'),
        figure=fa,documents=documents,local_links=links,scope='numerical-reference risk attribution and integrity, not a new-method success')
    (root/'results/round_174_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
