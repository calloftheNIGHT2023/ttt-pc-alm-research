"""Integrity and small independent identities for research rounds 177--186."""
import argparse,hashlib,json,re
from pathlib import Path
import numpy as np
import audit_credit_direction_geometry as spectral
import hybrid_rejection_materialization as hybrid


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve();code=root/'work/experiments';docs=root/'outputs/ttt-pc-alm-research'
    manifests={};protocols={};checks=0
    paths=['credit_fiber/diagnostic','credit_fiber/geometry_v2','rejection_credit/diagnostic','rejection_materialization/development','hybrid_materialization/development']
    for path in paths:
        base=root/'results'/path;p=json.loads((base/'protocol.json').read_text());protocols[path]=p
        for name,value in p['source_sha256'].items():
            assert sha(code/name)==value,(path,name);checks+=1
            if name in manifests:assert manifests[name]==value
            manifests[name]=value
    geom=protocols['credit_fiber/geometry_v2'];assert sha(root/'results/credit_fiber/geometry/protocol.json')==geom['failed_protocol_sha256']
    assert (root/'results/credit_fiber/geometry/FAILURE.md').exists() and (root/'results/rejection_credit/preflight_failure.md').exists()
    assert sha(root/'results/credit_fiber/analysis/region_geometry.json')==geom['geometry_rows_sha256']
    assert sha(root/'results/credit_fiber/diagnostic/audits.json')==geom['input_audits_sha256']
    assert spectral.verify()==geom['verification']
    analyses=[('credit_fiber','credit_fiber_results.png','analyze_credit_fiber.py','diagnostic'),
              ('rejection_credit','rejection_credit.png','analyze_rejection_credit.py','diagnostic'),
              ('rejection_materialization','rejection_materialization.png','analyze_rejection_materialization.py','development'),
              ('hybrid_materialization','hybrid_materialization.png','analyze_hybrid_materialization.py','development')]
    for folder,figure,script,run in analyses:
        base=root/'results'/folder;summary=json.loads((base/'analysis/summary.json').read_text());assert summary['complete']
        assert sha(code/script)==summary['analysis_source_sha256']
        for name,value in summary['input_sha256'].items():assert sha(base/run/name)==value,(folder,name)
        fa=json.loads((base/'analysis/figure_audit.json').read_text());assert sha(code/script)==fa['source_sha256']
        assert sha(base/'analysis/summary.json')==fa['summary_sha256'] and sha(base/'analysis'/figure)==fa['figure_sha256']
    resources=0
    for folder,script in [('rejection_materialization','audit_rejection_materialization_resources.py'),('hybrid_materialization','audit_hybrid_materialization_resources.py')]:
        for method in hybrid.METHODS:
            record=json.loads((root/f'results/{folder}/resources/{method}.json').read_text());assert record['passed'] and record['state_and_prediction_bitwise']
            assert sha(code/script)==record['source_sha256'] and sha(root/f'results/{folder}/development/protocol.json')==record['protocol_sha256'];resources+=1
    sources=protocols['hybrid_materialization/development'];assert hybrid.verify_distribution()==sources['verification']['hybrid_distribution']
    for folder in ['rejection_materialization','hybrid_materialization']:
        rows=json.loads((root/f'results/{folder}/development/rows.json').read_text());assert len(rows)==320
        for row in rows:assert sha(root/f'results/{folder}/development'/row['state_file'])==row['state_sha256']
    links=0;documents=[]
    for number in range(177,187):
        matches=list(docs.glob(f'{number}_*.md'));assert len(matches)==1,(number,matches);doc=matches[0]
        for target in re.findall(r'\]\(([^)]+)\)',doc.read_text(encoding='utf-8')):
            if target.startswith(('http:','https:','#')):continue
            assert (doc.parent/target.split('#')[0].strip('<>')).exists(),(doc.name,target);links+=1
        documents.append(dict(name=doc.name,sha256=sha(doc)))
    result=dict(passed=True,source_sha256=sha(Path(__file__)),aggregate_frozen_source_checks=checks,unique_frozen_sources=len(manifests),
        documents=documents,local_links=links,figures=4,fresh_process_resources=resources,materialization_state_files=640,
        exact_hybrid_distribution=hybrid.verify_distribution(),spectral_primitive=spectral.verify(),
        scope='integrity and independent identities, not certification of research novelty or a complete task-level victory')
    (root/'results/round_186_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
