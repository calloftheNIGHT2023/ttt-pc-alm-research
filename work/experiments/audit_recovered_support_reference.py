"""Exact separation replay and independent current-task coverage accounting."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import enumerate_support_modes as reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_support_reference/development';online=root/'results/recovered_online_comparison/development'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'coverage.json').read_text());assert len(rows)==len(p['seeds'])==run['tasks']
    checks=dict(reference_hashes=0,positive_regions=0,exact_rejection_certificates=0,coverage_rows=0,support_hashes=0)
    minimum=float('inf');stats=[]
    for row in rows:
        file=inp/row['reference_file'];assert sha(file)==row['reference_sha256'];ref=json.loads(file.read_text());checks['reference_hashes']+=1
        source=online/row['support_state_file'];assert sha(source)==row['support_state_sha256']
        with np.load(source) as z:assert z['x'].tolist()==ref['x'] and z['v'].tolist()==ref['v']
        checks['support_hashes']+=1;x=np.array(ref['x']);v=np.array(ref['v']);d=p['depth'];polys={}
        for r in ref['positive_regions']:
            assert r['pattern'] not in polys and r['volume']>0;polys[r['pattern']]=r
            b=np.array(r['representative']);assert np.max(abs(b))<=.12+1e-12
            assert reference.base.pattern(x,b).astype(np.uint8).tobytes().hex()==r['pattern']
            assert np.max(abs(reference.base.forward(x,b)-v))<=.001+1e-7;checks['positive_regions']+=1
        for cert in ref['certificates']:
            ids=np.array(cert['observation_ids']);reg=np.array(cert['pattern'],np.uint8)
            aa,rhs=reference.constraints(x[ids],v[ids],reg,True)
            for j in range(d):
                vec=[F(0)]*d;vec[j]=F(1);aa.extend([vec,[-q for q in vec]]);rhs.extend([F(.12),F(.12)])
            coeff=[F(0)]*d;constant=F(0)
            for index,value in cert['multipliers']:
                weight=F(value);assert weight>=0;constant-=weight*rhs[index]
                for j in range(d):coeff[j]+=weight*aa[index][j]
            lower=constant-F(.12)*sum(map(abs,coeff),F(0))
            assert lower>0 and lower==F(int(cert['lower_numerator']),int(cert['lower_denominator']))
            minimum=min(minimum,float(lower));checks['exact_rejection_certificates']+=1
        assert row['complete']==ref['numerical_volume_reference_complete']
        if not row['complete']:assert not row['coverage'];continue
        assert ref['enumeration_completed'] and not ref['final_geometry_unresolved'];total=sum(r['volume'] for r in polys.values())
        for r in row['coverage']:
            detail=online/r['detail_file'];assert sha(detail)==r['detail_sha256'];meta=json.loads(detail.read_text())
            found=set(meta['positive_mode_keys']);missing=set(polys)-found
            assert found<=set(polys) and sorted(found)==r['found_patterns'] and sorted(missing)==r['missing_patterns']
            mass=sum(polys[k]['volume'] for k in found)/total;assert abs(mass-r['mass_fraction'])<1e-12 and mass<=1+1e-12
            checks['coverage_rows']+=1
        print(json.dumps(dict(seed=row['seed'],**checks)),flush=True)
    for cfg in p['configs']:
        records=[next(c for c in r['coverage'] if c['method']==cfg['name']) for r in rows if r['complete']]
        full=len(records)==len(rows)
        stats.append(dict(method=cfg['name'],complete_reference_tasks=len(records),
            mean_mass_fraction=float(np.mean([r['mass_fraction'] for r in records])) if full else None,
            minimum_mass_fraction=min(r['mass_fraction'] for r in records) if full else None,
            all_regions_found_tasks=sum(not r['missing_patterns'] for r in records),
            missing_region_tasks=[rows[i]['seed'] for i,r in enumerate(records) if r['missing_patterns']] if full else None))
    result=dict(passed=True,checks=checks,complete_references=sum(r['complete'] for r in rows),
        minimum_exact_positive_margin=minimum,summaries=stats,source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(inp/name) for name in ['protocol.json','coverage.json','run_audit.json']},
        scope='Support-only numerical complete-volume coverage; exact rational LP exclusions, no conditional function-risk inference yet')
    out=inp.parent/'analysis';out.mkdir(parents=True,exist_ok=True)
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
