"""Independent strict rejection, zero-volume and positive-region accounting."""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
import enumerate_support_modes as reference
import shared_mode_readout as shared
from audit_shared_mode_readout import check_cube
from analyze_recovered_online_comparison import forward
from run_multiplier_fixed_point_screen import sha,dump


def constraints(x,v,pattern):
    a,r=reference.constraints(x,v,pattern,True)
    for j in range(4):
        row=[F(0)]*4;row[j]=F(1);a.extend([row,[-t for t in row]]);r.extend([F(.12),F(.12)])
    return a,r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/confirmation_conditional_risk';inp=base/'reference';boundary=base/'boundary_certificates';out=base/'reference_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'summary.json').read_text());bd=json.loads((boundary/'summary.json').read_text());assert run['passed'] and bd['passed'] and bd['unresolved']==0
    hashes=dict(json.loads((boundary/'protocol.json').read_text())['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    assert sha(inp/'coverage.json')==run['coverage_sha256'] and sha(inp/'files.json')==run['files_sha256'];assert sha(boundary/'certificates.json')==bd['certificates_sha256']
    files=json.loads((inp/'files.json').read_text());tasks=json.loads((inp/'coverage.json').read_text());zeros={(r['seed'],r['pattern']):r for r in json.loads((boundary/'certificates.json').read_text())};assert len(zeros)==bd['cells']
    old=root/'results/matched_budget_confirmation/conditioned_confirmation';before=json.loads((old/'before_query_manifest.json').read_text());assert sha(old/'before_query_manifest.json')==p['frozen_predictor_manifest_sha256'] and sha(old/'rows.json')==before['rows_sha256']
    methods={(r['seed'],r['method']):r for r in json.loads((old/'rows.json').read_text())}
    dump(out/'protocol.json',dict(source_sha256=hashes,reference_summary_sha256=sha(inp/'summary.json'),boundary_summary_sha256=sha(boundary/'summary.json'),phase_accesses_query_targets=False,
        scope='all64 enumerations; strict rational rejection, rational measure-zero hyperplanes, scalar exact interior cubes and all35 saved pools; numerical volumes remain numerical'))
    counts=Counter();coverage=[];minimum=float('inf');begin=time.perf_counter();seen_zero=set()
    for task in tasks:
        path=inp/task['file'];assert sha(path)==task['sha256']==files[task['file']];data=json.loads(path.read_text());ref=data['reference'];x=np.array(data['x_observed']);v=np.array(data['v_observed']);assert ref['enumeration_completed'] and not task['failure'];counts['complete_enumerations']+=1
        original=methods[task['seed'],p['primary']];support=old/original['file'];assert sha(support)==original['sha256']==before['prediction_files'][original['file']]
        with np.load(support) as z:assert x.tobytes()==z['x_observed'].tobytes() and v.tobytes()==z['v_observed'].tobytes()
        for cert in ref['certificates']:
            ids=np.array(cert['observation_ids']);a,r=constraints(x[ids],v[ids],np.array(cert['pattern'],np.uint8));coeff=[F(0)]*4;constant=F(0)
            for i,w0 in cert['multipliers']:
                w=F(w0);assert w>=0;constant-=w*r[i]
                for j in range(4):coeff[j]+=w*a[i][j]
            lower=constant-F(.12)*sum(map(abs,coeff));assert lower>0 and lower==F(int(cert['lower_numerator']),int(cert['lower_denominator']));minimum=min(minimum,float(lower));counts['exact_strict_rejections']+=1
        for unknown in ref['final_geometry_unresolved']:
            key=(task['seed'],unknown['pattern']);cert=zeros[key];seen_zero.add(key);pattern=np.frombuffer(bytes.fromhex(key[1]),dtype=np.uint8).reshape(4,4);a,r=constraints(x,v,pattern);coeff=[F(0)]*4;value=F(0);active=[]
            for term in cert['multipliers']:
                i=term['row'];w=F(int(term['numerator']),int(term['denominator']));assert w>0;value+=w*r[i]
                for j in range(4):coeff[j]+=w*a[i][j]
                if any(a[i]):active.append(i)
            assert not any(coeff) and value==F(int(cert['weighted_rhs']['numerator']),int(cert['weighted_rhs']['denominator']))
            if cert['kind']=='zero_volume':assert value==0 and cert['nonconstant_equality_row'] in active;counts['exact_zero_volume_cells']+=1
            else:assert cert['kind']=='infeasible' and value<0;counts['additional_exact_infeasible_cells']+=1
        polys={r['pattern']:r for r in ref['positive_regions']};assert len(polys)==len(ref['positive_regions']) and set(polys)==set(data['geometry']);total=0.
        for key,region in polys.items():
            item=data['geometry'][key];path=inp/item['file'];assert sha(path)==item['sha256']==files[item['file']];assert region['volume']>0;total+=region['volume']
            cube=check_cube(x,v,key,item['proof']);counts['exact_interior_cube_corners']+=cube['corners']
            with np.load(path) as z:
                point=z['center']+z['scale']*z['interior'];assert shared.geometry.base.pattern(x,point).astype(np.uint8).tobytes().hex()==key
                assert np.max(abs(forward(x,point[None])[0]-v))<=.001+1e-7
                numerical=np.abs(np.linalg.det(z['facets']-z['interior'])).sum()/24*np.prod(z['scale']);assert np.isclose(numerical,region['volume'],rtol=1e-8,atol=1e-22) and float(z['volume'])==region['volume']
                assert abs(float(z['simplex_probs'].sum())-1)<1e-12
            counts['positive_volume_regions']+=1
        rr=[]
        for name in p['methods']:
            row=methods[task['seed'],name];assert not row['metadata']['execution_failed'];found=set(row['metadata']['positive_modes']);assert found<=set(polys),(task['seed'],name);mass=sum(polys[k]['volume'] for k in sorted(found))/total
            assert 0<=mass<=1+1e-12;rr.append(dict(method=name,found_modes=sorted(found),missing_modes=sorted(set(polys)-found),numerical_posterior_mass_fraction=mass));counts['frozen_pool_coverage_rows']+=1
        coverage.append(dict(seed=task['seed'],complete_up_to_certified_zero_volume=True,positive_regions=len(polys),total_numerical_volume=total,coverage=rr))
        if len(coverage)%4==0:print(json.dumps(dict(audited_tasks=len(coverage),total=64,seconds=time.perf_counter()-begin)),flush=True)
    assert seen_zero==set(zeros) and counts['complete_enumerations']==64 and counts['exact_strict_rejections']==run['certified_rejections'] and counts['positive_volume_regions']==run['positive_regions']
    stats=[]
    for name in p['methods']:
        rr=[next(r for r in t['coverage'] if r['method']==name) for t in coverage];mass=np.array([r['numerical_posterior_mass_fraction'] for r in rr]);stats.append(dict(method=name,mean_mass_fraction=float(mass.mean()),minimum_mass_fraction=float(mass.min()),all_positive_modes_found=sum(not r['missing_modes'] for r in rr),mean_squared_missing_mass=float(np.mean((1-mass)**2))))
    dump(out/'coverage.json',coverage);dump(out/'mass_summary.json',stats);ans=dict(passed=True,counts=counts,complete_up_to_certified_zero_volume=64,minimum_exact_rejection_margin=minimum,seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'),coverage_sha256=sha(out/'coverage.json'),mass_summary_sha256=sha(out/'mass_summary.json'),phase_accesses_query_targets=False,
        scope='full positive-volume numerical posterior reference after exact exclusion of boundary cells; not exact-real volume integration; no query improvement inferred from mass alone')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
