"""Independent rational face proof audit plus finite coverage display."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import enumerate_support_modes as reference


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/recovered_support_reference/exact_faces';orig=inp.parent/'development';online=root/'results/recovered_online_comparison/development'
    p=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'coverage.json').read_text());proofs=json.loads((inp/'proofs.json').read_text())
    summary=json.loads((inp/'summary.json').read_text());assert summary['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    checked=0;coverage=0;expected=set();proof_keys=set();lookup={r['seed']:r for r in rows};stats=[]
    for row in rows:
        path=orig/row['reference_file'];assert sha(path)==row['reference_sha256'];ref=json.loads(path.read_text())
        expected.update((row['seed'],r['pattern']) for r in ref['final_geometry_unresolved'])
        polys={r['pattern']:r for r in ref['positive_regions']};total=sum(r['volume'] for r in polys.values())
        assert row['complete'] and not row['remaining_unresolved']
        for item in row['coverage']:
            detail=online/item['detail_file'];assert sha(detail)==item['detail_sha256'];found=set(json.loads(detail.read_text())['positive_mode_keys'])
            assert found<=set(polys) and item['missing_patterns']==sorted(set(polys)-found)
            assert abs(item['mass_fraction']-sum(polys[k]['volume'] for k in found)/total)<1e-12;coverage+=1
    for proof in proofs:
        key=proof['seed'],proof['pattern'];assert key not in proof_keys;proof_keys.add(key)
        ref=json.loads((orig/lookup[proof['seed']]['reference_file']).read_text());x=np.array(ref['x']);v=np.array(ref['v'])
        reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);aa,rhs=reference.constraints(x,v,reg,True)
        for j in range(4):
            a=[F(0)]*4;a[j]=F(1);aa.extend([a,[-t for t in a]]);rhs.extend([F(.12),F(.12)])
        i=proof['first_row'];k=proof['second_row'];s=F(proof['positive_scale']);assert s>0
        assert any(aa[i]) and all(a+s*b==0 for a,b in zip(aa[i],aa[k]))
        gap=rhs[i]+s*rhs[k];assert gap==F(proof['rhs_sum']) and gap<=0
        assert proof['normal']==list(map(str,aa[i])) and proof['rhs']==str(rhs[i])
        assert proof['kind']==('exact_affine_hyperplane_zero_volume' if gap==0 else 'exact_opposite_infeasible');checked+=1
    assert expected==proof_keys and checked==summary['exact_face_proofs']==6 and len(rows)==summary['complete_references']==16
    for item in summary['summaries']:
        values=[next(c['mass_fraction'] for c in r['coverage'] if c['method']==item['method']) for r in rows]
        assert abs(np.mean(values)-item['mean_mass_fraction'])<1e-12
        truncation_bound=float(np.mean((1-np.array(values))**2))
        stats.append(dict(method=item['method'],mean_missing_mass_squared=truncation_bound,
            sampling_uniform_bound=1/(4*2048),expected_excess_bound_using_numerical_weights=truncation_bound+1/(4*2048)))
    names=['alm_c5','adam60_c5','pc_c5','nodual_c5','direct4096_c5'];labels=['ALM','Adam60','PC','No dual','Direct4096']
    values=np.array([[next(c['mass_fraction'] for c in r['coverage'] if c['method']==name) for name in names] for r in rows])
    fig,ax=plt.subplots(figsize=(8.2,6.1),layout='constrained');im=ax.imshow(values,vmin=.5,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(names)),labels);ax.set_yticks(range(len(rows)),[str(r['seed']) for r in rows])
    for i in range(len(rows)):
        for j in range(len(names)):ax.text(j,i,f'{100*values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>.85 else 'black')
    ax.set_title('Initial 4-support posterior-mass coverage (%)\nAll 16 current development tasks; numerical volumes')
    fig.colorbar(im,ax=ax,label='Covered posterior mass');file=inp/'coverage.png';fig.savefig(file,dpi=170);plt.close(fig)
    result=dict(passed=True,exact_opposing_row_proofs=checked,complete_references=len(rows),coverage_rows=coverage,
        finite_bounds=stats,source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),figure_sha256=sha(file),
        scope='Exact zero-volume classification; remaining weights numerical. Risk bounds are theoretical at true masses, numerical evaluations are not certified risk values')
    (inp/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
