"""Independent certificate replay and first-write posterior-mass accounting."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import enumerate_support_modes as reference


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    protocol=json.loads((a.input/'protocol.json').read_text());rows=json.loads((a.input/'coverage.json').read_text());assert len(rows)==16
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h
    certificate_count=0;minimum_margin=np.inf;all_modes=0;lp=0
    for row in rows:
        path=a.input/row['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['reference_sha256']
        data=json.loads(path.read_text());x=np.array(data['x']);v=np.array(data['v']);d=protocol['depth'];lp+=data['lp_calls']
        assert data['enumeration_completed'] and not data['final_geometry_unresolved'];keys=set()
        for p in data['positive_regions']:
            assert p['pattern'] not in keys;keys.add(p['pattern']);b=np.array(p['representative'])
            assert reference.base.pattern(x,b).astype(np.uint8).tobytes().hex()==p['pattern']
            assert np.max(np.abs(reference.base.forward(x,b)-v))<=reference.base.EPS+1e-7
            assert p['volume']>0
        all_modes+=len(keys)
        for cert in data['certificates']:
            ids=np.array(cert['observation_ids']);regs=np.array(cert['pattern'],np.uint8)
            aa,rr=reference.constraints(x[ids],v[ids],regs,True)
            for j in range(d):
                rowa=[F(0)]*d;rowa[j]=F(1);aa.extend([rowa,[-z for z in rowa]]);rr.extend([F(.12),F(.12)])
            # Recompute the Lagrangian bound directly, not via exact_separation.
            coeff=[F(0)]*d;constant=F(0)
            for i,val in cert['multipliers']:
                weight=F(val);assert weight>=0;constant-=weight*rr[i]
                for j in range(d):coeff[j]+=weight*aa[i][j]
            lower=constant-F(.12)*sum((abs(z) for z in coeff),F(0))
            assert lower>0 and lower==F(int(cert['lower_numerator']),int(cert['lower_denominator']))
            minimum_margin=min(minimum_margin,float(lower));certificate_count+=1
    names=[c['name'] for c in protocol['configs']];values=np.array([[next(c for c in row['coverage'] if c['method']==name)['numerical_posterior_mass_fraction'] for name in names] for row in rows])
    assert np.all(values<=1+1e-8) and np.all(values>0);stats=[]
    for j,name in enumerate(names):
        val=values[:,j];counts=[next(c for c in row['coverage'] if c['method']==name)['positive_regions'] for row in rows]
        stats.append(dict(method=name,mean_mass_fraction=float(val.mean()),minimum_mass_fraction=float(val.min()),
            all_numerical_modes_found=sum(len(row['coverage'][j]['missing_patterns'])==0 for row in rows),
            mean_positive_regions=float(np.mean(counts)),mean_squared_missing_mass=float(np.mean((1-val)**2))))
    paired=[];rng=np.random.default_rng(61993);indices=rng.integers(0,16,(20000,16))
    for j,name in enumerate(names[1:],start=1):
        delta=values[:,0]-values[:,j]
        paired.append(dict(comparison=names[0]+' minus '+name,mean_mass_fraction_difference=float(delta.mean()),
            descriptive_ci95=np.quantile(delta[indices].mean(1),[.025,.975]).tolist(),
            greater_tasks=int((delta>1e-10).sum()),equal_tasks=int((np.abs(delta)<=1e-10).sum()),lower_tasks=int((delta<-1e-10).sum())))
    output=dict(audit=dict(source_hashes=len(protocol['source_sha256']),reference_file_hashes=16,
        exact_lp_rejection_certificates=certificate_count,minimum_exact_positive_margin=minimum_margin,
        positive_regions=all_modes,lp_calls=lp,all_enumerations_complete=True,unresolved_final_geometry=0),
        summary=stats,paired=paired,scope='Numerical complete-volume reference after exhaustive pattern search with exact LP exclusions; old development support only; no query advantage inferred from mass coverage.')
    a.out.mkdir(parents=True,exist_ok=True);(a.out/'summary.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    fig,ax=plt.subplots(figsize=(9,6));im=ax.imshow(values,vmin=0,vmax=1,cmap='YlGnBu',aspect='auto')
    ax.set_xticks(range(len(names)),['ALM20','Adam60','Adam240','Direct128','No dual20','PC80']);ax.set_yticks(range(16),[str(r['seed']) for r in rows])
    for i in range(16):
        for j in range(len(names)):ax.text(j,i,f'{values[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if values[i,j]>.65 else 'black')
    ax.set_xlabel('Same F256 first-write proposals and common region geometry');ax.set_ylabel('Old development task');ax.set_title('Fraction of enumerated posterior mass discovered at 4 observations')
    fig.colorbar(im,ax=ax,label='Discovered mass / complete numerical reference');fig.tight_layout();fig.savefig(a.out/'complete_mode_coverage.png',dpi=160);plt.close(fig)
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
