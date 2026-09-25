"""Exact constructive witness for tied repair, not PC-specific necessity."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import tied_local_block_repair as tied


def f(q,b):
    for shift in b:q=max(F(0),1-abs(2*(q+shift)-1))
    return q


def knots(b):
    first,second=b;out=[F(0),F(1)]+[k-first for k in [F(0),F(1,2),F(1)]]
    for k in [F(0),F(1,2),F(1)]:
        y=k-second
        if 0<=y<=1:out.extend([y/2-first,1-y/2-first])
    return sorted(set(x for x in out if 0<=x<=1))


def moments_and_risk(truth,predictor=None,affine=None):
    mesh=sorted(set(knots(truth)+(knots(predictor) if predictor is not None else [])));m0=F(0);m1=F(0);m2=F(0);risk=F(0)
    for lo,hi in zip(mesh[:-1],mesh[1:]):
        fl=f(lo,truth);fh=f(hi,truth);s=(fh-fl)/(hi-lo);c=fl-s*lo
        assert f((lo+hi)/2,truth)==s*(lo+hi)/2+c
        i0=hi-lo;i1=(hi**2-lo**2)/2;i2=(hi**3-lo**3)/3
        m0+=s*i1+c*i0;m1+=s*i2+c*i1;m2+=s*s*i2+2*s*c*i1+c*c*i0
        if predictor is not None:
            pl=f(lo,predictor);ph=f(hi,predictor);ps=(ph-pl)/(hi-lo);pc=pl-ps*lo
            assert f((lo+hi)/2,predictor)==ps*(lo+hi)/2+pc
        else:pc,ps=affine if affine is not None else (F(0),F(0))
        ds=s-ps;dc=c-pc;risk+=ds*ds*i2+2*ds*dc*i1+dc*dc*i0
    return m0,m1,m2,risk


def rationals(value):return dict(numerator=str(value.numerator),denominator=str(value.denominator),float=float(value))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    x=np.array([5/32,9/32]);b=np.array([1/16,1/16]);truth=[F(0),F(1,16)];v=np.array([3/4,3/4]);h=np.array([[1.,1.],[3/4,3/4]]);u=np.zeros_like(h)
    reg=tied.scalar.old.split_many(x,b[None],h[None])[0];clauses=[]
    for i in range(2):
        a=np.zeros_like(h);a[-1,i]=1.;p=np.zeros_like(a);clause=tied.scalar.old.conflict.extract(x,v,reg,p,a)
        assert clause is not None and clause['cardinality']==1;clauses.append(clause)
    pool,_,stats=tied.candidate_bank(x,b,h,u,clauses)
    sb,sh,sm=tied.select(x,b,h,clauses,pool,stats,True,False)
    tb,th,tm=tied.select(x,b,h,clauses,pool,stats,True,True)
    eb,eh,em=tied.select(x,b,h,clauses,pool,stats,False,True)
    assert not sm['accepted'] and sm['allowed_candidates']==0
    assert tm['accepted'] and tm['selected_kind']=='tied' and np.array_equal(tb,np.array([0.,1/16]))
    assert np.array_equal(th,np.array([[5/16,9/16],[3/4,3/4]]))
    assert all(tied.scalar.phi_exact(x,tb,th,c['a'])==0 for c in clauses)
    assert all(f(F(float(xx)),truth)==F(float(yy)) for xx,yy in zip(x,v))
    m0,m1,m2,_=moments_and_risk(truth);intercept=4*m0-6*m1;slope=-6*m0+12*m1;oracle_affine=m2-intercept*m0-slope*m1
    before=moments_and_risk(truth,[F(float(t)) for t in b])[-1];after=moments_and_risk(truth,[F(float(t)) for t in tb])[-1]
    energy_risk=moments_and_risk(truth,[F(float(t)) for t in eb])[-1];closed_head=moments_and_risk(truth,affine=(F(3,4),F(0)))[-1]
    assert before>0 and after==0 and oracle_affine>0 and closed_head>=oracle_affine
    # A simpler internal parametric inverse also recovers t=0. This must be
    # disclosed; the witness is not a unique-PC superiority example.
    internal_inverse=(F(3,4)/2-F(1,16))/2-F(5,32);assert internal_inverse==0
    result=dict(passed=True,x=x.tolist(),target=v.tolist(),before_b=b.tolist(),before_h=h.tolist(),proofs=clauses,
                scalar=sm,tied=tm,energy=em,tied_b=tb.tolist(),energy_b=eb.tolist(),tied_h=th.tolist(),
                exact_uniform_risks=dict(before=rationals(before),tied=rationals(after),same_pool_energy=rationals(energy_risk),
                                         fitted_affine_head=rationals(closed_head),oracle_best_affine=rationals(oracle_affine)),
                oracle_affine_coefficients=[rationals(intercept),rationals(slope)],simple_internal_inverse=rationals(internal_inverse),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_sha256=hashlib.sha256(Path(tied.__file__).read_bytes()).hexdigest(),
                scope='explicit one-step local-repair witness; simple parametric inverse also solves it; easy C20-detectable conflicts, not online bank admission or unique PC-ALM contribution')
    out=args.project/'results/tied_local_block';out.mkdir(parents=True,exist_ok=True);assert not (out/'constructive_witness.json').exists()
    (out/'constructive_witness.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
