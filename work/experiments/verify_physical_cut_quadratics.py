"""Independent full-network rational objective check (not runtime code)."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import credit_residual_cut as cut


def full_objective(x,b,h,u,b0,h0):
    # Recompute all layers rather than using the candidate's local delta.
    prev=[F(float(t)) for t in x];energy=F(0);d,n=h.shape
    for j in range(d):
        for i in range(n):
            z=prev[i]+F(float(b[j]));prediction=max(F(0),1-abs(2*z-1))
            energy+=(F(float(h[j,i]))-prediction+F(float(u[j,i])))**2
        prev=[F(float(t)) for t in h[j]]
    energy+=F(.01)*(sum((F(float(v))-F(float(w)))**2 for v,w in zip(h.ravel(),h0.ravel()))
                   +n*sum((F(float(v))-F(float(w)))**2 for v,w in zip(b,b0)))
    return energy


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    rng=np.random.default_rng(782212);checks=0;neighbor_checks=0;cut_checks=0
    for d,n in [(2,3),(4,4)]:
        for _ in range(6):
            x=rng.uniform(0,1,n);b=rng.uniform(-.12,.12,d);h=rng.uniform(0,1,(d,n));u=rng.normal(0,.1,(d,n));reg=cut.old.split_many(x,b[None],h[None])[0]
            clauses=[dict(a=rng.normal(size=(d,n)).tolist(),positions=list(range(d*n)),codes=reg.ravel().tolist()) for _ in range(4)]
            bank,segments,stats=cut.candidate_bank(x,b,h,u,clauses)
            for row in bank:
                nb=b.copy();nh=h.copy()
                if row['kind']=='activity':nh[row['j'],row['i']]=float(row['value'])
                elif row['kind']=='bias':nb[row['j']]=float(row['value'])
                assert full_objective(x,nb,nh,u,b,h)==row['objective'];checks+=1
                allowed=all(cut.phi_exact(x,nb,nh,c['a'])<=0 for c in clauses);assert allowed==row['allowed'];cut_checks+=len(clauses)
                if row['kind']=='unchanged':continue
                # The constrained representative is at least as good as either
                # adjacent binary64 point in its own affine piece, when feasible.
                if not row['allowed']:continue
                seg=next(s for s in segments if all(s[k]==row[k] for k in ['kind','j','i','segment']))
                if not seg['feasible']:continue
                for direction in [-np.inf,np.inf]:
                    value=float(np.nextafter(float(row['value']),direction));t=F(value)
                    if not seg['cutlo']<=t<=seg['cuthi']:continue
                    cb=nb.copy();ch=nh.copy()
                    if row['kind']=='activity':ch[row['j'],row['i']]=value
                    else:cb[row['j']]=value
                    assert full_objective(x,cb,ch,u,b,h)>=row['objective'];neighbor_checks+=1
    result=dict(passed=True,independent_full_objective_checks=checks,exact_candidate_cut_checks=cut_checks,adjacent_feasible_binary64_optimum_checks=neighbor_checks,
                verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),runtime_source_sha256=hashlib.sha256(Path(cut.__file__).read_bytes()).hexdigest(),
                scope='independent algebra and binary64 neighbor checks; no task efficacy or performance claim')
    output=args.project/'results/physical_residual_cut/quadratic_verification.json';output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
