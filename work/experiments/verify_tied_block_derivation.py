"""Pure rational algebra tests for 156; not an adaptation implementation."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import credit_residual_cut as scalar


def residual(x,b,h):
    prev=x;out=[]
    for j in range(len(b)):
        out.append([h[j][i]-max(F(0),1-abs(2*(prev[i]+b[j])-1)) for i in range(len(x))]);prev=h[j]
    return out


def full_energy(x,b,h,u,b0,h0):
    r=residual(x,b,h);n=len(x)
    return sum((v+w)**2 for rr,uu in zip(r,u) for v,w in zip(rr,uu))+scalar.TAU*(sum((v-w)**2 for row,old in zip(h,h0) for v,w in zip(row,old))+n*sum((v-w)**2 for v,w in zip(b,b0)))


def dot(a,r):return sum(v*w for row,rr in zip(a,r) for v,w in zip(row,rr))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();rng=np.random.default_rng(734491)
    cases=0;blocks=0;segments=0;points=0;objectives=0;cut_checks=0;rounded_affine_disagreements=0;max_disagreement=0.
    for d,n in [(2,2),(4,4),(4,8)]:
        for _ in range(4):
            x=[F(float(v)) for v in rng.uniform(0,1,n)];b=[F(float(v)) for v in rng.uniform(-.12,.12,d)]
            h=scalar.frac_array(rng.uniform(0,1,(d,n)));u=scalar.frac_array(rng.normal(0,.1,(d,n)))
            directions=[scalar.frac_array(rng.normal(size=(d,n))) for _ in range(3)];r0=residual(x,b,h);e0=full_energy(x,b,h,u,b,h);cases+=1
            for j in range(d-1):
                prev=x if j==0 else h[j-1];neighbor=b[j+1];knots=[F(0),F(1,2),F(1)];ev=[-scalar.B,scalar.B]
                for p in prev:
                    ev.extend(k-p for k in knots)
                    for k in knots:
                        y=k-neighbor
                        if 0<=y<=1:ev.extend([y/2-p,1-y/2-p])
                ev=sorted(set(v for v in ev if -scalar.B<=v<=scalar.B));assert len(ev)-1<=9*n+1;blocks+=1
                old_local=sum((r0[j][i]+u[j][i])**2+(r0[j+1][i]+u[j+1][i])**2 for i in range(n))
                for lo,hi in zip(ev[:-1],ev[1:]):
                    middle=(lo+hi)/2
                    first=[sum(p+middle>=k for k in knots) for p in prev]
                    p1=[F(scalar.S[k]) for k in first];q1=[p1[i]*prev[i]+scalar.C[first[i]] for i in range(n)]
                    second=[sum(p1[i]*middle+q1[i]+neighbor>=k for k in knots) for i in range(n)]
                    m=[scalar.S[second[i]]*p1[i] for i in range(n)]
                    offset=[scalar.S[second[i]]*(q1[i]+neighbor)+scalar.C[second[i]] for i in range(n)]
                    target=[h[j+1][i]+u[j+1][i] for i in range(n)]
                    A=sum(v*v for v in m)+scalar.TAU*(sum(v*v for v in p1)+n)
                    B=sum(m[i]*(target[i]-offset[i]) for i in range(n))+scalar.TAU*sum(p1[i]*(h[j][i]-q1[i]) for i in range(n))+scalar.TAU*n*b[j]
                    constant=e0-old_local+sum(u[j][i]**2+(target[i]-offset[i])**2 for i in range(n))+scalar.TAU*sum((q1[i]-h[j][i])**2 for i in range(n))+scalar.TAU*n*b[j]**2
                    alpha=[-sum(a[j+1][i]*m[i] for i in range(n)) for a in directions]
                    beta=[dot(a,r0)-sum(a[j][i]*r0[j][i]+a[j+1][i]*r0[j+1][i] for i in range(n))+sum(a[j+1][i]*(h[j+1][i]-offset[i]) for i in range(n)) for a in directions]
                    assert A>0;domain=scalar.solve_interval(lo,hi,alpha,beta)
                    test=[lo,middle,hi,min(max(B/A,lo),hi)]
                    if domain is not None:test.extend([domain[0],domain[1],min(max(B/A,domain[0]),domain[1])])
                    for t in test:
                        nb=b.copy();nh=[row.copy() for row in h];nb[j]=t;nh[j]=[scalar.g(p+t) for p in prev]
                        assert all(nh[j][i]==p1[i]*t+q1[i] for i in range(n))
                        assert all(scalar.g(nh[j][i]+neighbor)==m[i]*t+offset[i] for i in range(n))
                        assert all(v==0 for v in residual(x,nb,nh)[j]);points+=1
                        assert full_energy(x,nb,nh,u,b,h)==A*t*t-2*B*t+constant;objectives+=1
                        rr=residual(x,nb,nh)
                        for a,c,aa in zip(alpha,beta,directions):assert dot(aa,rr)==a*t+c;cut_checks+=1
                    # Round a derived point as an actual binary64 state would:
                    # the ideal affine formula need not remain exact afterwards.
                    t=F(float(middle));nb=b.copy();nh=[row.copy() for row in h];nb[j]=t;nh[j]=[F(float(scalar.g(p+t))) for p in prev]
                    rr=residual(x,nb,nh)
                    for a,c,aa in zip(alpha,beta,directions):
                        delta=dot(aa,rr)-(a*t+c)
                        rounded_affine_disagreements+=int(delta!=0);max_disagreement=max(max_disagreement,abs(float(delta)))
                    segments+=1
    result=dict(passed=True,cases=cases,blocks=blocks,segments=segments,exact_points=points,full_objective_checks=objectives,
                exact_cut_coefficient_checks=cut_checks,rounded_affine_disagreements=rounded_affine_disagreements,max_rounding_cut_difference=max_disagreement,
                segment_bound_9n_plus_1=True,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),scalar_source_sha256=hashlib.sha256(Path(scalar.__file__).read_bytes()).hexdigest(),
                scope='rational formula and segment coverage tests, not candidate generation or online adaptation; rounded states must be re-certified')
    out=args.project/'results/tied_local_block';out.mkdir(parents=True,exist_ok=True);assert not (out/'derivation_verification.json').exists()
    (out/'derivation_verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
