"""Rational piecewise reference and correlated-fiber constructive witness."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import conditional_fiber_readout as fiber


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rational(query,point,direction,lo,hi):
    # Exact binary64 inputs, independent Fraction arithmetic and integration.
    states=[(F(lo),F(hi),F(0),F(query))]
    slopes=[0,2,-2,0];offsets=[0,0,2,0];knots=[F(0),F(1,2),F(1)]
    for bias,rate in zip(point,direction):
        following=[]
        for left,right,s,c in states:
            s+=F(float(rate));c+=F(float(bias));edges={left,right}
            if s:
                for knot in knots:
                    value=(knot-c)/s
                    if left<value<right:edges.add(value)
            edges=sorted(edges)
            for a,b in zip(edges[:-1],edges[1:]):
                code=sum(s*(a+b)/2+c>=k for k in knots)
                following.append((a,b,slopes[code]*s,slopes[code]*c+offsets[code]))
        states=following
    integral=sum(s*(b*b-a*a)/2+c*(b-a) for a,b,s,c in states)
    second=sum(s*s*(b**3-a**3)/3+s*c*(b*b-a*a)+c*c*(b-a) for a,b,s,c in states)
    return integral/(F(hi)-F(lo)),second/(F(hi)-F(lo)),len(states)


def verify():
    rng=np.random.default_rng(715263);checks=0;bound_checks=0;max_error=0.;piece_count=0
    for depth in [1,2,4,6]:
        for _ in range(12):
            point=rng.uniform(-.08,.08,depth);direction=rng.integers(-3,4,depth).astype(float)
            if not direction.any():direction[0]=1
            matrix=np.r_[np.eye(depth),-np.eye(depth)];rhs=np.full(2*depth,.12)
            lo,hi=fiber.fiber_interval(matrix,rhs,point,direction)
            for t in [lo,(lo+hi)/2,hi]:
                assert np.max(matrix@(point+t*direction)-rhs)<1e-14;bound_checks+=1
            for q in [.0,.125,.5,.75,1.]:
                a,b,n=fiber.line_moments(q,point,direction,lo,hi)
                ra,rb,rn=rational(q,point,direction,lo,hi)
                error=max(abs(a-float(ra)),abs(b-float(rb)))
                assert error<1e-11 and n==rn,(depth,error,n,rn)
                assert -1e-12<=a<=1+1e-12 and b>=a*a-1e-12
                # Reversing direction must leave conditional moments unchanged.
                aa,bb,nn=fiber.line_moments(q,point,-direction,-hi,-lo)
                assert max(abs(a-aa),abs(b-bb))<1e-11 and nn==n
                checks+=1;piece_count+=n;max_error=max(max_error,error)
    # Uniform parallelogram: U~Unif[-1/16,1/16], Z~Unif[0,e],
    # B=(U,Z-2U), q=3/4. This is a readout witness, not a task-family claim.
    e=F(1,64);a=F(1,16)
    matrix=np.array([[1,0],[-1,0],[2,1],[-2,-1]],dtype=float)
    rhs=np.array([float(a),float(a),float(e),0.]);point=np.zeros(2)
    lo,hi=fiber.fiber_interval(matrix,rhs,point,np.array([1.,-2.]))
    mean,second,_=fiber.line_moments(.75,point,[1.,-2.],lo,hi)
    assert abs(mean-.75)<1e-15 and abs(second-7/12)<1e-15
    original=F(1,48)+2*e**2/3-16*e**4/9
    joint_remaining=64*e**4/45
    joint_removed=F(1,48)+2*e**2/3-16*e**4/5
    coordinate_removed_ceiling=e**2/3
    assert original-joint_remaining==joint_removed>coordinate_removed_ceiling
    # Independently integrate the exact conditional formulas over Z.
    mean_exact=F(3,4)-4*e**2/3
    second_exact=F(7,12)-4*e**2/3
    assert second_exact-mean_exact**2==original
    for z in [F(0),e/3,e]:
        point=np.array([0.,float(z)]);d=np.array([1.,-2.]);lo,hi=fiber.fiber_interval(matrix,rhs,point,d)
        mu,sec,_=fiber.line_moments(.75,point,d,lo,hi)
        assert abs(mu-float(F(3,4)-4*z*z))<1e-14
        assert abs(sec-float(F(7,12)-4*z*z))<1e-14
    return dict(passed=True,rational_moment_checks=checks,fiber_feasibility_checks=bound_checks,
                propagated_pieces=piece_count,maximum_absolute_moment_error=max_error,
                witness=dict(original_variance=str(original),joint_remaining_variance=str(joint_remaining),
                             joint_removed_variance=str(joint_removed),coordinate_removed_ceiling=str(coordinate_removed_ceiling),
                             joint_removed_fraction=float(joint_removed/original),coordinate_max_removed_fraction=float(coordinate_removed_ceiling/original)),
                scope='exact mathematical witness and floating primitive tests; no new online quality, PC uniqueness or same-time superiority')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'protocol.json').exists()
    protocol=dict(source_sha256={Path(__file__).name:sha(Path(__file__)),Path(fiber.__file__).name:sha(Path(fiber.__file__))},
                  rng_seed=715263,numpy_version=np.__version__,depths=[1,2,4,6],cases_per_depth=12,queries=[0,.125,.5,.75,1],
                  scope='mathematical primitive, not development or held-out task experiment')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    result=verify();(args.out/'result.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
