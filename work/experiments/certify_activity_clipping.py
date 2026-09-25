"""Exact rational witness: clipping a global unbounded block can increase energy."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists()
    weights=[[F(3,5),F(-4,5)],[F(4,5),F(3,5)]];bias=[F(1,5),F(0)]
    star=[F(3,2),F(1)];old=[F(1),F(87,119)];clipped=[F(1),F(1)];trust=F(1,100)
    center=[s+trust*(s-o) for s,o in zip(star,old)]
    def forward(h):
        z=[sum(w*t for w,t in zip(row,h))+b for row,b in zip(weights,bias)]
        return [max(F(-1),1-2*abs(t)) for t in z]
    target=forward(star)
    def energy(h):
        return sum((a-c)**2+trust*(a-o)**2 for a,c,o in zip(h,center,old))+sum((a-t)**2 for a,t in zip(forward(h),target))
    # Completing squares proves star is a global unbounded minimizer:
    # Q(h)-Q(star)=(1+trust)||h-star||^2+||g(Wh+b)-g(Wstar+b)||^2.
    identity_constant=trust*(1+trust)*sum((s-o)**2 for s,o in zip(star,old))
    assert energy(star)==identity_constant
    assert all(-1<=t<=1 for t in old+clipped)
    assert energy(clipped)-energy(old)==F(768,2975)>0
    result=dict(passed=True,weights=[[str(t) for t in row] for row in weights],bias=list(map(str,bias)),
                unbounded_global_minimizer=list(map(str,star)),feasible_old=list(map(str,old)),clipped=list(map(str,clipped)),
                center=list(map(str,center)),target=list(map(str,target)),trust=str(trust),
                old_energy=str(energy(old)),clipped_energy=str(energy(clipped)),
                exact_energy_increase=str(energy(clipped)-energy(old)),numeric_energy_increase=float(energy(clipped)-energy(old)),
                scope='conditional activity-block witness; not task-risk advantage or publication novelty',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
