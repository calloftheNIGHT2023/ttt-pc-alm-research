"""Analytic minimal example: best affine closed-form risk vs conditional mean.

This is a one-layer subfamily mechanism proof, not the risk of the four-layer
confirmation task and not a PC-only lower bound.
"""
import argparse,json
from pathlib import Path
from scipy.integrate import quad


def risk(a=.12,d=.06,eps=0.):
    ey=1-a-d*d/a
    vary=a*a/3+2*d*d-d**4/(a*a)
    cov=a*a/3-d*d+2*d**3/(3*a)
    varv=(a*a+eps*eps)/3
    affine=vary-cov*cov/varv
    if eps==0:
        bayes=4*d*d-8*d**3/(3*a)
        predicted_gap=4*a*a*(d/a)**3*(1-d/a)**3/3
        assert abs(affine-bayes-predicted_gap)<1e-14
    else:
        def integrand(v):
            lo=max(0.,(1-v-eps)/2); hi=min(a,(1-v+eps)/2)
            if hi<=lo: return 0.
            if hi<=d: expected_max=d
            elif lo>=d: expected_max=(lo+hi)/2
            else: expected_max=(d*(d-lo)+(hi*hi-d*d)/2)/(hi-lo)
            conditional_mean=1-2*expected_max
            density=(hi-lo)/(2*a*eps)
            return density*(conditional_mean-ey)**2
        knots=sorted({1-2*a-eps,1-2*a+eps,1-2*d-eps,1-2*d+eps,1-eps,1+eps})
        explained=sum(quad(integrand,l,h,epsabs=1e-13,epsrel=1e-11)[0] for l,h in zip(knots[:-1],knots[1:]))
        bayes=vary-explained
    assert affine>bayes>0
    return {"a":a,"d":d,"epsilon":eps,"best_affine_population_risk":affine,"conditional_mean_population_risk":bayes,
        "strict_excess_risk":affine-bayes,"scope":"one-layer analytic subfamily; not PC-specific; no four-layer quantitative transfer"}


if __name__=="__main__":
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path); args=p.parse_args()
    result={"passed":True,"cases":[risk(),risk(eps=.001)]}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))
