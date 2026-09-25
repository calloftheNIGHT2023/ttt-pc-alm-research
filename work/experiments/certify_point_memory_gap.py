"""Analytic Bayes risk gap for single-point shifted-tent memory; not PC exclusivity."""
import argparse,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.integrate import quad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def excess(a,r,b):
    b=abs(b)
    if b<=r:return 4*r**3/(3*a)+4*(a-r)*b*b/a
    return 4*b*b-4*b**3/(3*a)-4*r*r*b/a+8*r**3/(3*a)


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    a=F(3,25);bayes=2*a*a/3;point=a*a;affine=bayes+a*a/105
    assert point-bayes==a*a/3 and affine-bayes==a*a/105
    rng=np.random.default_rng(486655);errors=[];violations=[]
    for _ in range(96):
        aa=float(a);r=rng.uniform(0,aa);b=rng.uniform(-aa,aa)
        fn=lambda d:4*(abs(d+b)-max(abs(d),r))**2
        numeric=quad(fn,-aa,aa,points=sorted({-r,0.,r,-b}),epsabs=1e-12,epsrel=1e-12)[0]/(2*aa)
        errors.append(abs(numeric-excess(aa,r,b)));violations.append(excess(aa,r,0)-excess(aa,r,b))
    assert max(errors)<1e-12 and max(violations)<=1e-12
    d=np.linspace(-float(a),float(a),2001);r=.06;v=1-2*r
    plus=1-2*np.abs(d+r);minus=1-2*np.abs(d-r);mean=(plus+minus)/2;point_prediction=1-2*np.abs(d)
    shallow=v-2*np.maximum(d-(1-v)/2,0)-2*np.maximum(-d-(1-v)/2,0)
    assert np.max(np.abs(mean-shallow))<1e-14
    result=dict(prior_a=str(a),bayes_risk=str(bayes),optimal_single_point_risk=str(point),optimal_affine_label_risk=str(affine),
                point_excess=str(point-bayes),affine_excess=str(affine-bayes),numeric_bayes=float(bayes),numeric_point=float(point),numeric_affine=float(affine),
                random_piecewise_integral_checks=len(errors),max_integral_error=max(errors),max_global_minimum_violation=max(violations),
                explicit_two_relu_context_decoder_max_error=float(np.max(np.abs(mean-shallow))),
                scope='noiseless one-layer illustrative subproblem; strict point/affine estimator class gaps, not impossibility for shallow context networks or PC-specific advantage')
    (args.out/'risk_gap.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained')
    axes[0].plot(d+.5,plus,'--',label='feasible b=+r',alpha=.65);axes[0].plot(d+.5,minus,'--',label='feasible b=-r',alpha=.65)
    axes[0].plot(d+.5,mean,lw=2.5,label='two-mode conditional mean');axes[0].plot(d+.5,point_prediction,':',lw=2,label='best single-point memory b=0')
    axes[0].set(xlabel='Independent query input q',ylabel='Prediction',title='Same observation, different internal explanations');axes[0].legend(fontsize=8);axes[0].grid(alpha=.2)
    values=[float(bayes),float(affine),float(point)];labels=['Bayes mean','Best affine in V','Best single b']
    axes[1].bar(labels,values,color=['#2a9d8f','#457b9d','#e76f51']);axes[1].set(ylabel='Exact population query MSE',title='Uniform b and query, a=0.12',ylim=(0,.016))
    for i,z in enumerate(values):axes[1].text(i,z+.0002,f'{z:.8f}',ha='center',fontsize=9)
    fig.savefig(args.out/'point_memory_gap.png',dpi=160);plt.close(fig);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
