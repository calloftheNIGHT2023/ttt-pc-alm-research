import argparse,hashlib,itertools,json,time
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
import exact_shared_bias_block as block
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def brute_branches(c,t,old,trust,bound):
    # Independent convex QP for every assignment; only used for n<=3.
    best=np.inf;n=len(c)
    for branch in itertools.product(range(4),repeat=n):
        s=np.array(block.SLOPES)[list(branch)];k=np.array(block.INTERCEPTS)[list(branch)]
        def objective(vector):
            b=vector[0];z=vector[1:];r=z-c-b;e=s*z+k-t;f=z-old
            return float(np.sum(r*r+e*e+trust*f*f)),np.r_[-2*r.sum(),2*r+2*s*e+2*trust*f]
        bounds=[(-bound,bound)]+[(block.LO[j],block.HI[j]) for j in branch]
        initial=np.r_[0,np.clip(c,np.array(block.LO)[list(branch)],np.array(block.HI)[list(branch)])]
        result=minimize(objective,initial,jac=True,bounds=bounds,method='L-BFGS-B',options={'ftol':1e-14,'gtol':1e-11,'maxiter':500})
        low=np.array([q[0] for q in bounds]);high=np.array([q[1] for q in bounds])
        projected=result.x-np.clip(result.x-result.jac,low,high)
        if not result.success and np.linalg.norm(projected)>=1e-7:
            result=minimize(objective,initial,jac=True,bounds=bounds,method='SLSQP',options={'ftol':1e-13,'maxiter':500})
            projected=result.x-np.clip(result.x-result.jac,low,high)
        assert np.linalg.norm(projected)<2e-6,(result.message,float(np.linalg.norm(projected)))
        best=min(best,float(result.fun))
    return best


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    assert not (args.out/'audit.json').exists()
    rng=np.random.default_rng(662509);cases=[];worst=0.;brute_gaps=[];times=[]
    for index in range(80):
        n=[1,2,3,8,24][index%5];c=rng.uniform(-1.8,1.8,n);t=rng.uniform(-2,2,n);old=rng.uniform(-1.8,1.8,n)
        trust=.01 if index%2 else .2;bound=.3
        before=time.perf_counter();b,z,meta=block.solve(c,t,old,trust,bound);times.append(time.perf_counter()-before)
        # Dense grid is an upper bound on optimum, not an exact optimality proof.
        grid=np.linspace(-bound,bound,20001);gridbest=np.inf
        for values in np.array_split(grid,20):
            s=np.array(block.SLOPES)[:,None,None];k=np.array(block.INTERCEPTS)[:,None,None]
            zz=np.clip((c[None,:,None]+values[None,None,:]+s*(t[None,:,None]-k)+trust*old[None,:,None])/(1+s*s+trust),np.array(block.LO)[:,None,None],np.array(block.HI)[:,None,None])
            ee=(zz-c[None,:,None]-values[None,None,:])**2+(block.activation(zz)-t[None,:,None])**2+trust*(zz-old[None,:,None])**2
            gridbest=min(gridbest,float(ee.min(axis=0).sum(axis=0).min()))
        gap=meta['energy']-gridbest;worst=max(worst,gap);assert gap<1e-8
        if index<15 and n<=3:
            brute=brute_branches(c,t,old,trust,bound);brute_gaps.append(meta['energy']-brute);assert abs(meta['energy']-brute)<1e-7
        # Same z-reference held fixed in the coordinate descent comparator.
        coord_b=0.;coord_z=old.copy()
        for _ in range(100):
            coord_z=block.prox(c+coord_b,t,old,trust);coord_b=float(np.clip(np.mean(coord_z-c),-bound,bound))
        coord_energy=block.energy(coord_b,coord_z,c,t,old,trust)
        assert meta['energy']<=coord_energy+1e-8
        cases.append({'n':n,'energy':meta['energy'],'coordinate100_energy':coord_energy,'seconds':times[-1],**meta})
    # Strict stationary coordinate example, exact rational energy comparison.
    c=np.array([-1.1]);target=np.array([-.95]);old=c.copy()
    bias,z,meta=block.solve(c,target,old)
    coordinate_z=block.prox(c,target,old,.01);assert np.allclose(coordinate_z,old,atol=1e-14)
    coordinate_bias=float(np.clip(np.mean(coordinate_z-c),-.3,.3));assert abs(coordinate_bias)<1e-14
    z_exact=Fraction(-3911,4010);b_exact=z_exact+Fraction(11,10)
    e_exact=(1+2*z_exact+Fraction(19,20))**2+Fraction(1,100)*(z_exact+Fraction(11,10))**2
    e_old=Fraction(1,400)
    assert e_exact<e_old and abs(float(z_exact)-z[0])<1e-12
    witness={'c':-1.1,'target':-.95,'old_z':-1.1,'old_bias':0.,'trust':.01,'bias_bound':.3,
        'new_bias':bias,'new_z':float(z[0]),'old_energy':float(e_old),'new_energy':meta['energy'],
        'exact_z':str(z_exact),'exact_bias':str(b_exact),'exact_new_energy':str(e_exact),'exact_old_energy':str(e_old)}
    output={'passed':True,'random_cases':len(cases),'max_grid_excess':worst,'independent_branch_qp_cases':len(brute_gaps),
        'max_abs_branch_qp_gap':max(abs(a) for a in brute_gaps),'median_block_seconds':float(np.median(times)),
        'source_sha256':{Path(p).name:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in [__file__,block.__file__]},
        'strict_coordinate_stall':witness,'cases':cases,
        'scope':'conditional block optimality; no end-to-end task result or novelty claim'}
    (args.out/'audit.json').write_text(json.dumps(output,indent=2),encoding='utf-8')
    grid=np.linspace(-.3,.3,1001);values=[block.energy(b,block.prox(c+b,target,old,.01),c,target,old,.01) for b in grid]
    fig,ax=plt.subplots(figsize=(8,4.5),layout='constrained');ax.plot(grid,values,label='After exact elimination of z')
    ax.scatter([0,bias],[float(e_old),meta['energy']],c=['#bc5a65','#007d8a'],zorder=5)
    ax.annotate('Coordinate fixed point',(0,float(e_old)),xytext=(-.26,.02),arrowprops={'arrowstyle':'->'})
    ax.annotate('Joint global minimum',(bias,meta['energy']),xytext=(.1,.04),arrowprops={'arrowstyle':'->'})
    ax.set(xlabel='Shared bias b',ylabel='Conditional local energy',title='Exact shared-bias block crosses a coordinate barrier');ax.legend();ax.grid(alpha=.12)
    fig.savefig(args.out/'shared_bias_witness.png',dpi=160);plt.close(fig)
    print(json.dumps({k:v for k,v in output.items() if k!='cases'}),flush=True)


if __name__=='__main__':main()
