"""Local preactivation-dual corrections recover omitted branch-normal terms.

Fix a from existing credit memory and initialize p=s_R*a. Every p coordinate
uses exact breakpoints of its concave piecewise-linear dual bound. Mandatory
outward certification remains the only deletion rule. No global derivatives.
"""
import time
import numpy as np
import local_region_screen as screen
base=screen.base


def refine(x,v,regs,a,sweeps=4):
    p=base.SLOPES[regs]*a; sa=p.copy(); zl,zh,hl,hh=screen.boxes(v,regs)
    snapshots={0:p.copy()}; initial=screen.float_bound(x,v,regs,p,a); bounds={0:initial}
    rows=np.arange(len(regs))
    for sweep in range(1,sweeps+1):
        for j in reversed(range(regs.shape[1])):
            for i in range(regs.shape[2]):
                old=p[:,j,i].copy(); other=p[:,j].sum(1)-old
                prior=np.zeros(len(regs)) if j==0 else a[:,j-1,i]
                options=np.column_stack([old,-other,sa[:,j,i],prior])
                zcoeff=options-sa[:,j,i,None]
                value=-screen.B*np.abs(other[:,None]+options)+zcoeff*np.where(zcoeff>=0,zl[:,j,i,None],zh[:,j,i,None])
                if j:
                    hcoeff=prior[:,None]-options
                    value+=hcoeff*np.where(hcoeff>=0,hl[:,j-1,i,None],hh[:,j-1,i,None])
                else:value-=options*x[i]
                p[:,j,i]=options[rows,np.argmax(value,axis=1)]
        bounds[sweep]=screen.float_bound(x,v,regs,p,a)
        assert np.all(bounds[sweep]>=bounds[sweep-1]-1e-10*(1+np.abs(a).sum((1,2))))
        if sweep in [1,sweeps]:snapshots[sweep]=p.copy()
    return p,bounds,snapshots


def screen_bank(x,v,regs,bank,sweeps=4):
    begin=time.perf_counter(); count=len(regs); rejected=np.zeros(count,bool); proofs=[]
    if not count or not len(bank):return rejected,proofs,dict(seconds=time.perf_counter()-begin,directions=len(bank),coordinate_updates=0,main_array_bytes_subtotal=0,sweep_positive_counts={})
    k=len(bank); rr=np.repeat(regs,k,axis=0); aa=np.tile(bank,(count,1,1))
    pp,bounds,snapshots=refine(x,v,rr,aa,sweeps)
    records={}; sweep_counts={}
    for sweep in [1,sweeps]:
        values=bounds[sweep].reshape(count,k); chosen=values.argmax(1); ids=np.arange(count)*k+chosen
        p=snapshots[sweep]; scale=1+np.abs(p[ids]).sum((1,2))+np.abs(aa[ids]).sum((1,2))
        valid=np.flatnonzero(values[np.arange(count),chosen]>1e-10*scale); flat=ids[valid]
        lower=screen.certified_lower_bound(x,v,rr[flat],p[flat],aa[flat]) if len(flat) else []
        for index,fi,lb in zip(valid,flat,lower):
            if lb<=0:continue
            if index not in records:
                records[int(index)]=dict(index=int(index),pattern=regs[index].tobytes().hex(),a=aa[fi].tolist(),p=p[fi].tolist(),lower=float(lb),
                    sweep=sweep,direction=int(fi%k),initial_best_bound=float(bounds[0].reshape(count,k)[index].max()),normal_norm=float(np.linalg.norm(p[fi]-base.SLOPES[rr[fi]]*aa[fi])))
            rejected[index]=True
        sweep_counts[sweep]=int(rejected.sum())
    proofs=list(records.values())
    arrays=rr.nbytes+aa.nbytes+pp.nbytes+sum(a.nbytes for a in snapshots.values())+sum(a.nbytes for a in bounds.values())
    return rejected,proofs,dict(seconds=time.perf_counter()-begin,directions=len(bank),coordinate_updates=sweeps*len(rr)*regs.shape[1]*regs.shape[2],
        main_array_bytes_subtotal=arrays,direction_bytes=bank.nbytes,sweep_positive_counts=sweep_counts,
        scope='array subtotal omits coordinate temporaries and objects; diagnostic time not whole online speed')


def verify():
    rng=np.random.default_rng(555391); checked=0; gaps=[]
    for d,n in [(1,2),(2,4),(4,4),(4,8)]:
        x=rng.uniform(0,1,n); v=base.forward(x,rng.uniform(-.12,.12,d)); regs=rng.integers(0,4,(7,d,n),dtype=np.uint8); a=rng.normal(size=regs.shape)
        p,bounds,snaps=refine(x,v,regs,a,4)
        assert np.all(bounds[4]>=bounds[0]-1e-10)
        # Independently evaluate the full bound at each one-coordinate option.
        for j in range(d):
            for i in range(n):
                baseline=snaps[0].copy(); other=baseline[:,j].sum(1)-baseline[:,j,i]
                choices=np.column_stack([baseline[:,j,i],-other,base.SLOPES[regs[:,j,i]]*a[:,j,i],a[:,j-1,i] if j else np.zeros(len(regs))])
                direct=[]
                for col in range(4):
                    pp=baseline.copy();pp[:,j,i]=choices[:,col];direct.append(screen.float_bound(x,v,regs,pp,a))
                zl,zh,hl,hh=screen.boxes(v,regs); zc=choices-base.SLOPES[regs[:,j,i,None]]*a[:,j,i,None]
                local=-screen.B*np.abs(other[:,None]+choices)+zc*np.where(zc>=0,zl[:,j,i,None],zh[:,j,i,None])
                if j:
                    hc=a[:,j-1,i,None]-choices;local+=hc*np.where(hc>=0,hl[:,j-1,i,None],hh[:,j-1,i,None])
                else:local-=choices*x[i]
                direct=np.array(direct).T; gap=np.max(np.abs((direct-direct[:,:1])-(local-local[:,:1])));gaps.append(float(gap));assert gap<1e-10;checked+=len(regs)
        zero=np.zeros_like(a);_,zb,_=refine(x,v,regs,zero,4);assert np.all(zb[4]>=-1e-12)
    return dict(passed=True,independent_full_bound_coordinate_cases=checked,max_local_vs_full_bound_difference=max(gaps),monotone_sweep_bounds=True,
        scope='best among local breakpoints includes old point; no claim of joint dual optimum')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
