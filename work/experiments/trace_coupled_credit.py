"""Read-only-in-algorithm tracing of the frozen split/joint trajectories.

No query data enters this diagnostic. Gradients below are adjacent-block partials,
not whole-chain derivatives. Traced outputs are checked against frozen runs.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import vector_interval_memory as base
import exact_shared_bias_block as exact


def support(b,x,v,weights):
    raw=base.forward(b,x,weights)-v
    return .5*np.mean(np.sum(base.residuals(raw)**2,axis=2),axis=1),np.max(np.abs(raw),axis=(1,2))


def snapshot(b,h,z,p,a,x,v,weights):
    d=len(weights);aff=np.stack([z[j]-(x if j==0 else h[j-1])@weights[j].T-b[:,j,None,:] for j in range(d)])
    non=h-base.family.activation(z);ra=aff+p;rn=non+a
    gz=ra-base.family.deriv(z)*rn
    gh=rn.copy()
    for j in range(d-1):gh[j]-=ra[j+1]@weights[j+1]
    projected_h=h-gh
    for j in range(d):
        projected_h[j]=np.clip(projected_h[j],np.maximum(-1,v-base.EPS) if j==d-1 else -1,np.minimum(1,v+base.EPS) if j==d-1 else 1)
    gh=h-projected_h
    gb=-ra.mean(axis=2).transpose(1,0,2);gb=b-np.clip(b-gb,-base.BOUND,base.BOUND)
    point=base.forward(b,x,weights);actual=np.linalg.norm(point-h[-1],axis=2)
    weighted=np.stack([2**(d-j-1)*(2*np.linalg.norm(aff[j],axis=2)+np.linalg.norm(non[j],axis=2)) for j in range(d)])
    bound=weighted.sum(0);assert np.all(actual<=bound+1e-10)
    value,error=support(b,x,v,weights)
    return {'band_loss':value,'max_support_error':error,'forward_activity_gap_rms':np.sqrt(np.mean(actual**2,axis=1)),
        'path_bound_rms':np.sqrt(np.mean(bound**2,axis=1)),
        'path_bound_by_layer_rms':np.sqrt(np.mean(weighted**2,axis=2)).T,
        'affine_rms_by_layer':np.sqrt(np.mean(aff**2,axis=(2,3))).T,
        'nonlinear_rms_by_layer':np.sqrt(np.mean(non**2,axis=(2,3))).T,
        'primal_residual_rms':np.sqrt(np.mean(aff**2+non**2,axis=(0,2,3))),
        'local_stationarity_rms':np.sqrt(np.mean(gz**2+gh**2,axis=(0,2,3))+np.mean(gb**2,axis=(1,2))),
        'scaled_dual_rms':np.sqrt(np.mean(p**2+a**2,axis=(0,2,3)))}


def trace(starts,x,v,weights,anchor,sweeps,joint=False):
    b=starts.copy();r,d,width=b.shape;n=len(x);trust=.01
    z=np.empty((d,r,n,width));h=np.empty_like(z);p=np.zeros_like(z);a=np.zeros_like(z);previous=x
    for j,w in enumerate(weights):z[j]=previous@w.T+b[:,j,None,:];h[j]=base.family.activation(z[j]);previous=h[j]
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor);trajectory=[];events=[]
    for iteration in range(sweeps):
        before=snapshot(b,h,z,p,a,x,v,weights)
        for j in reversed(range(d)):
            target=base.family.activation(z[j])-a[j]
            if j==d-1:h[j]=np.clip((target+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                t=z[j+1]-b[:,j+1,None,:]+p[j+1]
                h[j]=np.clip((target+t@weights[j+1]+trust*h[j])/(2+trust),-1,1)
            previous=x if j==0 else h[j-1];c=previous@weights[j].T-p[j];target=h[j]+a[j];old=z[j].copy()
            if joint and iteration%60==0:
                prior_b=b.copy();ref_z=base.family.nonlinear_prox(c+b[:,j,None,:],target,old,trust)
                ref_b=b.copy();ref_b[:,j]=np.clip(np.mean(ref_z-c,axis=1),-base.BOUND,base.BOUND)
                ref_energy=np.sum((ref_z-c-ref_b[:,j,None,:])**2+(base.family.activation(ref_z)-target)**2+trust*(ref_z-old)**2,axis=(1,2))
                energies=np.zeros(r)
                for ri in range(r):
                    for channel in range(width):
                        newb,newz,meta=exact.solve(c[ri,:,channel],target[ri,:,channel],old[ri,:,channel],trust,base.BOUND)
                        b[ri,j,channel]=newb;z[j,ri,:,channel]=newz;energies[ri]+=meta['energy']
                assert np.all(energies<=ref_energy+1e-8)
                lv,le=support(b,x,v,weights);rv,re=support(ref_b,x,v,weights);ov,oe=support(prior_b,x,v,weights)
                for ri in range(r):events.append({'iteration':iteration,'layer':j,'restart':ri,
                    'conditional_energy_gain':float(ref_energy[ri]-energies[ri]),'joint_band_loss':float(lv[ri]),
                    'coordinate_band_loss':float(rv[ri]),'before_band_loss':float(ov[ri]),
                    'joint_max_error':float(le[ri]),'coordinate_max_error':float(re[ri]),'before_max_error':float(oe[ri])})
            else:z[j]=base.family.nonlinear_prox(c+b[:,j,None,:],target,old,trust)
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            b[:,j]=np.clip(np.mean(z[j]-previous@w.T+p[j],axis=1),-base.BOUND,base.BOUND)
        after_primal=snapshot(b,h,z,p,a,x,v,weights)
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            p[j]+=.5*(z[j]-previous@w.T-b[:,j,None,:]);a[j]+=.5*(h[j]-base.family.activation(z[j]))
        err,move=base.score(b,x,v,weights,anchor);update=base.better(err,move,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=move[update]
        for ri in range(r):
            row={'iteration':iteration,'restart':ri,'best_max_support_error':float(errors[ri]),
                'band_loss_before_sweep':float(before['band_loss'][ri]),'residual_before_sweep':float(before['primal_residual_rms'][ri]),
                'stationarity_before_sweep':float(before['local_stationarity_rms'][ri])}
            row.update({key:np.asarray(value[ri]).tolist() for key,value in after_primal.items()});trajectory.append(row)
    return best,trajectory,events


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists()
    sources=[Path(__file__),Path(base.__file__),Path(base.family.__file__),Path(exact.__file__)]
    proto={'seeds':list(range(5400000,5400004)),'width':8,'depth':3,'contexts':8,'configs':[('split',480),('joint',240)],
        'scope':'context-only trajectory diagnosis; not a method search or confirmation','source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}}
    (args.out/'protocol.json').write_text(json.dumps(proto,indent=2),encoding='utf-8');all_rows=[];all_events=[];checks=[]
    old=json.loads((args.out.parents[1]/'exact_shared_bias/development/episodes.json').read_text())
    weights=base.family.make_weights(3,8)
    for seed in proto['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-base.PRIOR,base.PRIOR,(3,8));x=rng.uniform(-1,1,(24,8))
        v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
        anchor=np.zeros_like(truth);starts,_=base.proposals(x[:8],v[:8],anchor,weights)
        for name,sweeps in proto['configs']:
            best,rows,events=trace(starts,x[:8],v[:8],weights,anchor,sweeps,joint=name=='joint')
            point=base.select(np.concatenate([starts,best]),x[:8],v[:8],weights,anchor)
            ref=next(r for r in old if r['seed']==seed and r['n_context']==8 and r['method']==('joint240' if name=='joint' else 'local480'))
            delta=float(np.max(np.abs(point-np.array(ref['anchor_output']))));assert delta<1e-12,delta
            checks.append({'seed':seed,'method':name,'max_parameter_change_vs_frozen':delta})
            all_rows.extend({'seed':seed,'method':name,**r} for r in rows);all_events.extend({'seed':seed,**r} for r in events)
        (args.out/'trace.json').write_text(json.dumps({'rows':all_rows,'events':all_events,'equivalence_checks':checks},indent=2),encoding='utf-8')
        print(json.dumps({'seed':seed,'trace_rows':len(all_rows),'events':len(all_events),'exact_equivalence':max(r['max_parameter_change_vs_frozen'] for r in checks)==0}),flush=True)
    print(json.dumps({'complete':True}),flush=True)


if __name__=='__main__':main()
