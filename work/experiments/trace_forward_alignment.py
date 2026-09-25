"""Read-only instrumentation of frozen local and Adam trajectories.

No query inputs or query targets are generated. Global gradients for local
proposals are computed only after each complete, non-mutated local trajectory.
"""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import guarded_activity_memory as guarded
import event_affine_memory as original
base=original.base
CHECKPOINTS={1,2,4,8,16,32,64,128}
ALPHAS=np.array([0.,.125,.25,.5,1.])


def pattern(bank,x,weights):
    h=np.broadcast_to(x,(len(bank),*x.shape));patterns=[]
    for j,weight in enumerate(weights):
        z=h@weight.T+bank[:,j,None,:];patterns.append(np.searchsorted([-1.,0.,1.],z));h=base.family.activation(z)
    return np.stack(patterns)


def loss(bank,x,v,weights):
    raw=base.forward(bank,x,weights)-v
    return .5*np.mean(np.sum(base.residuals(raw)**2,axis=2),axis=1),np.max(np.abs(raw),axis=(1,2))


def conditional(bank,h,a,x,weights):
    value=np.zeros(len(bank))
    for j,weight in enumerate(weights):
        previous=x if j==0 else h[j-1]
        residual=h[j]+a[j]-base.family.activation(previous@weight.T+bank[:,j,None,:])
        value+=.5*np.mean(np.sum(residual**2,axis=-1),axis=-1)
    return value


def records(old,new,x,v,weights,iteration,qbefore=None,qafter=None,h=None):
    bank=old[None]+ALPHAS[:,None,None,None]*(new-old)[None]
    values,errors=loss(bank.reshape(-1,*old.shape[1:]),x,v,weights)
    values=values.reshape(len(ALPHAS),len(old));errors=errors.reshape(len(ALPHAS),len(old))
    true_pattern=pattern(old,x,weights);changed=np.mean(true_pattern!=pattern(new,x,weights),axis=(0,2,3))
    mismatch=None
    if h is not None:
        free=np.stack([np.searchsorted([-1.,0.,1.],(x if j==0 else h[j-1])@weight.T+old[:,j,None,:]) for j,weight in enumerate(weights)])
        mismatch=np.mean(free!=true_pattern,axis=(0,2,3))
    return [dict(iteration=iteration,restart=i,old_parameters=old[i].tolist(),proposed_parameters=new[i].tolist(),
                 line_losses=values[:,i].tolist(),line_max_errors=errors[:,i].tolist(),branch_change_fraction=float(changed[i]),
                 free_true_branch_mismatch=None if mismatch is None else float(mismatch[i]),
                 conditional_before=None if qbefore is None else float(qbefore[i]),
                 conditional_after=None if qafter is None else float(qafter[i]),
                 old_within_prior=bool(np.max(np.abs(old[i]))<=base.BOUND+1e-12),
                 proposed_within_prior=bool(np.max(np.abs(new[i]))<=base.BOUND+1e-12)) for i in range(len(old))]


def local(starts,x,v,weights,anchor,*,sweeps=128,guard=False):
    b=starts.copy();r,d,w=b.shape;trust=.01
    h=np.empty((d,r,len(x),w));a=np.zeros_like(h);previous=x
    for j,weight in enumerate(weights):h[j]=base.family.activation(previous@weight.T+b[:,j,None,:]);previous=h[j]
    low,high=guarded.reference.activity.interval_bounds(x,weights,base.BOUND)
    cache=original.first.prepare(x@weights[0].T,base.BOUND,trust)
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor);rows=[]
    initial_loss,initial_error=loss(b,x,v,weights)
    for iteration in range(1,sweeps+1):
        old=b.copy()
        for j in reversed(range(d)):
            previous=x if j==0 else h[j-1]
            center=base.family.activation(previous@weights[j].T+b[:,j,None,:])-a[j]
            if j==d-1:h[j]=np.clip((center+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            elif guard:h[j],_=guarded.update(h[j],center,h[j+1]+a[j+1],weights[j+1],b[:,j+1,None,:],low[j],high[j],trust,True)
            else:
                target=h[j+1]+a[j+1];old_u=h[j]@weights[j+1].T+b[:,j+1,None,:]
                u_center=center@weights[j+1].T+b[:,j+1,None,:]
                u=base.family.nonlinear_prox(u_center,target,old_u,trust)
                h[j]=(u-b[:,j+1,None,:])@weights[j+1]
        qbefore=conditional(b,h,a,x,weights) if iteration in CHECKPOINTS else None
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];c=previous@weight.T;target=h[j]+a[j]
            if j==0:b[:,j],_=original.first.solve(cache,target,b[:,j])
            else:b[:,j],_=original.bias_solver.solve(c,target,b[:,j],trust,base.BOUND)
        if iteration in CHECKPOINTS:
            rows.extend(records(old,b,x,v,weights,iteration,qbefore,conditional(b,h,a,x,weights),h))
        for j,weight in enumerate(weights):
            previous=x if j==0 else h[j-1];a[j]+=.5*(h[j]-base.family.activation(previous@weight.T+b[:,j,None,:]))
        err,move=base.score(b,x,v,weights,anchor);change=base.better(err,move,errors,moves)
        best[change]=b[change];errors[change]=err[change];moves[change]=move[change]
    return best,rows,dict(initial_losses=initial_loss.tolist(),initial_max_errors=initial_error.tolist())


def adam(starts,x,v,weights,anchor,steps):
    b=starts.copy();best=b.copy();errors,moves=base.score(best,x,v,weights,anchor)
    first=np.zeros_like(b);second=first.copy();rows=[]
    def retain(params,raw):
        err=np.max(np.abs(raw),axis=(1,2));move=.5*np.sum((params-anchor)**2,axis=(1,2));change=base.better(err,move,errors,moves)
        best[change]=params[change];errors[change]=err[change];moves[change]=move[change]
    for iteration in range(1,steps+1):
        old=b.copy();_,grad,raw=base.evaluate(b,x,v,weights);retain(b,raw)
        first=.9*first+.1*grad;second=.999*second+.001*grad**2
        b=np.clip(b-.01*(first/(1-.9**iteration))/(np.sqrt(second/(1-.999**iteration))+1e-8),-base.BOUND,base.BOUND)
        if iteration in CHECKPOINTS:rows.extend(records(old,b,x,v,weights,iteration))
    retain(b,base.forward(b,x,weights)-v)
    return best,rows


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();base.BOUND=.2
    sources=[Path(s) for s in [__file__,guarded.__file__,guarded.reference.__file__,guarded.reference.activity.__file__,
             original.__file__,original.original.__file__,original.first.__file__,original.bias_solver.__file__,base.__file__,base.family.__file__]]
    protocol=dict(seeds=list(range(5400000,5400012)),contexts=8,depth=3,width=8,restarts=16,features=256,
                  methods={'orthogonal':128,'guard':128,'adam64':64,'adam240':240},checkpoints=sorted(CHECKPOINTS),
                  zero='initial state recorded separately; checkpoint k describes the kth actual update',alphas=ALPHAS.tolist(),
                  scope='support-only diagnostic; no query generation; gradients added after full candidate trajectories; no feedback',
                  source_sha256={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources})
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];audits=[];weights=base.family.make_weights(3,8)
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.2,.2,(3,8));x=rng.uniform(-1,1,(24,8))[:8]
        v=base.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,x.shape)
        anchor=np.zeros_like(truth);starts,_=base.proposals(x,v,anchor,weights,features=256,restarts=16)
        for name,steps in protocol['methods'].items():
            if name in ['orthogonal','guard']:
                original_evaluate=base.evaluate
                def forbidden(*args,**kwargs):raise AssertionError('global gradient entered local trajectory')
                base.evaluate=forbidden
                try:
                    best,trace,initial=local(starts,x,v,weights,anchor,sweeps=steps,guard=name=='guard')
                    expected,_=(guarded.local if name=='guard' else original.local)(starts,x,v,weights,anchor,sweeps=steps)
                finally:base.evaluate=original_evaluate
            else:
                best,trace=adam(starts,x,v,weights,anchor,steps);initial={}
                expected,_=base.bp(starts,x,v,weights,anchor,solver='adam',steps=steps,lr=.01)
            gap=float(np.max(np.abs(best-expected)));assert gap==0,(name,seed,gap)
            # The whole local run has finished before any diagnostic credit is computed.
            old=np.array([r['old_parameters'] for r in trace]);new=np.array([r['proposed_parameters'] for r in trace])
            _,gradient,_=base.evaluate(old,x,v,weights)
            for index,row in enumerate(trace):
                direction=new[index]-old[index];dot=float(np.sum(gradient[index]*direction));denom=float(np.linalg.norm(gradient[index])*np.linalg.norm(direction))
                row.update(seed=seed,method=name,diagnostic_gradient_dot=dot,diagnostic_cosine=dot/denom if denom>1e-30 else None)
            rows.extend(trace);audits.append(dict(seed=seed,method=name,original_parameter_gap=gap,**initial))
        (args.out/'trace.json').write_text(json.dumps(dict(rows=rows,audits=audits),indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=len(protocol['seeds']),rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
