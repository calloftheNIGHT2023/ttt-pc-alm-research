"""Scheduled conditional-global shared-bias blocks in local ALM. No global BP."""
import numpy as np
import vector_interval_memory as base
import exact_shared_bias_block as exact


def local(starts,x,v,weights,anchor,*,sweeps=240,dual_rate=.5,joint_every=60,coordinate_rounds=0):
    b=starts.copy();r,d,width=b.shape;n=len(x);trust=.01
    z=np.empty((d,r,n,width));h=np.empty_like(z);p=np.zeros_like(z);a=np.zeros_like(z);previous=x
    for j,w in enumerate(weights):z[j]=previous@w.T+b[:,j,None,:];h[j]=base.family.activation(z[j]);previous=h[j]
    best=b.copy();errors,moves=base.score(best,x,v,weights,anchor);calls=0;gain=0.;max_intervals=0
    for iteration in range(sweeps):
        for j in reversed(range(d)):
            target=base.family.activation(z[j])-a[j]
            if j==d-1:h[j]=np.clip((target+trust*h[j])/(1+trust),np.maximum(-1,v-base.EPS),np.minimum(1,v+base.EPS))
            else:
                t=z[j+1]-b[:,j+1,None,:]+p[j+1]
                h[j]=np.clip((target+t@weights[j+1]+trust*h[j])/(2+trust),-1,1)
            previous=x if j==0 else h[j-1];c=previous@weights[j].T-p[j];target=h[j]+a[j];old=z[j].copy()
            # c has one restart dimension through p, including the first layer.
            if iteration%joint_every==0:
                if coordinate_rounds:
                    for _ in range(coordinate_rounds):
                        z[j]=base.family.nonlinear_prox(c+b[:,j,None,:],target,old,trust)
                        b[:,j]=np.clip(np.mean(z[j]-c,axis=1),-base.BOUND,base.BOUND)
                else:
                    for ri in range(r):
                        for channel in range(width):
                            cc=c[ri,:,channel];tt=target[ri,:,channel];oo=old[ri,:,channel]
                            oldb=b[ri,j,channel];sepz=exact.prox(cc+oldb,tt,oo,trust)
                            sep_b=float(np.clip(np.mean(sepz-cc),-base.BOUND,base.BOUND))
                            reference=exact.energy(sep_b,sepz,cc,tt,oo,trust)
                            newb,newz,meta=exact.solve(cc,tt,oo,trust,base.BOUND)
                            assert meta['energy']<=reference+1e-8*max(1,abs(reference))
                            b[ri,j,channel]=newb;z[j,ri,:,channel]=newz;calls+=1
                            gain+=reference-meta['energy'];max_intervals=max(max_intervals,meta['sum_intervals'])
            else:z[j]=base.family.nonlinear_prox(c+b[:,j,None,:],target,old,trust)
        # Refresh shared biases after lower h blocks changed. This also decreases
        # their conditional quadratic; it does not undo the exact-block audit.
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            b[:,j]=np.clip(np.mean(z[j]-previous@w.T+p[j],axis=1),-base.BOUND,base.BOUND)
        for j,w in enumerate(weights):
            previous=x if j==0 else h[j-1]
            p[j]+=dual_rate*(z[j]-previous@w.T-b[:,j,None,:]);a[j]+=dual_rate*(h[j]-base.family.activation(z[j]))
        err,move=base.score(b,x,v,weights,anchor);update=base.better(err,move,errors,moves)
        best[update]=b[update];errors[update]=err[update];moves[update]=move[update]
    return best,{'sweeps':sweeps,'dual_rate':dual_rate,'joint_every':joint_every,'coordinate_rounds':coordinate_rounds,
        'exact_block_calls':calls,'sum_conditional_gain_over_one_coordinate_pass':gain,'max_sum_envelope_intervals':max_intervals,
        'major_arrays_bytes_subtotal':sum(t.nbytes for t in [b,z,h,p,a,best])}


def fit(x,v,anchor,weights,cfg):
    starts,meta=base.proposals(x,v,anchor,weights)
    bank,more=local(starts,x,v,weights,anchor,sweeps=cfg['sweeps'],dual_rate=cfg.get('dual_rate',.5),
        joint_every=cfg.get('joint_every',60),coordinate_rounds=cfg.get('coordinate_rounds',0))
    point=base.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:base.forward(point[None],q,weights)[0],point,{**meta,**more,'persistent_state_bytes':point.nbytes,
        'known_weights_bytes':weights.nbytes,'anchor_output':point.tolist()}
