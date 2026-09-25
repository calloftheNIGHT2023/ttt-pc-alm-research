"""Two-set local projection ADMM for coupled interval memory writes.

Projection/consensus neural learning has extensive prior art. This is a new
controlled implementation of that solver family, not a novelty claim.
"""
import numpy as np
from scipy.optimize import minimize
import vector_interval_memory as task


def project_affine(z,h,b,x,weights):
    """Euclidean projection onto all affine equalities + the shared bias box.

    Blocks are disjoint (h[l-1], z[l], b[l]); orthogonal W gives closed forms.
    h[-1] is unconstrained by this set. Its noisy target belongs to graph set B.
    """
    az=z.copy(); ah=h.copy(); ab=b.copy(); d,r,n,width=z.shape
    for j,w in enumerate(weights):
        if j==0:
            fixed=x@w.T
            ab[:,j]=np.clip((np.sum(z[j]-fixed,axis=1)+b[:,j])/(n+1),-task.BOUND,task.BOUND)
            az[j]=fixed+ab[:,j,None,:]
        else:
            before=h[j-1]@w.T
            ab[:,j]=np.clip((np.sum(z[j]-before,axis=1)+2*b[:,j])/(n+2),-task.BOUND,task.BOUND)
            az[j]=(z[j]+before+ab[:,j,None,:])/2
            ah[j-1]=(h[j-1]+(z[j]-ab[:,j,None,:])@w)/2
    return az,ah,ab


def project_graph(z,h,v):
    d,r,n,width=z.shape; zz=np.empty_like(z); hh=np.empty_like(h)
    shape=(4,1,1,1); slope=np.array([0.,2.,-2.,0.]).reshape(shape); intercept=np.array([-1.,1.,1.,-1.]).reshape(shape)
    low=np.array([-np.inf,-1.,0.,1.]).reshape(shape); high=np.array([-1.,0.,1.,np.inf]).reshape(shape)
    for j in range(d):
        lo=np.broadcast_to(low,(4,r,n,width)).copy(); hi=np.broadcast_to(high,lo.shape).copy(); valid=np.ones_like(lo,dtype=bool)
        if j==d-1:
            vl=np.maximum(-1,v-task.EPS); vh=np.minimum(1,v+task.EPS)
            valid[0]=valid[3]=(vl<=-1)&(vh>=-1)
            lo[1]=np.maximum(lo[1],(vl-1)/2); hi[1]=np.minimum(hi[1],(vh-1)/2)
            lo[2]=np.maximum(lo[2],(1-vh)/2); hi[2]=np.minimum(hi[2],(1-vl)/2)
            valid&=lo<=hi
        candidates=np.minimum(np.maximum((z[j][None]+slope*(h[j][None]-intercept))/(1+slope**2),lo),hi)
        output=task.family.activation(candidates)
        cost=(candidates-z[j])**2+(output-h[j])**2; cost=np.where(valid,cost,np.inf)
        chosen=np.argmin(cost,axis=0)[None]
        assert np.isfinite(np.min(cost,axis=0)).all()
        zz[j]=np.take_along_axis(candidates,chosen,axis=0)[0]; hh[j]=np.take_along_axis(output,chosen,axis=0)[0]
    return zz,hh


def refine(starts,x,v,weights,anchor,*,steps=120,dual_rate=1.):
    b=starts.copy(); r,d,width=b.shape; z=np.empty((d,r,len(x),width)); h=np.empty_like(z); prev=x
    for j,w in enumerate(weights):z[j]=prev@w.T+b[:,j,None,:]; h[j]=task.family.activation(z[j]); prev=h[j]
    uz=np.zeros_like(z); uh=np.zeros_like(h); best=b.copy(); errors,moves=task.score(best,x,v,weights,anchor)
    for _ in range(steps):
        az,ah,b=project_affine(z-uz,h-uh,b,x,weights)
        z,h=project_graph(az+uz,ah+uh,v)
        uz+=dual_rate*(az-z); uh+=dual_rate*(ah-h)
        err,move=task.score(b,x,v,weights,anchor); update=task.better(err,move,errors,moves)
        best[update]=b[update]; errors[update]=err[update]; moves[update]=move[update]
    return best,{"consensus_steps":steps,"dual_rate":dual_rate,"consensus_residual_rms":float(np.sqrt(np.mean((az-z)**2+(ah-h)**2))),
        "major_arrays_bytes_subtotal":sum(t.nbytes for t in [b,best,z,h,uz,uh,az,ah]),
        "state_scope":"major live-array subtotal; not full temporary/native peak"}


def fit(x,v,anchor,weights,cfg):
    if cfg["method"]!="consensus":return task.fit_internal(x,v,anchor,weights,cfg)
    starts,meta=task.proposals(x,v,anchor,weights,features=256,restarts=16)
    bank,more=refine(starts,x,v,weights,anchor,steps=cfg["steps"],dual_rate=cfg.get("dual_rate",1.))
    point=task.select(np.concatenate([starts,bank]),x,v,weights,anchor)
    return lambda q:task.forward(point[None],q,weights)[0],point,{**meta,**more,"persistent_state_bytes":point.nbytes,"known_weights_bytes":weights.nbytes,
        "anchor_output":point.tolist(),"readout":"one support-selected parameter, no global geometric refinement"}


def verify():
    rng=np.random.default_rng(198559); w=task.family.make_weights(3,2); x=rng.uniform(-1,1,(3,2)); v=rng.uniform(-1,1,(3,2))
    z=rng.normal(size=(3,2,3,2)); h=rng.normal(size=z.shape); b=rng.uniform(-.3,.3,(2,3,2))
    az,ah,ab=project_affine(z,h,b,x,w)
    for j in range(3):
        previous=x if j==0 else ah[j-1]
        assert np.max(np.abs(az[j]-previous@w[j].T-ab[:,j,None,:]))<1e-12
    aa=project_affine(az,ah,ab,x,w); assert all(np.allclose(left,right,atol=1e-12) for left,right in zip(aa,[az,ah,ab]))
    # Independent equality-constrained quadratic solves, including clipping b.
    gaps=[]
    for j in [0,1,2]:
        for r in [0,1]:
            n,width=3,2; previous=x if j==0 else h[j-1,r]
            def unpack(flat):
                bb=flat[:width]; hh0=previous if j==0 else flat[width:width+n*width].reshape(n,width)
                zz0=flat[width:].reshape(n,width) if j==0 else flat[width+n*width:].reshape(n,width)
                return bb,hh0,zz0
            def fun(flat):
                bb,hh0,zz0=unpack(flat)
                return .5*np.sum((bb-b[r,j])**2)+.5*np.sum((zz0-z[j,r])**2)+(0 if j==0 else .5*np.sum((hh0-previous)**2))
            initial=np.r_[b[r,j],z[j,r].ravel()] if j==0 else np.r_[b[r,j],previous.ravel(),z[j,r].ravel()]
            result=minimize(fun,initial,method="SLSQP",bounds=[(-.3,.3)]*width+[(None,None)]*(len(initial)-width),
                constraints={"type":"eq","fun":lambda flat:(unpack(flat)[2]-unpack(flat)[1]@w[j].T-unpack(flat)[0]).ravel()},
                options={"maxiter":500,"ftol":1e-12})
            actual=np.r_[ab[r,j],az[j,r].ravel()] if j==0 else np.r_[ab[r,j],ah[j-1,r].ravel(),az[j,r].ravel()]
            gap=float(fun(actual)-result.fun); assert result.success and abs(gap)<1e-7; gaps.append(gap)
    gz,gh=project_graph(z,h,v); assert np.allclose(task.family.activation(gz),gh)
    assert np.max(np.abs(gh[-1]-v))<=task.EPS+1e-14
    gz2,gh2=project_graph(gz,gh,v); assert np.allclose(gz,gz2,atol=1e-12) and np.allclose(gh,gh2,atol=1e-12)
    # Each finite branch's unconstrained minimizer has exact closed form; dense
    # audit includes narrow feasible endpoints for the final output graph.
    branch_cases=0
    for j in range(3):
        for i in range(3):
            for k in range(2):
                a=z[j,0,i,k]; t=h[j,0,i,k]; grid=np.linspace(-5,5,100001)
                if j==2:
                    vl=max(-1,v[i,k]-task.EPS); vh=min(1,v[i,k]+task.EPS)
                    grid=np.r_[grid,np.linspace((vl-1)/2,(vh-1)/2,201),np.linspace((1-vh)/2,(1-vl)/2,201)]
                output=task.family.activation(grid); energy=(grid-a)**2+(output-t)**2
                if j==2:energy=np.where(np.abs(output-v[i,k])<=task.EPS+1e-14,energy,np.inf)
                exact=(gz[j,0,i,k]-a)**2+(gh[j,0,i,k]-t)**2
                assert exact<=energy.min()+1e-8; branch_cases+=1
    original=task.evaluate
    try:
        def forbidden(*args,**kwargs):raise RuntimeError("global BP forbidden")
        task.evaluate=forbidden
        guarded,_=refine(b,x,v,w,np.zeros((3,2)),steps=3)
        assert np.isfinite(guarded).all()
    finally:task.evaluate=original
    return {"passed":True,"candidate_global_bp_guard":True,"affine_projection_max_objective_gap":max(gaps),"independent_qp_cases":len(gaps),
        "activation_graph_dense_cases":branch_cases,"affine_and_graph_idempotence":True,"output_band_enforced":True}


if __name__=="__main__":
    import json
    print(json.dumps(verify()))
