"""Fixed support-only credit-scatter directions for conditional integration.

Local forces use only adjacent activities and local derivatives. An explicitly
separate BP control uses global Jacobians of the same observed support loss.
Geometry and eigenvectors are shared controls, not PC-exclusive contributions.
"""
import time
import numpy as np
import deferred_typed_memory as model

KINDS=('raw','augmented','residual','bp')
DIRECTIONS=('coordinate_0','coordinate_1','coordinate_2','coordinate_3','pca','support_tangent',
            'local_raw_cov','local_augmented_cov','residual_cov','bp_cov','local_augmented_plain')


def local_forces(x,b,h,u):
    r,d=b.shape;n=len(x);previous=np.broadcast_to(x,(r,n));derivative=[];residual=[]
    for layer in range(d):
        z=previous+b[:,layer,None];derivative.append(model.base.derivative(z));residual.append(h[layer]-model.base.g(z));previous=h[layer]
    derivative=np.array(derivative);residual=np.array(residual);raw=np.zeros_like(residual) if u is None else u
    return {name:-np.mean(derivative*credit,axis=2).T for name,credit in [('raw',raw),('augmented',raw+residual),('residual',residual)]}


def bp_forces(x,v,b):
    """Global-BP comparator ONLY: gradient of half mean squared band residual."""
    previous=np.broadcast_to(x,(len(b),len(x)));jac=np.zeros((len(b),len(x),b.shape[1]))
    for layer in range(b.shape[1]):
        z=previous+b[:,layer,None];derivative=model.base.derivative(z)
        jac*=derivative[:,:,None];jac[:,:,layer]+=derivative;previous=model.base.g(z)
    error=previous-v;band=error-np.clip(error,-model.base.EPS,model.base.EPS)
    return np.mean(jac*band[:,:,None],axis=1)


class Statistics:
    def __init__(self,depth=4):
        self.scatter={k:np.zeros((depth,depth)) for k in KINDS};self.count={k:0 for k in KINDS}
        self.seconds={k:0. for k in ['local','bp']};self.callbacks=0

    def add(self,name,forces):
        norms=np.linalg.norm(forces,axis=1);active=norms>1e-12
        unit=forces[active]/norms[active,None]
        self.scatter[name]+=unit.T@unit;self.count[name]+=len(unit)

    def local(self,x,b,h,u):
        begin=time.perf_counter()
        for name,force in local_forces(x,b,h,u).items():self.add(name,force)
        self.seconds['local']+=time.perf_counter()-begin;self.callbacks+=1

    def bp(self,x,v,b):
        begin=time.perf_counter();self.add('bp',bp_forces(x,v,b));self.seconds['bp']+=time.perf_counter()-begin


def capture(x,v,cfg,include_bp=True):
    previous=model.Collector
    class Instrumented(previous):
        def __init__(self,*args,**kwargs):super().__init__(*args,**kwargs);self.fiber_statistics=Statistics(len(cfg.get('anchor',np.zeros(4))))
        def activity(self,b,h,u=None,step=0,phase=''):
            super().activity(b,h,u,step,phase)
            self.fiber_statistics.local(self.x,b,h,u)
            if include_bp:self.fiber_statistics.bp(self.x,self.v,b)
    try:
        model.Collector=Instrumented
        result=model.prepare(x,v,cfg)
    finally:model.Collector=previous
    stats=Statistics()
    for collector in result[3]:
        other=collector.fiber_statistics
        for name in KINDS:stats.scatter[name]+=other.scatter[name];stats.count[name]+=other.count[name]
        for name in stats.seconds:stats.seconds[name]+=other.seconds[name]
        stats.callbacks+=other.callbacks
    return result,stats


def covariance(poly):
    d=len(poly['center']);k=d+1
    vertices=np.concatenate([poly['facets'],np.broadcast_to(poly['interior'],(len(poly['facets']),1,d))],axis=1)
    vertices=poly['center']+poly['scale']*vertices;means=vertices.mean(axis=1);centered=vertices-means[:,None,:]
    p=poly['simplex_probs']/poly['simplex_probs'].sum();mean=p@means
    within=np.einsum('ski,skj->sij',centered,centered)/(k*(k+1))
    diff=means-mean;cov=np.einsum('s,sij->ij',p,within+diff[:,:,None]*diff[:,None,:])
    return mean,(cov+cov.T)/2


def canonical(vector):
    vector=vector/np.linalg.norm(vector);index=np.argmax(np.abs(vector))
    return vector if vector[index]>=0 else -vector


def principal(matrix):
    values,vectors=np.linalg.eigh((matrix+matrix.T)/2)
    return canonical(vectors[:,-1]),float(values[-1])


def directions(cov,output_matrix,scatter):
    pca,_=principal(cov);result={f'coordinate_{j}':np.eye(len(cov))[j] for j in range(len(cov))}
    result['pca']=pca;_,sv,right=np.linalg.svd(output_matrix,full_matrices=True)
    tolerance=1e-12*max(1.,float(sv.max(initial=0.)));rank=int(np.sum(sv>tolerance))
    basis=right[rank:].T if rank<len(cov) else right[-1:, :].T
    inner,_=principal(basis.T@cov@basis);result['support_tangent']=canonical(basis@inner)
    fallback={}
    for name,kind in [('local_raw_cov','raw'),('local_augmented_cov','augmented'),('residual_cov','residual'),('bp_cov','bp'),('local_augmented_plain','augmented')]:
        matrix=scatter[kind];matrix=matrix/np.trace(matrix) if np.trace(matrix)>0 else matrix
        operator=matrix if name.endswith('_plain') else cov@matrix@cov
        maximum=float(np.max(np.abs(operator)))
        fallback[name]=maximum==0
        # Scale before eigh; otherwise small but valid covariance forces can
        # be silently treated as a numerically zero operator.
        result[name]=pca.copy() if maximum==0 else principal(operator/maximum)[0]
    assert tuple(result)==DIRECTIONS
    return result,dict(support_rank=rank,pca_fallback=fallback)


def verify():
    rng=np.random.default_rng(542197);local_checks=0;bp_checks=0;largest=0.
    x=rng.uniform(.1,.9,5);v=rng.uniform(.1,.9,5)
    for _ in range(8):
        b=rng.uniform(-.1,.1,(1,4));h=rng.uniform(.1,.9,(4,1,5));u=rng.normal(0,.2,h.shape)
        forces=local_forces(x,b,h,u)['augmented'][0]
        def energy(point):
            previous=x;total=0.
            for layer in range(4):
                residual=h[layer,0]-model.base.g(previous+point[layer]);total+=np.mean(u[layer,0]*residual+.5*residual**2);previous=h[layer,0]
            return total
        def loss(point):
            err=model.base.forward(x,point)-v;band=err-np.clip(err,-model.base.EPS,model.base.EPS);return .5*np.mean(band**2)
        bp=bp_forces(x,v,b)[0]
        for j in range(4):
            step=np.eye(4)[j]*1e-7
            a=(energy(b[0]+step)-energy(b[0]-step))/(2e-7)
            c=(loss(b[0]+step)-loss(b[0]-step))/(2e-7)
            error=max(abs(a-forces[j]),abs(c-bp[j]));assert error<1e-7
            local_checks+=1;bp_checks+=1;largest=max(largest,error)
    # Local statistics must not invoke the comparator helper.
    old=globals()['bp_forces']
    def forbidden(*a,**k):raise AssertionError('BP entered local force computation')
    try:globals()['bp_forces']=forbidden;Statistics().local(x,b,h,u)
    finally:globals()['bp_forces']=old
    import region_posterior_memory as geometry
    box,_=geometry.polytope(np.empty((0,4)),np.empty(0));mean,cov=covariance(box)
    assert np.max(np.abs(mean))<1e-12 and np.max(np.abs(cov-np.eye(4)*.12**2/3))<1e-12
    mats={k:np.eye(4) for k in KINDS};dirs,meta=directions(cov,np.array([[2.,1.,0.,0.]]),mats)
    assert abs(np.array([2.,1.,0.,0.])@dirs['support_tangent'])<1e-12
    return dict(passed=True,local_finite_differences=local_checks,bp_finite_differences=bp_checks,
                maximum_gradient_error=largest,box_full_covariance=True,local_statistics_no_bp=True)
