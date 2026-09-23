"""Matched shifted-tent population, learned closed-form and shallow controls.

All query targets live outside adaptation/prediction APIs. Outer training may
use independently sampled task query targets and is charged separately.
"""
import math
import numpy as np
import torch

STAGES=(4,8,16,24)
EPS=.001


def teacher(x,parameters):
    h=x
    for j in range(parameters.shape[-1]):
        h=(1-torch.abs(2*(h+parameters[:,j,None])-1)).clamp_min(0)
    return h


def training_batch(batch,queries,generator,device,dtype=torch.float64):
    truth=.24*torch.rand((batch,4),generator=generator,device=device,dtype=dtype)-.12
    x=torch.rand((batch,24),generator=generator,device=device,dtype=dtype)
    noise=.002*torch.rand((batch,24),generator=generator,device=device,dtype=dtype)-.001
    q=torch.rand((batch,queries),generator=generator,device=device,dtype=dtype)
    return x,teacher(x,truth)+noise,q,teacher(q,truth)


def fixed_batch(seeds,queries,device,dtype=torch.float64):
    xx=[];vv=[];qq=[];tt=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,queries)
        def forward(x):
            h=x.copy()
            for b in truth:h=np.maximum(0,1-np.abs(2*(h+b)-1))
            return h
        xx.append(x);vv.append(forward(x)+np.random.default_rng(seed+19000000).uniform(-EPS,EPS,24));qq.append(q);tt.append(forward(q))
    return tuple(torch.as_tensor(np.stack(a),device=device,dtype=dtype) for a in [xx,vv,qq,tt])


class PriorMetaRidge(torch.nn.Module):
    """Learn a kernel/mean from the exact same known 256-task prior bank.

    This is a fixed nonlinear feature model at test time, solved exactly in
    observation space. For fixed x and n its unclipped output is affine in v.
    """
    def __init__(self,rank=64,features=256,learn=True):
        super().__init__();self.learn=learn;self.rank=rank
        self.register_buffer('bank',torch.as_tensor(np.random.default_rng(731).uniform(-.12,.12,(features,4)),dtype=torch.float64))
        projection=torch.randn(features,rank,dtype=torch.float64)/math.sqrt(features) if learn else torch.eye(features,dtype=torch.float64)/math.sqrt(features)
        self.projection=torch.nn.Parameter(projection,requires_grad=learn)
        self.mean_weights=torch.nn.Parameter(torch.zeros(features,dtype=torch.float64),requires_grad=learn)
        self.log_ridge=torch.nn.Parameter(torch.full((4,),math.log(EPS**2/3),dtype=torch.float64),requires_grad=learn)

    def features(self,x):
        h=x[...,None].expand(*x.shape,len(self.bank))
        for j in range(4):h=(1-torch.abs(2*(h+self.bank[:,j])-1)).clamp_min(0)
        mu=h.mean(-1);centered=h-mu[...,None]
        return centered@self.projection,mu+centered@self.mean_weights

    def adapt(self,x,v,state=None):
        phi,mu=self.features(x);ridge=self.log_ridge[STAGES.index(x.shape[1])].exp().clamp(1e-9,.1)
        kernel=phi@phi.transpose(-2,-1)+ridge*torch.eye(x.shape[1],device=x.device,dtype=x.dtype)
        alpha=torch.linalg.solve(kernel,(v-mu)[...,None])
        return phi.transpose(-2,-1)@alpha

    def predict(self,state,q):
        phi,mu=self.features(q)
        return mu+(phi@state).squeeze(-1)


class MetaShallow(torch.nn.Module):
    """A single-hidden-layer 1D ReLU spline with learned fast initialization.

    f(x)=b+c*x+sum_j a_j*relu(x-t_j). All 2w+2 coefficients adapt.
    The affine bypass makes this at least as expressive as the usual hinge
    basis rather than arbitrarily fixing all hidden slopes to one sign.
    """
    def __init__(self,width=64,steps=5):
        super().__init__();self.width=width;self.steps=steps
        self.knots=torch.nn.Parameter(torch.linspace(0,1,width,dtype=torch.float64))
        self.amplitudes=torch.nn.Parameter(.01*torch.randn(width,dtype=torch.float64))
        self.offset=torch.nn.Parameter(torch.tensor(.5,dtype=torch.float64))
        self.slope=torch.nn.Parameter(torch.tensor(0.,dtype=torch.float64))
        # Separate positive, bounded rates for amplitudes, knots and affine part.
        self.rate_logits=torch.nn.Parameter(torch.full((4,),math.log(.01/(.2-.01)),dtype=torch.float64))

    def initial(self,batch):
        return tuple(p[None].expand(batch,*p.shape) for p in [self.amplitudes,self.knots,self.offset,self.slope])

    def predict(self,state,q):
        a,t,b,c=state
        return b[:,None]+c[:,None]*q+(a[:,None,:]*torch.relu(q[...,None]-t[:,None,:])).sum(-1)

    def adapt(self,x,v,state=None):
        tracking=torch.is_grad_enabled()
        with torch.enable_grad():
            if state is None:state=self.initial(len(x))
            if not tracking:state=tuple(p.detach().requires_grad_() for p in state)
            rates=.2*torch.sigmoid(self.rate_logits)
            for _ in range(self.steps):
                residual=(torch.abs(self.predict(state,x)-v)-EPS).clamp_min(0)
                # Sum episodes; average support within each, matching the online loss.
                loss=.5*residual.square().mean(1).sum()
                grads=torch.autograd.grad(loss,state,create_graph=tracking)
                state=tuple(p-r*g for p,r,g in zip(state,rates,grads))
        return state if tracking else tuple(p.detach() for p in state)


def make_model(config):
    if config['kind']=='ridge':return PriorMetaRidge(config['rank'])
    if config['kind']=='prior':return PriorMetaRidge(rank=256,learn=False)
    if config['kind']=='shallow':return MetaShallow(config['width'],config['inner_steps'])
    raise ValueError(config)


def trajectory_loss(model,batch,warm=True):
    x,v,q,target=batch;state=None;losses=[]
    for n in STAGES:
        state=model.adapt(x[:,:n],v[:,:n],state if warm else None)
        losses.append((model.predict(state,q)-target).square().mean())
    return torch.stack(losses).mean()


def verify():
    import streaming_branch_projection as base
    torch.manual_seed(874612);g=torch.Generator().manual_seed(974612)
    truth=torch.rand(3,4,generator=g,dtype=torch.float64)*.24-.12;x=torch.rand(3,24,generator=g,dtype=torch.float64)
    gap=float(np.max(np.abs(teacher(x,truth).numpy()-np.stack([base.forward(a,b) for a,b in zip(x.numpy(),truth.numpy())]))))
    assert gap<1e-14
    ridge=PriorMetaRidge(16);v=teacher(x,truth);q=x[:,:7]
    p=ridge.predict(ridge.adapt(x[:,:8],v[:,:8]),q)
    phi,mu=ridge.features(x[:,:8]);penalty=ridge.log_ridge[1].exp()
    beta=torch.linalg.solve(phi.transpose(-2,-1)@phi+penalty*torch.eye(16,dtype=x.dtype),phi.transpose(-2,-1)@(v[:,:8]-mu)[...,None])
    dual_gap=float((p-ridge.predict(beta,q)).abs().max().detach());assert dual_gap<1e-7
    mixed=.3*v[:,:8]+.7*(v[:,:8]+.1)
    expected=.3*p+.7*ridge.predict(ridge.adapt(x[:,:8],v[:,:8]+.1),q)
    affine_gap=float((ridge.predict(ridge.adapt(x[:,:8],mixed),q)-expected).abs().max().detach());assert affine_gap<1e-7
    shallow=MetaShallow(width=8,steps=2)
    batch=training_batch(3,7,g,'cpu');loss=trajectory_loss(shallow,batch);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in shallow.parameters())
    with torch.no_grad():
        bb=shallow.adapt(x[:,:8],v[:,:8]);single=torch.cat([shallow.predict(shallow.adapt(x[i:i+1,:8],v[i:i+1,:8]),q[i:i+1]) for i in range(3)])
        batch_gap=float((single-shallow.predict(bb,q)).abs().max());assert batch_gap<1e-12
    return dict(passed=True,teacher_numpy_max_error=gap,ridge_primal_dual_max_error=dual_gap,ridge_affinity_max_error=affine_gap,
                shallow_batched_individual_max_error=batch_gap,full_second_order_outer_gradients_finite=True,
                query_target_absent_from_adapt_and_predict=True)
