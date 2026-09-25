"""Joint affine/nonlinear constraint splitting for coupled vector fast memories.

Development-only prototype, with exact local branch solves. Queries and latent
teacher parameters are never arguments to fitting routines.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize

BETA=1e-6
BOUND=.3
SHIFT=.2


def activation(z):
    return np.maximum(-1.,1.-2.*np.abs(z))


def deriv(z):
    return np.where((z>-1)&(z<0),2.,np.where((z>0)&(z<1),-2.,0.))


def make_weights(depth,width):
    rng=np.random.default_rng(210)
    return np.stack([np.linalg.qr(rng.normal(size=(width,width)))[0] for _ in range(depth)])


def forward(x,b,weights):
    h=x
    for bias,w in zip(b,weights):
        h=activation(h@w.T+bias)
    return h


def objective_gradient(flat,x,v,weights):
    depth,width=weights.shape[:2]
    b=flat.reshape(depth,width)
    h=x
    zs=[]
    for bias,w in zip(b,weights):
        z=h@w.T+bias
        zs.append(z)
        h=activation(z)
    error=h-v
    loss=.5*np.mean(np.sum(error**2,axis=1))+.5*BETA*np.sum(b**2)
    adjoint=error/len(x)
    grad=np.zeros_like(b)
    for layer in reversed(range(depth)):
        adjoint=adjoint*deriv(zs[layer])
        grad[layer]=adjoint.sum(axis=0)+BETA*b[layer]
        adjoint=adjoint@weights[layer]
    return float(loss),grad.ravel()


def nonlinear_prox(a,t,old,trust):
    slope=np.array([0.,2.,-2.,0.]).reshape(4,1,1,1)
    intercept=np.array([-1.,1.,1.,-1.]).reshape(4,1,1,1)
    lo=np.array([-np.inf,-1.,0.,1.]).reshape(4,1,1,1)
    hi=np.array([-1.,0.,1.,np.inf]).reshape(4,1,1,1)
    candidates=np.clip((a[None]+slope*(t[None]-intercept)+trust*old[None])/(1+slope**2+trust),lo,hi)
    energy=(candidates-a)**2+(activation(candidates)-t)**2+trust*(candidates-old)**2
    return np.take_along_axis(candidates,np.argmin(energy,axis=0)[None],axis=0)[0]


def fit_split(x,v,weights,*,sweeps=200,rho=.1,dual_rate=.5,trust=.01,restarts=1,gradient_nonlinearity=False):
    depth,width=weights.shape[:2]
    rng=np.random.default_rng(912)
    b=np.concatenate([np.zeros((1,depth,width)),rng.uniform(-BOUND,BOUND,(restarts-1,depth,width))])
    z=np.empty((depth,restarts,len(x),width))
    h=np.empty_like(z)
    up=np.zeros_like(z)
    ua=np.zeros_like(z)
    previous=np.broadcast_to(x,(restarts,*x.shape))
    for layer in range(depth):
        z[layer]=previous@weights[layer].T+b[:,layer,None,:]
        h[layer]=activation(z[layer])
        previous=h[layer]
    def true_losses(params):
        current=np.broadcast_to(x,(restarts,*x.shape))
        for layer,w in enumerate(weights):
            current=activation(current@w.T+params[:,layer,None,:])
        return .5*np.mean(np.sum((current-v)**2,axis=-1),axis=1)+.5*BETA*np.sum(params**2,axis=(1,2))
    best_b=b.copy()
    best_loss=true_losses(b)
    for _ in range(sweeps):
        # Joint local block sweep, with both kinds of consistency residuals.
        for layer in reversed(range(depth)):
            a=activation(z[layer])-ua[layer]
            if layer==depth-1:
                h[layer]=(v+rho*a+rho*trust*h[layer])/(1+rho+rho*trust)
            else:
                t=z[layer+1]-b[:,layer+1,None,:]+up[layer+1]
                # W.T @ W=I, so this is the exact h block quadratic minimizer.
                h[layer]=(a+t@weights[layer+1]+trust*h[layer])/(2+trust)
            previous=x if layer==0 else h[layer-1]
            center=previous@weights[layer].T+b[:,layer,None,:]-up[layer]
            target=h[layer]+ua[layer]
            if gradient_nonlinearity:
                dz=z[layer]-center+deriv(z[layer])*(activation(z[layer])-target)
                z[layer]=z[layer]-dz/(5+trust)
            else:
                z[layer]=nonlinear_prox(center,target,z[layer],trust)
        for layer,w in enumerate(weights):
            previous=x if layer==0 else h[layer-1]
            b[:,layer]=np.clip(np.mean(z[layer]-previous@w.T+up[layer],axis=1)/(1+BETA/rho),-BOUND,BOUND)
        for layer,w in enumerate(weights):
            previous=x if layer==0 else h[layer-1]
            up[layer]+=dual_rate*(z[layer]-previous@w.T-b[:,layer,None,:])
            ua[layer]+=dual_rate*(h[layer]-activation(z[layer]))
        current=true_losses(b)
        better=current<best_loss
        best_b[better]=b[better]
        best_loss[better]=current[better]
    winner=int(np.argmin(best_loss))
    return best_b[winner],{"support_objective":float(best_loss[winner]),"winning_restart":winner,
                           "persistent_scalars":int(4*z.size+b.size),"sweeps":sweeps}


def fit_adam(x,v,weights,*,steps=500,lr=.01,restarts=1):
    depth,width=weights.shape[:2]
    rng=np.random.default_rng(912)
    best_b=np.zeros((depth,width))
    best_loss=objective_gradient(best_b.ravel(),x,v,weights)[0]
    for restart in range(restarts):
        b=np.zeros(depth*width) if restart==0 else rng.uniform(-BOUND,BOUND,depth*width)
        m=np.zeros_like(b)
        vsq=np.zeros_like(b)
        for step in range(1,steps+1):
            loss,grad=objective_gradient(b,x,v,weights)
            if loss<best_loss:
                best_b,best_loss=b.copy().reshape(depth,width),loss
            m=.9*m+.1*grad
            vsq=.999*vsq+.001*grad**2
            b=np.clip(b-lr*(m/(1-.9**step))/(np.sqrt(vsq/(1-.999**step))+1e-8),-BOUND,BOUND)
    return best_b,{"support_objective":float(best_loss),"persistent_scalars":3*depth*width}


def fit_lbfgs(x,v,weights,*,steps=300,restarts=4):
    depth,width=weights.shape[:2]
    rng=np.random.default_rng(912)
    best_b=np.zeros((depth,width))
    best_loss=objective_gradient(best_b.ravel(),x,v,weights)[0]
    calls=0
    for restart in range(restarts):
        b=np.zeros(depth*width) if restart==0 else rng.uniform(-BOUND,BOUND,depth*width)
        result=minimize(objective_gradient,b,args=(x,v,weights),jac=True,method="L-BFGS-B",bounds=[(-BOUND,BOUND)]*len(b),
                        options={"maxiter":steps,"maxfun":steps*3,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        calls+=result.nfev
        if result.fun<best_loss:
            best_b,best_loss=result.x.reshape(depth,width).copy(),result.fun
    return best_b,{"support_objective":float(best_loss),"function_gradient_calls":calls}


def data(seed,weights,n=24,nq=1024):
    rng=np.random.default_rng(seed)
    b=rng.uniform(-SHIFT,SHIFT,weights.shape[:2])
    x=rng.uniform(-1,1,(n,weights.shape[1]))
    q=rng.uniform(-1,1,(nq,weights.shape[1]))
    return x,forward(x,b,weights),q,forward(q,b,weights)


def verify():
    rng=np.random.default_rng(807)
    a,t,old=rng.uniform(-1.5,1.5,(3,1,4,3))
    exact=nonlinear_prox(a,t,old,.01)
    grid=np.linspace(-3,3,100001)
    for i in range(4):
        for j in range(3):
            def loss(v):
                return (v-a[0,i,j])**2+(activation(v)-t[0,i,j])**2+.01*(v-old[0,i,j])**2
            assert loss(exact[0,i,j])<=np.min(loss(grid))+1e-8
    weights=make_weights(3,4)
    assert np.max(np.abs(weights@weights.transpose(0,2,1)-np.eye(4)))<1e-12
    x,v,_,_=data(870,weights)
    b=rng.uniform(-.25,.25,12)
    analytic=objective_gradient(b,x,v,weights)[1]
    numerical=np.array([(objective_gradient(b+np.eye(12)[i]*1e-7,x,v,weights)[0]-objective_gradient(b-np.eye(12)[i]*1e-7,x,v,weights)[0])/2e-7 for i in range(12)])
    err=np.max(np.abs(analytic-numerical))
    assert err<1e-6,err
    return {"passed":True,"branch_grid_cases":12,"gradient_max_abs_error":float(err)}


def configurations():
    configs=[]
    for rho in [.03,.1,.3,1.0]:
        for rate in [0.,.1,.5]:
            configs.append({"method":"split","sweeps":240,"rho":rho,"dual_rate":rate,"restarts":1})
    for lr in [.001,.003,.01,.03]:
        configs.append({"method":"adam","steps":1000,"lr":lr,"restarts":1})
    for restarts in [1,4,16]:
        configs.append({"method":"lbfgs","steps":300,"restarts":restarts})
    return configs


def main():
    global BOUND,SHIFT
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--depth",type=int,default=3)
    parser.add_argument("--width",type=int,default=8)
    parser.add_argument("--count",type=int,default=12)
    parser.add_argument("--shift",type=float,default=.2)
    parser.add_argument("--config-file",type=Path)
    parser.add_argument("--confirm",action="store_true")
    args=parser.parse_args()
    SHIFT=args.shift
    BOUND=1.5*SHIFT
    args.out.mkdir(parents=True,exist_ok=True)
    checks=verify()
    print(json.dumps(checks),flush=True)
    (args.out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    weights=make_weights(args.depth,args.width)
    cfgs=configurations() if args.config_file is None else json.loads(args.config_file.read_text(encoding="utf-8"))["configs"]
    if args.confirm and args.config_file is None:
        raise ValueError("Freeze configurations before confirmation")
    seed0=5500000 if args.confirm else 5400000
    protocol={"phase":"confirmation" if args.confirm else "development","seed0":seed0,"count":args.count,
              "depth":args.depth,"width":args.width,"n_context":24,"n_query":1024,"configs":cfgs,
              "shift":SHIFT,"bound":BOUND,
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    summary=[]
    for ci,cfg in enumerate(cfgs):
        options={k:val for k,val in cfg.items() if k!="method"}
        fit={"split":fit_split,"adam":fit_adam,"lbfgs":fit_lbfgs}[cfg["method"]]
        for seed in range(seed0,seed0+args.count):
            x,v,q,target=data(seed,weights)
            start=time.perf_counter()
            b,metadata=fit(x,v,weights,**options)
            seconds=time.perf_counter()-start
            rows.append({"config_index":ci,"seed":seed,"config":cfg,"query_mse":float(np.mean((forward(q,b,weights)-target)**2)),
                         "seconds":seconds,"b":b.tolist(),**metadata})
        selected=rows[-args.count:]
        stats={"config":cfg,"mean_query_mse":float(np.mean([r["query_mse"] for r in selected])),
               "median_seconds":float(np.median([r["seconds"] for r in selected]))}
        summary.append(stats)
        print(json.dumps(stats),flush=True)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
