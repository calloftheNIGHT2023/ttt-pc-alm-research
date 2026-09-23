"""Support-only piecewise local ALM prototype; all labels for queries stay in evaluator.

This is a new block-solve variant, NOT the official PC-ALM or official TTT update.
Reference primal/dual sign convention follows the pinned official PC-ALM source.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

KNOTS = np.array([0.0, 0.5, 1.0])
SLOPES = np.array([0.0, 2.0, -2.0, 0.0])
INTERCEPTS = np.array([0.0, 0.0, 2.0, 0.0])
BOUND = 0.15
BETA = 1e-6


def g(z):
    return np.maximum(0.0, 1.0 - np.abs(2.0 * z - 1.0))


def derivative(z):
    return np.where((z > 0) & (z < 0.5), 2.0,
                    np.where((z > 0.5) & (z < 1), -2.0, 0.0))


def forward(x, b):
    out = np.asarray(x, dtype=float)
    for bias in b:
        out = g(out + bias)
    return out


def objective_gradient(b, x, v):
    h = x.copy()
    jac = np.zeros((x.size, len(b)))
    for layer, bias in enumerate(b):
        d = derivative(h + bias)
        jac *= d[:, None]
        jac[:, layer] += d
        h = g(h + bias)
    error = h - v
    return (0.5 * np.mean(error**2) + 0.5 * BETA * np.dot(b, b),
            jac.T @ error / x.size + BETA * b)


def objective(b, x, v):
    return 0.5 * np.mean((forward(x, b) - v)**2) + 0.5 * BETA * np.dot(b, b)


def activity_argmin(a, target, next_bias, old, trust):
    """Exact global scalar block solution across four affine pieces, vectorized by sample."""
    lo = np.maximum(0.0, np.array([-np.inf, 0.0, 0.5, 1.0]) - next_bias)
    hi = np.minimum(1.0, np.array([0.0, 0.5, 1.0, np.inf]) - next_bias)
    s = SLOPES[:, None]
    c = (SLOPES * next_bias + INTERCEPTS)[:, None]
    candidates = (a[None, :] + s * (target[None, :] - c) + trust * old[None, :]) / (1+s*s+trust)
    candidates = np.minimum(np.maximum(candidates, lo[:, None]), hi[:, None])
    values = ((candidates-a)**2 + (g(candidates+next_bias)-target)**2
              + trust*(candidates-old)**2)
    values[lo > hi, :] = np.inf
    return candidates[np.argmin(values, axis=0), np.arange(len(a))]


def bias_argmin(previous, target, old, rho, trust):
    breaks = np.unique(np.clip(np.r_[-BOUND, BOUND, (KNOTS[:, None]-previous).ravel()], -BOUND, BOUND))
    lo, hi = breaks[:-1], breaks[1:]
    mid = (lo+hi)/2
    z = previous[None, :] + mid[:, None]
    region = np.searchsorted(KNOTS, z, side="right")
    s = SLOPES[region]
    c = s * previous + INTERCEPTS[region]
    den = np.mean(s*s, axis=1) + BETA/rho + trust
    numer = np.mean(s*(target-c), axis=1) + trust*old
    candidates = np.divide(numer, den, out=np.full_like(den, old), where=den>0)
    candidates = np.clip(candidates, lo, hi)
    values = (np.mean((g(previous[None, :]+candidates[:, None])-target)**2, axis=1)
              + BETA/rho*candidates**2 + trust*(candidates-old)**2)
    return float(candidates[np.argmin(values)])


def bias_batch_sweep(previous, target, old, rho, trust):
    """Exact O(R*n*log(n)) solve via sorted changes to quadratic coefficients.

    Each sample changes affine branch at only 3 points. Prefix sums propagate
    A*b^2 - 2*B*b + C without evaluating every sample in every interval.
    """
    restarts,n=previous.shape
    initial_region=np.searchsorted(KNOTS,previous-BOUND,side="right")
    s0=SLOPES[initial_region]
    c0=s0*previous+INTERCEPTS[initial_region]
    aa0=np.sum(s0*s0,axis=1)
    bb0=np.sum(s0*(target-c0),axis=1)
    cc0=np.sum((c0-target)**2,axis=1)
    events=(KNOTS[None,:,None]-previous[:,None,:]).reshape(restarts,-1)
    before_s=SLOPES[:-1][None,:,None]
    after_s=SLOPES[1:][None,:,None]
    before_c=before_s*previous[:,None,:]+INTERCEPTS[:-1][None,:,None]
    after_c=after_s*previous[:,None,:]+INTERCEPTS[1:][None,:,None]
    da=np.broadcast_to(after_s**2-before_s**2,(restarts,3,n)).reshape(restarts,-1)
    db=(after_s*(target[:,None,:]-after_c)-before_s*(target[:,None,:]-before_c)).reshape(restarts,-1)
    dc=((after_c-target[:,None,:])**2-(before_c-target[:,None,:])**2).reshape(restarts,-1)
    interior=(events>-BOUND)&(events<BOUND)
    positions=np.clip(events,-BOUND,BOUND)
    order=np.argsort(positions,axis=1,kind="stable")
    positions=np.take_along_axis(positions,order,axis=1)
    coefs=[]
    for initial,delta in [(aa0,da),(bb0,db),(cc0,dc)]:
        ordered=np.take_along_axis(np.where(interior,delta,0),order,axis=1)
        coefs.append(np.concatenate([initial[:,None],initial[:,None]+np.cumsum(ordered,axis=1)],axis=1)/n)
    aa,bb,cc=coefs
    lo=np.concatenate([np.full((restarts,1),-BOUND),positions],axis=1)
    hi=np.concatenate([positions,np.full((restarts,1),BOUND)],axis=1)
    candidates=np.clip((bb+trust*old[:,None])/(np.maximum(aa,0)+BETA/rho+trust),lo,hi)
    values=(aa+BETA/rho)*candidates**2-2*bb*candidates+cc+trust*(candidates-old[:,None])**2
    return candidates[np.arange(restarts),np.argmin(values,axis=1)]


def fit_local(x, v, depth, *, sweeps=120, rho=0.1, dual_rate=0.1, trust=0.01,
              rho_growth=1.0, rho_period=40, reverse_weights=False):
    b = np.zeros(depth)
    h = np.empty((depth, len(x)))
    previous = x
    for layer in range(depth):
        h[layer] = g(previous)
        previous = h[layer]
    u = np.zeros_like(h)
    best_b, best_value = b.copy(), objective(b, x, v)
    best_iteration = 0
    switches = 0
    for it in range(sweeps):
        for layer in reversed(range(depth)):
            previous = x if layer == 0 else h[layer-1]
            a = g(previous+b[layer])-u[layer]
            old = h[layer].copy()
            if layer == depth-1:
                h[layer] = np.clip((v+rho*a+rho*trust*old)/(1+rho+rho*trust), 0, 1)
            else:
                h[layer] = activity_argmin(a, h[layer+1]+u[layer+1], b[layer+1], old, trust)
                switches += int(np.sum(np.searchsorted(KNOTS, old+b[layer+1]) !=
                                       np.searchsorted(KNOTS, h[layer]+b[layer+1])))
        for layer in (reversed(range(depth)) if reverse_weights else range(depth)):
            previous = x if layer == 0 else h[layer-1]
            b[layer] = bias_argmin(previous, h[layer]+u[layer], b[layer], rho, trust)
        previous = x
        residual = np.empty_like(h)
        for layer in range(depth):
            residual[layer] = h[layer]-g(previous+b[layer])
            previous = h[layer]
        u += dual_rate * residual
        value = objective(b, x, v)
        if value < best_value:
            best_b, best_value, best_iteration = b.copy(), value, it+1
        if (it+1) % rho_period == 0 and rho_growth != 1.0:
            new_rho = min(3.0, rho*rho_growth)
            u *= rho/new_rho  # Preserve the unscaled dual lambda when rho changes.
            rho = new_rho
    return best_b, {"support_objective":float(best_value), "best_iteration":best_iteration,
                    "last_residual_rms":float(np.sqrt(np.mean(residual**2))),
                    "activity_branch_switches":switches,
                    "persistent_scalars":int(2*h.size+depth), "iterations":sweeps}


def fit_adam(x, v, depth, *, steps=500, lr=0.01, restarts=1):
    rng = np.random.default_rng(912)
    best_b = np.zeros(depth)
    best_value = objective(best_b, x, v)
    for restart in range(restarts):
        b = np.zeros(depth) if restart == 0 else rng.uniform(-BOUND, BOUND, depth)
        m, vsq = np.zeros(depth), np.zeros(depth)
        for it in range(1, steps+1):
            value, grad = objective_gradient(b, x, v)
            if value < best_value:
                best_value, best_b = value, b.copy()
            m = 0.9*m + 0.1*grad
            vsq = 0.999*vsq + 0.001*grad**2
            b = np.clip(b-lr*(m/(1-0.9**it))/(np.sqrt(vsq/(1-0.999**it))+1e-8), -BOUND, BOUND)
        value = objective(b, x, v)
        if value < best_value:
            best_value, best_b = value, b.copy()
    return best_b, {"support_objective":float(best_value), "iterations":steps*restarts,
                    "persistent_scalars":3*depth}


def fit_batched_local(x, v, depth, *, sweeps=120, rho=0.03, dual_rate=0.5,
                      trust=0.01, restarts=8, gradient_blocks=False):
    """Vectorize identical independent restart blocks; no extra observations or labels."""
    rng = np.random.default_rng(912)
    b = np.vstack([np.zeros(depth),rng.uniform(-BOUND,BOUND,(restarts-1,depth))])
    h=np.empty((depth,restarts,len(x)))
    u=np.zeros_like(h)
    previous=np.broadcast_to(x,(restarts,len(x)))
    for layer in range(depth):
        h[layer]=g(previous+b[:,layer,None])
        previous=h[layer]
    def values(params):
        out=np.broadcast_to(x,(restarts,len(x)))
        for j in range(depth):
            out=g(out+params[:,j,None])
        return .5*np.mean((out-v)**2,axis=1)+.5*BETA*np.sum(params**2,axis=1)
    best_b=b.copy()
    best_value=values(b)
    for _ in range(sweeps):
        for layer in reversed(range(depth)):
            previous=x if layer==0 else h[layer-1]
            a=g(previous+b[:,layer,None])-u[layer]
            old=h[layer].copy()
            if layer==depth-1:
                h[layer]=np.clip((v+rho*a+rho*trust*old)/(1+rho+rho*trust),0,1)
            elif gradient_blocks:
                delta=old-a+derivative(old+b[:,layer+1,None])*(g(old+b[:,layer+1,None])-h[layer+1]-u[layer+1])
                h[layer]=np.clip(old-delta/(5+trust),0,1)
            else:
                next_b=b[:,layer+1]
                lo=np.maximum(0,np.array([-np.inf,0,.5,1.])[:,None]-next_b[None,:])
                hi=np.minimum(1,np.array([0,.5,1.,np.inf])[:,None]-next_b[None,:])
                s=SLOPES[:,None,None]
                c=(SLOPES[:,None]*next_b[None,:]+INTERCEPTS[:,None])[:,:,None]
                t=h[layer+1]+u[layer+1]
                candidates=(a[None,:,:]+s*(t[None,:,:]-c)+trust*old[None,:,:])/(1+s*s+trust)
                candidates=np.minimum(np.maximum(candidates,lo[:,:,None]),hi[:,:,None])
                costs=(candidates-a)**2+(g(candidates+next_b[None,:,None])-t)**2+trust*(candidates-old)**2
                costs=np.where((lo>hi)[:,:,None],np.inf,costs)
                indices=np.argmin(costs,axis=0)
                h[layer]=np.take_along_axis(candidates,indices[None,:,:],axis=0)[0]
        for layer in range(depth):
            previous=np.broadcast_to(x,(restarts,len(x))) if layer==0 else h[layer-1]
            target=h[layer]+u[layer]
            old=b[:,layer].copy()
            if gradient_blocks:
                z=previous+old[:,None]
                grad=np.mean((g(z)-target)*derivative(z),axis=1)+BETA/rho*old
                b[:,layer]=np.clip(old-grad/(4+BETA/rho+trust),-BOUND,BOUND)
            else:
                b[:,layer]=bias_batch_sweep(previous,target,old,rho,trust)
        previous=x
        residual=np.empty_like(h)
        for layer in range(depth):
            residual[layer]=h[layer]-g(previous+b[:,layer,None])
            previous=h[layer]
        u+=dual_rate*residual
        current=values(b)
        better=current<best_value
        best_value[better]=current[better]
        best_b[better]=b[better]
    winner=int(np.argmin(best_value))
    return best_b[winner], {"support_objective":float(best_value[winner]),"winning_restart":winner,
                           "persistent_scalars":int(2*h.size+b.size),"iterations":sweeps,
                           "total_restart_sweeps":sweeps*restarts,
                           "last_residual_rms":float(np.sqrt(np.mean(residual[:,winner]**2)))}


def fit_lbfgs(x, v, depth, *, steps=300, restarts=4):
    rng = np.random.default_rng(912)
    best_b, best_value = np.zeros(depth), objective(np.zeros(depth), x, v)
    calls = 0
    for restart in range(restarts):
        b0 = np.zeros(depth) if restart == 0 else rng.uniform(-BOUND, BOUND, depth)
        result = minimize(objective_gradient, b0, args=(x, v), jac=True, method="L-BFGS-B",
                          bounds=[(-BOUND, BOUND)]*depth,
                          options={"maxiter":steps,"maxfun":steps*3,"ftol":1e-13,"gtol":1e-9,"maxls":40})
        calls += result.nfev
        if result.fun < best_value:
            best_value, best_b = result.fun, result.x.copy()
    return best_b, {"support_objective":float(best_value), "function_gradient_calls":calls}


def fit_regression(x, v, depth, *, family="rbf", features=256):
    """Use only support LOOCV for ridge; the prior bank uses the same known task family."""
    if family == "linear":
        phi=np.column_stack([np.ones_like(x),x])
        coef=np.linalg.lstsq(phi,v,rcond=None)[0]
        return lambda q:np.column_stack([np.ones_like(q),q])@coef,{"persistent_scalars":2}
    if family == "prior_bank":
        bank=np.random.default_rng(731).uniform(-.12,.12,(features,depth))
        def features_for(inputs):
            out=np.broadcast_to(inputs,(features,len(inputs)))
            for layer in range(depth):
                out=g(out+bank[:,layer,None])
            return out.T
        phi=features_for(x)
        feature_mean=phi.mean(axis=0)
        phi=(phi-feature_mean)/np.sqrt(features)
        kernels=[(None,phi@phi.T)]
    elif family == "rbf":
        kernels=[(width,np.exp(-.5*((x[:,None]-x[None,:])/width)**2)) for width in [.01,.025,.05,.1,.2,.5]]
    else:
        raise ValueError(family)
    best=None
    for width,kernel in kernels:
        # Include the constant function as an unpenalized intercept via centering.
        center=np.eye(len(x))-np.ones((len(x),len(x)))/len(x)
        centered=center@kernel@center
        eig,vec=np.linalg.eigh(centered)
        eig=np.maximum(eig,0)
        for ridge in [1e-6,1e-4,.01,.1,1.0,10.0]:
            shrink=eig/(eig+ridge)
            hat=(vec*shrink)@vec.T+np.ones((len(x),len(x)))/len(x)
            loo=(v-hat@v)/np.maximum(1-np.diag(hat),1e-10)
            score=np.mean(loo**2)
            if best is None or score<best[0]:
                coef=np.linalg.solve(centered+ridge*np.eye(len(x)),v-v.mean())
                best=(score,width,ridge,coef,kernel.mean(axis=0),float(kernel.mean()))
    _,width,ridge,coef,column_mean,total_mean=best
    def predict(q):
        if family=="prior_bank":
            cross=((features_for(q)-feature_mean)/np.sqrt(features))@phi.T
        else:
            cross=np.exp(-.5*((q[:,None]-x[None,:])/width)**2)
        cross=cross-cross.mean(axis=1)[:,None]-column_mean[None,:]+total_mean
        return cross@coef+v.mean()
    return predict,{"ridge":ridge,"width":width,"persistent_scalars":int(len(x)**2+len(x)+(features*depth if family=="prior_bank" else len(x)))}


def fit_shallow(x,v,depth,*,width=12,steps=1000,lr=.01,restarts=4):
    rng=np.random.default_rng(912)
    best_loss=np.inf
    best_params=None
    for restart in range(restarts):
        params=np.r_[v.mean(),0.,rng.normal(0,.03,width),np.linspace(0,1,width+2)[1:-1]+rng.normal(0,.01,width)]
        m=np.zeros_like(params)
        vsq=np.zeros_like(params)
        for step in range(1,steps+1):
            a0,a1=params[:2]
            weights=params[2:width+2]
            knots=params[width+2:]
            hinge=np.maximum(x[:,None]-knots[None,:],0)
            error=a0+a1*x+hinge@weights-v
            loss=.5*np.mean(error**2)+.5*BETA*np.sum(params[:width+2]**2)
            if loss<best_loss:
                best_loss,best_params=loss,params.copy()
            grad=np.r_[error.mean(),np.mean(error*x),hinge.T@error/len(x),
                       -np.mean(error[:,None]*(x[:,None]>knots)*weights,axis=0)]
            grad[:width+2]+=BETA*params[:width+2]
            m=.9*m+.1*grad
            vsq=.999*vsq+.001*grad**2
            params-=lr*(m/(1-.9**step))/(np.sqrt(vsq/(1-.999**step))+1e-8)
            params[width+2:]=np.clip(params[width+2:],0,1)
    def predict(q):
        return best_params[0]+best_params[1]*q+np.maximum(q[:,None]-best_params[width+2:][None,:],0)@best_params[2:width+2]
    return predict,{"support_objective":float(best_loss),"persistent_scalars":int(3*(2+2*width)),"iterations":steps*restarts}


def episode(seed, depth, n=24, nq=2048):
    rng = np.random.default_rng(seed)
    truth = rng.uniform(-0.12, 0.12, depth)
    x, q = rng.uniform(0, 1, n), rng.uniform(0, 1, nq)
    return x, forward(x, truth), q, forward(q, truth)


def configs():
    out = []
    for rho in [0.03, 0.1, 0.3, 1.0]:
        for rate in [0.0, 0.1, 0.5]:
            out.append({"method":"local", "rho":rho, "dual_rate":rate,"sweeps":120,"trust":0.01})
    for lr in [0.003,0.01,0.03]:
        out.append({"method":"adam","lr":lr,"steps":500,"restarts":1})
    out.append({"method":"lbfgs","steps":300,"restarts":4})
    return out


def run_one(cfg, x, v, depth):
    options = {k: val for k,val in cfg.items() if k != "method"}
    if cfg["method"] == "local":
        return fit_local(x,v,depth,**options)
    if cfg["method"] == "adam":
        return fit_adam(x,v,depth,**options)
    if cfg["method"] == "lbfgs":
        return fit_lbfgs(x,v,depth,**options)
    if cfg["method"] == "batched_local":
        return fit_batched_local(x,v,depth,**options)
    raise ValueError(cfg)


def verify():
    rng = np.random.default_rng(882)
    grid = np.linspace(0,1,100001)
    max_gap = -np.inf
    for _ in range(30):
        a,t,old = rng.uniform(-0.2,1.2,3)
        bias = rng.uniform(-BOUND,BOUND)
        value = activity_argmin(np.array([a]),np.array([t]),bias,np.array([old]),0.01)[0]
        loss=lambda h:(h-a)**2+(g(h+bias)-t)**2+0.01*(h-old)**2
        gap=float(loss(value)-np.min(loss(grid)))
        assert gap < 1e-9, gap
        max_gap=max(max_gap,gap)
    grid_b=np.linspace(-BOUND,BOUND,100001)
    for _ in range(8):
        x,t=rng.uniform(0,1,5),rng.uniform(-.2,1.2,5)
        value=bias_argmin(x,t,0,0.1,0.01)
        loss=lambda b:np.mean((g(x[None,:]+b[:,None])-t)**2,axis=1)+(BETA/.1+.01)*b*b
        assert loss(np.array([value]))[0] <= np.min(loss(grid_b))+1e-9
    previous=rng.uniform(0,1,(40,24))
    target=rng.uniform(-.5,1.5,(40,24))
    old=rng.uniform(-BOUND,BOUND,40)
    swept=bias_batch_sweep(previous,target,old,.03,.01)
    reference=np.array([bias_argmin(previous[j],target[j],old[j],.03,.01) for j in range(40)])
    assert np.max(np.abs(swept-reference)) < 1e-10
    x,v,_,_=episode(882,3)
    b=rng.uniform(-.1,.1,3)
    analytic=objective_gradient(b,x,v)[1]
    numerical=np.array([(objective(b+np.eye(3)[i]*1e-7,x,v)-objective(b-np.eye(3)[i]*1e-7,x,v))/2e-7 for i in range(3)])
    assert np.max(np.abs(analytic-numerical)) < 1e-6
    # Exact branch escape: derivative is zero at z=-0.2, but global block solution is positive.
    # Here h in [0,1], next_bias=-0.3, a=old=0.1 and target=1.
    escaped=activity_argmin(np.array([.1]),np.array([1.]),-.3,np.array([.1]),.01)[0]
    assert escaped > .3
    scalar,_=fit_local(x,v,3,sweeps=12,rho=.03,dual_rate=.5)
    batched,_=fit_batched_local(x,v,3,sweeps=12,restarts=1)
    assert np.max(np.abs(scalar-batched)) < 1e-10
    return {"activity_grid_cases":30,"bias_grid_cases":8,"gradient_max_abs_error":float(np.max(np.abs(analytic-numerical))),
            "branch_escape_from":.1,"branch_escape_to":float(escaped),
            "sorted_sweep_cases":40,"sorted_sweep_max_abs_error":float(np.max(np.abs(swept-reference))),"passed":True}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--mode",choices=["verify","screen","confirm"],default="verify")
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--depth",type=int,default=3)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--frozen",type=Path)
    parser.add_argument("--config-file",type=Path)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    checks=verify()
    (args.out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks),flush=True)
    if args.mode=="verify":
        return
    if args.mode=="screen":
        cfgs=configs() if args.config_file is None else json.loads(args.config_file.read_text(encoding="utf-8"))["configs"]
        seed0=5200000
    else:
        if args.frozen is None:
            raise ValueError("Confirmation requires a previously frozen config file")
        cfgs=json.loads(args.frozen.read_text(encoding="utf-8"))["configs"]
        seed0=5300000
    protocol={"mode":args.mode,"depth":args.depth,"n_context":24,"n_query":2048,"seed0":seed0,
              "count":args.count,"configs":cfgs,"source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for ci,cfg in enumerate(cfgs):
        for seed in range(seed0,seed0+args.count):
            x,v,q,target=episode(seed,args.depth)
            start=time.perf_counter()
            options={k:val for k,val in cfg.items() if k != "method"}
            if cfg["method"]=="regression":
                predict,diagnostics=fit_regression(x,v,args.depth,**options)
                b=[]
            elif cfg["method"]=="shallow":
                predict,diagnostics=fit_shallow(x,v,args.depth,**options)
                b=[]
            else:
                b,diagnostics=run_one(cfg,x,v,args.depth)
                predict=lambda inputs:forward(inputs,b)
            seconds=time.perf_counter()-start
            rows.append({"config_index":ci,"config":cfg,"seed":seed,"query_mse":float(np.mean((predict(q)-target)**2)),
                         "seconds":seconds,"b":np.asarray(b).tolist(),**diagnostics})
        selected=rows[-args.count:]
        print(json.dumps({"config":cfg,"mean_query_mse":float(np.mean([r["query_mse"] for r in selected])),
                          "median_seconds":float(np.median([r["seconds"] for r in selected]))}),flush=True)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    summaries=[]
    for ci,cfg in enumerate(cfgs):
        selected=[r for r in rows if r["config_index"]==ci]
        errors=np.array([r["query_mse"] for r in selected])
        summaries.append({"config":cfg,"mean_query_mse":float(np.mean(errors)),"median_query_mse":float(np.median(errors)),
                          "p90_query_mse":float(np.quantile(errors,.9)),"median_seconds":float(np.median([r["seconds"] for r in selected]))})
    (args.out/"summary.json").write_text(json.dumps(summaries,indent=2),encoding="utf-8")
    if args.mode=="screen":
        groups=[lambda c:c["method"] in {"local","batched_local"} and c.get("dual_rate",.5)>0,
                lambda c:c["method"] in {"local","batched_local"} and c.get("dual_rate",.5)==0,
                lambda c:c["method"]=="adam",lambda c:c["method"]=="lbfgs"]
        chosen=[]
        for predicate in groups:
            eligible=[s for s in summaries if predicate(s["config"])]
            if eligible:
                chosen.append(min(eligible,key=lambda s:s["mean_query_mse"])["config"])
        # Also force a rho-matched dual ablation, regardless of its validation score.
        if chosen and chosen[0]["method"] in {"local","batched_local"}:
            chosen.append({**chosen[0],"dual_rate":0.0})
        (args.out/"frozen_configs.json").write_text(json.dumps({"configs":chosen,"selection":"validation query mean; plus exact rho-matched dual ablation"},indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
