"""NumPy reference for frozen regression/meta heads; no query targets.

Separate arithmetic and explicit band-loss gradients, not model.adapt/predict.
Checkpoint values and initial fast parameters are the same frozen prior.
"""
import numpy as np
from analyze_recovered_online_comparison import forward,regression_replay


def values(tensor):return tensor.detach().numpy().copy()


def replay(cfg,x,v,q,a,meta,loaded):
    if cfg['family']=='prior':
        n=cfg['features'];bank=np.random.default_rng(731).uniform(-.12,.12,(n,4))
        support=forward(x,bank);mu=support.mean(0);centered=support-mu
        alpha=np.linalg.solve(centered.T@centered/n+np.eye(len(x))*.001**2/3,v-mu)
        weights=(1+centered@alpha)/n
        # Bounded-size chunks are an audit memory choice, not a model change.
        return np.concatenate([weights@forward(q[i:i+128],bank) for i in range(0,len(q),128)]),{}
    name=cfg['name'].removeprefix('cold__')
    if not name.startswith('meta_'):
        return regression_replay(x,v,q,name,a,meta['metadata']),{}
    model=loaded[name]
    if name.startswith('meta_ridge'):
        bank=values(model.bank);projection=values(model.projection);mean_weights=values(model.mean_weights)
        def features(z):
            raw=forward(z,bank).T;mu=raw.mean(1);centered=raw-mu[:,None]
            return centered@projection,mu+centered@mean_weights
        phi,mu=features(x);ridge=float(np.clip(np.exp(values(model.log_ridge)[0]),1e-9,.1))
        beta=phi.T@np.linalg.solve(phi@phi.T+ridge*np.eye(len(x)),v-mu)
        qp,qmu=features(q);return np.clip(qmu+qp@beta,0,1),{'fast_0':beta[None,:,None]}
    assert name.startswith('meta_shallow')
    amp=values(model.amplitudes);knots=values(model.knots);offset=values(model.offset);slope=values(model.slope)
    rates=.2/(1+np.exp(-values(model.rate_logits)))
    for _ in range(model.steps):
        h=np.maximum(x[:,None]-knots,0);res=offset+slope*x+h@amp-v
        grad=np.sign(res)*np.maximum(abs(res)-.001,0)/len(x)
        ga=h.T@grad;gt=-amp*((x[:,None]>knots)*grad[:,None]).sum(0)
        gb=grad.sum();gc=grad@x
        amp=amp-rates[0]*ga;knots=knots-rates[1]*gt;offset=offset-rates[2]*gb;slope=slope-rates[3]*gc
    pred=offset+slope*q+np.maximum(q[:,None]-knots,0)@amp
    states={'fast_0':amp[None],'fast_1':knots[None],'fast_2':np.asarray(offset)[None],'fast_3':np.asarray(slope)[None]}
    return np.clip(pred,0,1),states
