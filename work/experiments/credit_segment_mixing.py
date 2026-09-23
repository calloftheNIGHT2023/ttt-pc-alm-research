"""Cheap exact line maximization of existing local-credit separation bounds.

Coordinates are local linear maps of supplied credits, not global BP. Floating
line maxima are proposals only; original outward-rounded bound must be positive
before rejection. BP and random direction controls get exactly the same rule.
"""
import time
import numpy as np
import local_region_screen as screen
base = screen.base


def coordinates(x, v, regs, bank):
    count, depth, n = regs.shape
    _, _, hl, hh = screen.boxes(v, regs[:1]); center = (hl[0]+hh[0])/2; radius = (hh[0]-hl[0])/2
    p = base.SLOPES[regs][:, None] * bank[None]
    hc = np.broadcast_to(bank, p.shape).copy(); hc[:, :, :-1] -= p[:, :, 1:]
    linear = (hc * center).sum((2, 3)) - (bank[None] * base.INTERCEPTS[regs][:, None]).sum((2, 3)) - (p[:, :, 0] * x).sum(2)
    z = np.concatenate([p.sum(3), hc.reshape(count, len(bank), -1)], axis=2)
    w = np.r_[np.full(depth, screen.B), radius.ravel()]
    return linear, z, w


def maximize_star(linear, z, weights):
    """All segments from the best endpoint to every other endpoint.

    Along each segment, l(t)-sum w*abs(z(t)) is concave PL; its maximum is at
    an endpoint or a zero crossing. This is not the whole convex-hull optimum.
    """
    scores = linear - np.abs(z) @ weights; root = int(np.argmax(scores))
    dz = z-z[root]; dl = linear-linear[root]
    zeros = np.divide(-z[root], dz, out=np.zeros_like(dz), where=dz != 0)
    zeros = np.where((zeros > 0) & (zeros < 1), zeros, 0)
    t = np.column_stack([np.zeros(len(z)), np.ones(len(z)), zeros])
    values = linear[root] + dl[:, None]*t - (np.abs(z[root] + dz[:, None]*t[:, :, None]) * weights).sum(2)
    index = np.unravel_index(np.argmax(values), values.shape)
    return root, int(index[0]), float(t[index]), float(values[index]), float(scores[root]), values.size


def screen_bank(x, v, regs, bank):
    begin=time.perf_counter(); proofs=[]; count=len(regs)
    if not count or not len(bank): return np.zeros(count, bool), proofs, dict(seconds=time.perf_counter()-begin, directions=len(bank), positive_proposals=0, line_evaluations=0, coordinate_bytes=0)
    linear,z,w=coordinates(x,v,regs,bank); choices=[]; aa=[]; old=[]; evaluations=0
    for i in range(count):
        first,second,t,bound,endpoint,tested=maximize_star(linear[i],z[i],w); evaluations+=tested
        a=(1-t)*bank[first]+t*bank[second]
        scale=1+np.abs(a).sum()+np.abs(base.SLOPES[regs[i]]*a).sum()
        if bound > 1e-10*scale:
            choices.append(i); aa.append(a); old.append((first,second,t,bound,endpoint))
    reject=np.zeros(count, bool)
    if choices:
        rr=regs[choices]; aa=np.array(aa); pp=base.SLOPES[rr]*aa
        lower=screen.certified_lower_bound(x,v,rr,pp,aa)
        for j,index in enumerate(choices):
            if lower[j] <= 0: continue
            reject[index]=True; first,second,t,bound,endpoint=old[j]
            proofs.append(dict(index=int(index),pattern=regs[index].tobytes().hex(),a=aa[j].tolist(),p=pp[j].tolist(),lower=float(lower[j]),
                first=first,second=second,t=t,rough_mixed_bound=bound,best_endpoint_rough_bound=endpoint))
    return reject,proofs,dict(seconds=time.perf_counter()-begin,directions=len(bank),positive_proposals=len(choices),line_evaluations=evaluations,
        coordinate_bytes=linear.nbytes+z.nbytes+w.nbytes,direction_bytes=bank.nbytes,
        scope='line-workspace and Python objects additional; no complete peak or whole-pipeline speed claim')


def verify():
    rng=np.random.default_rng(838921); coordinate_error=0.; optimum_gap=0.; count=0
    for d,n in [(1,2),(2,4),(4,4),(4,8)]:
        x=rng.uniform(0,1,n); v=base.forward(x,rng.uniform(-.12,.12,d)); regs=rng.integers(0,4,(6,d,n),dtype=np.uint8); bank=rng.normal(size=(9,d,n))
        linear,z,w=coordinates(x,v,regs,bank)
        for i,reg in enumerate(regs):
            rr=np.repeat(reg[None],len(bank),axis=0); direct=screen.float_bound(x,v,rr,base.SLOPES[rr]*bank,bank)
            error=float(np.max(np.abs(direct-(linear[i]-np.abs(z[i])@w)))); coordinate_error=max(coordinate_error,error)
            first,second,t,bound,endpoint,evaluations=maximize_star(linear[i],z[i],w)
            aa=(1-t)*bank[first]+t*bank[second]; exact=screen.float_bound(x,v,reg[None],(base.SLOPES[reg]*aa)[None],aa[None])[0]
            assert abs(exact-bound)<1e-10 and bound>=endpoint-1e-11
            grid=np.linspace(0,1,1001)
            values=linear[i,first]+(linear[i]-linear[i,first])[:,None]*grid-(np.abs(z[i,first]+(z[i]-z[i,first])[:,None,:]*grid[None,:,None])*w).sum(2)
            gap=float(values.max()-bound); optimum_gap=max(optimum_gap,gap); assert gap<1e-10; count+=1
    # Elementary strict synergy: both endpoints -0.6, midpoint +0.4.
    first,second,t,bound,endpoint,_=maximize_star(np.array([.4,.4]),np.array([[-1.],[1.]]),np.array([1.]))
    assert abs(bound-.4)<1e-15 and endpoint<0 and t==.5
    assert coordinate_error<1e-10
    return dict(passed=True,coordinate_and_line_cases=count,max_coordinate_error=coordinate_error,max_dense_grid_advantage=optimum_gap,
        strict_synergy_example=dict(endpoint_bound=endpoint,mixed_bound=bound,t=t),scope='valid algebraic example, not a unique PC-ALM task advantage')


if __name__=='__main__':
    import json
    print(json.dumps(verify()))
