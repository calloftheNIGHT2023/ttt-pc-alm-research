"""Paired-start diagnostics, not a new adaptive method or confirmation.

Independent local and BP runs. No BP-derived initialization enters local ALM.
Numerical fixed-cell LP bounds refer to the assembled floating-point affine model.
"""
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import vector_interval_memory as m


def cell_lp(point, x, v, weights):
    """Best possible max context error within this entire activation cell."""
    d, w = point.shape; p = point.size; theta = point.ravel()
    h = x.copy(); jac = np.zeros((len(x), w, p)); matrices=[]; rhs=[]
    for layer, weight in enumerate(weights):
        z = h @ weight.T + point[layer]
        jz = np.einsum('ab,nbp->nap', weight, jac, optimize=True)
        jz[:,np.arange(w),layer*w+np.arange(w)] += 1
        offset = z - np.einsum('nwp,p->nw', jz, theta)
        regions = np.searchsorted([-1.,0.,1.], z)
        lower = np.array([-np.inf,-1.,0.,1.])[regions]
        upper = np.array([-1.,0.,1.,np.inf])[regions]
        for sign, bound in [(1.,upper),(-1.,-lower)]:
            finite = np.isfinite(bound)
            matrices.append(sign*jz[finite]); rhs.append(bound[finite]-sign*offset[finite])
        jac = jz*m.family.deriv(z)[:,:,None]; h=m.family.activation(z)
    design=jac.reshape(v.size,p); offset=(h-v).ravel()-design@theta
    branch=np.concatenate(matrices); brhs=np.concatenate(rhs)
    a=np.vstack([np.c_[branch,np.zeros(len(branch))],np.c_[design,-np.ones(v.size)],np.c_[-design,-np.ones(v.size)]])
    b=np.r_[brhs,-offset,offset]; c=np.r_[np.zeros(p),1.]
    limit=2+m.EPS
    sol=linprog(c,A_ub=a,b_ub=b,bounds=[(-m.BOUND,m.BOUND)]*p+[(0,limit)],method='highs')
    result={'lp_status':int(sol.status),'affine_constraint_count':len(b),'parameter_count':p,
        'certificate_scope':'numerical bound for assembled float affine cell; not rational end-to-end certificate'}
    if sol.success:
        multipliers=np.maximum(-sol.ineqlin.marginals,0)
        coefficient=c+multipliers@a
        lower_bound=-float(multipliers@b)-m.BOUND*float(np.sum(np.abs(coefficient[:-1])))+min(0,float(coefficient[-1])*limit)
        result.update(minimum_max_error=float(sol.fun),dual_box_lower_bound=lower_bound,
            duality_gap=float(sol.fun-lower_bound),cell_infeasible_at_noise_level=bool(lower_bound>m.EPS+1e-7))
    return result


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--out',type=Path,required=True); args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True); assert not (args.out/'protocol.json').exists()
    sources=[Path(__file__),Path(m.__file__),Path(m.family.__file__)]
    protocol={'widths':[8,32],'depth':3,'seeds':list(range(5400000,5400004)),
        'stages':[8,16,24],'restarts':16,'prior_features':256,'anchor':'zero at every stage for paired-start diagnosis',
        'methods':['local480','adam240_010','lbfgs300'],'max_cell_lps':8,
        'cell_selection':'first eight local-feasible/Adam-infeasible starts in width/seed/stage/restart order; no query selection',
        'source_sha256':{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}}
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]; cells=[]; timings=[]
    for width in protocol['widths']:
        weights=m.family.make_weights(3,width)
        for seed in protocol['seeds']:
            rng=np.random.default_rng(seed); truth=rng.uniform(-m.PRIOR,m.PRIOR,(3,width))
            x=rng.uniform(-1,1,(24,width)); q=rng.uniform(-1,1,(512,width))
            v=m.forward(truth[None],x,weights)[0]+np.random.default_rng(seed+19000000).uniform(-m.EPS,m.EPS,x.shape)
            target=m.forward(truth[None],q,weights)[0]; anchor=np.zeros_like(truth)
            for n in protocol['stages']:
                starts,_=m.proposals(x[:n],v[:n],anchor,weights)
                outputs={}
                for name in protocol['methods']:
                    before=time.perf_counter()
                    if name=='local480':bank,_=m.local(starts,x[:n],v[:n],weights,anchor,sweeps=480)
                    elif name=='adam240_010':bank,_=m.bp(starts,x[:n],v[:n],weights,anchor,solver='adam',steps=240,lr=.01)
                    else:bank,_=m.lbfgs(starts,x[:n],v[:n],weights,anchor,steps=300)
                    timings.append({'width':width,'seed':seed,'n':n,'method':name,'seconds':time.perf_counter()-before})
                    value,gradient,raw=m.evaluate(bank,x[:n],v[:n],weights)
                    err=np.max(np.abs(raw),axis=(1,2)); feasible=err<=m.EPS+m.TOL
                    outputs[name]=(bank,feasible)
                    query_mse=np.mean((m.forward(bank,q,weights)-target)**2,axis=(1,2))
                    # Projected gradient mapping uses unit step, for a box-constrained stationarity diagnostic.
                    pg=bank-np.clip(bank-gradient,-m.BOUND,m.BOUND)
                    for restart in range(len(starts)):
                        rows.append({'width':width,'seed':seed,'n_context':n,'restart':restart,'method':name,
                            'support_feasible':bool(feasible[restart]),'max_support_error':float(err[restart]),
                            'band_objective':float(value[restart]),'projected_gradient_norm':float(np.linalg.norm(pg[restart])),
                            'query_mse_evaluator_only':float(query_mse[restart])})
                chosen=np.flatnonzero(outputs['local480'][1]&~outputs['adam240_010'][1])
                for restart in chosen:
                    if len(cells)>=protocol['max_cell_lps']:break
                    cell=cell_lp(outputs['adam240_010'][0][restart],x[:n],v[:n],weights)
                    cells.append({'width':width,'seed':seed,'n_context':n,'restart':int(restart),**cell})
            (args.out/'diagnostics.json').write_text(json.dumps({'rows':rows,'cells':cells,'timings':timings},indent=2),encoding='utf-8')
            print(json.dumps({'width':width,'seed':seed,'rows':len(rows),'cell_lps':len(cells)}),flush=True)
    print(json.dumps({'complete':True,'rows':len(rows)}),flush=True)


if __name__=='__main__':main()
