"""301: support-only projected KKT census and exact rational checks of near states."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import time
import numpy as np
import batched_bp_discovery as bp
from diagnose_gradient_flat_split_states_v1 import independent_reverse, residuals, tent
from diagnose_counterfactual_conditional_risk_v1 import read, save, sha

BOUND = .12


def projected(b, g):
    return b-np.clip(b-g,-BOUND,BOUND)


def kkt_violation(b, g):
    return np.where(b == -BOUND,np.minimum(g,0),np.where(b == BOUND,np.maximum(g,0),g))


def exact_state(b, x, v):
    bias = list(map(lambda a:F(float(a)),b))
    bound,epsilon = F(BOUND),F(.001)
    gradient = [F(0)]*4
    knots = 0
    loss,maximum = F(0),F(0)
    for xx,vv in zip(x,v):
        h = F(float(xx)); slopes = []
        for bb in bias:
            z = h+bb
            knots += int(z in [F(0),F(1,2),F(1)])
            slopes.append(2 if 0<z<F(1,2) else -2 if F(1,2)<z<1 else 0)
            h = max(F(0),min(2*z,2-2*z))
        raw = h-F(float(vv))
        maximum = max(maximum,abs(raw))
        excess = max(abs(raw)-epsilon,F(0))
        credit = excess if raw>=0 else -excess
        loss += credit*credit/(2*len(x))
        for j in range(3,-1,-1):
            credit *= slopes[j]
            gradient[j] += credit/len(x)
    valid = [(-bound<=bb<=bound) for bb in bias]
    conditions = [(gg>=0 if bb==-bound else gg<=0 if bb==bound else gg==0) for bb,gg in zip(bias,gradient)]
    return dict(bias_fractions=list(map(str,bias)),gradient_fractions=list(map(str,gradient)),
        gradient_float=list(map(float,gradient)),box_valid=all(valid),coordinate_kkt=conditions,
        exact_kkt=all(valid) and all(conditions),exact_knots=knots,
        smooth_activation_local_minimum=all(valid) and all(conditions) and knots==0,
        exact_loss=str(loss),exact_max_raw=str(maximum),exact_band_feasible=maximum<=epsilon)


def run(root,out):
    tick = time.perf_counter()
    btest = np.array([[-BOUND,BOUND,0.,-BOUND]])
    gtest = np.array([[2.,-3.,.02,-.01]])
    target = np.array([[0.,0.,.02,-.01]])
    np.testing.assert_allclose(projected(btest,gtest),target,rtol=0,atol=1e-16)
    assert np.array_equal(kkt_violation(btest,gtest),target)
    exact = exact_state(np.zeros(4),np.array([.1]),np.array([.4]))
    assert exact['exact_band_feasible'] and exact['exact_kkt'] and exact['exact_knots']==0
    rawdir = root/'results/certificate_activity_attribution/development'
    census = root/'results/stagnant_local_states/development_v2'
    assert sha(census/'summary.json') == 'e96bffa8a65d57021c9302d525d4fe01e9c1d9eca2e92169b711e2fd96c8756e'
    cs = read(census/'summary.json')
    for name,digest in cs['outputs_sha256'].items():
        assert sha(census/name)==digest,name
    selections = {r['seed']:r for r in read(census/'tasks.json')}
    original = {r['seed']:r for r in read(rawdir/'rows.json') if r['method']=='credit_control_probe33'}
    manifest = read(rawdir/'before_evaluation_manifest.json')
    assert sha(rawdir/'rows.json')==manifest['rows_sha256']
    frozen = read(root/'results/round_287_audit_v6.json')
    for name,digest in frozen['source_sha256'].items():
        assert sha(root/'work/experiments'/name)==digest,name
    names=['census_projected_stationarity_v1.py','batched_bp_discovery.py',
           'diagnose_gradient_flat_split_states_v1.py','diagnose_counterfactual_conditional_risk_v1.py']
    sources={name:sha(root/'work/experiments'/name) for name in names}
    save(out/'protocol.json',dict(source_sha256=sources,seeds=list(range(5910000,5910064)),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/301_projected_stationarity_protocol.md'),
        thresholds=[0.,1e-12,1e-8],eta=1.,bound=BOUND,gradient_only_offline=True,
        exact_fraction_source='binary64 inputs, true rational forward, not float execution',
        query_targets_accessed=False,reference_accessed=False,new_adaptation=False))
    totals, intersections = Counter(),{g:Counter() for g in ['dwell','parameter_stall','union']}
    rows,metrics,exactrows = [],[],[]
    max_reverse_gap = max_gnorm_gap = 0.
    with np.load(census/'metrics.npz',allow_pickle=False) as z:
        oldmetrics=z['values']
    for task_index,seed in enumerate(range(5910000,5910064)):
        path=rawdir/original[seed]['file']
        assert sha(path)==original[seed]['sha256']==selections[seed]['source_sha256']
        with np.load(path,allow_pickle=False) as z:
            x,v=z['x_observed'],z['v_observed']
            traces={p:{n:z[f'{p}_{n}'] for n in ['b','h','u']} for p in ['prefix','anchor']}
        masks={g:{tuple(loc) for loc in selections[seed]['locations'][g]} for g in ['dwell','parameter_stall']}
        masks['union']=masks['dwell']|masks['parameter_stall']
        counts=Counter(); tm=[]; cache={}; taskexact=[]
        for phase_id,phase in enumerate(['prefix','anchor']):
            trace=traces[phase]; bb,hh=trace['b'],trace['h']
            for t in range(32):
                b,h=bb[t],hh[t]
                assert np.all(abs(b)<=BOUND)
                _,res,jac,raw=bp.evaluate(b,x,v)
                g=np.einsum('rni,rn->ri',jac,res,optimize=False)/len(x)
                reverse,independent_raw=independent_reverse(b,x,v)
                max_reverse_gap=max(max_reverse_gap,float(np.max(abs(g-reverse))))
                assert np.array_equal(independent_raw,raw)
                G=projected(b,g); V=kkt_violation(b,g)
                gn=np.max(abs(g),axis=1); pn=np.max(abs(G),axis=1); vn=np.max(abs(V),axis=1)
                er=np.max(abs(raw),axis=1)
                split=np.max(abs(residuals(b,h,x)),axis=(0,2))
                boundary=np.sum(abs(b)==BOUND,axis=1)
                knots=np.zeros(len(b),dtype=int); prev=np.broadcast_to(x,(len(b),len(x)))
                for j in range(4):
                    pre=prev+b[:,j,None]
                    knots+=np.count_nonzero((pre==0)|(pre==.5)|(pre==1),axis=1)
                    prev=tent(pre)
                if t>0:
                    change=np.maximum(np.max(abs(b-bb[t-1]),axis=1),np.max(abs(h-hh[t-1]),axis=(0,2)))
                elif phase_id==1:
                    pb,ph=traces['prefix']['b'],traces['prefix']['h']
                    change=np.maximum(np.max(abs(pb[-1,:1]-pb[-2,:1]),axis=1),np.max(abs(ph[-1,:,:1]-ph[-2,:,:1]),axis=(0,2)))
                else: change=np.full(len(b),np.inf)
                for origin in range(len(b)):
                    loc=(phase_id,t+1,origin)
                    non=er[origin]>.001001
                    flags=dict(states=True,nonfeasible=non,boundary=bool(boundary[origin]),activation_knot=bool(knots[origin]),
                        nonfeasible_projected_zero=non and pn[origin]==0,
                        nonfeasible_direct_kkt_zero=non and vn[origin]==0,
                        nonfeasible_projected_1e12=non and pn[origin]<=1e-12,
                        nonfeasible_projected_1e8=non and pn[origin]<=1e-8,
                        nonfeasible_direct_kkt_1e8=non and vn[origin]<=1e-8,
                        nonfeasible_primal_change_1e8=non and change[origin]<=1e-8 and split[origin]>1e-6)
                    counts.update({k:int(a) for k,a in flags.items()})
                    for group,selected in masks.items():
                        if loc in selected: intersections[group].update({k:int(a) for k,a in flags.items()})
                    tm.append([*loc,float(er[origin]),float(gn[origin]),float(pn[origin]),float(vn[origin]),
                        int(boundary[origin]),int(knots[origin]),float(split[origin]),float(change[origin])])
                    if non and min(pn[origin],vn[origin])<=1e-8:
                        key=b[origin].tobytes()
                        if key not in cache: cache[key]=exact_state(b[origin],x,v)
                        case=dict(seed=seed,location=list(loc),projected_norm=float(pn[origin]),
                            direct_kkt_norm=float(vn[origin]),**cache[key])
                        taskexact.append(case)
        tm=np.array(tm)
        assert tm.shape==(1088,11)
        assert np.array_equal(tm[:,:3],oldmetrics[task_index,:,:3])
        assert np.array_equal(tm[:,3],oldmetrics[task_index,:,3])
        max_gnorm_gap=max(max_gnorm_gap,float(np.max(abs(tm[:,4]-oldmetrics[task_index,:,8]))))
        totals.update(counts); metrics.append(tm); exactrows.extend(taskexact)
        rows.append(dict(seed=seed,counts=dict(counts),exact_candidate_locations=len(taskexact),
            exact_unique_parameter_states=len(cache),source_sha256=sha(path)))
    assert totals['states']==69632 and max_reverse_gap<1e-12 and max_gnorm_gap==0
    assert intersections['dwell']['states']==2068 and intersections['parameter_stall']['states']==38 and intersections['union']['states']==2105
    save(out/'tasks.json',rows)
    save(out/'exact_states.json',exactrows)
    save(out/'columns.json',['phase','step','origin','support_max_error','gradient_max','projected_gradient_max',
        'direct_kkt_max','boundary_coordinates','activation_knots','split_max','primal_last_change'])
    with (out/'metrics.npz').open('xb') as stream:
        np.savez_compressed(stream,values=np.array(metrics),seeds=np.arange(5910000,5910064))
    for name,digest in sources.items(): assert sha(root/'work/experiments'/name)==digest,name
    summary=dict(passed=True,tasks=64,counts=dict(totals),intersection_counts={g:dict(c) for g,c in intersections.items()},
        exact_locations=len(exactrows),exact_kkt=sum(r['exact_kkt'] for r in exactrows),
        exact_smooth_local_minima=sum(r['smooth_activation_local_minimum'] for r in exactrows),
        max_reverse_gradient_gap=max_reverse_gap,max_299_gradient_gap=max_gnorm_gap,
        query_targets_accessed=False,reference_accessed=False,new_adaptation=False,
        no_exhaustive_exact_stationarity_claim=True,core_research_goal_complete=False,
        seconds=time.perf_counter()-tick,outputs_sha256={p.name:sha(p) for p in sorted(out.iterdir()) if p.is_file()})
    save(out/'summary.json',summary)
    print({k:v for k,v in summary.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve()
    out=root/'results/projected_stationarity/development_v1'
    out.mkdir(parents=True,exist_ok=False)
    try: run(root,out)
    except BaseException as exc:
        save(out/'failure.json',dict(error_type=type(exc).__name__,message=str(exc),no_automatic_retry=True))
        raise
