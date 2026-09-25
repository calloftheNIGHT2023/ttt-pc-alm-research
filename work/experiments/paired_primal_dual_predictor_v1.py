"""312 causal paired extrapolation followed by the frozen local primal update."""
from fractions import Fraction as F
import time
import numpy as np
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from complete_credit_amplitude_events_v2 import rational_state
from complete_credit_rational_reference_v1 import direct_step, tent
from layer_credit_interaction_v1 import mode_list

NAMES=['normal_one','normal_two','paired_bu','bias_only','dual_only','dual_full',
       'full_bhu','paired_b_zero_u','paired_b_last_du','paired_b_random_du',
       'random_db_paired_u','paired_b_current_residual_u']


def admissible_alpha(b, delta, bound=.12):
    bound=F(bound);alpha=F(1)
    for value,d in zip(b,delta):
        value,d=F(float(value)),F(float(d))
        assert abs(value)<=bound
        if d>0:alpha=min(alpha,(bound-value)/d)
        elif d<0:alpha=min(alpha,(-bound-value)/d)
    assert 0<=alpha<=1
    approximate=float(alpha)
    if F(approximate)>alpha:approximate=float(np.nextafter(approximate,0.))
    assert 0<=F(approximate)<=alpha
    return approximate,alpha


def predictions(current,previous,x,v,key):
    b,h,u=[np.asarray(current[k]) for k in ['b','h','u']]
    db,dh,du=[current[k]-previous[k] for k in ['b','h','u']]
    alpha,exact_alpha=admissible_alpha(b,db)
    sign_u=np.random.default_rng(np.random.SeedSequence([312901,*key,0])).choice([-1.,1.],size=u.shape)
    sign_b=np.random.default_rng(np.random.SeedSequence([312901,*key,1])).choice([-1.,1.],size=b.shape)
    random_alpha,random_exact_alpha=admissible_alpha(b,db*sign_b)
    predicted_b=np.clip(b+alpha*db,-.12,.12)
    predicted_h=np.clip(h+alpha*dh,0,1)
    predicted_h[-1]=np.clip(predicted_h[-1],np.maximum(0,v-.001),np.minimum(1,v+.001))
    residual=np.stack([h[j]-cold.base.g((x if j==0 else h[j-1])+b[j]) for j in range(len(b))])
    versions={
        'normal_one':(b,h,u), 'normal_two':(b,h,u),
        'paired_bu':(predicted_b,h,u+alpha*du),
        'bias_only':(predicted_b,h,u),
        'dual_only':(b,h,u+alpha*du),
        'dual_full':(b,h,u+du),
        'full_bhu':(predicted_b,predicted_h,u+alpha*du),
        'paired_b_zero_u':(predicted_b,h,np.zeros_like(u)),
        'paired_b_last_du':(predicted_b,h,alpha*du),
        'paired_b_random_du':(predicted_b,h,u+alpha*du*sign_u),
        'random_db_paired_u':(np.clip(b+random_alpha*db*sign_b,-.12,.12),h,u+random_alpha*du),
        'paired_b_current_residual_u':(predicted_b,h,u+alpha*.5*residual)}
    assert list(versions)==NAMES
    assert np.array_equal(abs(du*sign_u),abs(du)) and np.array_equal(abs(db*sign_b),abs(db))
    actual={n:{k:np.array(a,copy=True) for k,a in zip(['b','h','u'],args)} for n,args in versions.items()}
    return actual,dict(alpha=alpha,exact_alpha=exact_alpha,random_alpha=random_alpha,
        random_exact_alpha=random_exact_alpha,db=db.tolist(),dh=dh.tolist(),du=du.tolist(),
        current_residual=residual.tolist(),du_vs_half_current_residual_max_gap=float(np.max(abs(du-.5*residual))),
        alpha_zero=alpha==0,alpha_less_than_one=alpha<1,
        predictor_b_rounding_projection_changes=int(np.count_nonzero(predicted_b!=(b+alpha*db))),
        extra_primary_named_history_bytes=previous['b'].nbytes+previous['u'].nbytes,
        state_cost_scope='Extra prior b/u arrays only; not peak memory, workspaces or full online fit.')


def local(current,x,v):
    state=cold.Local(current['b'][None],x,v,'alm')
    state.h=current['h'][:,None].copy();state.u=current['u'][:,None].copy()
    return state


def reference(current,after,x,v):
    rstate=rational_state(current['b'],current['h'],current['u'],x,v)
    rr=direct_step(rstate,1)
    unew=[]
    for j in range(len(rr['b'])):
        prev=rstate['x'] if j==0 else rr['h'][j-1]
        unew.append([q+F(1,2)*(hh-tent(pp+rr['b'][j]))
                     for q,hh,pp in zip(rstate['direction'][j],rr['h'][j],prev)])
    errors={k:float(np.max(abs(np.array(val,dtype=float)-after[k])))
            for k,val in [('b',rr['b']),('h',rr['h']),('u',unew)]}
    exact_mode=bytes(map(int,rr['mode'])).hex()
    float_mode=mode_list(after['b'][None],x)[0]
    return dict(b=rr['b'],h=rr['h'],u=unew,exact_mode=exact_mode,float_mode=float_mode,
                gaps=errors,value_discrepancy=max(errors.values())>1e-10,mode_discrepancy=exact_mode!=float_mode)


def evaluate(current,previous,x,v,key):
    tick=time.perf_counter();starts,metadata=predictions(current,previous,x,v,key)
    construction_seconds=time.perf_counter()-tick
    rows=[]
    with discovery_box(.12):
        for name in NAMES:
            tick=time.perf_counter();state=local(starts[name],x,v)
            setup_seconds=time.perf_counter()-tick
            trace=[{k:a.copy() for k,a in starts[name].items()}];seconds=[]
            for _ in range(2 if name=='normal_two' else 1):
                tick=time.perf_counter();state.step();seconds.append(time.perf_counter()-tick)
                trace.append(dict(b=state.b[0].copy(),h=state.h[:,0].copy(),u=state.u[:,0].copy()))
            refs=[reference(a,b,x,v) for a,b in zip(trace,trace[1:])]
            modes=mode_list(np.stack([s['b'] for s in trace]),x)
            rows.append(dict(method=name,states=[{k:a.tolist() for k,a in s.items()} for s in trace],
                point_modes=modes,proposal_modes=sorted(set(modes)),references=refs,
                costs=dict(setup_seconds=setup_seconds,step_seconds=seconds,local_steps=len(seconds),
                           diagnostic_scalar_run=True,resources_matched=False)))
    return rows,dict(**metadata,prediction_construction_seconds=construction_seconds)
