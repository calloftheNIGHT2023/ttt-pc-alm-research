"""374 native history initializes established local PDHG; no novelty claim for PDHG."""
import time
import numpy as np
import cold_stagnation_switch as cold
import local_region_screen as local
from posterior_confirmation_pipeline import discovery_box
from test_region_conditioned_credit_v1 import guarded

METHODS=['cold_zero128','cold_zero1024','alm_zero128','alm_dual128','alm_residual128','alm_random128','alm_bp128','pc_zero128','pc_residual128']
PRIMARY='alm_dual128'


def global_bp_credit(x,v,b):
    """Explicit global-chain control. KKT sign for h-f residual is negative."""
    h=x.copy();derivatives=[]
    for bias in b:
        z=h+bias;derivatives.append(cold.base.derivative(z));h=cold.base.g(z)
    residual=np.sign(h-v)*np.maximum(abs(h-v)-.001,0.)
    credit=np.empty((4,len(x)));credit[-1]=-residual
    for j in range(2,-1,-1):credit[j]=credit[j+1]*derivatives[j+1]
    return credit


def generate(x,v,seed,name):
    tick=time.perf_counter();n=len(x);arrays={};meta=dict(generator_steps=0,generator_restarts=0,generator_state_bytes=0,global_bp_used=name=='alm_bp128')
    if name.startswith('cold_'):
        b=np.zeros(4);h=[];previous=x
        for _ in range(4):previous=cold.base.g(previous);h.append(previous.copy())
        h=np.array(h);credit=np.zeros((4,n));index=-1
    else:
        method='pc' if name.startswith('pc_') else 'alm'
        with discovery_box(.12),guarded():
            state=cold.Local(cold.starts(),x,v,method)
            for _ in range(64):state.step()
            err,move=cold.base.score(state.b,x,v,np.zeros(4))
        index=min(range(17),key=lambda i:(float(err[i]),float(move[i]),i))
        b=state.b[index].copy();h=state.h[:,index].copy();previous=x;residual=[]
        for j in range(4):residual.append(h[j]-cold.base.g(previous+b[j]));previous=h[j]
        residual=np.array(residual)
        arrays.update(mother_b=state.b.copy(),mother_h=state.h.copy(),mother_u=state.u.copy(),mother_errors=err,mother_moves=move,selected_residual=residual)
        if '_zero' in name:credit=np.zeros((4,n))
        elif '_dual' in name:credit=state.u[:,index].copy()
        elif '_residual' in name:credit=residual.copy()
        elif '_random' in name:credit=np.random.default_rng(np.random.SeedSequence([374137,seed,n])).choice(np.array([-1.,1.]),size=(4,n))
        else:assert name=='alm_bp128';credit=global_bp_credit(x,v,b)
        meta.update(generator_steps=64,generator_restarts=17,generator_state_bytes=state.numeric_state_bytes())
    raw=credit.copy();norm=float(np.linalg.norm(credit));credit=credit/norm if norm>0 else credit.copy()
    arrays.update(start_b=b,start_h=h,credit_raw=raw,credit=credit)
    meta.update(selected_restart=index,credit_raw_norm=norm,generation_seconds=time.perf_counter()-tick,prefix_observations=n,full_context_suffix_accessed=False)
    return arrays,meta


def solve(x,v,regs,b0,h0,credit,steps):
    started=time.perf_counter();r,d,n=regs.shape;zl,zh,hl,hh=local.boxes(v,regs)
    s=cold.base.SLOPES[regs];c=cold.base.INTERCEPTS[regs]
    b=np.broadcast_to(b0,(r,d)).copy();h=np.broadcast_to(h0,(r,d,n)).copy()
    previous=np.concatenate([np.broadcast_to(x,(r,1,n)),h[:,:-1]],axis=1);z=previous+b[:,:,None]
    a=np.broadcast_to(credit,regs.shape).copy();p=s*a
    initial=dict(initial_b=b.copy(),initial_h=h.copy(),initial_z=z.copy(),initial_p=p.copy(),initial_a=a.copy())
    bb=b.copy();zb=z.copy();hb=h.copy();lastp=p.copy();lasta=a.copy()
    tb=.99/n;tz=.99/(1+abs(s));th=np.full((1,d,1),.99/2);th[:,-1]=.99
    sp=np.full((1,d,1),.99/3);sp[:,0]=.99/2;sa=.99/(1+abs(s))
    first=np.full(r,-1,np.int32);proofp=np.zeros_like(p);proofa=np.zeros_like(a);lower=np.zeros(r);proposal=np.full(r,-1,np.int8)
    checkpoints=[]
    def check(step):
        nonlocal lastp,lasta
        for code,(pp,aa) in enumerate([(p,a),(p-lastp,a-lasta)]):
            rough=local.float_bound(x,v,regs,pp,aa)
            ids=np.flatnonzero((rough>1e-10*(1+abs(pp).sum((1,2))+abs(aa).sum((1,2))))&(first<0))
            if len(ids):
                bound=local.certified_lower_bound(x,v,regs[ids],pp[ids],aa[ids]);good=bound>0;accepted=ids[good]
                first[accepted]=step;proofp[accepted]=pp[accepted];proofa[accepted]=aa[accepted];lower[accepted]=bound[good];proposal[accepted]=code
        lastp=p.copy();lasta=a.copy()
        checkpoints.append(dict(step=step,rejected=int((first>=0).sum()),solver_seconds=time.perf_counter()-started))
    check(0)
    with guarded():
        for it in range(steps):
            previous=np.concatenate([np.broadcast_to(x,(r,1,n)),hb[:,:-1]],axis=1)
            p+=sp*(zb-previous-bb[:,:,None]);a+=sa*(hb-s*zb-c)
            oldb,oldz,oldh=b,z,h
            b=np.clip(oldb+tb*p.sum(axis=2),-.12,.12);z=np.clip(oldz-tz*(p-s*a),zl,zh)
            hc=a.copy();hc[:,:-1]-=p[:,1:];h=np.clip(oldh-th*hc,hl,hh)
            bb,zb,hb=2*b-oldb,2*z-oldz,2*h-oldh
            if (it+1)%32==0 or it+1==steps:check(it+1)
    arrays={**initial,'first_step':first,'proof_p':proofp,'proof_a':proofa,'proof_lower':lower,'proof_proposal':proposal,
        'final_b':b,'final_h':h,'final_z':z,'final_p':p,'final_a':a}
    named=[b,z,h,p,a,bb,zb,hb,lastp,lasta,zl,zh,hl,hh,s,c,tz,sa,first,proofp,proofa,lower,proposal]
    return arrays,dict(steps=steps,checkpoints=checkpoints,solver_seconds=time.perf_counter()-started,
        live_named_array_bytes=sum(t.nbytes for t in named),persistent_initial_array_bytes=sum(t.nbytes for t in initial.values()),
        memory_scope='Named arrays; not Python/native allocator peak; archive bytes recorded separately',lp_accessed=False)


def fit(x,v,regs,seed,name):
    begin=time.perf_counter();assert name in METHODS
    init,meta=generate(x,v,seed,name)
    arrays,note=solve(x,v,regs,init['start_b'],init['start_h'],init['credit'],1024 if name=='cold_zero1024' else 128)
    arrays.update(init);meta.update(note,total_seconds=time.perf_counter()-begin,query_targets_accessed=False,
        returned_array_bytes=sum(t.nbytes for t in arrays.values()),historical_generation_charged=True)
    return arrays,meta
