"""318 exact integer first-exit proposal along a constant-primal dual line."""
from fractions import Fraction as F
from affine_root_validation_v1 import state,domain,exact_step,forward_metrics
from audit_affine_root_proposal_v1 import minimize_by_energy,g


def blocks(s,delta):
    d,n=len(s['b']),len(s['x']);dim=d+d*n
    du=[delta[dim+j*n:dim+(j+1)*n] for j in range(d)]
    result=[];knots=[F(0),F(1,2),F(1)];B=s['bound']
    for j in reversed(range(d)):
        for i in range(n):
            lo,hi=(max(F(0),s['v'][i]-s['eps']),min(F(1),s['v'][i]+s['eps'])) if j==d-1 else (F(0),F(1))
            cuts=[lo,hi]
            if j<d-1:cuts += [k-s['b'][j+1] for k in knots if lo<k-s['b'][j+1]<hi]
            result.append(dict(kind='h',j=j,i=i,old=s['h'][j][i],breaks=sorted(set(cuts))))
    for j in range(d):
        prev=s['x'] if j==0 else s['h'][j-1]
        cuts=[-B,B]+[k-p for p in prev for k in knots if -B<k-p<B]
        result.append(dict(kind='b',j=j,old=s['b'][j],breaks=sorted(set(cuts))))
    return result,du


def gap(s,du,block,z,t):
    j=block['j'];old=block['old'];tau=s['trust'];n=len(s['x']);d=len(s['b'])
    if block['kind']=='h':
        i=block['i'];prev=s['x'][i] if j==0 else s['h'][j-1][i]
        a=g(prev+s['b'][j])-s['direction'][j][i]-t*du[j][i]
        value=(z-a)**2-(old-a)**2+tau*(z-old)**2
        if j<d-1:
            target=s['h'][j+1][i]+s['direction'][j+1][i]+t*du[j+1][i]
            value+=(g(z+s['b'][j+1])-target)**2-(g(old+s['b'][j+1])-target)**2
        return value
    prev=s['x'] if j==0 else s['h'][j-1]
    target=[h+u+t*e for h,u,e in zip(s['h'][j],s['direction'][j],du[j])]
    return sum(((g(p+z)-y)**2-(g(p+old)-y)**2 for p,y in zip(prev,target)),F(0))/n+tau*(z-old)**2


def beta(s,du,block,z):
    j=block['j'];old=block['old'];n=len(s['x']);d=len(s['b'])
    if block['kind']=='h':
        i=block['i'];value=2*du[j][i]*(z-old)
        if j<d-1:value-=2*du[j+1][i]*(g(z+s['b'][j+1])-g(old+s['b'][j+1]))
        return value
    prev=s['x'] if j==0 else s['h'][j-1]
    return -F(2,n)*sum((e*(g(p+z)-g(p+old)) for p,e in zip(prev,du[j])),F(0))


def escape(q,delta,depth,x,v,method='alm',cap=64):
    n=len(x);dim=depth+depth*n;s=state(q,depth,x,v)
    result=dict(cap=cap,applicable=False,status=None,query_targets_accessed=False)
    if any(delta[:dim]):result['status']='nonzero_primal_drift';return result
    bad=domain(s)
    if bad:result.update(status='q_outside_domain',domain_violations=bad);return result
    bs,du=blocks(s,delta);residual=[]
    for j,b in enumerate(s['b']):
        prev=s['x'] if j==0 else s['h'][j-1]
        residual.extend(h-g(z+b) for h,z in zip(s['h'][j],prev))
    rate=F(1,2) if method=='alm' else F(0)
    if [rate*r for r in residual]!=delta[dim:]:result['status']='drift_not_actual_dual_increment';return result
    checks=[]
    def stays(t):
        evaluated=0;intervals=0;first=None
        for index,block in enumerate(bs):
            best,k=minimize_by_energy(block['breaks'],lambda z:gap(s,du,block,z,F(t)))
            evaluated+=1;intervals+=k
            if best!=block['old']:
                first=dict(block=index,selected=best,old=block['old']);break
        checks.append(dict(phase=t,stays=first is None,evaluated_blocks=evaluated,intervals=intervals,first_changed=first))
        return first is None
    if not stays(0):result.update(status='phase_zero_not_primal_stationary',checks=checks);return result
    result['applicable']=True;endpoints=[];negative=[]
    for index,block in enumerate(bs):
        for z in block['breaks']:
            initial=gap(s,du,block,z,F(0));slope=beta(s,du,block,z)
            assert initial>=0
            witness=dict(block=index,candidate=z,delta=initial,beta=slope)
            if slope<0:
                ratio=initial/(-slope);upper=ratio.numerator//ratio.denominator+1
                witness['strict_integer_exit_upper_bound']=upper;negative.append(witness)
            endpoints.append(witness)
    result.update(endpoint_witnesses=endpoints,negative_endpoint_count=len(negative))
    if not negative:
        result.update(status='primal_stasis_for_all_nonnegative_phases',checks=checks)
        return result
    witness=min(negative,key=lambda w:(w['strict_integer_exit_upper_bound'],w['block']))
    result['finite_exit_witness']=witness
    hi=min(cap,witness['strict_integer_exit_upper_bound'])
    if stays(hi):
        assert hi==cap and witness['strict_integer_exit_upper_bound']>cap
        result.update(status='finite_exit_beyond_cap',checks=checks);return result
    lo=0
    while hi-lo>1:
        middle=(lo+hi)//2
        if stays(middle):lo=middle
        else:hi=middle
    assert stays(hi-1) and not stays(hi)
    current=[z+hi*d for z,d in zip(q,delta)];expected=[z+(hi+1)*d for z,d in zip(q,delta)]
    actual,rr=exact_step(state(current,depth,x,v),method)
    assert actual[:dim]!=expected[:dim]
    # Direct full sweep immediately before the exit, independent of bisection.
    previous=[z+(hi-1)*d for z,d in zip(q,delta)]
    last,_=exact_step(state(previous,depth,x,v),method);assert last==current
    result.update(status='first_primal_exit_found',first_exit_input_phase=hi,
                  first_changed_update_number=hi+1,exit_input=current,exit_output=actual,
                  exit_metrics=forward_metrics(state(actual,depth,x,v)),checks=checks)
    return result
