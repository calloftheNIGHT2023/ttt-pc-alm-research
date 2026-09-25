"""O(R*w*n log n) exact conditional bias solve by sorted coefficient events.

Float64 implementation of the same finite-piece objective as batched_bias_block.
"""
import numpy as np
import anchored_input_block as reference


def solve(c,target,old,trust=.01,bound=.2):
    r,n,w=c.shape;cc=c.transpose(0,2,1).reshape(r*w,n);tt=target.transpose(0,2,1).reshape(r*w,n);oo=old.reshape(-1)
    z=cc-bound
    cuts=np.array([-1.,0.,1.])[None,:,None]-cc[:,None,:]
    # Determine the initial branch using the exact same floating event locations.
    # Computing c-bound separately can round a near-boundary event to the other side.
    region=np.sum(cuts<=-bound,axis=1)
    slopes=np.array([0.,2.,-2.,0.])[region]
    offsets=slopes*cc+np.array([-1.,1.,1.,-1.])[region]
    initial_a=np.sum(slopes**2,axis=1)+trust*n
    initial_b=np.sum(slopes*(offsets-tt),axis=1)-trust*n*oo
    initial_c=np.sum(offsets**2-2*offsets*tt,axis=1)
    before_s=np.array([0.,2.,-2.])[None,:,None];after_s=np.array([2.,-2.,0.])[None,:,None]
    before_d=before_s*cc[:,None,:]+np.array([-1.,1.,1.])[None,:,None]
    after_d=after_s*cc[:,None,:]+np.array([1.,1.,-1.])[None,:,None]
    active=(cuts>-bound)&(cuts<bound)
    da=np.broadcast_to(after_s**2-before_s**2,cuts.shape)*active
    db=(after_s*(after_d-tt[:,None,:])-before_s*(before_d-tt[:,None,:]))*active
    dc=(after_d**2-before_d**2-2*(after_d-before_d)*tt[:,None,:])*active
    cuts=np.clip(cuts,-bound,bound).reshape(r*w,3*n);order=np.argsort(cuts,axis=1,kind='stable')
    cuts=np.take_along_axis(cuts,order,axis=1)
    def accumulate(initial,events):
        changes=np.take_along_axis(events.reshape(r*w,3*n),order,axis=1)
        return np.c_[initial,initial[:,None]+np.cumsum(changes,axis=1)]
    aa=accumulate(initial_a,da);bb=accumulate(initial_b,db);constant=accumulate(initial_c,dc)
    assert np.min(aa)>0
    left=np.c_[np.full(r*w,-bound),cuts];right=np.c_[cuts,np.full(r*w,bound)]
    trial=np.clip(-bb/aa,left,right);values=aa*trial**2+2*bb*trial+constant
    winner=np.argmin(values,axis=1);answer=trial[np.arange(r*w),winner].reshape(r,w)
    actual=np.sum((reference.activation(c+answer[:,None,:])-target)**2,axis=1)+trust*n*(answer-old)**2
    oldenergy=np.sum((reference.activation(c+old[:,None,:])-target)**2,axis=1)
    assert np.all(actual<=oldenergy+1e-8*np.maximum(1,oldenergy))
    return answer,{'conditional_objective':actual,'candidate_intervals':3*n+1,
        'major_workspace_bytes_subtotal':sum(t.nbytes for t in [cc,tt,z,slopes,offsets,before_d,after_d,cuts,order,da,db,dc,aa,bb,constant,left,right,trial,values]),
        'bias_solver':'sorted quadratic coefficient events'}
