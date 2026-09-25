"""Batched finite-piece conditional bias solver with changing layer inputs."""
import numpy as np
import anchored_input_block as reference


def solve(c,target,old,trust=.01,bound=.2):
    r,n,w=c.shape;cc=c.transpose(0,2,1).reshape(r*w,n);tt=target.transpose(0,2,1).reshape(r*w,n);oo=old.reshape(-1)
    knots=np.clip(np.concatenate([np.full((r*w,1),-bound),np.full((r*w,1),bound),-1-cc,-cc,1-cc],axis=1),-bound,bound)
    knots.sort(axis=1);left=knots[:,:-1];right=knots[:,1:];middle=(left+right)/2
    zz=cc[:,:,None]+middle[:,None,:];slopes=np.where((zz>-1)&(zz<0),2.,np.where((zz>0)&(zz<1),-2.,0.))
    offsets=reference.activation(zz)-slopes*middle[:,None,:]
    aa=np.sum(slopes**2,axis=1)+trust*n
    bb=np.sum(slopes*(offsets-tt[:,:,None]),axis=1)-trust*n*oo[:,None]
    constant=np.sum(offsets**2-2*offsets*tt[:,:,None],axis=1) # target-only constant cancels in argmin
    trial=np.clip(-bb/aa,left,right)
    values=aa*trial**2+2*bb*trial+constant
    winner=np.argmin(values,axis=1);answer=trial[np.arange(r*w),winner].reshape(r,w)
    actual=np.sum((reference.activation(c+answer[:,None,:])-target)**2,axis=1)+trust*n*(answer-old)**2
    oldenergy=np.sum((reference.activation(c+old[:,None,:])-target)**2,axis=1)
    assert np.all(actual<=oldenergy+1e-8*np.maximum(1,oldenergy))
    return answer,{'conditional_objective':actual,'candidate_intervals':left.shape[1],
        'major_workspace_bytes_subtotal':sum(t.nbytes for t in [cc,tt,knots,zz,slopes,offsets,aa,bb,constant,trial,values])}
