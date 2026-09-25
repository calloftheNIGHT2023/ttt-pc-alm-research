"""Exact scalar bias fit through a known-input piecewise-linear activation.

Precomputation is shared across restarts and iterations for observed inputs only.
"""
import numpy as np


def activation(z):return np.maximum(-1.,1-2*np.abs(z))


def prepare(c,bound=.3,trust=.01):
    n,width=c.shape;channels=[]
    for channel in range(width):
        values=c[:,channel]
        points=np.unique(np.r_[-bound,bound,np.clip(np.array([-1.,0.,1.])[:,None]-values,-bound,bound).ravel()])
        middle=(points[:-1]+points[1:])/2;z=values[:,None]+middle[None]
        slopes=np.where((z>-1)&(z<0),2.,np.where((z>0)&(z<1),-2.,0.))
        offsets=activation(z)-slopes*middle
        channels.append({'left':points[:-1],'right':points[1:],'slopes':slopes,'offsets':offsets,
            'curvature':np.sum(slopes**2,axis=0)+trust*n,'linear_base':np.sum(slopes*offsets,axis=0)})
    return {'c':c.copy(),'channels':channels,'trust':trust,'bound':bound,'n':n}


def solve(cache,target,old_bias):
    n=cache['n'];trust=cache['trust'];r,_,width=target.shape;answer=np.empty((r,width));objective=np.zeros((r,width))
    for channel,part in enumerate(cache['channels']):
        t=target[:,:,channel];old=old_bias[:,channel]
        linear=part['linear_base'][None]-t@part['slopes']-trust*n*old[:,None]
        trial=np.clip(-linear/part['curvature'][None],part['left'][None],part['right'][None])
        # Exact polynomial evaluation; constant term included for cross-interval comparison.
        constant=np.sum(part['offsets']**2,axis=0)[None]-2*t@part['offsets']+np.sum(t*t,axis=1)[:,None]+trust*n*old[:,None]**2
        value=part['curvature'][None]*trial**2+2*linear*trial+constant
        idx=np.argmin(value,axis=1);answer[:,channel]=trial[np.arange(r),idx]
        z=cache['c'][:,channel][None]+answer[:,channel,None]
        objective[:,channel]=np.sum((activation(z)-t)**2,axis=1)+trust*n*(answer[:,channel]-old)**2
    return answer,{'conditional_objective':objective,'cache_bytes':sum(a.nbytes for p in cache['channels'] for a in p.values())+cache['c'].nbytes,
        'max_intervals':max(len(p['left']) for p in cache['channels'])}
