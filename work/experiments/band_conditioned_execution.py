"""Preserve initialization evidence even when shared recovery replaces a fit."""
import band_conditioned_online as model

def fit(x,v,state,cfg,rng,count=2048):
    saved=model.conditioned_discover;attempts=[]
    def observed(*args,**kwargs):
        bank,meta=saved(*args,**kwargs)
        if 'band_initialization' in meta:attempts.append(meta['band_initialization'])
        return bank,meta
    try:
        model.conditioned_discover=observed
        pred,new,meta=model.fit(x,v,state,cfg,rng,count)
    except Exception as exc:
        exc.conditioning_attempts=attempts
        raise
    finally:model.conditioned_discover=saved
    return pred,new,dict(meta,conditioning_attempts=attempts)
