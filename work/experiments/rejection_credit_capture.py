"""Same-trajectory local/BP/residual banks for volume-weighted rejection."""
import time
import numpy as np
import deferred_typed_memory as model
import retained_credit_memory as retained

# Nested prepare() functions temporarily replace retained.Collector with a
# factory. Capture the real class before entering any such patch scope.
WatchCollector = retained.Collector


def capture(x,v,cfg,include_bp=True):
    saved=model.Collector
    class Observer(saved):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs)
            self.rejection_watchers={'residual':WatchCollector(self.x,self.v,'residual')}
            if include_bp:self.rejection_watchers['bp']=WatchCollector(self.x,self.v,'bp')
        def parameter(self,b,*args,**kwargs):
            super().parameter(b,*args,**kwargs)
            for watcher in self.rejection_watchers.values():watcher.parameter(b,*args,**kwargs)
        def activity(self,b,h,u=None,step=0,phase=''):
            super().activity(b,h,u,step,phase)
            for watcher in self.rejection_watchers.values():watcher.activity(b,h,u,step,phase)
    try:model.Collector=Observer;result=model.prepare(x,v,cfg)
    finally:model.Collector=saved
    roles={'local':result[3],'residual':[c.rejection_watchers['residual'] for c in result[3]]}
    if include_bp:roles['bp']=[c.rejection_watchers['bp'] for c in result[3]]
    return result,roles


def pool(result):
    keys={r.tobytes() for r in result[1]}
    for collector in result[3]:
        keys.update(collector.forward);keys.update(collector.split);keys.update(collector.deferred.seen)
    return np.array([np.frombuffer(k,np.uint8).reshape(4,len(result[3][0].x)) for k in sorted(keys)],np.uint8)


def screen(x,v,regs,volumes,collectors,cap=32):
    begin=time.perf_counter();reject,matching,matchmeta=retained.matching(x,v,regs,collectors)
    selected=sorted(matching,key=lambda p:(-volumes[p['index']],p['pattern']))
    bank=[];seen=set()
    for proof in selected:
        a=np.array(proof['a']);a/=np.max(np.abs(a));token=a.tobytes()
        if token in seen:continue
        seen.add(token);bank.append(a)
        if len(bank)==cap:break
    bank=np.array(bank).reshape(-1,4,len(x))
    cross,crossproof,crossmeta=retained.normal.screen_bank(x,v,regs,bank)
    return reject|cross,dict(matching_proofs=matching,cross_proofs=crossproof,matching_meta=matchmeta,
                             cross_meta=crossmeta,directions=bank.tolist(),seconds=time.perf_counter()-begin,
                             retained_numeric_bytes=sum(len(k[0])+8+item['a'].nbytes for c in collectors for k,item in c.retained.items()),
                             bank_direction_bytes=bank.nbytes)
