"""Delay row-independent C20 only; causal first-arrival clauses stay online."""
import time
import numpy as np
import conflict_feedback_memory as model


class Router:
    def __init__(self,x,v,chunk_size=256):
        self.x=x;self.v=v;self.chunk_size=chunk_size;self.seen=set();self.pending={};self.kept={};self.flushed=False
        self.stats=dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.)
        self.c20_calls=0;self.c20_rows=0;self.max_c20_batch=0;self.peak_pending_keys=0

    def observe(self,regs,clauses,event,restart,old_proofs):
        assert not self.flushed
        start=time.perf_counter();keys=list(dict.fromkeys(r.tobytes() for r in regs));fresh=[k for k in keys if k not in self.seen];self.seen.update(fresh)
        self.stats['candidate_mode_visits']+=len(regs);self.stats['distinct_modes']+=len(fresh)
        if fresh:
            arr=np.array([np.frombuffer(k,np.uint8).reshape(len(regs[0]),len(self.x)) for k in fresh])
            mask=model.conflict.clause_mask(arr,clauses);self.stats['causal_clause_removed']+=int(mask.sum())
            for reg in arr[~mask]:self.pending[reg.tobytes()]=dict(event=event,restart=restart,old_proofs=old_proofs)
        self.peak_pending_keys=max(self.peak_pending_keys,len(self.pending));self.stats['routing_seconds']+=time.perf_counter()-start

    def flush(self):
        assert not self.flushed
        start=time.perf_counter();keys=list(self.pending);d=len(keys[0])//len(self.x) if keys else 4
        for first in range(0,len(keys),self.chunk_size):
            part=keys[first:first+self.chunk_size];arr=np.array([np.frombuffer(k,np.uint8).reshape(d,len(self.x)) for k in part])
            reject=model.conflict.screen.contract(self.x,self.v,arr,20)
            self.c20_calls+=1;self.c20_rows+=len(part);self.max_c20_batch=max(self.max_c20_batch,len(part));self.stats['c20_removed']+=int(reject.sum())
            for key,bad in zip(part,reject):
                if not bad:self.kept[key]=self.pending[key]
        self.flushed=True;self.stats['routing_seconds']+=time.perf_counter()-start
        assert len(self.seen)==len(self.kept)+self.stats['causal_clause_removed']+self.stats['c20_removed']
        return self.kept


def verify():
    from diagnose_light_tied_routing import route
    rng=np.random.default_rng(811693);rows=0;cases=0;visits=0
    for n in [2,4,8]:
        for count in [1,17,257,519]:
            x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);regs=rng.integers(0,4,(count,4,n),dtype=np.uint8)
            whole=model.conflict.screen.contract(x,v,regs,20)
            chunks=np.concatenate([model.conflict.screen.contract(x,v,regs[i:i+31],20) for i in range(0,count,31)])
            assert np.array_equal(whole,chunks);rows+=count
            deferred=Router(x,v);seen=set();kept={};stats=dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.)
            for event,part in enumerate(np.array_split(regs,min(9,count)),1):
                clauses=[] if event<4 else [dict(positions=[0],codes=[2])]
                part=np.concatenate([part,regs[:min(5,count)]])
                route(x,v,part,clauses,seen,kept,stats,event,0,len(clauses));deferred.observe(part,clauses,event,0,len(clauses));visits+=len(part)
            deferred.flush();assert list(kept.items())==list(deferred.kept.items()) and seen==deferred.seen
            for key in stats:
                if key!='routing_seconds':assert stats[key]==deferred.stats[key]
            cases+=1
    return dict(passed=True,batch_partition_rows=rows,causal_prefix_cases=cases,candidate_visits=visits,
        scope='row independence and exact queue equivalence; synthetic clauses check mechanics, not learned-certificate validity; no task superiority')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2))
