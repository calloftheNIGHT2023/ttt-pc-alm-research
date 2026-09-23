"""Read-only full residual history on the unchanged ALM discovery trajectory."""
import hashlib
import time
from unittest.mock import patch
import numpy as np
import diagnose_h2_full_bank_ceiling as dense
import light_h2_credit as light

VARIANTS=('history_full','history_uniform12','history_recent12')
SELECTED={'history_full':tuple(range(1,17)),
          'history_uniform12':(2,3,4,6,7,8,10,11,12,14,15,16),
          'history_recent12':tuple(range(5,17))}


def digest_event(hasher,label,arrays):
    hasher.update(label.encode())
    for a in arrays:
        hasher.update(str((a.shape,a.dtype.str)).encode());hasher.update(a.tobytes())


class Bridge:
    def __init__(self,observer,x,collect):
        self.observer=observer;self.x=x;self.collect=collect;self.hasher=hashlib.sha256()
        self.residual=[];self.dual=[];self.shadow=None;self.shadow_checks=0;self.events=0;self.collection_seconds=0.

    def parameter(self,b,step):
        digest_event(self.hasher,f'parameter:{step}',[b]);self.events+=1;self.observer.parameter(b,step)

    def activity(self,b,h,u,step,phase):
        digest_event(self.hasher,f'activity:{step}:{phase}',[b,h,u]);self.events+=1
        self.observer.activity(b,h,u,step,phase)
        if not self.collect or phase!='after_parameters_before_dual':return
        started=time.perf_counter();prev=np.broadcast_to(self.x,(len(b),len(self.x)));rr=[]
        for j in range(len(h)):
            rr.append(h[j]-light.base.g(prev+b[:,j,None]));prev=h[j]
        residual=np.array(rr)
        if self.shadow is None:self.shadow=np.zeros_like(u)
        assert self.shadow.tobytes()==u.tobytes(),('shadow mismatch',step)
        self.shadow_checks+=1;self.residual.append(np.ascontiguousarray(residual.transpose(1,0,2)))
        self.dual.append(np.ascontiguousarray(u.transpose(1,0,2)).copy())
        self.shadow+=.5*residual;self.collection_seconds+=time.perf_counter()-started


def capture(x,v,collect=True):
    previous=light.trace.refine;bridges=[];best=[]
    def watched(starts,x,v,anchor,method,sweeps,observer):
        assert method=='alm' and sweeps==16
        bridge=Bridge(observer,x,collect);bridges.append(bridge)
        answer=previous(starts,x,v,anchor,method,sweeps,bridge);best.append(answer.copy());return answer
    started=time.perf_counter()
    with patch.object(light.trace,'refine',watched):data,meta,regs,bank,labels,steps=dense.capture(x,v,light.config('alm','native'))
    assert len(bridges)==len(best)==1;bridge=bridges[0]
    details=dict(trajectory_sha256=bridge.hasher.hexdigest(),best_sha256=hashlib.sha256(best[0].tobytes()).hexdigest(),
        events=bridge.events,shadow_checks=bridge.shadow_checks,collection_seconds=bridge.collection_seconds,
        diagnostic_capture_seconds=time.perf_counter()-started)
    arrays=dict(residual=np.array(bridge.residual),dual=np.array(bridge.dual),best=best[0])
    if collect:assert arrays['residual'].shape==arrays['dual'].shape==(16,64,4,len(x)) and bridge.shadow_checks==16
    return data,meta,regs,bank,labels,steps,arrays,details


def normalize(raw,threshold=False):
    norm=np.max(abs(raw),axis=(1,2));keep=np.linalg.norm(raw,axis=(1,2))>1e-14 if threshold else norm>0
    selected=[];seen={};mapping=np.full(len(raw),-1,np.int32);rows=[]
    for i in np.flatnonzero(keep):
        row=raw[i]/norm[i];key=row.tobytes()
        if key not in seen:seen[key]=len(rows);selected.append(i);rows.append(row)
        mapping[i]=seen[key]
    return np.array(rows),np.array(selected,dtype=np.int32),mapping,norm


def build(history):
    residual=history['residual'];dual=history['dual'];t,restarts,d,n=residual.shape
    assert t==16 and restarts==64
    arrays={};metadata={}
    for name in VARIANTS:
        chosen=np.array(SELECTED[name])-1;raw=residual[chosen].reshape(-1,d,n);bank,ids,mapping,norm=normalize(raw)
        arrays[name]=bank
        if name=='history_full':arrays.update(full_map=mapping,full_norm=norm)
        metadata[name]=dict(steps=SELECTED[name],raw_rows=len(raw),directions=len(bank),archive_raw_bytes=raw.nbytes,
            bank_bytes=bank.nbytes,step_index_bytes=chosen.nbytes,scope='Raw archive subtotal, not deployable allocator peak')
    raw=[];steps=[];kinds=[];restart=[]
    for step in [4,8,12,16]:
        for kind,value in enumerate([residual[step-1],dual[step-1],dual[step-1]+residual[step-1]]):
            raw.extend(value);steps.extend([step]*restarts);kinds.extend([kind]*restarts);restart.extend(range(restarts))
    native,ids,_,norm=normalize(np.array(raw),threshold=True)
    arrays.update(native=native,native_steps=np.array(steps,np.int32)[ids],native_kinds=np.array(kinds,np.int8)[ids],
        native_restart=np.array(restart,np.int32)[ids],native_norm=norm[ids])
    return arrays,metadata


def equal_tree(a,b):
    if isinstance(a,np.ndarray):assert isinstance(b,np.ndarray) and a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for key in a:equal_tree(a[key],b[key])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):equal_tree(x,y)
    else:assert a==b
