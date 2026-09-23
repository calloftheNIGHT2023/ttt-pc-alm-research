"""Read all already-executed baseline credits without changing old observers."""
import hashlib
import time
from unittest.mock import patch
import numpy as np
import light_h2_credit as light
import diagnose_h2_full_bank_ceiling as dense
from credit_history_capture import digest_event,normalize,equal_tree

LEARNERS=('pc','nodual','adam60')
VARIANTS=('pc_history_full','nodual_history_full','adam60_history_full','strong_history_union')


def capture(x,v,learner,collect=True):
    assert learner in LEARNERS
    parent=light.Snapshots;instances=[]
    class Shadow(parent):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.digest=hashlib.sha256();self.trace_events=0;self.evaluation_calls=0
            self.extra=[];self.extra_labels=[];self.extra_steps=[];self.collection_seconds=0.;instances.append(self)

        def record(self,label,arrays):
            digest_event(self.digest,label,arrays);self.trace_events+=1

        def save(self,value,label,step):
            self.extra.append(np.ascontiguousarray(value).copy());self.extra_labels.extend([label]*len(value));self.extra_steps.extend([step]*len(value))

        def parameter(self,b,step=0):
            self.record(f'parameter:{step}',[b]);return super().parameter(b,step)

        def activity(self,b,h,u=None,step=0,phase=''):
            self.record(f'activity:{step}:{phase}',[b,h]+([] if u is None else [u]));result=super().activity(b,h,u,step,phase)
            if collect and phase=='after_parameters_before_dual':
                started=time.perf_counter();prev=np.broadcast_to(self.x,(len(b),len(self.x)));rr=[]
                for j in range(len(h)):rr.append(h[j]-light.base.g(prev+b[:,j,None]));prev=h[j]
                self.save(np.array(rr).transpose(1,0,2),'residual',int(step));self.collection_seconds+=time.perf_counter()-started
            return result

        def evaluate(self,bank,x,v,with_jacobian=True):
            self.record(f'evaluate_in:{self.evaluation_calls}:{with_jacobian}',[bank,x,v]);result=super().evaluate(bank,x,v,with_jacobian)
            self.record(f'evaluate_out:{self.evaluation_calls}:{with_jacobian}',[a for a in result if a is not None]);self.evaluation_calls+=1
            # Parent already computed these credits on this exact call, including
            # its final with_jacobian=False evaluation. No extra forward/BP call.
            if collect:
                started=time.perf_counter();r,d=bank.shape;residual=np.zeros((r,d,len(x)));residual[:,-1]=-result[1]
                self.save(residual,'residual',self.steps);self.save(self.previous,'current_bp',self.steps)
                self.save(self.history,'history_bp',self.steps);self.save(self.previous+self.history,'combined_bp',self.steps)
                self.collection_seconds+=time.perf_counter()-started
            return result

    started=time.perf_counter()
    with patch.object(light,'Snapshots',Shadow):old=dense.capture(x,v,light.config(learner,'native'))
    elapsed=time.perf_counter()-started;assert len(instances)==1;shadow=instances[0]
    if learner=='adam60':assert shadow.evaluation_calls==61 and shadow.trace_events==122
    else:assert shadow.evaluation_calls==0 and shadow.trace_events==(242 if learner=='pc' else 50)
    details=dict(learner=learner,collect=collect,trajectory_sha256=shadow.digest.hexdigest(),trace_events=shadow.trace_events,
        evaluation_calls=shadow.evaluation_calls,collection_seconds=shadow.collection_seconds,diagnostic_capture_seconds=elapsed,
        scope='Read-only observer overhead included, not deployment timing')
    arrays={}
    if collect:
        raw=np.concatenate(shadow.extra);bank,selected,mapping,norm=normalize(raw)
        arrays=dict(raw=raw,bank=bank,raw_labels=np.array(shadow.extra_labels),raw_steps=np.array(shadow.extra_steps,dtype=np.int32),
            labels=np.array(shadow.extra_labels)[selected],steps=np.array(shadow.extra_steps,dtype=np.int32)[selected],selected=selected,mapping=mapping,norm=norm)
        keys={row.tobytes():i for i,row in enumerate(bank)};old_mapping=np.array([keys[row.tobytes()] for row in old[3]],dtype=np.int32)
        assert len(raw)==dict(pc=5120,nodual=1024,adam60=15616)[learner]
        assert all(old[3][i].tobytes()==bank[j].tobytes() for i,j in enumerate(old_mapping));arrays['old_mapping']=old_mapping
        details.update(raw_rows=len(raw),directions=len(bank),raw_bytes=raw.nbytes,bank_bytes=bank.nbytes,
            diagnostic_array_bytes=sum(a.nbytes for a in arrays.values()),old_directions=len(old[3]),old_subset_bytewise=True)
    return old,arrays,details


def verify_pair(reference,actual):
    a,_,am=reference;b,_,bm=actual;equal_tree(a[0],b[0])
    for i in [2,3,4,5]:assert a[i].shape==b[i].shape and a[i].dtype==b[i].dtype and a[i].tobytes()==b[i].tobytes()
    for k in ['credit_bank','credit_proofs','positive_mode_keys','discovery_bank_sha256','completion_proposals']:assert a[1][k]==b[1][k]
    for k in ['trajectory_sha256','trace_events','evaluation_calls']:assert am[k]==bm[k]
    return dict(trajectory_events=am['trace_events'],evaluation_calls=am['evaluation_calls'],old_arrays=4,old_metadata_fields=5,final_data_equal=True)
