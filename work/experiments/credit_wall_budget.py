"""Cooperative wall-clock cutoff with deterministic event-clock replay.

Only certificates completed by the deadline are admitted. Already running
atomic operations can overrun; all observed time and late work are recorded.
"""
import time
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
import credit_moment_step as moment
from online_endpoint_h2 import EndpointIntervalBank
import verify_finite_credit_game as guard


class Gate:
    def __init__(self,budget,clock=None,replay=None):
        self.started=time.perf_counter();self.budget=budget;self.clock=clock;self.replay=replay;self.events=[];self.position=0
    def stamp(self,name,step=0,index=-1):
        if self.replay is not None:
            row=self.replay[self.position];assert (row['name'],row['step'],row['index'])==(name,step,index)
            elapsed=row['seconds'];self.position+=1
        elif self.clock is not None:elapsed=float(self.clock(name,step,index))
        else:elapsed=time.perf_counter()-self.started
        assert elapsed>=0 and (not self.events or elapsed>=self.events[-1]['seconds'])
        self.events.append(dict(name=name,step=int(step),index=int(index),seconds=elapsed));return elapsed
    def complete(self):
        if self.replay is not None:assert self.position==len(self.replay)


def solve(x,v,regs,bank,budget,max_steps=4096,clock=None,replay_events=None):
    gate=Gate(budget,clock,replay_events);assert budget>=0 and max_steps>0
    count_all,d,n=regs.shape;k=len(bank);assert k>0 and np.isfinite(bank).all() and np.max(abs(bank))<=1
    raw_old=np.zeros(count_all,bool);old=np.zeros(count_all,bool);old_proofs=[];raw_old_proofs=[];late_old=[]
    stop_reason='step_cap';stop_event=None;screen_seconds=0.
    before=gate.stamp('old_before')
    if before<budget:
        screen=EndpointIntervalBank(x,v,bank);raw_old,raw_old_proofs,_=screen.screen(regs)
        ended=gate.stamp('old_after');screen_seconds=ended-before
        if ended<=budget:old=raw_old.copy();old_proofs=raw_old_proofs
        else:late_old=raw_old_proofs;stop_reason='deadline';stop_event='old_after'
    else:stop_reason='deadline';stop_event='old_before'
    indices=np.flatnonzero(~old);rr=regs[indices];count=len(rr);flat=bank.reshape(k,d*n)
    logits=np.zeros((count,k));best_weights=np.zeros_like(logits);best_credit=np.zeros((count,d,n));best=np.full(count,-np.inf)
    first=np.zeros(count,np.int32);best_step=np.zeros(count,np.int32);terminal=np.zeros(count,np.int8);stop_step=np.zeros(count,np.int32)
    oracle_count=np.zeros(count,np.int32);mean_count=np.zeros(count,np.int64);variance_count=np.zeros(count,np.int64)
    max_beta=np.zeros(count);prefix_values=np.full((count,len(moment.PREFIXES)),-np.inf)
    proofs=[];late_proofs=[];exact_checks=0;late_exact_checks=0;responses=0;updates=0;last_step=0
    ready=gate.stamp('state_ready')
    if stop_reason!='deadline' and ready>=budget:stop_reason='deadline';stop_event='state_ready'
    if stop_reason!='deadline':
        for step in range(1,max_steps+1):
            ids=np.flatnonzero(terminal==0)
            if not len(ids):
                for j,prefix in enumerate(moment.PREFIXES):
                    if prefix>=step:prefix_values[:,j]=best
                stop_reason='all_terminal';break
            if gate.stamp('response_before',step)>=budget:stop_reason='deadline';stop_event='response_before';break
            last_step=step;weights=moment.probabilities(logits[ids]);a=(weights@flat).reshape(len(ids),d,n)
            out=moment.response(x,v,rr[ids],a);oracle_count[ids]+=1;responses+=1
            assert out['valid'].all() and np.max(abs(out['residual']))<=1+1e-12
            if gate.stamp('response_after',step)>=budget:stop_reason='deadline';stop_event='response_after';break
            values=out['objective'];improved=values>best[ids];ii=ids[improved]
            best[ii]=values[improved];best_weights[ii]=weights[improved];best_credit[ii]=a[improved];best_step[ii]=step
            for local in np.flatnonzero(values>0):
                index=int(ids[local])
                if gate.stamp('proof_before',step,index)>=budget:stop_reason='deadline';stop_event='proof_before';break
                exact=moment.original.exact_optimum(x,v,rr[index],a[local]);exact_checks+=1
                completed=gate.stamp('proof_after',step,index)
                proof=dict(index=index,step=step,pattern=rr[index].tobytes().hex(),exact=exact,completed_seconds=completed)
                if completed>budget:
                    late_exact_checks+=1
                    if exact['positive']:late_proofs.append(proof)
                    stop_reason='deadline';stop_event='proof_after';break
                if exact['positive']:
                    assert 'empty_layer' not in exact
                    terminal[index]=1;stop_step[index]=step;first[index]=step
                    best_weights[index]=weights[local];best_credit[index]=a[local];best[index]=values[local];best_step[index]=step;proofs.append(proof)
            if stop_reason=='deadline':break
            if gate.stamp('update_before',step)>=budget:stop_reason='deadline';stop_event='update_before';break
            todo=np.flatnonzero(terminal[ids]==0);gains=out['residual'][todo].reshape(len(todo),d*n)@flat.T
            if len(todo):
                target=ids[todo];new,um=moment.update(logits[target],gains);updates+=1
                logits[target]=new;mean_count[target]+=um['mean_count'];variance_count[target]+=um['variance_count']
                max_beta[target]=np.maximum(max_beta[target],um['beta'])
                for status,code in [(1,2),(2,3),(3,4),(4,5)]:
                    retired=target[um['status']==status];terminal[retired]=code;stop_step[retired]=step
            if step in moment.PREFIXES:prefix_values[:,moment.PREFIXES.index(step)]=best
            if gate.stamp('update_after',step)>=budget:stop_reason='deadline';stop_event='update_after';break
    accepted=old.copy();accepted[indices[first>0]]=True
    arrays=dict(credit=best_credit,weights=best_weights,first_step=first,best_step=best_step,best_value=best,prefix_values=prefix_values,
        terminal=terminal,stop_step=stop_step,oracle_count=oracle_count,mean_count=mean_count,variance_count=variance_count,max_beta=max_beta,
        indices=indices,old_positive=old,accepted=accepted,raw_old_positive=raw_old)
    meta=dict(budget_seconds=budget,max_steps=max_steps,delta=moment.DELTA,old_proofs=old_proofs,late_old_proofs=late_old,proofs=proofs,late_proofs=late_proofs,
        old_count=int(old.sum()),new_count=len(proofs),total_positive=int(accepted.sum()),late_old_count=len(late_old),late_new_count=len(late_proofs),
        regions=count,directions=k,old_stage_seconds=screen_seconds,response_batches=responses,update_batches=updates,last_step=last_step,
        oracle_pairs=int(oracle_count.sum()),mean_evaluations=int(mean_count.sum()),variance_evaluations=int(variance_count.sum()),exact_checks=exact_checks,
        late_exact_checks=late_exact_checks,stop_reason=stop_reason,stop_event=stop_event,events=gate.events,
        target_limited=int(np.sum(terminal==2)),zero_variance=int(np.sum(terminal==3)),nonfinite_failures=int(np.sum(terminal==4)),already_at_target=int(np.sum(terminal==5)),
        time_source='replay' if replay_events is not None else 'artificial' if clock is not None else 'wall',
        scope='Cooperative cutoff; excludes all late certificates, includes atomic overrun; not a hard real-time return bound')
    elapsed=gate.stamp('finish');gate.complete();meta.update(total_seconds=elapsed,overrun_seconds=max(0.,elapsed-budget))
    return arrays,meta


def guarded_solve(x,v,regs,bank,**kwargs):
    wrapped=SimpleNamespace(solve=lambda xx,vv,rr,bb:solve(xx,vv,rr,bb,**kwargs))
    with patch.object(guard,'model',wrapped):return guard.guarded_solve(x,v,regs,bank)
