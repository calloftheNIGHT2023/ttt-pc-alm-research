"""Full existing checkpoint credits, causal H2 waves, shared layer bounds.

No compression, future candidate pool, query answers, or changed trajectories.
All learners have their own credit bank and receive the same factorization.
"""
import hashlib
import time
import numpy as np
import light_h2_credit as light
import factorized_credit_bank as factor

prior=light.prior
base=light.base
neighbor=light.neighbor


def configs():
    answer=[]
    for learner in ['alm','adam60','pc','nodual','direct4096']:
        for rounds in [5,20]:
            answer.append(light.config(learner,f'c{rounds}'))
            if learner=='direct4096':continue
            for mode in (['native','residual'] if learner in ['alm','adam60'] else ['native']):
                cfg=light.config(learner,mode)
                cfg.update(name=f'{learner}_{mode}_full_c{rounds}',factorized_full=True,contract_rounds=rounds,
                    direction_budget=1024)
                answer.append(cfg)
    return answer


class FullSnapshots(light.Snapshots):
    def directions(self):
        start=time.perf_counter();shape=(4,len(self.x))
        raw=np.concatenate(self.values) if self.values else np.empty((0,*shape))
        labels=[e['label'] for a,e in zip(self.values,self.events) for _ in a]
        steps=[e['step'] for a,e in zip(self.values,self.events) for _ in a]
        keep=np.linalg.norm(raw,axis=(1,2))>1e-14;ids=np.flatnonzero(keep)
        raw=raw[keep];raw/=np.max(abs(raw),axis=(1,2),keepdims=True)
        unique=[];seen=set()
        for i,row in enumerate(raw):
            key=row.tobytes()
            if key not in seen:seen.add(key);unique.append(i)
        bank=raw[unique]
        assert len(bank)<=self.cfg['direction_budget']
        return bank,dict(selection_seconds=time.perf_counter()-start,retained_rows=sum(len(a) for a in self.values),
            nonzero_rows=len(raw),selected_rows=len(bank),retained_snapshot_bytes=sum(a.nbytes for a in self.values),
            selected_bank_bytes=bank.nbytes,snapshot_events=self.events,collection_seconds=self.seconds,
            checkpoints=sorted(self.checkpoints),labels=[labels[ids[i]] for i in unique],
            steps=[steps[ids[i]] for i in unique],compression='none; normalized bytewise deduplication only')


def prepare(x,v,cfg):
    saved_discover=neighbor.discover;saved_contract=neighbor.previous.screen.contract;snapshots=light.Snapshots
    holder=[];proofs=[];calls=[];discovery_hash=[];construction=[]
    def discover(x,v,state,cfg):
        assert state is None
        neighbor.discover=saved_discover
        try:
            light.Snapshots=FullSnapshots
            bank,meta,directions=light.discover(x,v,cfg)
        finally:neighbor.discover=discover;light.Snapshots=snapshots
        start=time.perf_counter();holder.append(factor.Bank(x,v,directions));construction.append(time.perf_counter()-start)
        discovery_hash.append(hashlib.sha256(bank.tobytes()).hexdigest())
        return bank,meta
    def contract(x,v,regs,rounds=5):
        start=time.perf_counter();rounds=cfg['contract_rounds'];mask=saved_contract(x,v,regs,rounds)
        ids=np.flatnonzero(~mask);preliminary=time.perf_counter()-start;solver=holder[0]
        start=time.perf_counter();reject,pp,detail=solver.screen(regs[ids]);elapsed=time.perf_counter()-start
        mask[ids[reject]]=True
        for proof in pp:
            proof=dict(proof,wave=len(calls));proofs.append(proof)
        detail.update(seconds=elapsed,exact_checks=len(pp))
        calls.append(dict(input_cells=len(regs),contract_rounds=rounds,contract_rejected=len(regs)-len(ids),
            credit_rejected=len(pp),contraction_seconds=preliminary,credit=detail,
            input_pattern_hash=hashlib.sha256(regs.tobytes()).hexdigest(),
            screened_pattern_hash=hashlib.sha256(regs[ids].tobytes()).hexdigest()))
        return mask
    try:
        neighbor.discover=discover;neighbor.previous.screen.contract=contract
        data,meta=prior.h2_prepare(x,v,cfg)
    finally:neighbor.discover=saved_discover;neighbor.previous.screen.contract=saved_contract;light.Snapshots=snapshots
    solver=holder[0]
    return data,dict(meta,learner=cfg['learner'],credit_mode=cfg['credit_mode'],screen_calls=calls,credit_proofs=proofs,
        discovery_bank_sha256=discovery_hash[0],credit_bank=solver.bank.tolist(),
        factorized=dict(layer_solves=solver.layer_solves,region_rows=solver.region_rows,calls=solver.calls,
            bank_construction_seconds=construction[0],bank_bytes=solver.bank.nbytes,
            cached_values_numeric_bytes=sum(len(k)+v[0].nbytes+1 for cache in solver.cache for k,v in cache.items())),
        causal_waves=True)


def fit(x,v,state,cfg,rng,count=2048):
    if not cfg.get('factorized_full'):return light.fit(x,v,state,cfg,rng,count)
    if state is not None:return prior.fit(x,v,state,cfg,rng,count)
    data,meta=prepare(x,v,cfg);points,labels,sampling=prior.hybrid.sample(data,count,rng)
    anchor=points[0].copy();anchor.setflags(write=False);points.setflags(write=False);state=prior.State(anchor,points)
    return prior.posterior.make_predict(points),state,dict(meta,sampling=sampling,previous_state_digest=None,
        new_state_digest=prior.state_digest(state),persistent_state_bytes=anchor.nbytes+points.nbytes,
        actual_mode_keys=sorted({data['regs'][int(i)].tobytes().hex() for i in labels}))


def verify():
    import diagnose_h2_full_bank_ceiling as dense
    rng=np.random.default_rng(5900001);b=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=base.forward(xx,b)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    cases=0;waves=0;proof_count=0;original_screen=factor.Bank.screen
    def checked(self,regs):
        nonlocal waves,proof_count
        result=original_screen(self,regs)
        pp,_=dense.full_screen(self.x,self.v,regs,self.bank)
        assert pp==result[1];waves+=1;proof_count+=len(pp)
        return result
    try:
        factor.Bank.screen=checked
        for learner in ['alm','adam60','pc','nodual','direct4096']:
            baseline=light.config(learner,'c5');ref,rm=light.prepare(x,v,baseline)
            samples,labels,_=prior.hybrid.sample(ref,64,np.random.default_rng(482201))
            for cfg in [c for c in configs() if c['learner']==learner]:
                if cfg.get('factorized_full'):data,meta=prepare(x,v,cfg)
                else:data,meta=light.prepare(x,v,cfg)
                assert meta['discovery_bank_sha256']==rm['discovery_bank_sha256']
                assert meta['positive_mode_keys']==rm['positive_mode_keys'] and meta['completion_proposals']==rm['completion_proposals']
                assert [c['input_pattern_hash'] for c in meta['screen_calls']]==[c['input_pattern_hash'] for c in rm['screen_calls']]
                pp,ll,_=prior.hybrid.sample(data,64,np.random.default_rng(482201))
                assert np.array_equal(pp,samples) and np.array_equal(ll,labels);cases+=1
    finally:factor.Bank.screen=original_screen
    core=neighbor.previous.core;jac=base.forward_jacobian;refine=core.batched.refine
    def forbidden(*a,**k):raise AssertionError('global BP in local full-credit H2')
    try:
        base.forward_jacobian=forbidden;core.batched.refine=forbidden;state=None
        cfg=next(c for c in configs() if c['name']=='alm_native_full_c5')
        for n in [4,8,16,24]:
            _,state,_=fit(xx[:n],vv[:n],state,cfg,np.random.default_rng(482201+n),64)
    finally:base.forward_jacobian=jac;core.batched.refine=refine
    return dict(passed=True,configs_equivalent=cases,causal_waves_dense_proofs_equal=waves,proofs=proof_count,
        local_four_stage_no_global_bp=True,scope='old seed preflight; no timing claim')


if __name__=='__main__':
    import json
    print(json.dumps(verify(),indent=2),flush=True)
