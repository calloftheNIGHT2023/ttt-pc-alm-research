"""Wide-budget equivalence and deterministic artificial deadline boundaries."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import credit_wall_budget as model
import common_pool_credit as common
import credit_history_capture as history


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


class JumpClock:
    def __init__(self,target):self.target=target;self.value=0.
    def __call__(self,name,step,index):
        if (name,step,index)==self.target:self.value=.2
        return self.value


def checks(arrays,meta):
    budget=meta['budget_seconds'];assert all(p['completed_seconds']<=budget for p in meta['proofs'])
    assert all(p['completed_seconds']>budget for p in meta['late_proofs'])
    assert len(meta['proofs'])==int(np.sum(arrays['first_step']>0))
    assert meta['total_positive']==int(arrays['accepted'].sum())==meta['old_count']+meta['new_count']
    assert all(a['seconds']<=b['seconds'] for a,b in zip(meta['events'],meta['events'][1:]))
    if meta['old_count']:assert next(e for e in meta['events'] if e['name']=='old_after')['seconds']<=budget


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/credit_history/development/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_232_audit.json';assert json.loads(audit.read_text())['passed']
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=history.light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    captured=history.capture(x,v,True);regs=captured[2];banks,_=history.build(captured[6]);counts=dict(wide_cases=0,wide_arrays=0,replay_arrays=0,deadline_cases=0,late_positive_cases=0)
    native_meta=None;events=[]
    for name in ['native',*history.VARIANTS]:
        bank=banks[name];gold,gm=common.solve(x,v,regs,bank);arrays,meta=model.guarded_solve(x,v,regs,bank,budget=60.,max_steps=128)
        for key,value in gold.items():assert value.tobytes()==arrays[key].tobytes(),(name,key);counts['wide_arrays']+=1
        assert meta['old_proofs']==gm['old_proofs'] and [{k:v for k,v in p.items() if k!='completed_seconds'} for p in meta['proofs']]==gm['proofs']
        assert meta['stop_reason']=='step_cap';checks(arrays,meta)
        again,repeated=model.guarded_solve(x,v,regs,bank,budget=60.,max_steps=128,replay_events=meta['events'])
        for key,value in arrays.items():assert value.tobytes()==again[key].tobytes();counts['replay_arrays']+=1
        assert repeated['proofs']==meta['proofs'] and repeated['events']==meta['events'];counts['wide_cases']+=1
        if name=='native':native_meta=meta
    for name in ['old_before','old_after','state_ready','response_before','response_after','proof_before','proof_after','update_before','update_after']:
        event=next(e for e in native_meta['events'] if e['name']==name);target=(event['name'],event['step'],event['index'])
        arrays,meta=model.guarded_solve(x,v,regs,banks['native'],budget=.1,max_steps=128,clock=JumpClock(target));checks(arrays,meta)
        assert meta['stop_reason']=='deadline' and meta['stop_event']==name
        if name=='proof_after':assert meta['late_new_count']==1;counts['late_positive_cases']+=1
        repeated,rm=model.guarded_solve(x,v,regs,banks['native'],budget=.1,max_steps=128,replay_events=meta['events'])
        for key,value in arrays.items():assert value.tobytes()==repeated[key].tobytes();counts['replay_arrays']+=1
        events.append(dict(target=target,old=meta['old_count'],new=meta['new_count'],late_old=meta['late_old_count'],late_new=meta['late_new_count']));counts['deadline_cases']+=1
    tiny,tm=model.guarded_solve(x,v,regs,banks['native'],budget=0.,max_steps=128);checks(tiny,tm);assert tm['total_positive']==0
    for name in ['audit_credit_history.py','audit_credit_history_resources.py','plot_credit_history.py','audit_research_round_232.py',
                 'credit_wall_budget.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    result=dict(passed=True,checks=counts,zero_budget_positive=tm['total_positive'],artificial_boundaries=events,source_sha256=hashes,
        parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit),design_sha256=sha(root/'outputs/ttt-pc-alm-research/233_credit_wall_budget_protocol.md'),
        scope='Old-seed deterministic semantic checks; no task superiority claim')
    out=root/'results/credit_wall_budget/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
