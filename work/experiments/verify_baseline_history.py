"""Read-only full-history preflight: old 5900001, no new research samples."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
import baseline_history_capture as history
import common_pool_credit as common
import exact_credit_hull as hull
from verify_exact_credit_hull import validate


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/exact_credit_hull/development/protocol.json';p=json.loads(parent.read_text());hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_236_audit.json';assert json.loads(audit.read_text())['passed']
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=history.light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    captures=[];banks={};allregs=set();checks=Counter()
    for learner in history.LEARNERS:
        reference=history.capture(x,v,learner,False);actual=history.capture(x,v,learner,True);q=history.verify_pair(reference,actual)
        old,arrays,meta=actual;checks['trajectory_events']+=q['trajectory_events'];checks['evaluation_calls']+=q['evaluation_calls'];checks['old_arrays']+=q['old_arrays']
        checks['old_direction_inclusions']+=len(arrays['old_mapping']);banks[learner+'_history_full']=arrays['bank'];allregs.update(r.tobytes() for r in old[2])
        captures.append(dict(learner=learner,reference=reference[2],actual=meta,checks=q));print(json.dumps(captures[-1]),flush=True)
    banks['strong_history_union']=np.concatenate([banks[m+'_history_full'] for m in history.LEARNERS])
    regs=np.array([np.frombuffer(b,np.uint8).reshape(4,4) for b in sorted(allregs)])
    # Deterministic small structural preflight; main experiment still covers all 700.
    ids=sorted({0,1,len(regs)//2,len(regs)-1});rows=[];statuses={}
    for method in history.VARIANTS:
        bank=banks[method];rb=hull.RationalBank(bank);counter=Counter()
        for index in ids:
            reg=regs[index];result=hull.solve(x,v,reg,rb);check=validate(x,v,reg,bank,result);counter[result['status']]+=1;checks['hull_cases']+=1
            rows.append(dict(method=method,index=index,pattern=reg.tobytes().hex(),result=result,verified=check))
        statuses[method]=dict(counter);print(json.dumps(dict(method=method,statuses=counter)),flush=True)
    for name in ['audit_exact_credit_hull.py','analyze_exact_credit_separations.py','plot_exact_credit_hull.py','plot_exact_credit_hull_v2.py',
                 'audit_research_round_236.py','baseline_history_capture.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    out=root/'results/baseline_history/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'records.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(passed=True,checks=dict(checks),captures=captures,preflight_pool_regions=len(regs),preflight_indices=ids,statuses=statuses,source_sha256=hashes,
        parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit),design_sha256=sha(root/'outputs/ttt-pc-alm-research/237_full_baseline_history_protocol.md'),records_sha256=sha(out/'records.json'))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k not in ['source_sha256','captures']}),flush=True)


if __name__=='__main__':main()
