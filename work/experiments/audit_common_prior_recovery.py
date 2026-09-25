"""Independent file/state/readout audit, including incomplete recovery paths."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(x,points):
    h=np.broadcast_to(x,(len(points),len(x))).copy()
    for j in range(4):
        h=2*(h+points[:,j,None])
        h=np.maximum(np.minimum(h,2-h),0)
    return h


def digest(z):return hashlib.sha256(z['anchor'].tobytes()+z['points'].tobytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/common_prior_recovery/development';old=root/'results/online_interval_h2/development'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    assert sha(old/'failures.json')==p['old_failures_sha256']
    rows=json.loads((inp/'rows.json').read_text());episodes=json.loads((inp/'episodes.json').read_text())
    pre=json.loads((inp/'rows_without_query.json').read_text());assert len(pre)==len(rows)
    for a,b in zip(pre,rows):assert a=={k:v for k,v in b.items() if k!='query_mse'} and 'query_mse' not in a
    checks=dict(initial_states=0,recovered_states=0,support_particles=0,prediction_replays=0,risk_replays=0,charged_attempts=0,own_state_links=0)
    for ep in episodes:
        path=old/ep['initial_state_file'];assert sha(path)==ep['initial_state_sha256']
        with np.load(path) as z:previous=digest(z)
        checks['initial_states']+=1
        rr=[r for r in rows if (r['seed'],r['method'],r['repetition'])==(ep['seed'],ep['method'],ep['repetition'])]
        assert len(rr)==ep['stages'] and [r['n'] for r in rr]==p['stages'][:len(rr)]
        if ep['complete']:assert len(rr)==3
        else:assert ep['failed_n']==p['stages'][len(rr)]
        for row in rr:
            file=inp/row['state_file'];assert sha(file)==row['state_sha256']
            with np.load(file) as z:
                assert row['previous_state_digest']==previous and row['state_digest']==digest(z);previous=digest(z);checks['own_state_links']+=1
                points=z['points'];assert len(points)==p['posterior_samples'] and points.max()<=.12+1e-12 and points.min()>=-.12-1e-12
                assert np.max(abs(evaluate(z['x'],points)-z['v']))<.001+1e-8;checks['support_particles']+=len(points)
                replay=np.zeros(len(z['q']))
                for first in range(0,len(points),179):replay+=evaluate(z['q'],points[::-1][first:first+179]).sum(0)/len(points)
                assert np.max(abs(replay-z['prediction']))<1e-10;checks['prediction_replays']+=1
                truth=evaluate(z['q'],np.random.default_rng(row['seed']).uniform(-.12,.12,(1,4)))[0]
                risk=float(np.trapezoid((replay-truth)**2,x=z['q']));assert abs(risk-row['query_mse'])<1e-12;checks['risk_replays']+=1
            rec=row['recovery'];events=rec['attempts'];assert events[-1]['success'] and all(not e['success'] for e in events[:-1])
            assert all(e['seconds']>=0 for e in events) and sum(e['seconds'] for e in events)<=row['write_seconds']+1e-6
            if rec['triggered']:
                assert events[0]['role']=='original' and events[0]['error']=='Geometry found no nonzero cell'
                assert [e['features'] for e in events[1:]]==p['feature_budgets'][:len(events)-1]
                assert rec['all_observations_reused']==row['n'];checks['recovered_states']+=1
            else:assert len(events)==1 and events[0]['role']=='original'
            checks['charged_attempts']+=len(events)
    assert checks['risk_replays']==len(rows) and run['checks']['old_failures_reproduced']==len(episodes)==8
    result=dict(passed=True,checks=checks,complete_paths=sum(e['complete'] for e in episodes),total_paths=len(episodes),
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),rows_sha256=sha(inp/'rows.json'),
        scope='Independent state and query audit of selected old failure paths, not global posterior completeness or full comparison')
    (inp/'independent_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
