"""Frozen causal four-stage benchmark, equal outputs and explicit failures."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import online_endpoint_h2 as memory
import run_light_h2_credit as driver


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/light_h2_credit/development';out=root/'results/online_endpoint_h2/development'
    parent=json.loads((inp/'protocol.json').read_text());bp=root/'results/endpoint_factorization/development/protocol.json'
    hashes=dict(json.loads(bp.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(memory.__file__)]:hashes[path.name]=sha(path)
    cfgs=memory.configs()+[dict(name='prior4096_ridge',family='regression')]
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),component_protocol_sha256=sha(bp),
        configs=cfgs,seeds=parent['seeds'],repetitions=2,stages=parent['stages'],posterior_samples=parent['posterior_samples'],
        rng_seed=parent['rng_seed'],order_seed=482511,query_points=257,verification=memory.verify(),
        scope='same 16 development tasks; endpoint-interval composition against bare C5/C20 and rational/interval C5 controls; all own learner states byte-identical to round191 C5')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    originals={(r['seed'],r['method'],r['repetition'],r['n']):r for r in json.loads((inp/'rows.json').read_text())}
    terminals={(r['seed'],r['method'],r['repetition']):r for r in json.loads((inp/'episodes.json').read_text())}
    original_memory=driver.memory;driver.memory=memory
    rows=[];episodes=[];failures=[];checks=dict(original191_states_bitwise=0,original191_terminal_equal=0,first_frontier_equal=0)
    order=np.random.default_rng(protocol['order_seed']);q=np.linspace(0,1,257);done=0;jobs=[(cfg,rep) for cfg in cfgs for rep in range(2)]
    try:
        for seed in protocol['seeds']:
            xx,vv=driver.observations(seed);seedrows=[]
            for index in order.permutation(len(jobs)):
                cfg,rep=jobs[index];rr,_,failure=driver.trial(xx,vv,q,cfg,seed,rep,protocol,out)
                if failure:failures.append(failure)
                episode=dict(seed=seed,method=cfg['name'],repetition=rep,complete=failure is None,stages=len(rr),
                    failed_n=None if failure is None else failure['failed_n'])
                baseline=cfg['learner']+'_c5' if cfg['family']!='regression' else cfg['name'];ref=terminals[seed,baseline,rep]
                assert episode['stages']==ref['stages'] and episode['failed_n']==ref['failed_n'];checks['original191_terminal_equal']+=1
                for row in rr:
                    old=originals[seed,baseline,rep,row['n']];assert sha(inp/old['state_file'])==old['state_sha256']
                    with np.load(out/row['state_file']) as a,np.load(inp/old['state_file']) as b:
                        for key in a.files:assert a[key].tobytes()==b[key].tobytes(),(seed,cfg['name'],rep,row['n'],key)
                    checks['original191_states_bitwise']+=1
                    if row['n']==4 and cfg['family']!='regression':
                        assert row['positive_mode_keys']==old['positive_mode_keys'] and row['completion_proposals']==old['completion_proposals']
                        assert row['discovery_bank_sha256']==old['discovery_bank_sha256']
                        assert [c['input_pattern_hash'] for c in row['screen_calls']]==[c['input_pattern_hash'] for c in old['screen_calls']]
                        checks['first_frontier_equal']+=1
                seedrows.extend(rr);episodes.append(episode);done+=1
                print(json.dumps(dict(completed=done,total=len(jobs)*len(protocol['seeds']),seed=seed,method=cfg['name'],rep=rep,
                    stages=len(rr),status='ok' if failure is None else 'failed',seconds=round(sum(r['total_seconds'] for r in rr),5))),flush=True)
            # No query answers before every state and terminal check for seed.
            truth=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
            for row in seedrows:
                with np.load(out/row['state_file']) as z:prediction=z['prediction']
                row['query_mse']=float(np.trapezoid((prediction-truth)**2,x=q))
            rows.extend(seedrows)
            for name,value in [('rows.json',rows),('episodes.json',episodes),('failures.json',failures)]:
                (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    finally:driver.memory=original_memory
    result=dict(execution_complete=True,episodes=done,stages=len(rows),failures=len(failures),checks=checks,sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
