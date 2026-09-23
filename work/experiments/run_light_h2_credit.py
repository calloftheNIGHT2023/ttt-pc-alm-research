"""Charged four-stage light-H2 experiment, preserving equivalent failures."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import light_h2_credit as memory
from run_independent_hybrid_memory import observations


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def trial(xx,vv,q,cfg,seed,rep,p,out=None):
    state=None;rows=[];artifacts=[];failure=None
    for n in p['stages']:
        rng=np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,n,p['posterior_samples']]))
        start=time.perf_counter()
        try:
            if cfg['family']=='regression':pred,state,meta=memory.prior.regression(xx[:n],vv[:n],cfg['name'],state)
            else:pred,state,meta=memory.fit(xx[:n],vv[:n],state,cfg,rng,p['posterior_samples'])
            write=time.perf_counter()-start;start=time.perf_counter();prediction=pred(q);read=time.perf_counter()-start
            assert np.all(np.isfinite(prediction))
            arrays=dict(x=xx[:n],v=vv[:n],q=q,prediction=prediction)
            if cfg['family']!='regression':
                arrays.update(points=state.samples,anchor=state.anchor)
                codes,h=memory.prior.capture.model.light.forward_many(xx[:n],state.samples)
                assert np.max(abs(h[:,-1]-vv[:n]))<=.001+1e-8
                assert sorted({r.tobytes().hex() for r in codes})==meta['actual_mode_keys']
            row=dict(seed=seed,method=cfg['name'],learner=cfg.get('learner'),mode=cfg.get('credit_mode'),family=cfg['family'],
                repetition=rep,n=n,write_seconds=write,read_seconds=read,total_seconds=write+read,
                state_bytes=meta['persistent_state_bytes'],state_digest=meta.get('new_state_digest'),
                previous_state_digest=meta.get('previous_state_digest'),
                geometry_calls=meta.get('geometry_calls',0),screen_calls=meta.get('screen_calls',[]),
                credit_snapshot=meta.get('discovery',{}).get('credit_snapshot',{}),
                positive_mode_keys=meta.get('positive_mode_keys',[]),completion_proposals=meta.get('completion_proposals'),
                discovery_bank_sha256=meta.get('discovery_bank_sha256'))
            if out is not None:
                path=out/f'state_{seed}_{cfg["name"]}_{rep}_{n}.npz';np.savez_compressed(path,**arrays)
                row.update(state_file=path.name,state_sha256=sha(path))
                if rep==0 and n==4 and cfg['family']!='regression':
                    detail=out/f'detail_{seed}_{cfg["name"]}.json';detail.write_text(json.dumps(meta,indent=2),encoding='utf-8')
                    row.update(detail_file=detail.name,detail_sha256=sha(detail))
            rows.append(row);artifacts.append(arrays)
        except Exception:
            failure=dict(seed=seed,method=cfg['name'],repetition=rep,failed_n=n,traceback=traceback.format_exc());break
    return rows,artifacts,failure


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';out=root/'results/light_h2_credit/development'
    parent=json.loads((inp/'protocol.json').read_text());hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(memory.__file__)]:hashes[path.name]=sha(path)
    cfgs=memory.configs()+[dict(name='prior4096_ridge',family='regression')]
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),configs=cfgs,
        seeds=parent['seeds'],repetitions=2,stages=[4,8,16,24],posterior_samples=2048,rng_seed=parent['rng_seed'],
        order_seed=482033,query_points=257,direction_budget=32,checkpoints='ceil(j * iterations / 4), j=1..4',
        verification=memory.verify(),scope='same 16 development tasks as round189; five own learners; exact-output resource comparison, not new confirmation or official TTT')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    original_rows={(r['seed'],r['method'],r['repetition'],r['n_context']):r for r in json.loads((inp/'rows.json').read_text())}
    old_methods={'alm':'alm_h2','adam60':'adam60_h2','direct4096':'direct4096_h2'}
    rows=[];episodes=[];failures=[];done=0;rngorder=np.random.default_rng(protocol['order_seed']);q=np.linspace(0,1,257)
    jobs=[(cfg,rep) for cfg in cfgs for rep in range(2)];checks=dict(within_learner_bitwise=0,original189_bitwise=0,identical_terminal_status=0)
    for seed in protocol['seeds']:
        xx,vv=observations(seed);seedrows=[];seedepisodes=[]
        for index in rngorder.permutation(len(jobs)):
            cfg,rep=jobs[index];rr,_,failure=trial(xx,vv,q,cfg,seed,rep,protocol,out)
            if failure:failures.append(failure)
            episode=dict(seed=seed,method=cfg['name'],repetition=rep,complete=failure is None,stages=len(rr),failed_n=None if failure is None else failure['failed_n'])
            seedrows.extend(rr);seedepisodes.append(episode);done+=1
            print(json.dumps(dict(completed=done,total=len(jobs)*len(protocol['seeds']),seed=seed,method=cfg['name'],rep=rep,
                stages=len(rr),status='ok' if failure is None else 'failed',seconds=round(sum(r['total_seconds'] for r in rr),4))),flush=True)
        lookup={(r['method'],r['repetition'],r['n']):r for r in seedrows};ep={(e['method'],e['repetition']):e for e in seedepisodes}
        for cfg in memory.configs():
            name=cfg['name'];learner=cfg['learner'];baseline=learner+'_c5'
            for rep in range(2):
                e=ep[name,rep];b=ep[baseline,rep];assert e['stages']==b['stages'] and e['failed_n']==b['failed_n'];checks['identical_terminal_status']+=1
                for n in protocol['stages'][:e['stages']]:
                    row=lookup[name,rep,n];ref=lookup[baseline,rep,n]
                    with np.load(out/row['state_file']) as a,np.load(out/ref['state_file']) as b:
                        for key in a.files:assert np.array_equal(a[key],b[key]),(seed,name,rep,n,key)
                    if n==4:
                        assert row['discovery_bank_sha256']==ref['discovery_bank_sha256']
                        assert row['positive_mode_keys']==ref['positive_mode_keys'] and row['completion_proposals']==ref['completion_proposals']
                    checks['within_learner_bitwise']+=1
                    if learner in old_methods:
                        oldname=old_methods[learner];key=(seed,oldname,rep,n)
                        # Failed old trials still committed their first stage.
                        oldpath=inp/f'state_{seed}_{oldname}_{rep}_{n}.npz'
                        assert oldpath.exists()
                        if key in original_rows:assert sha(oldpath)==original_rows[key]['state_sha256']
                        with np.load(out/row['state_file']) as a,np.load(oldpath) as b:
                            for field in ['q','prediction','points','anchor']:assert np.array_equal(a[field],b[field])
                        checks['original189_bitwise']+=1
        # Query targets are revealed only after all states/terminal checks.
        truth=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            with np.load(out/row['state_file']) as z:prediction=z['prediction']
            row['query_mse']=float(np.trapezoid((prediction-truth)**2,x=q))
        rows.extend(seedrows);episodes.extend(seedepisodes)
        for name,value in [('rows.json',rows),('episodes.json',episodes),('failures.json',failures)]:
            (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,episodes=done,stages=len(rows),failures=len(failures),checks=checks,sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
