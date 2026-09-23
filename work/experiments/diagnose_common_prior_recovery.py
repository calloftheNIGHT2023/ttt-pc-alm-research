"""Frozen, support-only recovery diagnostic on all eight old bare-C5 failures.

This selected failure subset is not a new confirmation set or full timing
comparison. Query answers are computed only after all recovered states commit.
"""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import common_prior_recovery as recovery
import run_light_h2_credit as driver

memory=recovery.memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def independent_forward(x,points):
    h=np.broadcast_to(x,(len(points),len(x)))
    for j in range(4):h=np.maximum(0.,1.-np.abs(2.*(h+points[:,j,None])-1.))
    return h


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/online_endpoint_h2/development';old=root/'results/online_interval_h2/development'
    assert json.loads((parent/'run_audit.json').read_text())['execution_complete'],'Do not overlap timed run200'
    out=root/'results/common_prior_recovery/development';out.mkdir(parents=True,exist_ok=True)
    assert not (out/'protocol.json').exists(),'Preserve prior diagnostic; do not overwrite'
    p=json.loads((parent/'protocol.json').read_text());hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(recovery.__file__)]:hashes[path.name]=sha(path)
    failures=json.loads((old/'failures.json').read_text())
    selected=sorted((r for r in failures if r['method'] in ['pc_c5','adam60_c5','nodual_c5']),key=lambda r:(r['seed'],r['method'],r['repetition']))
    assert len(selected)==8 and all(r['failed_n']==8 and recovery.EMPTY in r['traceback'] for r in selected)
    cfgs={c['name']:c for c in p['configs']};oldrows={(r['seed'],r['method'],r['repetition'],r['n']):r for r in json.loads((old/'rows.json').read_text())}
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parent/'protocol.json'),old_failures_sha256=sha(old/'failures.json'),
        cases=[{k:r[k] for k in ['seed','method','repetition','failed_n']} for r in selected],
        posterior_samples=p['posterior_samples'],rng_seed=p['rng_seed'],stages=[8,16,24],
        feature_budgets=list(recovery.FEATURE_BUDGETS),verification=recovery.verify(),
        scope='All eight old bare-C5 failed flows; conditional development diagnostic, not whole-task ranking or confirmation')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];episodes=[];checks=dict(old_failures_reproduced=0,state_replays=0,support_particles=0,own_state_links=0)
    q=np.linspace(0,1,257)
    for case in selected:
        seed=case['seed'];name=case['method'];rep=case['repetition'];cfg=cfgs[name];xx,vv=driver.observations(seed)
        ref=oldrows[seed,name,rep,4];path=old/ref['state_file'];assert sha(path)==ref['state_sha256']
        with np.load(path) as z:
            anchor=z['anchor'].copy();points=z['points'].copy();assert np.array_equal(xx[:4],z['x']) and np.array_equal(vv[:4],z['v'])
        anchor.setflags(write=False);points.setflags(write=False);state=memory.prior.State(anchor,points)
        assert memory.prior.state_digest(state)==ref['state_digest']
        rng_for=lambda n:np.random.default_rng(np.random.SeedSequence([p['rng_seed'],seed,rep,n,p['posterior_samples']]))
        before=memory.prior.state_digest(state);start=time.perf_counter()
        try:
            memory.fit(xx[:8],vv[:8],state,cfg,rng_for(8),p['posterior_samples'])
            raise AssertionError('Old failure no longer reproduced')
        except RuntimeError as exc:assert str(exc)==recovery.EMPTY
        assert memory.prior.state_digest(state)==before;checks['old_failures_reproduced']+=1
        reproduction_seconds=time.perf_counter()-start
        episode=dict(seed=seed,method=name,repetition=rep,initial_state_file=ref['state_file'],initial_state_sha256=ref['state_sha256'],
            reproduction_seconds_not_recovery_benchmark=reproduction_seconds,complete=True,failed_n=None,stages=0)
        for n in protocol['stages']:
            previous=memory.prior.state_digest(state);start=time.perf_counter()
            try:pred,new,meta=recovery.fit(xx[:n],vv[:n],state,cfg,rng_for(n),p['posterior_samples'])
            except recovery.RecoveryExhausted as exc:
                episode.update(complete=False,failed_n=n,error=str(exc),events=exc.events,failed_seconds=time.perf_counter()-start);break
            except Exception:
                episode.update(complete=False,failed_n=n,traceback=traceback.format_exc(),failed_seconds=time.perf_counter()-start);break
            write=time.perf_counter()-start;start=time.perf_counter();prediction=pred(q);read=time.perf_counter()-start
            assert memory.prior.state_digest(state)==previous and meta['previous_state_digest']==previous
            assert np.array_equal(new.anchor,new.samples[0]) and np.max(abs(new.samples))<=.12+1e-12
            assert np.max(abs(independent_forward(xx[:n],new.samples)-vv[:n]))<=.001+1e-8
            replay=independent_forward(q,new.samples[::-1]).mean(0);assert np.max(abs(replay-prediction))<1e-10
            checks['state_replays']+=1;checks['support_particles']+=len(new.samples);checks['own_state_links']+=1
            file=out/f'state_{seed}_{name}_{rep}_{n}.npz'
            np.savez_compressed(file,x=xx[:n],v=vv[:n],q=q,prediction=prediction,points=new.samples,anchor=new.anchor)
            row=dict(seed=seed,method=name,repetition=rep,n=n,write_seconds=write,read_seconds=read,total_seconds=write+read,
                previous_state_digest=previous,state_digest=memory.prior.state_digest(new),state_file=file.name,state_sha256=sha(file),
                recovery=meta['recovery'],geometry_calls=meta['geometry_calls'],state_bytes=meta['persistent_state_bytes'])
            assert sum(e['seconds'] for e in row['recovery']['attempts'])<=write+1e-6
            rows.append(row);episode['stages']+=1;state=new
            print(json.dumps(dict(seed=seed,method=name,rep=rep,n=n,recovered=meta['recovery']['triggered'],seconds=write+read)),flush=True)
        episodes.append(episode)
        for filename,value in [('rows_without_query.json',rows),('episodes.json',episodes)]:
            (out/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    # All adaptation is over before the evaluator constructs any query answer.
    for row in rows:
        truth=independent_forward(q,np.random.default_rng(row['seed']).uniform(-.12,.12,(1,4)))[0]
        with np.load(out/row['state_file']) as z:
            assert sha(out/row['state_file'])==row['state_sha256']
            row['query_mse']=float(np.trapezoid((z['prediction']-truth)**2,x=q))
    for filename,value in [('rows.json',rows),('episodes.json',episodes)]:
        (out/filename).write_text(json.dumps(value,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,cases=len(episodes),complete=sum(e['complete'] for e in episodes),checks=checks,
        triggered_stages=sum(r['recovery']['triggered'] for r in rows),source_hashes=len(hashes),
        source_sha256=sha(Path(__file__)),scope=protocol['scope'])
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
