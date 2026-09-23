"""Frozen fresh-development four-stage own-state experiment; evaluation last."""
import argparse
import hashlib
import json
import time
import traceback
from pathlib import Path
import numpy as np
import independent_hybrid_memory as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def observations(seed):
    rng=np.random.default_rng(seed);latent=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=memory.base.forward(xx,latent)+np.random.default_rng(seed+19000000).uniform(-.001,.001,24)
    return xx,vv


def trial(xx,vv,q,cfg,seed,rep,protocol,out=None):
    state=None;rows=[];artifacts=[]
    for n in protocol['stages']:
        rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'],seed,rep,n,protocol['posterior_samples']]))
        begin=time.perf_counter()
        if cfg['family']=='regression':pred,state,meta=memory.regression(xx[:n],vv[:n],cfg['name'],state)
        else:pred,state,meta=memory.fit(xx[:n],vv[:n],state,cfg,rng,protocol['posterior_samples'])
        write_seconds=time.perf_counter()-begin;begin=time.perf_counter();prediction=pred(q)
        read_seconds=time.perf_counter()-begin
        assert np.all(np.isfinite(prediction))
        arrays=dict(q=q,prediction=prediction,x=xx[:n],v=vv[:n])
        max_error=None
        if cfg['family']!='regression':
            arrays.update(points=state.samples,anchor=state.anchor)
            codes,h=memory.capture.model.light.forward_many(xx[:n],state.samples)
            max_error=float(np.max(abs(h[:,-1]-vv[:n])))
            assert max_error<=.001+1e-8
            assert sorted({r.tobytes().hex() for r in codes})==meta['actual_mode_keys']
        elif state is not None:arrays.update(rls_w=state['w'],rls_p=state['p'])
        name=f'state_{seed}_{cfg["name"]}_{rep}_{n}.npz'
        row=dict(seed=seed,method=cfg['name'],family=cfg['family'],repetition=rep,n_context=n,
            write_seconds=write_seconds,read_seconds=read_seconds,total_seconds=write_seconds+read_seconds,
            persistent_state_bytes=meta['persistent_state_bytes'],max_support_error=max_error,
            state_file=name,previous_state_digest=meta.get('previous_state_digest'),
            new_state_digest=meta.get('new_state_digest'),discovery_seconds=meta.get('discovery_seconds',0.),
            post_credit_seconds=meta.get('post_credit_seconds',0.),post_credit_rejected=meta.get('post_credit_rejected',0),
            sampling=meta.get('sampling',{}),effective_generator=meta.get('discovery',{}).get('effective_generator'),
            previous_posterior_samples=meta.get('discovery',{}).get('previous_posterior_samples',0),
            own_state_only=meta.get('own_state_only'),collector_kinds=meta.get('collector_kinds',[]),
            actual_mode_keys=meta.get('actual_mode_keys',[]),pool=meta.get('pool'),geometry_calls=meta.get('geometry_calls',0))
        if out is not None:
            path=out/name;np.savez_compressed(path,**arrays);row['state_sha256']=sha(path)
            if rep==0 and n==4 and cfg['family']!='regression':
                detail=out/f'detail_{seed}_{cfg["name"]}.json'
                detail.write_text(json.dumps(meta,indent=2),encoding='utf-8');row['detail_file']=detail.name;row['detail_sha256']=sha(detail)
        rows.append(row);artifacts.append(arrays)
    return rows,artifacts


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    out=root/'results/independent_hybrid/development';parent=root/'results/compact_hybrid_materialization/development/protocol.json'
    old=json.loads(parent.read_text());hashes=dict(old['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(memory.__file__)]:hashes[path.name]=sha(path)
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parent),configs=memory.configs(),
        seeds=list(range(5920000,5920016)),repetitions=2,stages=[4,8,16,24],posterior_samples=2048,
        rng_seed=481853,order_seed=481859,query_points=257,proposal_cap_per_particle=128,
        anchor_rule='first_emitted_particle',later_generator='direct128 with own posterior plus common prior256',
        untouched_confirmation_ranges=['5910000+','5500000+'],verification=memory.verify(),
        scope='new development tasks, independent discovery and own four-stage state; no confirmation/official TTT/downstream claim')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[];failures=[];rngorder=np.random.default_rng(protocol['order_seed']);q=np.linspace(0,1,protocol['query_points'])
    jobs=[(cfg,rep) for cfg in protocol['configs'] for rep in range(protocol['repetitions'])]
    total=len(jobs)*len(protocol['seeds']);done=0
    for seed in protocol['seeds']:
        xx,vv=observations(seed);seedrows=[]
        for index in rngorder.permutation(len(jobs)):
            cfg,rep=jobs[index]
            try:
                rr,_=trial(xx,vv,q,cfg,seed,rep,protocol,out);seedrows.extend(rr)
                note=dict(seconds=round(sum(r['total_seconds'] for r in rr),4),status='ok')
            except Exception:
                failure=dict(seed=seed,method=cfg['name'],repetition=rep,traceback=traceback.format_exc())
                failures.append(failure);(out/'failures.json').write_text(json.dumps(failures,indent=2),encoding='utf-8')
                note=dict(status='failed',error=failure['traceback'])
            done+=1;print(json.dumps(dict(completed=done,total=total,seed=seed,method=cfg['name'],rep=rep,**note)),flush=True)
        # Only after every method and repetition has committed its predictions
        # on this task does the evaluator reconstruct any query answer.
        targets=memory.base.forward(q,np.random.default_rng(seed).uniform(-.12,.12,4))
        for row in seedrows:
            path=out/row['state_file'];assert sha(path)==row['state_sha256']
            with np.load(path) as z:prediction=z['prediction']
            row['query_mse']=float(np.trapezoid((prediction-targets)**2,x=q))
        rows.extend(seedrows);(out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,all_trials_passed=not failures,failures=len(failures),
        episodes=done,stages=len(rows),expected_stages=total*len(protocol['stages']),sources=len(hashes))
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
