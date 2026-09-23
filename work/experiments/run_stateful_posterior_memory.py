"""Frozen online state-reuse development experiment, evaluator-only queries."""
import argparse,hashlib,json,platform,time
from pathlib import Path
import numpy as np
import stateful_posterior_memory as model
base=model.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    previous=json.loads(Path('results/posterior_state_reuse/support_diagnostic/protocol.json').read_text())
    assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in previous['source_sha256'].items())
    sources=[Path(__file__),Path(model.__file__)]
    protocol=dict(**cfg,source_sha256={**previous['source_sha256'],**{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}},
        config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),audit=model.verify(),
        python=platform.python_version(),numpy=np.__version__,platform=platform.platform(),
        timing_scope='wall time includes pool construction, all discovery, screening, global geometry, sampling and 2048-query read; artifact serialization excluded',
        state_scope='sample bank and anchor shared by predictor and next write; old plus new state coexist during adaptation; array subtotals are not full native peak')
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];order=np.random.default_rng(884713)
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        q=rng.uniform(0,1,cfg['queries']);target=base.forward(q,truth)
        for ci in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][ci];state=None
            for n in cfg['stages']:
                old_bytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                begin=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,posterior_samples=cfg['posterior_samples']));fit=time.perf_counter()-begin
                begin=time.perf_counter();pred=predict(q);read=time.perf_counter()-begin
                error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                filename=f'{seed}_{c["name"]}_n{n}.npz'
                np.savez_compressed(a.out/filename,samples=state.samples,anchor=state.anchor)
                rows.append(dict(method=c['name'],seed=seed,n_context=n,
                    query_mse=float(np.mean((np.clip(pred,0,1)-target)**2)),raw_query_mse=float(np.mean((pred-target)**2)),
                    support_max_error=error,support_feasible=bool(error<=base.EPS+base.TOL),
                    adaptation_seconds=fit,read_queries_seconds=read,previous_state_bytes=old_bytes,
                    state_file=filename,state_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),**meta))
        (a.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-cfg['seed0']+1,total=cfg['count'],rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
