"""Frozen causal neighborhood-completion development run."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import neighbor_mode_memory as model
base=model.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());old=json.loads(Path('results/posterior_state_reuse/component_frontier/protocol.json').read_text())
    for name,h in old['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    sources=[Path(__file__),Path(model.__file__)]
    protocol=dict(**cfg,source_sha256={**old['source_sha256'],**{s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources}},
        config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),audit=model.verify(),
        invariants='no reference modes or query inputs/answers in fit; zero-initialized local multipliers; global common LP/QP explicitly counted',
        state_scope='readonly anchor and sampled memory; record old-plus-new coexistence and temporary array subtotals, not full native peaks')
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists();(a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];order=np.random.default_rng(117324)
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=base.forward(q,truth)
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24)
        for ci in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][ci];state=None
            for n in cfg['stages']:
                oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                begin=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']));fit=time.perf_counter()-begin
                begin=time.perf_counter();prediction=predict(q);read=time.perf_counter()-begin;error=float(np.max(np.abs(predict(x[:n])-v[:n])))
                filename=f'{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(a.out/filename,samples=state.samples,anchor=state.anchor)
                rows.append(dict(seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(prediction,0,1)-target)**2)),
                    raw_query_mse=float(np.mean((prediction-target)**2)),support_max_error=error,support_feasible=bool(error<=base.EPS+base.TOL),
                    adaptation_seconds=fit,read_queries_seconds=read,previous_state_bytes=oldbytes,
                    state_file=filename,state_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),**meta))
        (a.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed=seed-cfg['seed0']+1,total=cfg['count'],rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
