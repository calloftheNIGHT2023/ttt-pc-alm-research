import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import scalar_matched_batched_controls as model
base=model.base


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists()
    names=['scalar_matched_batched_controls.py','posterior_confirmation_pipeline.py','batched_bp_discovery.py','contextual_candidate_bank.py',
           'streaming_branch_projection.py','local_branch_memory.py','region_posterior_memory.py','region_posterior_controls.py',
           'hybrid_discovery_bank.py','matched_discovery_baselines.py','certified_branch_solver_v2.py']
    sources=[Path(__file__)]+[Path(__file__).with_name(name) for name in names]
    protocol=dict(**cfg,source_sha256={s.name:hashlib.sha256(s.read_bytes()).hexdigest() for s in sources},
                  config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),audit=model.verify(),
                  scope='same original study34 posterior, batched BP audit on old streams; not confirmation; contextual alternatives explicitly separated')
    (args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];order=np.random.default_rng(778619)
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,cfg['depth']);x=rng.uniform(0,1,max(cfg['stages']))
        v=base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,len(x));q=rng.uniform(0,1,cfg['queries']);target=base.forward(q,truth)
        for ci in order.permutation(len(cfg['configs'])):
            config=cfg['configs'][ci];anchor=np.zeros(cfg['depth'])
            for n in cfg['stages']:
                before=time.perf_counter();predict,anchor,meta=model.fit(x[:n],v[:n],anchor,dict(**config,posterior_samples=cfg['posterior_samples']))
                fit=time.perf_counter()-before;before=time.perf_counter();prediction=predict(q);read=time.perf_counter()-before
                support=np.clip(predict(x[:n]),0,1);error=float(np.max(np.abs(support-v[:n])))
                rows.append(dict(method=config['name'],seed=seed,n_context=n,query_mse=float(np.mean((np.clip(prediction,0,1)-target)**2)),
                                 raw_query_mse=float(np.mean((prediction-target)**2)),support_max_error=error,support_feasible=bool(error<=base.EPS+base.TOL),
                                 adaptation_seconds=fit,read_queries_seconds=read,**meta))
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed=seed-cfg['seed0']+1,total=cfg['count'],rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
