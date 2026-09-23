"""Frozen online bank implementation; all cost-bearing work stays inside fit."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import batched_credit_bank_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());prior=json.loads(Path('results/cross_mode_credit_bank/diagnostic/protocol.json').read_text())
    hashes=prior['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    for path in [Path(__file__),Path(model.__file__)]:hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),verification=model.verify(),
        primary_endpoint='task-level mean over 2 repeats of full stream fit+read time; mandatory bitwise preserved states and predictions',
        frozen_selection='one direction per pattern maximizing rough/(1+L1(p)+L1(a)); strict check only after discovery; canonical accepted pattern order with first K distinct normalized directions',
        novelty_boundary='common batching/banking given to ALM and BP; local candidate never reads BP credit; global geometry remains explicitly present',
        scope='16 old tasks, 2 timing repeats, not 32 independent tasks; random interleaving; no parallel heavy benchmark')
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];order=np.random.default_rng(934511)
    blocks=[(rep,seed) for rep in range(cfg['repetitions']) for seed in range(cfg['seed0'],cfg['seed0']+cfg['count'])];order.shuffle(blocks)
    for index,(rep,seed) in enumerate(blocks):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for ci in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][ci];state=None
            for n in cfg['stages']:
                oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                start=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,
                    dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']))
                fit=time.perf_counter()-start;start=time.perf_counter();prediction=predict(q);read=time.perf_counter()-start
                error=float(np.max(np.abs(predict(x[:n])-v[:n])));filename=f'r{rep}_{seed}_{c["name"]}_n{n}.npz'
                np.savez_compressed(a.out/filename,samples=state.samples,anchor=state.anchor)
                rows.append(dict(repetition=rep,seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(prediction,0,1)-target)**2)),
                    raw_query_mse=float(np.mean((prediction-target)**2)),support_max_error=error,support_feasible=bool(error<=model.base.EPS+model.base.TOL),
                    adaptation_seconds=fit,read_queries_seconds=read,previous_state_bytes=oldbytes,
                    state_file=filename,state_sha256=hashlib.sha256((a.out/filename).read_bytes()).hexdigest(),**meta))
        (a.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed_blocks=index+1,total_blocks=len(blocks),rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
