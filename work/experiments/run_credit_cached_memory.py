"""Matched online cost test; two repeats are NOT independent new tasks."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import credit_cached_mode_memory as model


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    cfg=json.loads(a.config.read_text());prior=json.loads(Path('results/reused_local_credit/history_increment/protocol.json').read_text())
    hashes=prior['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    for path in [Path(__file__),Path(model.__file__)]:hashes[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
    verification=model.verify()
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=hashlib.sha256(a.config.read_bytes()).hexdigest(),verification=verification,
        primary_endpoint='per-stream full fit+read cost, averaged across two independently interleaved timing repeats; exact output preservation is mandatory',
        scope='16 old streams, 2 timing repeats, not 32 independent tasks; no parallel benchmark; serialization/audits outside timing',
        safety='only positive outward-rounded bounds reject; all other patterns use unchanged common geometry; historical local credit never initialized by BP')
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    order=np.random.default_rng(873114)
    blocks=[(rep,seed) for rep in range(cfg['repetitions']) for seed in range(cfg['seed0'],cfg['seed0']+cfg['count'])]
    order.shuffle(blocks)
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
