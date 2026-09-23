"""Frozen full-causal benchmark for fused retained-credit matching and reuse."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import retained_credit_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text())
    verification_root=Path('results/retained_credit/fused_verification');parent=json.loads((verification_root/'protocol.json').read_text());verified=json.loads((verification_root/'summary.json').read_text());assert verified['cases']==144
    hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=sha(args.config),verification_sha256=sha(verification_root/'summary.json'),
        timing='all original work, fused retention, exact rational matching/cross checks, geometry, state sampling and query read; two interleaved repeats per task',
        scope='16 old tasks, fixed K24 and full controls, causal own-state later updates, query targets evaluator-only')
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    order=np.random.default_rng(741105);blocks=[(rep,s) for rep in range(cfg['repetitions']) for s in range(cfg['seed0'],cfg['seed0']+cfg['count'])];order.shuffle(blocks);rows=[]
    for block,(rep,seed) in enumerate(blocks):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=model.base.forward(q,truth);v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for i in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][i];state=None
            for n in cfg['stages']:
                oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                begin=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']));fit=time.perf_counter()-begin
                begin=time.perf_counter();pred=predict(q);read=time.perf_counter()-begin;support=float(np.max(np.abs(predict(x[:n])-v[:n])));filename=f'r{rep}_{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(args.out/filename,anchor=state.anchor,samples=state.samples)
                rows.append(dict(repetition=rep,seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(pred,0,1)-target)**2)),raw_query_mse=float(np.mean((pred-target)**2)),
                    support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),adaptation_seconds=fit,read_queries_seconds=read,previous_state_bytes=oldbytes,state_file=filename,state_sha256=sha(args.out/filename),**meta))
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed_blocks=block+1,total_blocks=len(blocks),rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
