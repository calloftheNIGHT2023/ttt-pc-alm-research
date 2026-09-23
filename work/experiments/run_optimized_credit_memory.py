"""Interleaved full causal online benchmark with analytic dual-block screening."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import optimized_credit_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    cfg=json.loads(args.config.read_text());diagdir=Path('results/optimized_branch_dual/diagnostic');parent=json.loads((diagdir/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(model.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    diag={(r['seed'],r['method']):r for r in json.loads((diagdir/'rows.json').read_text()) if r['bank']=='native'}
    verification=model.verify();checks=0
    for seed in range(cfg['seed0'],cfg['seed0']+cfg['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);v=model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for c in cfg['configs']:
            if 'diagnostic_reference' not in c:continue
            _,regs,meta=model.prepare(xx[:4],v[:4],c)
            assert [r.tobytes().hex() for r in regs]==diag[seed,c['diagnostic_reference']]['after_keys'];checks+=1
    verification['diagnostic_sequence_replays']=checks
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=sha(args.config),verification=verification,
        timing='all discovery, credit, rational certification, global geometry, sampling and query read; two interleaved repeats averaged within task',
        scope='16 old development streams; full-state equivalence is required for uncapped screening; query labels evaluator-only; limited writes continue with own states')
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    order=np.random.default_rng(103591);blocks=[(rep,s) for rep in range(cfg['repetitions']) for s in range(cfg['seed0'],cfg['seed0']+cfg['count'])];order.shuffle(blocks);rows=[]
    for block,(rep,seed) in enumerate(blocks):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for i in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][i];state=None
            for n in cfg['stages']:
                oldbytes=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                start=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']));fit=time.perf_counter()-start
                start=time.perf_counter();pred=predict(q);read=time.perf_counter()-start
                support=float(np.max(np.abs(predict(x[:n])-v[:n])));filename=f'r{rep}_{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(args.out/filename,anchor=state.anchor,samples=state.samples)
                rows.append(dict(repetition=rep,seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(pred,0,1)-target)**2)),raw_query_mse=float(np.mean((pred-target)**2)),
                    support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),adaptation_seconds=fit,read_queries_seconds=read,previous_state_bytes=oldbytes,
                    state_file=filename,state_sha256=sha(args.out/filename),**meta))
        (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed_blocks=block+1,total_blocks=len(blocks),rows=len(rows))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
