"""Fixed-budget dynamic physical-cut development, with all costs recorded."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import physical_cut_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--out',type=Path,required=True);args=p.parse_args();cfg=json.loads(args.config.read_text())
    parentroot=Path('results/physical_residual_cut/diagnostic');parent=json.loads((parentroot/'protocol.json').read_text());audit=json.loads((parentroot/'summary.json').read_text())
    assert audit['old_repair_records_replayed']==10121 and audit['original_paths_replayed']==64
    physical=[r for r in audit['summary'] if r['variant']=='physical_all_old_cuts'];assert all(r['closed_conflict_after']==r['still_positive_cuts']==0 for r in physical)
    assert next(r for r in physical if r['method']=='alm_feedback_full')['accepted']>0
    hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(model.__file__).name,'verify_physical_cut_quadratics.py']:hashes[name]=sha(Path(__file__).with_name(name))
    primitive=Path('results/physical_residual_cut/quadratic_verification.json');assert json.loads(primitive.read_text())['passed']
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=sha(args.config),verification=model.verify(),admission_sha256=sha(parentroot/'summary.json'),quadratic_audit_sha256=sha(primitive),
                  timing='all fit, exact rational cut solving, causal proof construction, geometry, memory and query readout; one interleaved repetition, descriptive timing only',
                  scope='16 old development streams; no tuning or fresh confirmation; own-state later updates; query answers evaluator-only')
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    order=np.random.default_rng(844509);seeds=list(range(cfg['seed0'],cfg['seed0']+cfg['count']));order.shuffle(seeds);rows=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for index in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][index];state=None
            for n in cfg['stages']:
                previous=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                begin=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']));fit=time.perf_counter()-begin
                begin=time.perf_counter();pred=predict(q);read=time.perf_counter()-begin;support=float(np.max(np.abs(predict(x[:n])-v[:n])))
                filename=f'r0_{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(args.out/filename,anchor=state.anchor,samples=state.samples)
                rows.append(dict(repetition=0,seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(pred,0,1)-target)**2)),raw_query_mse=float(np.mean((pred-target)**2)),
                                 support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),adaptation_seconds=fit,read_queries_seconds=read,
                                 previous_state_bytes=previous,state_file=filename,state_sha256=sha(args.out/filename),**meta))
            (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,method=c['name'],rows=len(rows),total=cfg['count']*len(cfg['configs'])*len(cfg['stages']))),flush=True)
    print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
