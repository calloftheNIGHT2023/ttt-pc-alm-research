"""Frozen randomly interleaved full-stream implementation comparison."""
import argparse,hashlib,json,time,platform,sys
from pathlib import Path
import numpy as np
import deferred_typed_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    settings=json.loads(args.config.read_text());basepath=Path(__file__).with_name(settings['base_config']);baseconfig=json.loads(basepath.read_text())
    configs=[]
    for cfg in baseconfig['configs']:
        configs.append(dict(**cfg,routing_implementation='eager'))
        if cfg.get('route_mode','none')!='none':configs.append(dict(**cfg,routing_implementation='deferred'))
    assert len(configs)==34
    parentroot=Path('results/deferred_router/diagnostic');parent=json.loads((parentroot/'protocol.json').read_text());diagnostic=json.loads((parentroot/'summary.json').read_text())
    assert diagnostic['exact_online_states']==16 and diagnostic['original_inputs']==2331
    hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for name in [Path(__file__).name,Path(model.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    protocol=dict(**settings,configs=configs,source_sha256=hashes,config_sha256=sha(args.config),base_config_sha256=sha(basepath),
        parent_protocol_sha256=sha(parentroot/'protocol.json'),parent_summary_sha256=sha(parentroot/'summary.json'),verification=model.verify(),
        environment=dict(python=sys.version,numpy=np.__version__,platform=platform.platform()),
        timing='2 fixed-seed randomly interleaved full-stream repetitions; no concurrent heavy job; evaluation and stored-state replay outside timed fit/read')
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    oldroot=Path('results/typed_routing_memory/development');old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'episodes.json').read_text())}
    oldhash={};oldstates={}
    for key,row in old.items():
        path=oldroot/row['state_file'];assert sha(path)==row['state_sha256'];oldhash[key]=row['state_sha256']
        with np.load(path) as z:oldstates[key]=(z['anchor'].copy(),z['samples'].copy())
    order=np.random.default_rng(settings['order_seed']);rows=[];state_checks=0;queue_checks=0;proof_checks=0;future_checks=0
    total=settings['repetitions']*settings['count']*len(configs)*len(settings['stages'])
    for repetition in range(settings['repetitions']):
        seeds=list(range(settings['seed0'],settings['seed0']+settings['count']));order.shuffle(seeds)
        for seed in seeds:
            rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,settings['queries']);target=model.base.forward(q,truth)
            v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
            for index in order.permutation(len(configs)):
                cfg=configs[index];implementation=cfg['routing_implementation'];state=None;fit=model.fit if implementation=='deferred' else model.original.fit
                for n in settings['stages']:
                    previous=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                    begin=time.perf_counter();predict,state,meta=fit(x[:n],v[:n],state,dict(**cfg,archive=True,pool='posterior_mix',posterior_samples=settings['posterior_samples'],proposal_budget=settings['proposal_budget']));adapt=time.perf_counter()-begin
                    begin=time.perf_counter();pred=predict(q);read=time.perf_counter()-begin
                    reference=old[seed,cfg['name'],n];a,b=oldstates[seed,cfg['name'],n]
                    assert np.array_equal(state.anchor,a) and np.array_equal(state.samples,b);state_checks+=1
                    mse=float(np.mean((np.clip(pred,0,1)-target)**2));raw=float(np.mean((pred-target)**2))
                    assert mse==reference['query_mse'] and raw==reference['raw_query_mse'] and meta['positive_mode_keys']==reference['positive_mode_keys']
                    if n==4 and cfg.get('route_mode','none')!='none':
                        for key in ['route_original_pattern_keys','route_extra_all_keys','route_extra_pattern_keys','route_extra_origins']:
                            assert meta[key]==reference[key],(cfg['name'],implementation,key)
                        queue_checks+=1
                        for now,before in zip(meta['feedback_details'],reference['feedback_details']):
                            assert now['proofs']==before['proofs'];proof_checks+=1
                            for key in ['candidate_mode_visits','distinct_modes','causal_clause_removed','c20_removed']:
                                assert now['route_stats'][key]==before['route_stats'][key]
                            if implementation=='deferred':assert now['deferred_route_flushed'] and now['deferred_max_c20_batch']<=256
                    if n!=4:
                        assert not meta['routing_active'] and meta['effective_generator']=='direct';future_checks+=1
                        if implementation=='deferred':assert not meta['deferred_routing_active']
                    filename=f'r{repetition}_{seed}_{cfg["name"]}_{implementation}_n{n}.npz';np.savez_compressed(args.out/filename,anchor=state.anchor,samples=state.samples)
                    support=float(np.max(np.abs(predict(x[:n])-v[:n])))
                    rows.append(dict(repetition=repetition,seed=seed,method=cfg['name'],implementation=implementation,n_context=n,
                        query_mse=mse,raw_query_mse=raw,support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),
                        adaptation_seconds=adapt,read_queries_seconds=read,previous_state_bytes=previous,state_file=filename,state_sha256=sha(args.out/filename),
                        original_state_sha256=oldhash[seed,cfg['name'],n],**meta))
                with (args.out/'episodes.jsonl').open('a',encoding='utf-8') as checkpoint:
                    for row in rows[-len(settings['stages']):]:checkpoint.write(json.dumps(row)+'\n')
                print(json.dumps(dict(repetition=repetition,seed=seed,method=cfg['name'],implementation=implementation,rows=len(rows),total=total)),flush=True)
            (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    assert len(rows)==state_checks==4352 and queue_checks==896 and proof_checks==896 and future_checks==3264
    result=dict(complete=True,rows=len(rows),original_states_and_query_mse_exact=state_checks,extra_queues_and_origins_exact=queue_checks,
        original_proof_sequences_exact=proof_checks,later_own_state_direct_checks=future_checks)
    (args.out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
