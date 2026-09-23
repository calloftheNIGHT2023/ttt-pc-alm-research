"""Frozen typed-route online development, including same-parent simple controls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import typed_routing_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);parser.add_argument('--out',type=Path,required=True);args=parser.parse_args();cfg=json.loads(args.config.read_text())
    parentroot=Path('results/light_tied_routing/diagnostic');parent=json.loads((parentroot/'protocol.json').read_text());summary=json.loads((parentroot/'summary.json').read_text())
    assert summary['original_states']==112 and summary['old_records']==21666
    local=next(r for r in summary['summary'] if r['method']=='alm_feedback_full' and r['variant']=='tied_forward');assert local['new_positive']==5
    hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for name in [Path(__file__).name,Path(model.__file__).name,'verify_local_vs_global_partition.py']:hashes[name]=sha(Path(__file__).with_name(name))
    protocol=dict(**cfg,source_sha256=hashes,config_sha256=sha(args.config),verification=model.verify(),admission_sha256=sha(parentroot/'summary.json'),
        timing='single randomly interleaved timing repetition; complete adaptation and prediction cost; diagnostic replay checks outside timed fit; no concurrent heavy tasks',
        scope='20 methods on 16 old development streams, own-state later direct updates; no fresh confirmation; queries evaluator-only; full same-parent and dual-zero proposal controls')
    args.out.mkdir(parents=True,exist_ok=True);assert not (args.out/'protocol.json').exists();(args.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    admission={(r['seed'],r['method'],r['variant']):r for r in json.loads((parentroot/'rows.json').read_text())}
    oldroot=Path('results/conflict_feedback_memory/development');old={(r['seed'],r['method'],r['n_context']):r for r in json.loads((oldroot/'episodes.json').read_text()) if r['repetition']==0}
    order=np.random.default_rng(913601);seeds=list(range(cfg['seed0'],cfg['seed0']+cfg['count']));order.shuffle(seeds);rows=[];reference_checks=0;admission_checks=0;proof_sequences=0
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24);q=rng.uniform(0,1,cfg['queries']);target=model.base.forward(q,truth)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for index in order.permutation(len(cfg['configs'])):
            c=cfg['configs'][index];state=None
            for n in cfg['stages']:
                previous=0 if state is None else state.anchor.nbytes+state.samples.nbytes
                begin=time.perf_counter();predict,state,meta=model.fit(x[:n],v[:n],state,dict(**c,archive=True,pool='posterior_mix',posterior_samples=cfg['posterior_samples'],proposal_budget=cfg['proposal_budget']));fit=time.perf_counter()-begin
                begin=time.perf_counter();pred=predict(q);read=time.perf_counter()-begin;support=float(np.max(np.abs(predict(x[:n])-v[:n])))
                if 'reference_method' in c:
                    before=old[seed,c['reference_method'],n]
                    with np.load(oldroot/before['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples']);reference_checks+=1
                if n==4 and c.get('route_mode','none')!='none':
                    source=c.get('admission_method',c.get('feedback_reference'));before=old[seed,source,n]
                    assert meta['route_original_pattern_keys']==before['evaluated_pattern_keys']
                    for a,b in zip(meta['feedback_details'],before['feedback_details']):assert a['proofs']==b['proofs'];proof_sequences+=1
                    if 'admission_method' in c:
                        item=admission[seed,source,c['route_mode']];assert meta['route_extra_all_keys']==item['extra_geometry_keys'];admission_checks+=1
                        expected=set(before['positive_mode_keys'])|set(item['new_keys'] if c.get('route_extra_budget') is None else item['first24_new_keys'])
                        assert set(meta['positive_mode_keys'])==expected
                if n!=4:assert not meta['routing_active'] and meta['effective_generator']=='direct'
                filename=f'r0_{seed}_{c["name"]}_n{n}.npz';np.savez_compressed(args.out/filename,anchor=state.anchor,samples=state.samples)
                rows.append(dict(repetition=0,seed=seed,method=c['name'],n_context=n,query_mse=float(np.mean((np.clip(pred,0,1)-target)**2)),raw_query_mse=float(np.mean((pred-target)**2)),
                    support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),adaptation_seconds=fit,read_queries_seconds=read,
                    previous_state_bytes=previous,state_file=filename,state_sha256=sha(args.out/filename),**meta))
            (args.out/'episodes.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,method=c['name'],rows=len(rows),total=cfg['count']*len(cfg['configs'])*len(cfg['stages']))),flush=True)
    result=dict(complete=True,rows=len(rows),old_reference_states_bitwise=reference_checks,admission_queues_exact=admission_checks,original_proof_sequences_exact=proof_sequences)
    assert len(rows)==1280 and reference_checks==384 and admission_checks==192 and proof_sequences==224
    (args.out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
