"""Frozen same-pool finite-budget ranking diagnostic on old ALM paths."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import dual_priority_proposals as proposal
import conflict_feedback_memory as model
from diagnose_light_tied_routing import route

BUDGETS=[1,2,4,8,16,24,None]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    root=args.project/'results';out=root/'dual_priority/diagnostic';oldroot=root/'conflict_feedback_memory/development'
    parentroot=root/'typed_routing_memory/development';parent=json.loads((parentroot/'protocol.json').read_text())
    hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for name in [Path(__file__).name,Path(proposal.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parentroot/'protocol.json'),seeds=list(range(5900000,5900016)),
        budgets=BUDGETS,scorers=proposal.SCORERS,verification=proposal.verify(),posterior_samples=512,queries=2048,
        source_method='alm_feedback_full',primary_scorer='dual_gain',primary_comparator='zero_gain',
        scope='same-pool first-write diagnostic on 16 old tasks; no fresh confirmation or full-stream timing; all ranks frozen before reference load')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cfg=next(c for c in json.loads((oldroot/'protocol.json').read_text())['configs'] if c['name']==protocol['source_method'])
    saved={r['seed']:r for r in json.loads((oldroot/'episodes.json').read_text()) if r['n_context']==4 and r['repetition']==0 and r['method']==cfg['name']}
    rows=[];pools=[];state_checks=0;record_checks=0;full_equal=0;score_checks=0;readouts=0;clauses_safe=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);q=rng.uniform(0,1,2048);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        start=time.perf_counter();bank,regs,_,collectors=model.prepare(x,v,dict(**cfg,capture_repairs=True),True);path_seconds=time.perf_counter()-start
        old=saved[seed];assert [r.tobytes().hex() for r in regs]==old['evaluated_pattern_keys']
        _,before,_=model.old.old.old.old.materialize_retained(x,v,bank,regs,512)
        with np.load(oldroot/old['state_file']) as z:assert np.array_equal(before.anchor,z['anchor']) and np.array_equal(before.samples,z['samples'])
        state_checks+=1;seen=set();kept={};features={};stats=dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.)
        counts=dict(inputs=0,raw_candidates=0,candidates=0,segments=0,construction_seconds=0.,local_score_seconds=0.,forward_pattern_seconds=0.,support_score_seconds=0.,bp_score_seconds=0.,aggregation_seconds=0.)
        for obs,oldobs in zip(collectors,old['feedback_details']):
            assert obs.feedback_bank.proofs==oldobs['proofs']
            for rec in obs.repair_records:
                b=np.array(rec['before_b']);h=np.array(rec['before_h']);u=np.array(rec['u']);clauses=obs.feedback_bank.clauses[:rec['old_proofs']]
                assert all(p['accepted_event']<rec['event'] for p in obs.feedback_bank.proofs[:rec['old_proofs']])
                ob,oh,om=model.repair.repair_one(x,b,h,u,clauses)
                assert np.array_equal(ob,np.array(rec['after_b'])) and np.array_equal(oh,np.array(rec['after_h'])) and om==rec['result'];record_checks+=1
                pb,ph,pm=proposal.shared_pool(x,b,h,u);patterns,values,tm=proposal.scores(x,v,b,h,u,pb,ph)
                counts['inputs']+=1
                for k,value in pm.items():counts[k]+=value
                for k,value in tm.items():counts[k]+=value
                # Full objective identity audited at every original trigger.
                move=np.sum((ph-h)**2,axis=(1,2))+len(x)*np.sum((pb-b)**2,axis=1)
                expected=(model.repair.energy_many(x,pb,ph,u)-model.repair.energy_many(x,b[None],h[None],u)[0]+proposal.light.TAU*move)/len(x)
                assert np.max(np.abs(expected-values['dual_gain']))<1e-11;score_checks+=len(pb)
                route(x,v,patterns,clauses,seen,kept,stats,rec['event'],rec['restart'],rec['old_proofs'])
                start=time.perf_counter();keys=[p.tobytes() for p in patterns]
                for i,key in enumerate(keys):
                    if key not in features:features[key]={s:float(a[i]) for s,a in values.items()}
                    else:
                        feature=features[key]
                        for scorer,arr in values.items():feature[scorer]=min(feature[scorer],float(arr[i]))
                counts['aggregation_seconds']+=time.perf_counter()-start
        assert stats['distinct_modes']==len(kept)+stats['causal_clause_removed']+stats['c20_removed']
        processed={r.tobytes() for r in regs};extras=[k for k in kept if k not in processed]
        orders={s:proposal.rank(extras,features,s) for s in proposal.SCORERS}
        selections={(s,k):proposal.canonical_select(extras,features,s,k) for s in proposal.SCORERS for k in BUDGETS}
        assert all(selections[s,None]==extras for s in proposal.SCORERS)
        pool=dict(seed=seed,original_patterns=[k.hex() for k in processed],extra_keys=[k.hex() for k in extras],
            features={k.hex():features[k] for k in extras},orders={s:[k.hex() for k in keys] for s,keys in orders.items()},
            counts=counts,routing=stats,original_path_capture_seconds=path_seconds,old_state_sha256=sha(oldroot/old['state_file']))
        poolfile=out/f'pool_{seed}.json';poolfile.write_text(json.dumps(pool,indent=2),encoding='utf-8');pools.append(dict(seed=seed,file=poolfile.name,sha256=sha(poolfile)))
        # No reference, query answer or volume above this boundary.
        cr=root/'posterior_state_reuse/conditional_risk';audit=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/audit['curve_file'];assert sha(cp)==audit['curve_sha256']
        with np.load(cp) as z:grid=z['q'];weights=z['weights'];means=z['region_means'];positive=z['patterns'];full=np.einsum('k,rkq->rq',weights,means)
        positive_keys=set(positive);target=model.base.forward(q,truth)
        assert {k.hex() for k in seen}&positive_keys=={k.hex() for k in kept}&positive_keys;clauses_safe+=len({k.hex() for k in seen}&positive_keys)
        fullstate=None;fullmse=None
        for scorer in proposal.SCORERS:
            for budget in BUDGETS:
                chosen=selections[scorer,budget];extra=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in chosen],np.uint8).reshape(-1,4,4)
                combined=np.concatenate([regs,extra]);start=time.perf_counter()
                predict,state,detail=model.old.old.old.old.materialize_retained(x,v,bank,combined,512);seconds=time.perf_counter()-start;readouts+=1
                pred=predict(q);mse=float(np.mean((pred-target)**2));actual=predict(grid)
                ce=float(np.mean([np.trapezoid((actual-full[a])*(actual-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
                mask=np.array([key in set(detail['positive_mode_keys']) for key in positive]);mass=float(weights[mask].sum())
                ideal=np.einsum('k,rkq->rq',weights*mask/mass,means)
                trunc=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=grid) for a,b in [(0,1),(2,3)]]))
                support=float(np.max(np.abs(predict(x)-v)));assert support<=model.base.EPS+model.base.TOL
                if budget is None:
                    if fullstate is None:fullstate=(state.anchor.copy(),state.samples.copy());fullmse=mse
                    else:assert np.array_equal(state.anchor,fullstate[0]) and np.array_equal(state.samples,fullstate[1]) and mse==fullmse;full_equal+=1
                label='full' if budget is None else str(budget);filename=f'{seed}_{scorer}_k{label}.npz'
                np.savez_compressed(out/filename,anchor=state.anchor,samples=state.samples)
                rows.append(dict(seed=seed,scorer=scorer,budget=budget,selected_keys=[k.hex() for k in chosen],extra_geometry_calls=len(chosen),
                    query_mse=mse,conditional_excess=ce,ideal_truncation=trunc,mass=mass,gained_modes=sorted(set(detail['positive_mode_keys'])-set(old['positive_mode_keys'])),
                    lost_modes=sorted(set(old['positive_mode_keys'])-set(detail['positive_mode_keys'])),positive_mode_keys=detail['positive_mode_keys'],
                    support_max_error=support,state_file=filename,state_sha256=sha(out/filename),readout_seconds=seconds))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'pools.json').write_text(json.dumps(pools,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,paths=state_checks,inputs=record_checks,readouts=readouts,extra_modes=len(extras))),flush=True)
    assert state_checks==16 and record_checks==2331 and readouts==784 and full_equal==96
    result=dict(complete=True,original_states=state_checks,original_records=record_checks,full_score_identity_points=score_checks,
        actual_first_write_readouts=readouts,full_budget_bitwise_controls=full_equal,positive_mode_preservation_checks=clauses_safe,
        source_hashes=len(hashes),scope=protocol['scope'])
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
