"""Frozen full-path admission of lightweight typed routing and coordinate scan."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import light_tied_proposals as light
import conflict_feedback_memory as model

METHODS=['alm_feedback_full','alm_energy_full','alm_residual_feedback_full','alm_bp_feedback_full','adam16_feedback_full','pc_feedback_full','nodual_feedback_full']
VARIANTS=['tied_forward','tied_both','coordinate_forward']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def route(x,v,regs,clauses,seen,kept,stats,event,restart,old_proofs):
    start=time.perf_counter();keys=list(dict.fromkeys(r.tobytes() for r in regs));fresh=[k for k in keys if k not in seen];seen.update(fresh)
    stats['candidate_mode_visits']+=len(regs);stats['distinct_modes']+=len(fresh)
    if fresh:
        arr=np.array([np.frombuffer(k,np.uint8).reshape(len(regs[0]),len(x)) for k in fresh]);mask=model.conflict.clause_mask(arr,clauses)
        stats['causal_clause_removed']+=int(mask.sum());rest=arr[~mask];coarse=model.conflict.screen.contract(x,v,rest,20) if len(rest) else np.zeros(0,bool)
        stats['c20_removed']+=int(coarse.sum())
        for reg in rest[~coarse]:kept[reg.tobytes()]=dict(event=event,restart=restart,old_proofs=old_proofs)
    stats['routing_seconds']+=time.perf_counter()-start


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    inp=root/'conflict_feedback_memory/development';parentroot=root/'tied_local_block/pool_audit';out=root/'light_tied_routing/diagnostic'
    parent=json.loads((parentroot/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for name in [Path(__file__).name,Path(light.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parentroot/'protocol.json'),seeds=list(range(5900000,5900016)),methods=METHODS,variants=VARIANTS,
        verification=light.verify(),proposal_rule='free tied piece: clipped optimum, in-piece adjacent doubles, midpoint, both endpoints, old bias; coordinate: same finite rule with midpoint optimum; first occurrence dedup',
        routing_rule='first seen per channel: prior causal clauses then C20; never current point Phi; old final processed modes excluded only when merging for evaluation',
        scope='same 21666 frozen old inputs; no new state trajectory or online query prediction; shared construction costs diagnostic only; all seven paths complete proposals before reference load')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    online=json.loads((inp/'protocol.json').read_text());cfgs={c['name']:c for c in online['configs']}
    saved={(r['seed'],r['method']):r for r in json.loads((inp/'episodes.json').read_text()) if r['n_context']==4 and r['repetition']==0}
    refroot=root/'posterior_state_reuse/first_write_reference';refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};rows=[];costs=[];replays=0;records=0;mode_safety_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4];pending=[]
        for name in METHODS:
            cfg=cfgs[name];bank,regs,_,collectors=model.prepare(x,v,dict(**cfg,capture_repairs=True),True);old=saved[seed,name]
            assert [r.tobytes().hex() for r in regs]==old['evaluated_pattern_keys']
            _,state,_=model.old.old.old.old.materialize_retained(x,v,bank,regs,512)
            with np.load(inp/old['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples'])
            replays+=1;seen={k:set() for k in VARIANTS};kept={k:{} for k in VARIANTS}
            stats={k:dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.) for k in VARIANTS}
            counts=dict(inputs=0,tied_candidates=0,tied_segments=0,coordinate_candidates=0,coordinate_segments=0,tied_seconds=0.,coordinate_seconds=0.,tied_pattern_seconds=0.,coordinate_pattern_seconds=0.)
            for obs,oldobs in zip(collectors,old['feedback_details']):
                assert obs.feedback_bank.proofs==oldobs['proofs']
                for rec in obs.repair_records:
                    b=np.array(rec['before_b']);h=np.array(rec['before_h']);u=np.array(rec['u']);clauses=obs.feedback_bank.clauses[:rec['old_proofs']]
                    ob,oh,om=model.repair.repair_one(x,b,h,u,clauses,enforce=cfg['feedback_mode']!='energy_only')
                    assert np.array_equal(ob,np.array(rec['after_b'])) and np.array_equal(oh,np.array(rec['after_h'])) and om==rec['result']
                    assert all(p['accepted_event']<rec['event'] for p in obs.feedback_bank.proofs[:rec['old_proofs']]);records+=1;counts['inputs']+=1
                    start=time.perf_counter();bb,hh,tm=light.tied_free(x,b,h,u);counts['tied_seconds']+=time.perf_counter()-start
                    start=time.perf_counter();fp,_=light.forward_many(x,bb);sp=model.repair.split_many(x,bb,hh);counts['tied_pattern_seconds']+=time.perf_counter()-start
                    start=time.perf_counter();cb,cm=light.coordinate_scan(x,b);counts['coordinate_seconds']+=time.perf_counter()-start
                    start=time.perf_counter();cp,_=light.forward_many(x,cb);counts['coordinate_pattern_seconds']+=time.perf_counter()-start
                    counts['tied_candidates']+=tm['candidates'];counts['tied_segments']+=tm['segments'];counts['coordinate_candidates']+=cm['candidates'];counts['coordinate_segments']+=cm['segments']
                    for variant,proposed in [('tied_forward',fp),('tied_both',np.concatenate([fp,sp])),('coordinate_forward',cp)]:
                        route(x,v,proposed,clauses,seen[variant],kept[variant],stats[variant],rec['event'],rec['restart'],rec['old_proofs'])
            for variant in VARIANTS:
                st=stats[variant];assert len(seen[variant])==st['distinct_modes']==len(kept[variant])+st['causal_clause_removed']+st['c20_removed']
            costs.append(dict(seed=seed,method=name,**counts));pending.append((name,seen,kept,stats,old,counts))
            print(json.dumps(dict(seed=seed,method=name,paths=replays,inputs=records)),flush=True)
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];positive={r['pattern'] for r in json.loads(rp.read_text())['positive_regions']}
        for name,seen,kept,stats,old,counts in pending:
            prior=set(old['positive_mode_keys']);processed=set(old['evaluated_pattern_keys'])
            for variant in VARIANTS:
                all_positive={k.hex() for k in seen[variant]}&positive;route_positive={k.hex() for k in kept[variant]}&positive
                assert all_positive==route_positive;mode_safety_checks+=len(all_positive)
                extra=[k.hex() for k in kept[variant] if k.hex() not in processed];new=route_positive-prior
                rows.append(dict(seed=seed,method=name,variant=variant,inputs=counts['inputs'],**stats[variant],mode_survivors=len(kept[variant]),
                    extra_geometry_candidates=len(extra),extra_geometry_keys=extra,new_positive=len(new),new_keys=sorted(new),
                    new_positive_first24=len(set(extra[:24])&positive-prior),first24_new_keys=sorted(set(extra[:24])&positive-prior),
                    new_witnesses=[dict(pattern=k,**kept[variant][bytes.fromhex(k)]) for k in sorted(new)],all_positive_preserved=True))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'costs.json').write_text(json.dumps(costs,indent=2),encoding='utf-8')
    summary=[]
    for name in METHODS:
        for variant in VARIANTS:
            rr=[r for r in rows if r['method']==name and r['variant']==variant];fields=['inputs','candidate_mode_visits','distinct_modes','causal_clause_removed','c20_removed','mode_survivors','extra_geometry_candidates','new_positive','new_positive_first24','routing_seconds']
            summary.append(dict(method=name,variant=variant,**{k:sum(r[k] for r in rr) for k in fields},tasks_new=sum(r['new_positive']>0 for r in rr)))
    assert replays==112 and records==21666 and len(rows)==336
    result=dict(source_hashes=len(hashes),original_states=replays,old_records=records,all_positive_preservation_checks=mode_safety_checks,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
