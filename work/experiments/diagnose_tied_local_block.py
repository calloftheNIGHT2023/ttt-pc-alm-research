"""Frozen same-input admission: scalar, tied-union, and union energy control."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import tied_local_block_repair as tied
import conflict_feedback_memory as model
import diagnose_credit_residual_cut as previous

METHODS=['alm_feedback_full','alm_energy_full','alm_residual_feedback_full','alm_bp_feedback_full','adam16_feedback_full','pc_feedback_full','nodual_feedback_full']
VARIANTS=['scalar_cuts','union_cuts','union_energy']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    inp=root/'conflict_feedback_memory/development';parentroot=root/'physical_cut_memory/development';out=root/'tied_local_block/diagnostic'
    parent=json.loads((parentroot/'protocol.json').read_text());online=json.loads((inp/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(tied.__file__).name,'verify_tied_block_derivation.py']:hashes[name]=sha(Path(__file__).with_name(name))
    derivation=root/'tied_local_block/derivation_verification.json';assert json.loads(derivation.read_text())['passed']
    protocol=dict(source_sha256=hashes,old_protocol_sha256=sha(inp/'protocol.json'),parent_protocol_sha256=sha(parentroot/'protocol.json'),
                  derivation_sha256=sha(derivation),seeds=list(range(5900000,5900016)),methods=METHODS,variants=VARIANTS,verification=tied.verify(),
                  proposal_set='per unconstrained and constrained segment: nearest feasible binary64 quadratic optimum, its in-interval adjacent doubles, midpoint, old bias; deduplicate per layer and value; strict actual-state cuts',
                  scope='all old triggered states from seven frozen paths; same-input variants; shared bank is diagnostic cost, not online timing; queries never used; reference loaded after all proposals')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cfgs={c['name']:c for c in online['configs']};lookup={(r['seed'],r['method'],r['n_context']):r for r in json.loads((inp/'episodes.json').read_text()) if r['repetition']==0}
    earlier=json.loads((root/'physical_residual_cut/diagnostic/rows.json').read_text());old_scalar={(r['seed'],r['method']):r for r in earlier if r['variant']=='physical_all_old_cuts'}
    refroot=root/'posterior_state_reuse/first_write_reference';refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};rows=[];events=[];costs=[]
    state_checks=0;record_checks=0;scalar_comparisons=0;exact_checks=0;rounding_rejections=0;extended_empty=0;proof_sequences=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4];pending=[]
        for name in METHODS:
            cfg=dict(**cfgs[name],capture_repairs=True);bank,regs,meta,collectors=model.prepare(x,v,cfg,True);saved=lookup[seed,name,4]
            assert [r.tobytes().hex() for r in regs]==saved['evaluated_pattern_keys']
            _,state,_=model.old.old.old.old.materialize_retained(x,v,bank,regs,512)
            with np.load(inp/saved['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples']);state_checks+=1
            per={v:[] for v in VARIANTS};proposed={v:dict(forward=set(),split=set()) for v in VARIANTS};shared_time=0.;shared_candidates=0;scalar_count=0;new_segments=0;rounding=0;old_empty_rescued=0;max_rounding=0.
            for obs,oldobs in zip(collectors,saved.get('feedback_details',[])):
                assert obs.feedback_bank.proofs==oldobs['proofs'];proof_sequences+=1
                for record in obs.repair_records:
                    b=np.array(record['before_b']);h=np.array(record['before_h']);u=np.array(record['u']);clauses=obs.feedback_bank.clauses[:record['old_proofs']]
                    ob,oh,om=model.repair.repair_one(x,b,h,u,clauses,enforce=cfg['feedback_mode']!='energy_only')
                    assert np.array_equal(ob,np.array(record['after_b'])) and np.array_equal(oh,np.array(record['after_h'])) and om==record['result'];record_checks+=1
                    assert all(p['accepted_event']<record['event'] for p in obs.feedback_bank.proofs[:record['old_proofs']])
                    token=hashlib.sha256(b.tobytes()+h.tobytes()+u.tobytes()+str((record['event'],record['restart'],record['old_proofs'])).encode()).hexdigest()
                    begin=time.perf_counter();pool,_,stats=tied.candidate_bank(x,b,h,u,clauses);shared_time+=time.perf_counter()-begin
                    shared_candidates+=len(pool);scalar_count+=stats['scalar_candidates'];new_segments+=stats['tied_segments'];rounding+=stats['tied_rounding_rejected_candidates'];max_rounding=max(max_rounding,stats['max_cut_rounding_difference'])
                    selected={}
                    for variant,enforce,include_tied in zip(VARIANTS,[True,True,False],[False,True,True]):
                        begin=time.perf_counter();nb,nh,result=tied.select(x,b,h,clauses,pool,stats,enforce,include_tied);seconds=time.perf_counter()-begin
                        selected[variant]=result
                        if result['accepted']:
                            closed=previous.exact_closed(x,nb,nh,clauses);values=[tied.scalar.phi_exact(x,nb,nh,c['a']) for c in clauses];exact_checks+=len(values)
                            if enforce:assert not closed and all(t<=0 for t in values)
                            forward=model.base.pattern(x,nb).astype(np.uint8).tobytes().hex();split=model.repair.split_many(x,nb[None],nh[None])[0].tobytes().hex()
                            proposed[variant]['forward'].add(forward);proposed[variant]['split'].add(split)
                            support=float(np.max(np.abs(model.base.forward(x,nb)-v)))
                            row=dict(**result,closed_conflict_after=closed,positive_cut_count=sum(t>0 for t in values),forward=forward,split=split,
                                     support_max_error=support,support_feasible=bool(support<=model.base.EPS+model.base.TOL),
                                     true_forward_pattern_changed=forward!=model.base.pattern(x,b).astype(np.uint8).tobytes().hex())
                        else:row=result
                        per[variant].append(dict(event=record['event'],restart=record['restart'],old_proofs=record['old_proofs'],input_sha256=token,selection_seconds=seconds,**row))
                    oldselected=selected['scalar_cuts'];newselected=selected['union_cuts']
                    if oldselected['accepted']:
                        assert newselected['accepted'] and newselected['proximal_objective']<=oldselected['proximal_objective']
                    else:old_empty_rescued+=int(newselected['accepted'])
            original=lookup[seed,'alm_retained_full' if name.startswith('alm') else 'adam16_retained_full',4]['positive_mode_keys']
            if (seed,name) in old_scalar:
                earlier=old_scalar[seed,name];rr=per['scalar_cuts'];assert len(rr)==earlier['inputs'] and sum(r['accepted'] for r in rr)==earlier['accepted'];scalar_comparisons+=1
            costs.append(dict(seed=seed,method=name,shared_bank_seconds=shared_time,shared_candidates=shared_candidates,scalar_candidates=scalar_count,tied_segments=new_segments,
                              rounding_rejected_candidates=rounding,max_cut_rounding_difference=max_rounding,scalar_empty_rescued=old_empty_rescued))
            rounding_rejections+=rounding;extended_empty+=old_empty_rescued;pending.append((name,per,proposed,original,saved['positive_mode_keys']))
            print(json.dumps(dict(seed=seed,method=name,paths=state_checks,inputs=record_checks,rescued=extended_empty)),flush=True)
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];positive={r['pattern'] for r in json.loads(rp.read_text())['positive_regions']}
        for name,per,proposed,original,final in pending:
            newsets={variant:((pp['forward']|pp['split'])&positive)-set(final) for variant,pp in proposed.items()}
            for variant,rr in per.items():
                pp=proposed[variant];fp=pp['forward']&positive;sp=pp['split']&positive;union=fp|sp;new=union-set(original);extra=union-set(final)
                rows.append(dict(seed=seed,method=name,variant=variant,inputs=len(rr),accepted=sum(r['accepted'] for r in rr),
                                 scalar_activity=sum(r.get('selected_kind')=='activity' for r in rr),scalar_bias=sum(r.get('selected_kind')=='bias' for r in rr),tied=sum(r.get('selected_kind')=='tied' for r in rr),
                                 no_point=sum(r.get('reason')=='no certified finite candidate' for r in rr),closed_conflict_after=sum(r.get('closed_conflict_after',False) for r in rr),
                                 still_positive_cuts=sum(r.get('positive_cut_count',0)>0 for r in rr),true_forward_pattern_changed=sum(r.get('true_forward_pattern_changed',False) for r in rr),
                                 feasible_written_parameters=sum(r.get('support_feasible',False) for r in rr),selection_seconds=sum(r['selection_seconds'] for r in rr),
                                 positive_forward=len(fp),positive_split=len(sp),positive_union=len(union),new_positive_vs_original=len(new),new_positive_vs_feedback=len(extra),
                                 new_positive_keys=sorted(new),new_vs_feedback_keys=sorted(extra),unique_new_vs_energy_keys=sorted(extra-newsets['union_energy']),
                                 unique_new_vs_scalar_keys=sorted(extra-newsets['scalar_cuts'])))
                events.append(dict(seed=seed,method=name,variant=variant,events=rr))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8');(out/'costs.json').write_text(json.dumps(costs,indent=2),encoding='utf-8')
    summary=[];fields=['inputs','accepted','scalar_activity','scalar_bias','tied','no_point','closed_conflict_after','still_positive_cuts','true_forward_pattern_changed','feasible_written_parameters',
                       'positive_forward','positive_split','positive_union','new_positive_vs_original','new_positive_vs_feedback','selection_seconds']
    for name in METHODS:
        for variant in VARIANTS:
            rr=[r for r in rows if r['method']==name and r['variant']==variant]
            summary.append(dict(method=name,variant=variant,**{k:sum(r[k] for r in rr) for k in fields},
                                unique_new_vs_energy=sum(len(r['unique_new_vs_energy_keys']) for r in rr),unique_new_vs_scalar=sum(len(r['unique_new_vs_scalar_keys']) for r in rr)))
    result=dict(source_hashes=len(hashes),original_state_replays=state_checks,original_proof_sequences=proof_sequences,old_records_replayed=record_checks,
                prior_scalar_task_replays=scalar_comparisons,exact_cut_checks=exact_checks,rounding_rejections=rounding_rejections,scalar_empty_rescued=extended_empty,summary=summary,scope=protocol['scope'])
    assert state_checks==112 and record_checks==21666 and scalar_comparisons==64
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
