"""Frozen-state admission of physical cuts, before altering online paths."""
import argparse,hashlib,json,time
from pathlib import Path
from fractions import Fraction as F
import numpy as np
import credit_residual_cut as cut
import conflict_feedback_memory as model

METHODS=['alm_feedback_full','alm_energy_full','alm_bp_feedback_full','adam16_feedback_full']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def exact_closed(x,b,h,clauses):
    prev=[F(float(t)) for t in x];z=[]
    for j in range(len(b)):
        z.extend(t+F(float(b[j])) for t in prev);prev=[F(float(t)) for t in h[j]]
    lo=[-cut.B,F(0),F(1,2),F(1)];hi=[F(0),F(1,2),F(1),F(float(model.conflict.screen.up(1+float(cut.B))))]
    return any(all(lo[k]<=z[t]<=hi[k] for t,k in zip(c['positions'],c['codes'])) for c in clauses)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results'
    inp=root/'conflict_feedback_memory/development';parentroot=root/'feedback_repair_paths/diagnostic';out=root/'physical_residual_cut/diagnostic'
    parent=json.loads((parentroot/'protocol.json').read_text());online=json.loads((inp/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for pp in [Path(__file__),Path(cut.__file__)]:hashes[pp.name]=sha(pp)
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parentroot/'protocol.json'),online_protocol_sha256=sha(inp/'protocol.json'),
                  seeds=parent['seeds'],methods=METHODS,verification=cut.verify(),
                  variants=['physical_all_old_cuts','same_piecewise_candidates_energy_only'],
                  scope='all old triggered states; proposals do not affect original trajectory; complete positive reference loaded only after proposals; no query answers')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cfgs={c['name']:c for c in online['configs']};lookup={(r['seed'],r['method'],r['n_context']):r for r in json.loads((inp/'episodes.json').read_text()) if r['repetition']==0}
    refroot=root/'posterior_state_reuse/first_write_reference';refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};rows=[];events=[];paths=0;replayed=0;cut_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4];pending=[]
        for name in METHODS:
            bank,regs,meta,collectors=model.prepare(x,v,dict(**cfgs[name],capture_repairs=True),True)
            saved=lookup[seed,name,4];assert [r.tobytes().hex() for r in regs]==saved['evaluated_pattern_keys'];paths+=1
            per={v:[] for v in protocol['variants']};proposed={v:dict(forward=set(),split=set()) for v in protocol['variants']}
            for obs in collectors:
                for record in obs.repair_records:
                    b=np.array(record['before_b']);h=np.array(record['before_h']);u=np.array(record['u']);clauses=obs.feedback_bank.clauses[:record['old_proofs']]
                    ob,oh,oldmeta=model.repair.repair_one(x,b,h,u,clauses,enforce=cfgs[name]['feedback_mode']!='energy_only')
                    assert np.array_equal(ob,np.array(record['after_b'])) and np.array_equal(oh,np.array(record['after_h'])) and oldmeta==record['result'];replayed+=1
                    assert all(p['accepted_event']<record['event'] for p in obs.feedback_bank.proofs[:record['old_proofs']])
                    token=hashlib.sha256(b.tobytes()+h.tobytes()+u.tobytes()+str((record['event'],record['restart'],record['old_proofs'])).encode()).hexdigest()
                    for variant,enforce in zip(protocol['variants'],[True,False]):
                        begin=time.perf_counter();nb,nh,result=cut.repair_one(x,b,h,u,clauses,enforce);elapsed=time.perf_counter()-begin
                        if result['accepted']:
                            closed=exact_closed(x,nb,nh,clauses);values=[cut.phi_exact(x,nb,nh,c['a']) for c in clauses];cut_checks+=len(values)
                            if enforce:assert not closed and all(t<=0 for t in values)
                            forward=model.base.pattern(x,nb).astype(np.uint8).tobytes().hex();split=model.repair.split_many(x,nb[None],nh[None])[0].tobytes().hex()
                            proposed[variant]['forward'].add(forward);proposed[variant]['split'].add(split)
                            same=bool(np.array_equal(nb,ob) and np.array_equal(nh,oh))
                            row=dict(**result,closed_conflict_after=closed,positive_cut_count=sum(t>0 for t in values),forward=forward,split=split,
                                     differs_from_original_repair=not same,forward_changed=not np.array_equal(nb,b),
                                     true_forward_pattern_changed=forward!=model.base.pattern(x,b).astype(np.uint8).tobytes().hex())
                        else:row=result
                        per[variant].append(dict(event=record['event'],restart=record['restart'],old_proofs=record['old_proofs'],input_sha256=token,seconds=elapsed,**row))
            original=lookup[seed,'alm_retained_full' if name.startswith('alm') else 'adam16_retained_full',4]['positive_mode_keys']
            pending.append((name,per,proposed,original,saved['positive_mode_keys']))
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];positive={r['pattern'] for r in json.loads(rp.read_text())['positive_regions']}
        for name,per,proposed,original,final in pending:
            for variant,rr in per.items():
                pp=proposed[variant];fp=pp['forward']&positive;sp=pp['split']&positive;union=fp|sp;new=union-set(original);extra=union-set(final)
                rows.append(dict(seed=seed,method=name,variant=variant,inputs=len(rr),accepted=sum(r['accepted'] for r in rr),
                                 activity=sum(r.get('selected_kind')=='activity' for r in rr),bias=sum(r.get('selected_kind')=='bias' for r in rr),
                                 no_feasible_point=sum(r.get('reason')=='no feasible representable one-coordinate point' for r in rr),
                                 segments=sum(r.get('segments',0) for r in rr),empty_segments=sum(r.get('empty_segments',0) for r in rr),
                                 unrepresentable_segments=sum(r.get('unrepresentable_segments',0) for r in rr),
                                 closed_conflict_after=sum(r.get('closed_conflict_after',False) for r in rr),
                                 still_positive_cuts=sum(r.get('positive_cut_count',0)>0 for r in rr),
                                 true_forward_pattern_changed=sum(r.get('true_forward_pattern_changed',False) for r in rr),
                                 differs_from_original_repair=sum(r.get('differs_from_original_repair',False) for r in rr),
                                 seconds=sum(r['seconds'] for r in rr),positive_forward=len(fp),positive_split=len(sp),
                                 positive_union=len(union),new_positive_vs_original=len(new),new_positive_vs_feedback=len(extra),
                                 new_positive_keys=sorted(new),new_vs_feedback_keys=sorted(extra)))
                events.append(dict(seed=seed,method=name,variant=variant,events=rr))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,paths=paths,inputs=replayed,exact_cut_checks=cut_checks)),flush=True)
    fields=['inputs','accepted','activity','bias','no_feasible_point','segments','empty_segments','unrepresentable_segments','closed_conflict_after','still_positive_cuts',
            'true_forward_pattern_changed','differs_from_original_repair','seconds','positive_forward','positive_split','positive_union','new_positive_vs_original','new_positive_vs_feedback']
    summary=[]
    for name in METHODS:
        for variant in protocol['variants']:
            rr=[r for r in rows if r['method']==name and r['variant']==variant];summary.append(dict(method=name,variant=variant,**{k:sum(r[k] for r in rr) for k in fields}))
    result=dict(source_hashes=len(hashes),original_paths_replayed=paths,old_repair_records_replayed=replayed,exact_cut_checks=cut_checks,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
