"""Replay accepted transient repairs, closure semantics and discarded modes."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import conflict_feedback_memory as model
repair=model.repair;conflict=model.conflict;base=model.base
METHODS=['alm_feedback_full','alm_energy_full','alm_bp_feedback_full','adam16_feedback_full']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def closed_matches(x,b,h,clauses):
    prev=x;zz=[]
    for j in range(len(b)):zz.append(prev+b[j]);prev=h[j]
    z=np.array(zz).ravel();low=np.array([-conflict.screen.B,0.,.5,1.]);high=np.array([0.,.5,1.,conflict.screen.up(1+conflict.screen.B)])
    return [bool(np.all((z[c['positions']]>=low[c['codes']])&(z[c['positions']]<=high[c['codes']]))) for c in clauses]


def phi(x,b,h,a):
    prev=x;value=0.
    for j in range(len(b)):value+=float(np.sum(a[j]*(h[j]-base.g(prev+b[j]))));prev=h[j]
    return value/(1+np.abs(a).sum())


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results';inp=root/'conflict_feedback_memory/development';out=root/'feedback_repair_paths/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((inp/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    hashes[Path(__file__).name]=sha(Path(__file__));cfgs={c['name']:c for c in parent['configs']};episodes={(r['seed'],r['method'],r['n_context']):r for r in json.loads((inp/'episodes.json').read_text()) if r['repetition']==0}
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=list(range(parent['seed0'],parent['seed0']+parent['count'])),methods=METHODS,
        parent_protocol_sha256=sha(inp/'protocol.json'),scope='frozen path replay; reference after trajectories; closure/phi float diagnostics not new proof; query targets unused')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];details=[];state_checks=0;record_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];pending=[]
        for name in METHODS:
            cfg=dict(**cfgs[name],capture_repairs=True);bank,regs,meta,collectors=model.prepare(x,v,cfg,True);saved=episodes[seed,name,4];assert [r.tobytes().hex() for r in regs]==saved['evaluated_pattern_keys']
            _,state,readout=model.old.old.old.old.materialize_retained(x,v,bank,regs,512)
            with np.load(inp/saved['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples']);state_checks+=1
            original=episodes[seed,'alm_retained_full' if name.startswith('alm') else 'adam16_retained_full',4];eq=[]
            for n in parent['stages']:
                left=episodes[seed,name,n];right=episodes[seed,'alm_retained_full' if name.startswith('alm') else 'adam16_retained_full',n]
                with np.load(inp/left['state_file']) as a,np.load(inp/right['state_file']) as b:eq.append(dict(n=n,anchor_equal=bool(np.array_equal(a['anchor'],b['anchor'])),samples_equal=bool(np.array_equal(a['samples'],b['samples']))))
            events=[];adopted=[]
            for obs in collectors:
                for record in obs.repair_records:
                    b=np.array(record['before_b']);h=np.array(record['before_h']);u=np.array(record['u']);clauses=obs.feedback_bank.clauses[:record['old_proofs']]
                    nb,nh,result=repair.repair_one(x,b,h,u,clauses,enforce=cfg['feedback_mode']!='energy_only')
                    assert np.array_equal(nb,np.array(record['after_b'])) and np.array_equal(nh,np.array(record['after_h'])) and result==record['result'];record_checks+=1
                    assert all(p['accepted_event']<record['event'] for p in obs.feedback_bank.proofs[:record['old_proofs']])
                    if not result['accepted']:continue
                    before=base.pattern(x,b).astype(np.uint8).tobytes().hex();after=base.pattern(x,nb).astype(np.uint8).tobytes().hex();adopted.append(after)
                    oldreg=repair.split_many(x,b[None],h[None])[0];triggered=[i for i,c in enumerate(clauses) if conflict.clause_mask(oldreg[None],[c])[0]]
                    closures=closed_matches(x,nb,nh,clauses);values=[phi(x,nb,nh,np.array(clauses[i]['a'])) for i in triggered]
                    events.append(dict(event=record['event'],restart=record['restart'],kind=result['selected_kind'],state_change_squared=result['state_change_squared'],
                        forward_pattern_changed=before!=after,after_forward_pattern=after,canonical_still_blocked=bool(conflict.clause_mask(repair.split_many(x,nb[None],nh[None]),clauses)[0]),
                        any_old_closed_conflict=any(closures),triggered_closed_conflict=sum(closures[i] for i in triggered),
                        normalized_phi_min=min(values),normalized_phi_max=max(values),triggered_credit_still_positive=any(t>1e-12 for t in values)))
            pending.append((name,events,adopted,eq,original['positive_mode_keys'],saved['positive_mode_keys']))
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];positive={r['pattern'] for r in json.loads(rp.read_text())['positive_regions']}
        for name,events,adopted,eq,original,final in pending:
            got=set(adopted)&positive;new=got-set(original);discarded=got-set(final)
            rows.append(dict(seed=seed,method=name,accepted=len(events),activity=sum(e['kind']=='activity' for e in events),bias=sum(e['kind']=='bias' for e in events),
                zero_or_roundoff_changes=sum(e['state_change_squared']<=1e-24 for e in events),closed_conflict_after=sum(e['any_old_closed_conflict'] for e in events),
                positive_trigger_credit_after=sum(e['triggered_credit_still_positive'] for e in events),canonical_blocked_after=sum(e['canonical_still_blocked'] for e in events),
                true_forward_pattern_changed=sum(e['forward_pattern_changed'] for e in events),unique_adopted_forward_modes=len(set(adopted)),adopted_positive_modes=len(got),
                new_positive_vs_original=len(new),discarded_positive=len(discarded),new_positive_keys=sorted(new),discarded_positive_keys=sorted(discarded),original_state_comparison=eq))
            details.append(dict(seed=seed,method=name,events=events))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'events.json').write_text(json.dumps(details,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,paths=len(rows),records=record_checks)),flush=True)
    fields=['accepted','activity','bias','zero_or_roundoff_changes','closed_conflict_after','positive_trigger_credit_after','canonical_blocked_after','true_forward_pattern_changed','unique_adopted_forward_modes','adopted_positive_modes','new_positive_vs_original','discarded_positive']
    summary=[]
    for name in METHODS:
        rr=[r for r in rows if r['method']==name];summary.append(dict(method=name,**{k:sum(r[k] for r in rr) for k in fields},
            original_anchor_equal_stages=sum(e['anchor_equal'] for r in rr for e in r['original_state_comparison']),original_samples_equal_stages=sum(e['samples_equal'] for r in rr for e in r['original_state_comparison'])))
    result=dict(source_hashes=len(hashes),captured_state_replays=state_checks,finite_candidate_record_replays=record_checks,summary=summary,scope=protocol['scope']);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
