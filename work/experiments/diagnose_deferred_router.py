"""All old local-path mode visits: same causal clauses, deferred C20."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import deferred_mode_router as deferred
import light_tied_proposals as light
from diagnose_light_tied_routing import route

model=deferred.model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results'
    out=root/'deferred_router/diagnostic';oldroot=root/'conflict_feedback_memory/development';typed=root/'typed_routing_memory/development'
    parent=json.loads((typed/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for name in [Path(__file__).name,Path(deferred.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    p=dict(source_sha256=hashes,parent_protocol_sha256=sha(typed/'protocol.json'),seeds=list(range(5900000,5900016)),chunk_size=256,verification=deferred.verify(),
        scope='read-only local-path equivalence and isolated routing components; no complete online benchmark, no new task advantage')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    cfg=next(c for c in json.loads((oldroot/'protocol.json').read_text())['configs'] if c['name']=='alm_feedback_full')
    refs={r['seed']:r for r in json.loads((typed/'episodes.json').read_text()) if r['method']=='alm_tied_forward_full' and r['n_context']==4}
    rows=[];records=0;visits=0;states=0
    for seed in p['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        bank,regs,_,collectors=model.prepare(x,v,dict(**cfg,capture_repairs=True),True);late=deferred.Router(x,v,256)
        seen=set();kept={};stats=dict(candidate_mode_visits=0,distinct_modes=0,causal_clause_removed=0,c20_removed=0,routing_seconds=0.);old_calls=0;old_rows=0
        for obs,reference in zip(collectors,refs[seed]['feedback_details']):
            assert obs.feedback_bank.proofs==reference['proofs']
            for rec in obs.repair_records:
                b=np.array(rec['before_b']);h=np.array(rec['before_h']);u=np.array(rec['u']);clauses=obs.feedback_bank.clauses[:rec['old_proofs']]
                assert all(z['accepted_event']<rec['event'] for z in obs.feedback_bank.proofs[:rec['old_proofs']])
                bb,hh,meta=light.tied_free(x,b,h,u);fp,_=light.forward_many(x,bb)
                # Count the original actual C20 calls without timing this audit.
                fresh=list(dict.fromkeys(r.tobytes() for r in fp if r.tobytes() not in seen))
                if fresh:
                    arr=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in fresh]);survive=~model.conflict.clause_mask(arr,clauses)
                    if survive.any():old_calls+=1;old_rows+=int(survive.sum())
                route(x,v,fp,clauses,seen,kept,stats,rec['event'],rec['restart'],rec['old_proofs'])
                late.observe(fp,clauses,rec['event'],rec['restart'],rec['old_proofs']);records+=1;visits+=len(fp)
        late.flush();assert list(kept.items())==list(late.kept.items()) and seen==late.seen
        for key in stats:
            if key!='routing_seconds':assert stats[key]==late.stats[key]
        assert old_rows==late.c20_rows
        original={r.tobytes() for r in regs};extra=[k for k in late.kept if k not in original]
        assert [k.hex() for k in extra]==refs[seed]['route_extra_all_keys']
        new=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in extra],np.uint8).reshape(-1,4,4)
        pred,state,meta=model.old.old.old.old.materialize_retained(x,v,bank,np.concatenate([regs,new]),512)
        with np.load(typed/refs[seed]['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples']);states+=1
        rows.append(dict(seed=seed,inputs=sum(len(c.repair_records) for c in collectors),candidate_visits=stats['candidate_mode_visits'],
            old_c20_calls=old_calls,new_c20_calls=late.c20_calls,c20_rows=old_rows,old_routing_seconds=stats['routing_seconds'],new_routing_seconds=late.stats['routing_seconds'],
            peak_pending_keys=late.peak_pending_keys,maximum_batch=late.max_c20_batch,extra_keys=[k.hex() for k in extra],state_bitwise=True,
            pending_key_numeric_bytes=late.peak_pending_keys*16,scope='key subtotal omits Python objects and contractor arrays; independent full resources pending'))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,states=states,inputs=records)),flush=True)
    assert records==2331 and visits==169370 and states==16
    summary=dict(complete=True,source_hashes=len(hashes),original_inputs=records,proposal_visits=visits,exact_online_states=states,
        old_c20_calls=sum(r['old_c20_calls'] for r in rows),new_c20_calls=sum(r['new_c20_calls'] for r in rows),c20_rows=sum(r['c20_rows'] for r in rows),
        old_routing_seconds=sum(r['old_routing_seconds'] for r in rows),new_routing_seconds=sum(r['new_routing_seconds'] for r in rows),
        maximum_pending_keys=max(r['peak_pending_keys'] for r in rows),maximum_batch=max(r['maximum_batch'] for r in rows),scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8');print(json.dumps(summary,indent=2),flush=True)


if __name__=='__main__':main()
