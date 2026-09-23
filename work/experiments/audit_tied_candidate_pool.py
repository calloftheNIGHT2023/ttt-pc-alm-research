"""Locate generation versus selection limits on all frozen local ALM inputs.

This is a development diagnostic, not a deployed archive or timing comparison.
Complete positive regions are loaded only after each task's proposals finish.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import tied_local_block_repair as tied
import conflict_feedback_memory as model

GROUPS=['scalar_all','scalar_certified','tied_all','tied_certified','union_all','union_certified']


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    root=args.project/'results';inp=root/'conflict_feedback_memory/development';parentroot=root/'tied_local_block/diagnostic';out=root/'tied_local_block/pool_audit'
    parent=json.loads((parentroot/'protocol.json').read_text());finished=json.loads((parentroot/'summary.json').read_text())
    assert finished['old_records_replayed']==21666
    hashes=parent['source_sha256'].copy()
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(parentroot/'protocol.json'),parent_summary_sha256=sha(parentroot/'summary.json'),
                  seeds=list(range(5900000,5900016)),method='alm_feedback_full',groups=GROUPS,
                  scope='all 2331 local old triggered states, every finite proposal; no new trajectory or query evaluation; no global selection using positive reference; diagnostic costs only',
                  decision='separate finite-pool coverage, strict-cut filtering, and minimum-energy selection; no automatic large online run from feasibility improvement alone')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    online=json.loads((inp/'protocol.json').read_text());cfg=next(c for c in online['configs'] if c['name']==protocol['method'])
    saved={(r['seed'],r['n_context']):r for r in json.loads((inp/'episodes.json').read_text()) if r['method']==protocol['method'] and r['repetition']==0}
    parent_events={(r['seed'],r['variant']):r['events'] for r in json.loads((parentroot/'events.json').read_text()) if r['method']==protocol['method']}
    refroot=root/'posterior_state_reuse/first_write_reference';refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    rows=[];events=[];new_examples=[];paths=0;records=0;candidate_count=0;selection_checks=0;certified_example_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        bank,regs,_,collectors=model.prepare(x,v,dict(**cfg,capture_repairs=True),True);old=saved[seed,4]
        assert [r.tobytes().hex() for r in regs]==old['evaluated_pattern_keys']
        _,state,_=model.old.old.old.old.materialize_retained(x,v,bank,regs,512)
        with np.load(inp/old['state_file']) as z:assert np.array_equal(state.anchor,z['anchor']) and np.array_equal(state.samples,z['samples'])
        assert len(collectors)==1
        paths+=1;sets={g:dict(forward=set(),split=set()) for g in GROUPS};counts={g:0 for g in GROUPS};fitcounts={g:0 for g in GROUPS}
        exemplars={};rr=[];seconds=0.;idx=0
        for obs,oldobs in zip(collectors,old['feedback_details']):
            assert obs.feedback_bank.proofs==oldobs['proofs']
            for record in obs.repair_records:
                b=np.array(record['before_b']);h=np.array(record['before_h']);u=np.array(record['u']);clauses=obs.feedback_bank.clauses[:record['old_proofs']]
                ob,oh,om=model.repair.repair_one(x,b,h,u,clauses)
                assert np.array_equal(ob,np.array(record['after_b'])) and np.array_equal(oh,np.array(record['after_h'])) and om==record['result']
                assert all(p['accepted_event']<record['event'] for p in obs.feedback_bank.proofs[:record['old_proofs']]);records+=1
                token=hashlib.sha256(b.tobytes()+h.tobytes()+u.tobytes()+str((record['event'],record['restart'],record['old_proofs'])).encode()).hexdigest()
                start=time.perf_counter();pool,_,stats=tied.candidate_bank(x,b,h,u,clauses);seconds+=time.perf_counter()-start;candidate_count+=len(pool)
                materialized=[tied.materialize(b,h,c) for c in pool];bb=np.array([t[0] for t in materialized]);hh=np.array([t[1] for t in materialized])
                split=model.repair.split_many(x,bb,hh);fp=[];prev=np.broadcast_to(x,(len(bb),len(x)))
                for j in range(len(b)):
                    z=prev+bb[:,j,None];fp.append(np.searchsorted(model.base.KNOTS,z,side='right').astype(np.uint8));prev=model.base.g(z)
                forward=np.array(fp).transpose(1,0,2);errors=np.max(np.abs(prev-v),axis=1)
                rank={i:k for k,i in enumerate(sorted(range(len(pool)),key=lambda i:(pool[i]['objective'],i)))}
                for i,c in enumerate(pool):
                    kind='scalar' if i<stats['scalar_candidates'] else 'tied';groups=[kind+'_all','union_all']
                    if c['allowed']:groups.extend([kind+'_certified','union_certified'])
                    for g in groups:
                        counts[g]+=1;fitcounts[g]+=int(errors[i]<=model.base.EPS+model.base.TOL)
                        for channel,arr in [('forward',forward),('split',split)]:
                            key=arr[i].tobytes().hex();sets[g][channel].add(key)
                            token2=g,channel,key
                            if token2 not in exemplars:
                                exemplars[token2]=dict(event=record['event'],restart=record['restart'],input_sha256=token,candidate_index=i,
                                    candidate_kind=c['kind'],energy_rank=rank[i],allowed=bool(c['allowed']),objective=float(c['objective']),
                                    b=bb[i].tolist(),h=hh[i].tolist(),old_proofs=record['old_proofs'],support_max_error=float(errors[i]),
                                    exact_phi=[[str(t.numerator),str(t.denominator)] for t in c.get('exact_phi',[])])
                selected={}
                for variant,enforce,include in [('scalar_cuts',True,False),('union_cuts',True,True),('union_energy',False,True)]:
                    _,_,meta=tied.select(x,b,h,clauses,pool,stats,enforce,include);prior=parent_events[seed,variant][idx]
                    assert prior['input_sha256']==token
                    for key,value in meta.items():assert prior[key]==value,(seed,idx,variant,key)
                    selected[variant]=dict(accepted=meta['accepted'],index=meta.get('selected_index'));selection_checks+=1
                rr.append(dict(input_sha256=token,event=record['event'],restart=record['restart'],candidates=len(pool),
                               allowed=sum(c['allowed'] for c in pool),selected=selected));idx+=1
        # Classification only: none of this reference has influenced proposals.
        ref=refs[seed];rp=refroot/ref['reference_file'];assert sha(rp)==ref['reference_sha256']
        positive={r['pattern'] for r in json.loads(rp.read_text())['positive_regions']};original=set(old['positive_mode_keys'])
        for g in GROUPS:
            forward=sets[g]['forward']&positive;split=sets[g]['split']&positive;found=forward|split;new=found-original
            rows.append(dict(seed=seed,group=g,inputs=idx,candidate_visits=counts[g],support_fitted_candidate_visits=fitcounts[g],
                distinct_forward=len(sets[g]['forward']),distinct_split=len(sets[g]['split']),positive_forward=len(forward),positive_split=len(split),
                positive_union=len(found),new_positive_vs_feedback=len(new),new_keys=sorted(new),positive_keys=sorted(found)))
            for key in sorted(new):
                for channel in ['forward','split']:
                    if key not in sets[g][channel]:continue
                    example=exemplars[g,channel,key]
                    if example['allowed']:
                        # Check the complete actual network, independently of
                        # the incremental candidate cut calculation.
                        obs=collectors[0];clauses=obs.feedback_bank.clauses[:example['old_proofs']]
                        values=[tied.scalar.phi_exact(x,np.array(example['b']),np.array(example['h']),c['a']) for c in clauses]
                        assert all(t<=0 for t in values);certified_example_checks+=len(values)
                    new_examples.append(dict(seed=seed,group=g,channel=channel,pattern=key,**example))
        events.append(dict(seed=seed,shared_construction_seconds=seconds,events=rr))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'events.json').write_text(json.dumps(events,indent=2),encoding='utf-8')
        (out/'new_examples.json').write_text(json.dumps(new_examples,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,paths=paths,inputs=records,candidates=candidate_count,new_union=sum(r['new_positive_vs_feedback'] for r in rows if r['group']=='union_all'))),flush=True)
    assert paths==16 and records==2331 and selection_checks==6993
    summary=[]
    for g in GROUPS:
        values=[r for r in rows if r['group']==g]
        fields=['inputs','candidate_visits','support_fitted_candidate_visits','distinct_forward','distinct_split','positive_forward','positive_split','positive_union','new_positive_vs_feedback']
        summary.append(dict(group=g,**{k:sum(r[k] for r in values) for k in fields},tasks_new=sum(r['new_positive_vs_feedback']>0 for r in values)))
    result=dict(passed=True,original_states=paths,old_records=records,candidate_visits=candidate_count,parent_selection_replays=selection_checks,
                certified_new_example_full_cut_checks=certified_example_checks,source_hashes=len(hashes),summary=summary,
                shared_construction_seconds=sum(r['shared_construction_seconds'] for r in events),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
