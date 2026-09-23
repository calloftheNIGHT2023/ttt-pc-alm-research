"""Chronological proof-bank audit; output references never enter observation."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import causal_conflict_observer as causal
import neighbor_mode_memory as neighbor
base=causal.base;conflict=causal.conflict;model=causal.model
PROFILES=['alm_local_k24','alm_residual_k24','alm_bp_c20_k24','nodual_residual_k24','pc_residual_k24','adam16_bp_k24','adam60_bp_k24']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';out=root/'causal_conflict/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((root/'matched_conflict_parents/diagnostic/protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(causal.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    oldcfg=json.loads((root/'budgeted_credit_memory/development/protocol.json').read_text());cfgs={c['name']:c for c in oldcfg['configs']}
    old={(r['seed'],r['method']):r for r in json.loads((root/'retained_credit/fused_verification/rows.json').read_text())}
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=parent['seeds'],profiles=PROFILES,banks=['c20_gated_primary','ungated_secondary'],
        configs={n:dict(**cfgs[n],retention_mode='bank') for n in PROFILES},verification=causal.verify(),
        selection='one new clause per callback per bank, only then-current directions; no initial local callback learning; lex pattern/label/row; skip already clause-covered parents',
        old_trajectory_file_sha256=sha(root/'retained_credit/fused_verification/rows.json'),scope='read-only causal diagnosis; repeated visits are not saved steps or independent tasks; no query answers')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];audits=[];details=[];lp_checks=0;proof_checks=0;positive_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];pending=[];core=[]
        for name in PROFILES:
            cfg=protocol['configs'][name];before=model.prepare(x,v,cfg,True);begin=time.perf_counter();after=causal.capture(x,v,cfg);elapsed=time.perf_counter()-begin
            assert causal.compare(before,after);obs=after[3][0];assert [c.trajectory.hexdigest() for c in after[3]]==old[seed,name]['trajectory_sha256']
            if name.startswith('alm_'):core.append(obs.core_trajectory.hexdigest())
            if name=='alm_local_k24':assert obs.current is None and obs.history is None
            pending.append((name,obs,elapsed));audits.append(dict(seed=seed,method=name,original_trajectory_bank_pending_retained_and_sequences_bitwise=True,
                trajectory_sha256=obs.trajectory.hexdigest(),core_trajectory_sha256=obs.core_trajectory.hexdigest(),diagnostic_seconds=elapsed,
                screen_unique_patterns=obs.screen_tested,screen_seconds=obs.screen_seconds,callbacks=obs.causal_events))
        assert len(set(core))==1
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];ref=json.loads(rp.read_text());positive=[np.frombuffer(bytes.fromhex(r['pattern']),np.uint8).reshape(4,4) for r in ref['positive_regions']]
        checked=set()
        for name,obs,elapsed in pending:
            for bank in obs.banks:
                for p in bank.proofs:
                    reg=np.frombuffer(bytes.fromhex(p['parent']),np.uint8).reshape(4,4);pp=np.array(p['p']);a=np.array(p['a']);exact=conflict.extract(x,v,reg,pp,a)
                    assert exact is not None and all(p[k]==v for k,v in exact.items());assert causal.normal.exact_optimum(x,v,reg,a)==p['optimized_exact'];proof_checks+=1
                    if bank.gate:assert not p['c20_detects_parent']
                    for field in ['first_future_clause_event','first_future_full_event']:
                        assert p[field] is None or p[field]>p['accepted_event']
                    cc,tt=conflict.rational_table(x,v,pp,a)
                    for r in positive:assert conflict.value(cc,tt,r)<=0;positive_checks+=1
                assert not np.any(conflict.clause_mask(np.array(positive),bank.clauses))
                for key in set(bank.hit_records)|{bytes.fromhex(p['parent']) for p in bank.proofs}:
                    if key in checked:continue
                    reg=np.frombuffer(key,np.uint8).reshape(4,4);_,_,g,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4)
                    assert lp.status==2;checked.add(key);lp_checks+=1
                rows.append(dict(seed=seed,method=name,**bank.report(),shared_screen_seconds=obs.screen_seconds,shared_screen_unique_patterns=obs.screen_tested))
                details.append(dict(seed=seed,method=name,gate_c20=bank.gate,proofs=bank.proofs,events=bank.events,
                    hits=[dict(pattern=k.hex(),**v) for k,v in bank.hit_records.items()]))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8');(out/'chronology.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,trajectories=len(audits),banks=len(rows),proofs=proof_checks,lp_modes=lp_checks)),flush=True)
    summary=[]
    count_fields=['clauses','clauses_with_future_hits','visits','clause_hits','full_hits','clause_beyond_c20','full_beyond_c20','forward_clause_hits','forward_full_hits','unique_clause_hits','unique_full_hits','unique_clause_beyond_c20','unique_full_beyond_c20','unique_cross_parent_clause','unique_cross_parent_full','exact_proposals','refinements']
    for name in PROFILES:
        for gate in [True,False]:
            rr=[r for r in rows if r['method']==name and r['gate_c20']==gate]
            summary.append(dict(method=name,gate_c20=gate,**{k:sum(r[k] for r in rr) for k in count_fields},
                tasks_with_clause_beyond_c20=sum(r['clause_beyond_c20']>0 for r in rr),max_library_numeric_bytes=max(r['library_numeric_bytes'] for r in rr),
                mean_seconds={k:float(np.mean([r[k] for r in rr])) for k in ['match_seconds','full_seconds','learning_seconds','shared_screen_seconds']}))
    result=dict(source_hashes=len(hashes),unchanged_trajectories=len(audits),bank_cases=len(rows),exact_proofs=proof_checks,independent_lp_modes=lp_checks,
        exact_complete_positive_checks=positive_checks,same_alm_trajectory_across_credit_controls=True,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
