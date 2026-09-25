"""Independent existence witnesses and evaluator-only potential of routed modes."""
import argparse,hashlib,json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import tied_local_block_repair as tied
import conflict_feedback_memory as model
import neighbor_mode_memory as neighbor


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';out=root/'tied_local_block/pool_audit'
    protocol=json.loads((out/'protocol.json').read_text())
    for name,value in protocol['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((out/'rows.json').read_text());examples=json.loads((out/'new_examples.json').read_text());summ=json.loads((out/'summary.json').read_text());assert summ['passed']
    inp=root/'conflict_feedback_memory/development';online=json.loads((inp/'protocol.json').read_text());cfg=next(c for c in online['configs'] if c['name']=='alm_feedback_full')
    old={r['seed']:r for r in json.loads((inp/'episodes.json').read_text()) if r['method']=='alm_feedback_full' and r['n_context']==4 and r['repetition']==0}
    refroot=root/'posterior_state_reuse/first_write_reference';refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    matched=json.loads((root/'matched_conflict_parents/diagnostic/rows.json').read_text());positive_volume={r['seed']:set(r['quality']['k24']['new_keys']) for r in matched if r['method']=='positive_volume'}
    quality=[];witnesses=[];curve_hashes=[];independent_regions=set();replays=0;full_cut_checks=0;safe_mode_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4]
        v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        ref=refs[seed];rp=refroot/ref['reference_file'];assert sha(rp)==ref['reference_sha256'];positive={r['pattern']:r for r in json.loads(rp.read_text())['positive_regions']}
        wanted=[r for r in examples if r['seed']==seed and r['group']=='union_all']
        if wanted:
            _,regs,_,collectors=model.prepare(x,v,dict(**cfg,capture_repairs=True),True);assert [r.tobytes().hex() for r in regs]==old[seed]['evaluated_pattern_keys'];assert len(collectors)==1
            obs=collectors[0];record_lookup={(r['event'],r['restart']):r for r in obs.repair_records};cache={}
            for example in wanted:
                key=example['pattern'];reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,4);rec=record_lookup[example['event'],example['restart']]
                b=np.array(rec['before_b']);h=np.array(rec['before_h']);u=np.array(rec['u']);clauses=obs.feedback_bank.clauses[:rec['old_proofs']]
                token=hashlib.sha256(b.tobytes()+h.tobytes()+u.tobytes()+str((rec['event'],rec['restart'],rec['old_proofs'])).encode()).hexdigest();assert token==example['input_sha256']
                if token not in cache:cache[token]=tied.candidate_bank(x,b,h,u,clauses)
                pool,_,stats=cache[token];item=pool[example['candidate_index']];nb,nh=tied.materialize(b,h,item)
                assert np.array_equal(nb,np.array(example['b'])) and np.array_equal(nh,np.array(example['h']));replays+=1
                values=[tied.scalar.phi_exact(x,nb,nh,c['a']) for c in clauses];full_cut_checks+=len(values);assert any(t>0 for t in values)
                if item['kind']=='tied':assert values==item['exact_phi']
                actual=model.base.pattern(x,nb).astype(np.uint8) if example['channel']=='forward' else model.repair.split_many(x,nb[None],nh[None])[0]
                assert np.array_equal(actual,reg)
                assert not model.conflict.clause_mask(reg[None],clauses)[0]
                reject,checks=model.conflict.full_mask(x,v,reg[None],clauses);assert not reject[0];safe_mode_checks+=len(clauses)
                if (seed,key) not in independent_regions:
                    _,_,g,rhs=neighbor.pattern_matrix(x,v,reg);poly,_=neighbor.posterior.polytope(g,rhs)
                    assert poly is not None and np.isclose(poly['volume'],positive[key]['volume'],rtol=1e-9,atol=1e-30)
                    center=poly['center']+poly['scale']*poly['interior'];bb=[F(float(t)) for t in center];prev=[F(float(t)) for t in x];codes=[]
                    assert all(abs(t)<=F(.12) for t in bb)
                    for j in range(4):
                        zz=[p+bb[j] for p in prev];codes.append([sum(z>=k for k in [F(0),F(1,2),F(1)]) for z in zz]);prev=[tied.scalar.g(z) for z in zz]
                    assert np.array_equal(np.array(codes,dtype=np.uint8),reg)
                    err=max(abs(p-F(float(y))) for p,y in zip(prev,v));assert err<=F(float(model.base.EPS))
                    independent_regions.add((seed,key))
                    primal=dict(bias=center.tolist(),bias_rationals=[[str(t.numerator),str(t.denominator)] for t in bb],exact_support_max_error=float(err),volume=poly['volume'])
                else:primal=None
                # Identify whether this finite proposal is already available
                # without any cut-dependent interval endpoint.
                free_sources=[]
                if item['kind']=='tied':
                    for segment in cache[token][1][1]:
                        if segment['j']!=item['j']:continue
                        values_free=tied.proposals(segment['linear']/segment['quad'],segment['lo'],segment['hi'],F(float(b[item['j']])))
                        if item['value'] in values_free:free_sources.append(segment['segment'])
                witnesses.append(dict(seed=seed,pattern=key,channel=example['channel'],kind=item['kind'],event=example['event'],restart=example['restart'],
                    candidate_index=example['candidate_index'],layer=item['j'],segment=item['segment'],value=float(item['value']),energy_rank=example['energy_rank'],
                    positive_current_point_cuts=sum(t>0 for t in values),max_point_cut=float(max(values)),old_clauses=len(clauses),
                    same_mode_passes_valid_clause_and_full_bound=True,available_from_free_tied_segment=bool(free_sources),free_tied_segments=free_sources,
                    already_in_old_positive_volume_k24=key in positive_volume[seed],exact_primal_witness=primal))
        cr=root/'posterior_state_reuse/conditional_risk';audit=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/audit['curve_file'];assert sha(cp)==audit['curve_sha256'];curve_hashes.append(dict(seed=seed,sha256=sha(cp)))
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        current=set(old[seed]['positive_mode_keys'])
        for group in ['before']+protocol['groups']:
            new=set() if group=='before' else set(next(r for r in rows if r['seed']==seed and r['group']==group)['new_keys'])
            mask=np.array([p in current|new for p in patterns]);mass=float(weights[mask].sum());estimate=np.einsum('k,rkq->rq',weights*mask/mass,means)
            risk=float(np.mean([np.trapezoid((estimate[a]-full[a])*(estimate[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            quality.append(dict(seed=seed,group=group,new_modes=len(new),mass=mass,ideal_truncation=risk))
    summary=[]
    for group in ['before']+protocol['groups']:
        rr=[r for r in quality if r['group']==group];summary.append(dict(group=group,new_modes=sum(r['new_modes'] for r in rr),mean_mass=float(np.mean([r['mass'] for r in rr])),mean_ideal_truncation=float(np.mean([r['ideal_truncation'] for r in rr]))))
    strong_path=root/'physical_cut_memory/development/episodes.json';strong_rows=json.loads(strong_path.read_text());strong=[]
    for method in ['alm_retained_full','adam16_retained_full','adam60_h2_full','direct4096_h2_full','alm_h2_full']:
        lookup={r['seed']:set(r['positive_mode_keys']) for r in strong_rows if r['method']==method and r['n_context']==4}
        missing=[dict(seed=s,pattern=k) for s,k in sorted(independent_regions) if k not in lookup[s]]
        strong.append(dict(method=method,already_contains=len(independent_regions)-len(missing),missing=missing))
    result=dict(source_sha256=sha(Path(__file__)),runtime_source_sha256=protocol['source_sha256'],input_sha256={name:sha(out/name) for name in ['protocol.json','rows.json','summary.json','new_examples.json']},
        independent_positive_regions=len(independent_regions),exact_new_examples_replayed=replays,full_actual_point_cut_checks=full_cut_checks,safe_mode_clause_tests=safe_mode_checks,
        witnesses=witnesses,quality=quality,summary=summary,curve_hashes=curve_hashes,strong_baseline_coverage=strong,strong_baseline_file_sha256=sha(strong_path),
        scope='evaluator-only union potential and exact existence witnesses; no online selection, causal state change, or runtime superiority')
    (out/'analysis.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k in ['independent_positive_regions','exact_new_examples_replayed','full_actual_point_cut_checks','safe_mode_clause_tests','witnesses','summary','scope']},indent=2))


if __name__=='__main__':main()
