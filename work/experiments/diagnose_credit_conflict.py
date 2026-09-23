"""Frozen conflict-guided radius-two search diagnostic on old support sets."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import credit_conflict as conflict
import neighbor_mode_memory as neighbor
base=conflict.base


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';out=root/'credit_conflict/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((root/'credit_primal_reconstruction/diagnostic_v2/protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(conflict.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    inputs=[root/'retained_credit/diagnostic/proofs.json',root/'retained_credit/cross_bank/proofs.json'];records=[]
    for p in inputs:records.extend(json.loads(p.read_text()))
    baseline_rows=json.loads((root/'retained_credit/cross_bank/rows.json').read_text());baseline={(r['seed'],r['method']):r for r in baseline_rows if r['bank']=='retained'}
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=parent['seeds'],profiles=parent['profiles'],verification=conflict.verify(),radius=2,geometry_budget=24,p_refinement_sweeps=4,
        input_sha256={str(p.relative_to(root)):sha(p) for p in inputs+[root/'retained_credit/cross_bank/rows.json']},
        order='minimum Hamming distance to any parent, then canonical bytes; exclude old processed and parent modes; shared C20',
        scope='old support diagnostic, complete reference and curves evaluator-only; not actual online timing or PC-specific advantage')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[];all_lp=0;all_geo=0;reference_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4];pending=[]
        for name in protocol['profiles']:
            pairs=sorted([r for r in records if r['seed']==seed and r['method']==name],key=lambda r:r['pattern']);old=baseline[seed,name]
            excluded={bytes.fromhex(k) for k in old['before_keys']+old['after_keys']}|{bytes.fromhex(r['pattern']) for r in pairs}
            begin=time.perf_counter();clauses=[];dist={};extract_seconds=0.
            for pair in pairs:
                reg=np.frombuffer(bytes.fromhex(pair['pattern']),np.uint8).reshape(4,4);a=np.array(pair['a']);start=time.perf_counter()
                pp,_,_=conflict.normal.refine(x,v,reg[None],a[None],4);c=conflict.extract(x,v,reg,pp[0],a);extract_seconds+=time.perf_counter()-start
                if c is not None:clauses.append(dict(parent=pair['pattern'],**c));proofs.append(dict(seed=seed,method=name,parent=pair['pattern'],**c))
                for key,distance in conflict.radius_two(reg).items():
                    if key not in excluded:dist[key]=min(distance,dist.get(key,3))
            keys=sorted(dist,key=lambda k:(dist[k],k));regs=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in keys],np.uint8).reshape(-1,4,4)
            generation_seconds=time.perf_counter()-begin;start=time.perf_counter()
            contracted=conflict.screen.contract(x,v,regs,20) if len(regs) else np.zeros(0,bool);survivors=regs[~contracted];contract_seconds=time.perf_counter()-start
            start=time.perf_counter();cm=conflict.clause_mask(survivors,clauses) if len(survivors) else np.zeros(0,bool);clause_seconds=time.perf_counter()-start
            start=time.perf_counter();fm,checks=conflict.full_mask(x,v,survivors,clauses) if len(survivors) else (np.zeros(0,bool),0);full_seconds=time.perf_counter()-start
            assert np.all(~cm|fm)
            for reg in survivors[fm]:
                _,_,g,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2;all_lp+=1
            paths={'c20':[r.tobytes().hex() for r in survivors], 'clause':[r.tobytes().hex() for r in survivors[~cm]], 'full_dual':[r.tobytes().hex() for r in survivors[~fm]]}
            pending.append(dict(method=name,parents=len(pairs),clauses=clauses,universe=len(keys),c20_removed=int(contracted.sum()),paths=paths,exact_deletion_checks=checks,
                generation_seconds=generation_seconds,extract_seconds=extract_seconds,contract_seconds=contract_seconds,clause_seconds=clause_seconds,full_seconds=full_seconds))
        # Algorithm outputs above are fixed before reference or risk curves load.
        item=refs[seed];rp=refroot/item['reference_file'];assert sha(rp)==item['reference_sha256'];reference=json.loads(rp.read_text());positive={r['pattern']:r for r in reference['positive_regions']}
        cr=root/'posterior_state_reuse/conditional_risk';audit=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/audit['curve_file'];assert sha(cp)==audit['curve_sha256']
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        for item in pending:
            name=item['method'];clauses=item.pop('clauses');paths=item['paths'];existing=set(baseline[seed,name]['after_keys'])&positive.keys();reference_regs=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,4) for k in positive])
            assert not np.any(conflict.clause_mask(reference_regs,clauses));reference_checks+=len(positive)*len(clauses)
            for label in ['clause','full_dual']:
                assert set(paths[label])&positive.keys()==set(paths['c20'])&positive.keys()
                for k in range(len(paths['c20'])+1):assert set(paths['c20'][:k])&positive.keys()<=set(paths[label][:k])&positive.keys()
            new=set(paths['c20'])&positive.keys()
            for key in new:
                reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,4);_,_,g,rhs=neighbor.pattern_matrix(x,v,reg);poly,_=neighbor.posterior.polytope(g,rhs)
                assert poly is not None and np.isclose(poly['volume'],positive[key]['volume'],rtol=1e-9,atol=1e-30);all_geo+=1
            quality={}
            for label,seq in [('before',[]),('all',paths['c20'])]+[(k,z[:24]) for k,z in paths.items()]:
                found=existing|(set(seq)&positive.keys());mask=np.array([p in found for p in patterns]);mass=float(weights[mask].sum());risk=None
                if mass:
                    ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
                quality[label]=dict(mass=mass,ideal_truncation=risk,new_positive_modes=len(found-existing))
            rows.append(dict(seed=seed,**item,clause_count=len(clauses),clause_cardinalities=[c['cardinality'] for c in clauses],required_edits=[c['minimum_required_edits'] for c in clauses],quality=quality))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'clauses.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,completed=len(rows)//9,clauses=len(proofs),independent_lp=all_lp,new_positive_geometry=all_geo)),flush=True)
    summary=[]
    for name in protocol['profiles']:
        rr=[r for r in rows if r['method']==name];cards=[c for r in rr for c in r['clause_cardinalities']]
        summary.append(dict(method=name,parents=sum(r['parents'] for r in rr),clauses=len(cards),mean_clause_cardinality=float(np.mean(cards)) if cards else None,
            universe=sum(r['universe'] for r in rr),remaining_calls={k:sum(len(r['paths'][k]) for r in rr) for k in ['c20','clause','full_dual']},
            quality={label:dict(mean_mass=float(np.mean([r['quality'][label]['mass'] for r in rr])),new_positive_modes=sum(r['quality'][label]['new_positive_modes'] for r in rr),
                mean_ideal_truncation=None if any(r['quality'][label]['ideal_truncation'] is None for r in rr) else float(np.mean([r['quality'][label]['ideal_truncation'] for r in rr]))) for label in ['before','all','c20','clause','full_dual']},
            mean_seconds={k:float(np.mean([r[k] for r in rr])) for k in ['generation_seconds','extract_seconds','contract_seconds','clause_seconds','full_seconds']}))
    result=dict(source_hashes=len(hashes),diagnostic_cases=len(rows),strict_clauses=len(proofs),independent_lp_rejections=all_lp,new_positive_geometry_checks=all_geo,complete_positive_clause_checks=reference_checks,all_prefix_inclusions=True,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
