"""Primal-witness discovery diagnostic with full-discovery and simple controls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import credit_primal_reconstruction as primal
import retained_credit_memory as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';out=root/'credit_primal_reconstruction/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((root/'retained_credit_memory/development/protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(primal.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    sourcepaths=[root/'retained_credit/diagnostic/proofs.json',root/'retained_credit/cross_bank/proofs.json'];records=[]
    for path in sourcepaths:records.extend(json.loads(path.read_text()))
    assert all(r.get('bank','retained')=='retained' for r in records)
    base_rows=json.loads((root/'retained_credit/cross_bank/rows.json').read_text());baseline={(r['seed'],r['method']):r['after_keys'] for r in base_rows if r['bank']=='retained'}
    oldprotocol=json.loads((root/'retained_credit/diagnostic/protocol.json').read_text());refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=oldprotocol['seeds'],profiles=oldprotocol['profiles'],modes=['credit','zero_credit','shuffled_credit','prior_random'],random_seed=135229,
        input_proofs_sha256={str(p.relative_to(root)):sha(p) for p in sourcepaths},verification=primal.verify(),scope='saved support-only certificate primal proposals, numerical complete reference evaluator-only, compared against full discovery not K24')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];details=[];geometry_checks=0;max_obj=0.
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4];pending=[]
        for mindex,name in enumerate(protocol['profiles']):
            pairs=sorted([r for r in records if r['seed']==seed and r['method']==name],key=lambda r:r['pattern'])
            if not pairs:
                for kind in protocol['modes']:pending.append((name,kind,[],[],0.,0,[]))
                continue
            regs=np.array([np.frombuffer(bytes.fromhex(r['pattern']),np.uint8).reshape(4,4) for r in pairs]);aa=np.array([r['a'] for r in pairs]);shuffled=[];random=[]
            for k,a in enumerate(aa):
                rng=np.random.default_rng(np.random.SeedSequence([135229,seed,mindex,k]));shuffled.append(rng.permutation(a.ravel()).reshape(4,4));random.append(rng.uniform(-.12,.12,4))
            for kind,credit in [('credit',aa),('zero_credit',np.zeros_like(aa)),('shuffled_credit',np.array(shuffled)),('prior_random',None)]:
                begin=time.perf_counter();proofdata=[]
                if kind=='prior_random':bb=np.array(random);valid=np.ones(len(bb),bool)
                else:
                    result=primal.reconstruct(x,v,regs,credit);bb=result['b'];valid=result['valid'];value=primal.dual.float_optimum(x,v,regs,credit)
                    difference=float(np.max(np.abs(value[valid]-result['objective'][valid]),initial=0));max_obj=max(max_obj,difference);assert difference<1e-10
                    for k in range(len(bb)):
                        proofdata.append(dict(parent_pattern=pairs[k]['pattern'],valid=bool(valid[k]),b=bb[k].tolist(),objective=float(result['objective'][k]),
                            forward_gap=float(result['forward_gap'][k]),propagated_residual_bound=float(result['propagated_residual_bound'][k]),support_max_error=float(result['observed_support_max_error'][k])))
                keys=[model.base.pattern(x,b).astype(np.uint8).tobytes().hex() for b in bb[valid]]
                pending.append((name,kind,keys,bb[valid].tolist(),time.perf_counter()-begin,len(pairs),proofdata))
        item=refs[seed];path=refroot/item['reference_file'];assert sha(path)==item['reference_sha256'];ref=json.loads(path.read_text());positive={r['pattern']:r for r in ref['positive_regions']}
        cr=root/'posterior_state_reuse/conditional_risk';ca=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        for name,kind,keys,bb,seconds,parents,proofdata in pending:
            found=set(baseline[seed,name])&positive.keys();new=(set(keys)&positive.keys())-found
            for key in new:
                reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,4);_,_,g,rhs=model.old.old.old.previous.neighbor.pattern_matrix(x,v,reg);poly,note=model.old.old.old.posterior.polytope(g,rhs)
                assert poly is not None and np.isclose(poly['volume'],positive[key]['volume'],rtol=1e-9,atol=1e-30);geometry_checks+=1
            quality={}
            for label,ss in [('before',found),('after',found|new)]:
                mask=np.array([p in ss for p in patterns]);mass=float(weights[mask].sum());risk=None
                if mass:
                    ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
                quality[label]=dict(mass=mass,ideal_truncation=risk,positive_modes=len(ss))
            rows.append(dict(seed=seed,method=name,proposal=kind,parents=parents,valid_proposals=len(keys),unique_proposed_modes=len(set(keys)),new_modes_outside_original_universe=len(set(keys)-set(baseline[seed,name])),
                new_positive_modes=len(new),new_positive_keys=sorted(new),quality=quality,proposal_seconds=seconds,
                support_feasible_points=sum(np.max(np.abs(model.base.forward(x,np.array(b))-v))<=model.base.EPS+model.base.TOL for b in bb)))
            details.append(dict(seed=seed,method=name,proposal=kind,proposed_keys=keys,primal_data=proofdata))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proposals.json').write_text(json.dumps(details,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,completed=len(rows)//36,geometry_checks=geometry_checks)),flush=True)
    summary=[]
    for name in protocol['profiles']:
        for kind in protocol['modes']:
            rr=[r for r in rows if r['method']==name and r['proposal']==kind]
            summary.append(dict(method=name,proposal=kind,**{k:sum(r[k] for r in rr) for k in ['parents','valid_proposals','unique_proposed_modes','new_modes_outside_original_universe','new_positive_modes','support_feasible_points']},
                mean_proposal_seconds=float(np.mean([r['proposal_seconds'] for r in rr])),quality={label:dict(mean_mass=float(np.mean([r['quality'][label]['mass'] for r in rr])),mean_ideal_truncation=None if any(r['quality'][label]['ideal_truncation'] is None for r in rr) else float(np.mean([r['quality'][label]['ideal_truncation'] for r in rr]))) for label in ['before','after']}))
    result=dict(source_hashes=len(hashes),diagnostic_cases=len(rows),new_positive_geometry_checks=geometry_checks,max_primal_objective_difference=max_obj,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
