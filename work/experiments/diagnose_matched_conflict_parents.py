"""Support-only matched parent counts, strong LP/positive/random/BP controls."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import credit_conflict as conflict
import retained_credit_memory as model
import neighbor_mode_memory as neighbor
base=model.base


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args();root=args.project/'results';out=root/'matched_conflict_parents/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    inherited=json.loads((root/'credit_conflict/diagnostic/protocol.json').read_text());hashes=inherited['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    inp=[root/'retained_credit/diagnostic/proofs.json',root/'retained_credit/cross_bank/proofs.json'];records=[]
    for p in inp:records.extend(json.loads(p.read_text()))
    baseline={(r['seed'],r['method']):r for r in json.loads((root/'retained_credit/cross_bank/rows.json').read_text()) if r['bank']=='retained'}
    cfg=next(c for c in json.loads((root/'retained_credit_memory/development/protocol.json').read_text())['configs'] if c['name']=='alm_retained_full')
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    methods=['local','canonical','lp_margin','positive_volume']+[f'random_{i}' for i in range(4)]+['shadow_bp_filled']
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=inherited['seeds'],methods=methods,config=cfg,random_seed=794183,geometry_budget=24,radius=2,
        input_sha256={str(p.relative_to(root)):sha(p) for p in inp+[root/'retained_credit/cross_bank/rows.json']},
        parent_budget='number of frozen local-credit proofs for this task',scope='all support-only controls; 16 old tasks; numerical ideal truncation, not actual online query benefit')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];audits=[];geometry_checks=0
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-base.EPS,base.EPS,24))[:4]
        start=time.perf_counter();_,regs,meta,collectors=model.prepare(x,v,cfg);assert [r.tobytes().hex() for r in regs]==baseline[seed,'alm_local_k24']['after_keys'];prepare=time.perf_counter()-start
        raw=set()
        for c in collectors:raw.update(c.forward);raw.update(c.split)
        ordered=sorted(raw);rr=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in ordered]);coarse=rr[~conflict.screen.contract(x,v,rr,5)];universe={r.tobytes() for r in coarse}
        lp_count=0;invalid=[];positive={};margin={};start=time.perf_counter()
        for reg in coarse:
            key=reg.tobytes();_,_,g,rhs=neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);lp_count+=1
            if lp.status==2:
                invalid.append(key);norm=np.maximum(np.linalg.norm(g,axis=1),1e-12)
                # Degenerate constant inequalities use unit scaling.
                norm[np.linalg.norm(g,axis=1)==0]=1.
                soft=linprog(np.r_[np.zeros(4),1.],A_ub=np.column_stack([g/norm[:,None],-np.ones(len(g))]),b_ub=rhs/norm,bounds=[(-.12,.12)]*4+[(0,None)])
                assert soft.success;margin[key]=float(soft.fun);lp_count+=1
            else:
                assert lp.success
                poly,_=neighbor.posterior.polytope(g,rhs)
                if poly is not None:positive[key]=float(poly['volume'])
        classification=time.perf_counter()-start
        local=sorted({bytes.fromhex(r['pattern']) for r in records if r['seed']==seed and r['method']=='alm_local_k24'});m=len(local);assert set(local)<=set(invalid)
        ranked=sorted(invalid,key=lambda k:(margin[k],k));assert len(ranked)>=m
        def fill(front):return list(dict.fromkeys(front+ranked))[:m]
        bp=sorted({bytes.fromhex(r['pattern']) for r in records if r['seed']==seed and r['method']=='alm_bp_c20_k24'}&set(invalid))
        selections={'local':local,'canonical':sorted(invalid)[:m],'lp_margin':ranked[:m],
            'positive_volume':fill(sorted(positive,key=lambda k:(-positive[k],k))),'shadow_bp_filled':fill(bp)}
        for i in range(4):
            rng=np.random.default_rng(np.random.SeedSequence([794183,seed,i]));selections[f'random_{i}']=[invalid[int(k)] for k in rng.choice(len(invalid),m,replace=False)]
        assert all(len(s)==m and len(set(s))==m for s in selections.values())
        pending=[]
        for method in methods:
            start=time.perf_counter();dist={}
            for key in selections[method]:
                reg=np.frombuffer(key,np.uint8).reshape(4,4)
                for new,d in conflict.radius_two(reg).items():
                    if new not in universe:dist[new]=min(d,dist.get(new,3))
            keys=sorted(dist,key=lambda k:(dist[k],k));newregs=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in keys],np.uint8).reshape(-1,4,4);generation=time.perf_counter()-start
            start=time.perf_counter();reject=conflict.screen.contract(x,v,newregs,20) if len(newregs) else np.zeros(0,bool);seconds=time.perf_counter()-start
            pending.append(dict(method=method,parents=m,parent_keys=[k.hex() for k in selections[method]],universe=len(keys),
                keys=[r.tobytes().hex() for r in newregs[~reject]],generation_seconds=generation,contract_seconds=seconds))
        # Reference classification occurs only after all selections and proposals.
        item=refs[seed];path=refroot/item['reference_file'];assert sha(path)==item['reference_sha256'];ref=json.loads(path.read_text());complete={r['pattern']:r for r in ref['positive_regions']}
        expected=set(baseline[seed,'alm_local_k24']['after_keys'])&complete.keys();assert {k.hex() for k in positive}==expected
        cr=root/'posterior_state_reuse/conditional_risk';a=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/a['curve_file'];assert sha(cp)==a['curve_sha256']
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        checked=set()
        for item in pending:
            quality={}
            for label,seq in [('before',[]),('all',item['keys']),('k24',item['keys'][:24])]:
                added=(set(seq)&complete.keys())-expected;found=expected|added;mask=np.array([p in found for p in patterns]);mass=float(weights[mask].sum());risk=None
                if mass:
                    ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
                quality[label]=dict(mass=mass,ideal_truncation=risk,new_positive_modes=len(added),new_keys=sorted(added))
                for key in added-checked:
                    reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,4);_,_,g,rhs=neighbor.pattern_matrix(x,v,reg);poly,_=neighbor.posterior.polytope(g,rhs)
                    assert poly is not None and np.isclose(poly['volume'],complete[key]['volume'],rtol=1e-9,atol=1e-30);checked.add(key);geometry_checks+=1
            rows.append(dict(seed=seed,**item,quality=quality))
        audits.append(dict(seed=seed,original_sequence_exact=True,original_positive_set_exact=True,coarse_modes=len(coarse),infeasible=len(invalid),positive=len(positive),parents=m,
            explicit_classification_lp_calls=lp_count,prepare_seconds=prepare,classification_seconds=classification,scope='polytope geometry has additional internal LP calls; explicit count is not total'))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,completed=len(rows)//9,new_geometry=geometry_checks)),flush=True)
    summary=[]
    for method in methods:
        rr=[r for r in rows if r['method']==method]
        summary.append(dict(method=method,parent_count=sum(r['parents'] for r in rr),universe=sum(r['universe'] for r in rr),geometry_candidates=sum(len(r['keys']) for r in rr),
            quality={label:dict(mean_mass=float(np.mean([r['quality'][label]['mass'] for r in rr])),new_positive_modes=sum(r['quality'][label]['new_positive_modes'] for r in rr),
                mean_ideal_truncation=float(np.mean([r['quality'][label]['ideal_truncation'] for r in rr]))) for label in ['before','all','k24']},
            mean_generation_seconds=float(np.mean([r['generation_seconds'] for r in rr])),mean_contract_seconds=float(np.mean([r['contract_seconds'] for r in rr]))))
    result=dict(source_hashes=len(hashes),cases=len(rows),original_trajectory_replays=len(audits),independent_new_positive_geometry=geometry_checks,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
