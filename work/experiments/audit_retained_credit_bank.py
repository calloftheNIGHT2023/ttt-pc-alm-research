"""Reuse newly certified matching directions; matched-size random controls."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import retained_credit_capture as capture
model=capture.model;normal=capture.normal


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results';inp=root/'retained_credit/diagnostic';out=root/'retained_credit/cross_bank';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((inp/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    hashes[Path(__file__).name]=sha(Path(__file__));source=json.loads((inp/'proofs.json').read_text());oldrows=json.loads((inp/'rows.json').read_text());refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=parent['seeds'],profiles=parent['profiles'],max_additional_directions=32,random_seed=17731,
        proof_file_sha256=sha(inp/'proofs.json'),rows_sha256=sha(inp/'rows.json'),rule='sort matching proof by pattern, normalize maxabs, deduplicate, first 32; random matched count; original screening never removed',scope='support-only reuse diagnostic; prior capture costs are not free')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[]
    random=np.random.default_rng(17731).normal(size=(32,4,4));random/=np.max(np.abs(random),axis=(1,2))[:,None,None]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4];pending=[]
        for name in protocol['profiles']:
            prior=next(r for r in oldrows if r['seed']==seed and r['method']==name);bank=[];seen=set()
            for proof in sorted([p for p in source if p['seed']==seed and p['method']==name],key=lambda p:p['pattern']):
                a=np.array(proof['a']);a/=np.max(np.abs(a));key=a.tobytes()
                if key not in seen:seen.add(key);bank.append(a)
                if len(bank)==32:break
            bank=np.array(bank).reshape(-1,4,4);regs=np.array([np.frombuffer(bytes.fromhex(k),np.uint8).reshape(4,4) for k in prior['after_keys']])
            for kind,dd in [('retained',bank),('random_matched',random[:len(bank)])]:
                reject,pf,cost=normal.screen_bank(x,v,regs,dd);pending.append((name,kind,prior,regs,reject,pf,cost))
        item=refs[seed];path=refroot/item['reference_file'];assert sha(path)==item['reference_sha256'];positive={r['pattern'] for r in json.loads(path.read_text())['positive_regions']}
        cr=root/'posterior_state_reuse/conditional_risk';ca=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        for name,kind,prior,regs,reject,pf,cost in pending:
            before=prior['after_keys'];after=[k for k,r in zip(before,reject) if not r];assert not {p['pattern'] for p in pf}&positive
            for proof in pf:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);exact=normal.exact_optimum(x,v,reg,np.array(proof['a']));assert exact==proof['exact'] and exact['positive']
                _,_,g,rhs=model.old.old.previous.neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                proofs.append(dict(seed=seed,method=name,bank=kind,**proof,lp_status=int(lp.status)))
            rp={k:i+1 for i,k in enumerate(before) if k in positive};rq={k:i+1 for i,k in enumerate(after) if k in positive};assert set(rp)==set(rq)
            for k in range(len(before)+1):assert set(before[:k])&positive<=set(after[:k])&positive
            mask=np.array([k in set(after[:24]) for k in patterns]);mass=float(weights[mask].sum());risk=None
            if mass:
                ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            rows.append(dict(seed=seed,method=name,bank=kind,before_calls=len(before),after_calls=len(after),new_strict_rejections=len(pf),
                advanced_positive_modes=sum(rp[k]>rq[k] for k in rp),rank_advancement=sum(rp[k]-rq[k] for k in rp),
                before_keys=before,after_keys=after,k24=dict(mass=mass,ideal_truncation=risk,positive_modes=int(mask.sum())),**cost))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8');print(json.dumps(dict(seed=seed,completed=len(rows)//18,proofs=len(proofs))),flush=True)
    summary=[]
    for name in protocol['profiles']:
        for kind in ['retained','random_matched']:
            rr=[r for r in rows if r['method']==name and r['bank']==kind]
            summary.append(dict(method=name,bank=kind,**{k:sum(r[k] for r in rr) for k in ['before_calls','after_calls','new_strict_rejections','advanced_positive_modes','rank_advancement','numeric_pairs','exact_checks']},
                mean_directions=float(np.mean([r['directions'] for r in rr])),mean_screen_seconds=float(np.mean([r['seconds'] for r in rr])),
                k24=dict(mean_mass=float(np.mean([r['k24']['mass'] for r in rr])),mean_ideal_truncation=None if any(r['k24']['ideal_truncation'] is None for r in rr) else float(np.mean([r['k24']['ideal_truncation'] for r in rr])),positive_modes=sum(r['k24']['positive_modes'] for r in rr))))
    result=dict(source_hashes=len(hashes),diagnostic_cases=len(rows),exact_fraction_and_lp_proofs=len(proofs),summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
