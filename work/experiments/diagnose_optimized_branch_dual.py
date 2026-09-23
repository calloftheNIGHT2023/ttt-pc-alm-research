"""Fixed-credit analytic dual-block optimum, exact proofs and prefix effects."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import diagnose_branch_normal_credit as parent
import optimized_branch_dual as opt
model=parent.model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);args=parser.parse_args()
    root=args.project/'results';online=root/'budgeted_credit_memory/development';refroot=root/'posterior_state_reuse/first_write_reference';out=root/'optimized_branch_dual/diagnostic'
    inherited=json.loads((root/'branch_normal_credit/diagnostic/protocol.json').read_text());hashes=inherited['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(opt.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    old=json.loads((online/'protocol.json').read_text());cfgs={c['name']:c for c in old['configs']};episodes={(r['seed'],r['method']):r for r in json.loads((online/'episodes.json').read_text()) if r['repetition']==0 and r['n_context']==4}
    prior={(r['seed'],r['method'],r['bank']):r for r in json.loads((root/'branch_normal_credit/diagnostic/rows.json').read_text())};refindex={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=inherited['seeds'],profiles=parent.PROFILES,verification=opt.verify(),random_seed=771931,
        online_protocol_sha256=sha(online/'protocol.json'),scope='support diagnostic only, exact rational fixed-credit optimized-p bounds, all controls treated identically; reference evaluator-only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    random=np.random.default_rng(771931).normal(size=(32,4,4));random/=np.max(np.abs(random),axis=(1,2))[:,None,None];rows=[];proofs=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        pending=[]
        for name,kind in parent.PROFILES:
            _,regs,meta,bank,detail=parent.inherited.capture(x,v,cfgs[name]);assert meta['unbudgeted_pattern_keys']==episodes[seed,name]['unbudgeted_pattern_keys']
            if kind=='random32':bank=random
            if kind=='zero':bank=np.zeros((1,4,4))
            reject,pf,cost=opt.screen_bank(x,v,regs,bank);pending.append((name,kind,regs,reject,pf,cost))
        entry=refindex[seed];path=refroot/entry['reference_file'];assert sha(path)==entry['reference_sha256'];ref=json.loads(path.read_text());positive={r['pattern'] for r in ref['positive_regions']}
        cr=root/'posterior_state_reuse/conditional_risk';ca=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as curve:
            patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        for name,kind,regs,reject,pf,cost in pending:
            before=[r.tobytes().hex() for r in regs];after=[k for k,skip in zip(before,reject) if not skip];assert not {p['pattern'] for p in pf}&positive
            pr=prior[seed,name,kind];assert before==pr['before_keys'];assert set(after)<=set(pr['after_keys'])
            for proof in pf:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);exact=opt.exact_optimum(x,v,reg,np.array(proof['a']));assert exact==proof['exact'] and exact['positive']
                _,_,g,rhs=model.old.previous.neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                proofs.append(dict(seed=seed,method=name,bank=kind,**proof,lp_status=int(lp.status)))
            rp={k:i+1 for i,k in enumerate(before) if k in positive};rq={k:i+1 for i,k in enumerate(after) if k in positive};assert set(rp)==set(rq)
            for k in range(len(before)+1):assert set(before[:k])&positive<=set(after[:k])&positive
            quality={}
            for label,seq in [('before',before),('coordinate',pr['after_keys']),('optimized',after)]:
                found=set(seq[:24])&positive;mask=np.array([k in found for k in patterns]);mass=float(weights[mask].sum());risk=None
                if mass:
                    ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
                quality[label]=dict(mass=mass,ideal_truncation=risk,positive_modes=len(found))
            rows.append(dict(seed=seed,method=name,bank=kind,before_calls=len(before),after_calls=len(after),new_strict_rejections=len(pf),
                beyond_coordinate=len(pr['after_keys'])-len(after),advanced_positive_modes=sum(rp[k]>rq[k] for k in rp),rank_advancement=sum(rp[k]-rq[k] for k in rp),
                before_keys=before,after_keys=after,k24=quality,**cost))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,completed=len(rows)//len(parent.PROFILES),proofs=len(proofs))),flush=True)
    summary=[]
    for name,kind in parent.PROFILES:
        rr=[r for r in rows if r['method']==name and r['bank']==kind]
        summary.append(dict(method=name,bank=kind,**{k:sum(r[k] for r in rr) for k in ['before_calls','after_calls','new_strict_rejections','beyond_coordinate','advanced_positive_modes','rank_advancement','numeric_pairs','exact_checks']},
            mean_screen_seconds=float(np.mean([r['seconds'] for r in rr])),max_main_array_bytes_subtotal=max(r['main_array_bytes_subtotal'] for r in rr),
            k24={label:dict(mean_mass=float(np.mean([r['k24'][label]['mass'] for r in rr])),mean_ideal_truncation=None if any(r['k24'][label]['ideal_truncation'] is None for r in rr) else float(np.mean([r['k24'][label]['ideal_truncation'] for r in rr])),
                positive_modes=sum(r['k24'][label]['positive_modes'] for r in rr)) for label in ['before','coordinate','optimized']},
            changed_k24_tasks=[r['seed'] for r in rr if r['k24']['optimized']['positive_modes']>r['k24']['before']['positive_modes']]))
    result=dict(source_hashes=len(hashes),original_sequence_replays=len(rows),exact_fraction_and_lp_proofs=len(proofs),all_coordinate_rejections_retained=True,summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
