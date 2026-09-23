"""Independent proof and prefix audit of local branch-normal corrections."""
import argparse,hashlib,json,time
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import diagnose_credit_segment_mixing as inherited
import branch_normal_credit as normal
from certified_branch_solver import exact_certificate
model=inherited.model
PROFILES=inherited.PROFILES+[('alm_none_k24','zero')]


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args()
    root=args.project/'results';online=root/'budgeted_credit_memory/development';refroot=root/'posterior_state_reuse/first_write_reference';out=root/'branch_normal_credit/diagnostic'
    parent=json.loads((root/'credit_segment_mixing/diagnostic/protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(normal.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    old=json.loads((online/'protocol.json').read_text());cfgs={c['name']:c for c in old['configs']};episodes={(r['seed'],r['method']):r for r in json.loads((online/'episodes.json').read_text()) if r['repetition']==0 and r['n_context']==4}
    refindex={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=parent['seeds'],profiles=PROFILES,sweeps=4,reported_sweeps=[1,4],random_seed=771931,verification=normal.verify(),
        online_protocol_sha256=sha(online/'protocol.json'),scope='observed-support only; four local p sweeps for every supplied credit; exact proof audits; reference and targets evaluator-only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    random=np.random.default_rng(protocol['random_seed']).normal(size=(32,4,4));random/=np.max(np.abs(random),axis=(1,2))[:,None,None]
    rows=[];proofs=[];audits=[]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        pending=[]
        for name,kind in PROFILES:
            begin=time.perf_counter();_,regs,meta,bank,detail=inherited.capture(x,v,cfgs[name]);assert meta['unbudgeted_pattern_keys']==episodes[seed,name]['unbudgeted_pattern_keys']
            if kind=='random32':bank=random
            if kind=='zero':bank=np.zeros((1,4,4))
            reject,pf,cost=normal.screen_bank(x,v,regs,bank,protocol['sweeps'])
            pending.append((name,kind,regs,reject,pf,cost,detail,time.perf_counter()-begin))
        entry=refindex[seed];path=refroot/entry['reference_file'];assert sha(path)==entry['reference_sha256'];ref=json.loads(path.read_text());positive={r['pattern'] for r in ref['positive_regions']}
        for name,kind,regs,reject,pf,cost,detail,elapsed in pending:
            before=[r.tobytes().hex() for r in regs];after=[key for key,skip in zip(before,reject) if not skip];assert not {p['pattern'] for p in pf}&positive
            for proof in pf:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);pp=np.array(proof['p']);aa=np.array(proof['a'])
                with model.old.old.core.pipeline.discovery_box(.12):exact=exact_certificate(x,v,reg,pp,aa)
                val=Fraction(int(exact['numerator']),int(exact['denominator']));assert val>0 and Fraction(proof['lower'])<=val
                _,_,g,rhs=model.old.previous.neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                proofs.append(dict(seed=seed,method=name,bank=kind,**proof,exact=exact,lp_status=int(lp.status)))
            rp={key:i+1 for i,key in enumerate(before) if key in positive};rq={key:i+1 for i,key in enumerate(after) if key in positive};assert set(rp)==set(rq)
            for k in range(len(before)+1):assert set(before[:k])&positive<=set(after[:k])&positive
            rows.append(dict(seed=seed,method=name,bank=kind,before_calls=len(before),after_calls=len(after),new_strict_rejections=len(pf),
                before_k24_positive=len(set(before[:24])&positive),after_k24_positive=len(set(after[:24])&positive),
                advanced_positive_modes=sum(rp[k]>rq[k] for k in rp),total_positive_rank_advancement=sum(rp[k]-rq[k] for k in rp),
                before_last_positive=max(rp.values(),default=0),after_last_positive=max(rq.values(),default=0),before_keys=before,after_keys=after,
                **cost,total_capture_and_screen_seconds=elapsed))
            audits.append(dict(seed=seed,method=name,bank=kind,original_sequence_bitwise=True,credit_replay=detail,reference_sha256=entry['reference_sha256']))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,completed=len(audits)//len(PROFILES),strict_proofs=len(proofs))),flush=True)
    summary=[]
    for name,kind in PROFILES:
        rr=[r for r in rows if r['method']==name and r['bank']==kind]
        summary.append(dict(method=name,bank=kind,**{k:sum(r[k] for r in rr) for k in ['before_calls','after_calls','new_strict_rejections','before_k24_positive','after_k24_positive','advanced_positive_modes','total_positive_rank_advancement','coordinate_updates']},
            first_sweep_new=sum(r['sweep_positive_counts'].get(1,0) for r in rr),mean_normal_seconds=float(np.mean([r['seconds'] for r in rr])),
            max_main_array_bytes_subtotal=max(r['main_array_bytes_subtotal'] for r in rr),max_direction_bytes=max(r.get('direction_bytes',0) for r in rr),
            changed_k24_tasks=[r['seed'] for r in rr if r['after_k24_positive']>r['before_k24_positive']]))
    result=dict(source_hashes=len(hashes),original_sequence_replays=len(audits),strict_fraction_and_lp_proofs=len(proofs),summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
