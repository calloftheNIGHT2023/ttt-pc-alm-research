"""Observed-support-only credit mixing, followed by independent proof audits."""
import argparse, hashlib, json, time
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import budgeted_credit_memory as model
import credit_segment_mixing as mix
from certified_branch_solver import exact_certificate

PROFILES = [
    ('alm_local_k24', 'native'), ('alm_residual_k24', 'native'),
    ('alm_bp_c20_k24', 'native'), ('alm_both_c20_k24', 'native'),
    ('adam8_bp_k24', 'native'), ('adam16_bp_k24', 'native'), ('adam60_bp_k24', 'native'),
    ('alm_none_k24', 'random32'),
]


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def capture(x,v,cfg):
    original=model.old.Collector; collectors=[]
    def factory(x,v,kind):
        instance=original(x,v,kind,capture=True); collectors.append(instance); return instance
    try:
        model.old.Collector=factory
        bank,regs,meta=model.prepare(x,v,cfg)
    finally: model.old.Collector=original
    directions=[]; seen=set(); artifacts=[]
    for collector in collectors:
        accepted,_=collector.finalize(); keys=[k.hex() for k in accepted]
        assert keys==meta['credit_details'][collector.kind]['matching_keys']
        assert [r[0] for r in accepted.values()]==meta['credit_details'][collector.kind]['matching_bounds']
        dd=model.old.direction_bank(accepted,32)
        for direction in dd:
            key=direction.tobytes()
            if key not in seen: seen.add(key); directions.append(direction)
        artifacts.append(dict(kind=collector.kind, matching_keys=keys, directions=len(dd)))
    return bank,regs,meta,np.array(directions).reshape(-1,4,len(x)),artifacts


def main():
    p=argparse.ArgumentParser(); p.add_argument('--project',type=Path,required=True); args=p.parse_args()
    root=args.project/'results'; online=root/'budgeted_credit_memory/development'; refroot=root/'posterior_state_reuse/first_write_reference'
    out=root/'credit_segment_mixing/diagnostic'; out.mkdir(parents=True,exist_ok=True)
    old=json.loads((online/'protocol.json').read_text()); hashes=old['source_sha256'].copy()
    for name,h in hashes.items(): assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(mix.__file__).name]: hashes[name]=sha(Path(__file__).with_name(name))
    assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,profiles=PROFILES,seeds=list(range(old['seed0'],old['seed0']+old['count'])),
        online_protocol_sha256=sha(online/'protocol.json'),verification=mix.verify(),random_seed=771931,
        candidate_rule='best endpoint to every supplied direction; exact piecewise-linear segment candidates followed by mandatory outward validation',
        scope='old support diagnostic, not online performance; random32 and BP get identical mixing; combined channel can have up to 64 directions and must be accounted')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    cfgs={c['name']:c for c in old['configs']}; online_rows={(r['seed'],r['method']):r for r in json.loads((online/'episodes.json').read_text()) if r['repetition']==0 and r['n_context']==4}
    references={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())}; rows=[]; allproof=[]; audit=[]
    random=np.random.default_rng(protocol['random_seed']).normal(size=(32,4,4)); random/=np.max(np.abs(random),axis=(1,2))[:,None,None]
    for seed in protocol['seeds']:
        rng=np.random.default_rng(seed); truth=rng.uniform(-.12,.12,4); xx=rng.uniform(0,1,24); x=xx[:4]
        v=model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24); v=v[:4]
        pending=[]
        # Reference targets/cells are not loaded until every method completes.
        for name,kind in PROFILES:
            begin=time.perf_counter(); _,regs,meta,bank,detail=capture(x,v,cfgs[name])
            assert meta['unbudgeted_pattern_keys']==online_rows[seed,name]['unbudgeted_pattern_keys']
            if kind=='random32': bank=random
            rejected,proofs,cost=mix.screen_bank(x,v,regs,bank)
            pending.append((name,kind,regs,rejected,proofs,cost,detail,time.perf_counter()-begin))
        item=references[seed]; path=refroot/item['reference_file']; assert sha(path)==item['reference_sha256']; ref=json.loads(path.read_text())
        assert np.array_equal(x,ref['x']) and np.array_equal(v,ref['v']); positive={r['pattern'] for r in ref['positive_regions']}
        for name,kind,regs,rejected,proofs,cost,detail,elapsed in pending:
            before=[r.tobytes().hex() for r in regs]; after=[key for key,skip in zip(before,rejected) if not skip]
            assert not {r['pattern'] for r in proofs}&positive
            for proof in proofs:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4); pp=np.array(proof['p']); aa=np.array(proof['a'])
                with model.old.old.core.pipeline.discovery_box(.12): exact=exact_certificate(x,v,reg,pp,aa)
                value=Fraction(int(exact['numerator']),int(exact['denominator'])); assert value>0 and Fraction(proof['lower'])<=value
                _,_,g,rhs=model.old.previous.neighbor.pattern_matrix(x,v,reg)
                lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4)
                assert lp.status==2
                allproof.append(dict(seed=seed,method=name,bank=kind,**proof,exact=exact,lp_status=int(lp.status)))
            rp={key:i+1 for i,key in enumerate(before) if key in positive}; rq={key:i+1 for i,key in enumerate(after) if key in positive}; assert set(rp)==set(rq)
            rankdelta={key:rp[key]-rq[key] for key in rp}; assert all(v>=0 for v in rankdelta.values())
            for k in range(len(before)+1): assert set(before[:k])&positive <= set(after[:k])&positive
            rows.append(dict(seed=seed,method=name,bank=kind,before_calls=len(before),after_calls=len(after),new_strict_rejections=len(proofs),
                before_k24_positive=len(set(before[:24])&positive),after_k24_positive=len(set(after[:24])&positive),
                advanced_positive_modes=sum(v>0 for v in rankdelta.values()),total_positive_rank_advancement=sum(rankdelta.values()),
                before_last_positive=max(rp.values(),default=0),after_last_positive=max(rq.values(),default=0),
                mixed_strictly_nonpositive_endpoints=sum(r['best_endpoint_rough_bound']<=0 for r in proofs),
                mixed_strict_interior=sum(0<r['t']<1 for r in proofs),before_keys=before,after_keys=after,**cost,total_capture_and_screen_seconds=elapsed))
            audit.append(dict(seed=seed,method=name,bank=kind,original_sequence_bitwise=True,credit_replays=detail,reference_sha256=item['reference_sha256']))
        (out/'audits.json').write_text(json.dumps(audit,indent=2),encoding='utf-8'); (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        (out/'proofs.json').write_text(json.dumps(allproof,indent=2),encoding='utf-8'); print(json.dumps(dict(seed=seed,completed=len(audit)//len(PROFILES),rows=len(rows),strict_proofs=len(allproof))),flush=True)
    summary=[]
    for name,kind in PROFILES:
        rr=[r for r in rows if r['method']==name and r['bank']==kind]
        summary.append(dict(method=name,bank=kind,**{k:sum(r[k] for r in rr) for k in ['before_calls','after_calls','new_strict_rejections','before_k24_positive','after_k24_positive','advanced_positive_modes','total_positive_rank_advancement','mixed_strictly_nonpositive_endpoints','mixed_strict_interior','line_evaluations']},
            mean_mixing_seconds=float(np.mean([r['seconds'] for r in rr])),mean_capture_and_screen_seconds=float(np.mean([r['total_capture_and_screen_seconds'] for r in rr])),
            max_direction_bytes=max(r.get('direction_bytes',0) for r in rr),max_coordinate_bytes=max(r['coordinate_bytes'] for r in rr),changed_k24_tasks=[r['seed'] for r in rr if r['after_k24_positive']>r['before_k24_positive']]))
    result=dict(source_hashes=len(hashes),original_sequence_replays=len(audit),strict_fraction_and_lp_proofs=len(allproof),summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
