"""Frozen support-only diagnosis of credit lost at the old positive gate."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import retained_credit_capture as capture
model=capture.model
PROFILES=['alm_local_k24','alm_residual_k24','alm_bp_c20_k24','alm_both_c20_k24','adam8_bp_k24','adam16_bp_k24','adam60_bp_k24','pc_residual_k24','nodual_residual_k24']


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def compare(left,right):
    assert np.array_equal(left[0],right[0]) and np.array_equal(left[1],right[1])
    assert left[2]['unbudgeted_pattern_keys']==right[2]['unbudgeted_pattern_keys'];records=[]
    for a,b in zip(left[3],right[3]):
        assert a.trajectory.hexdigest()==b.trajectory.hexdigest() and a.forward==b.forward and a.split==b.split
        assert list(a.pending)==list(b.pending)
        for key in a.pending:
            assert a.pending[key][0]==b.pending[key][0] and np.array_equal(a.pending[key][1],b.pending[key][1]) and a.pending[key][2]==b.pending[key][2]
        records.append(dict(kind=a.kind,trajectory_sha256=a.trajectory.hexdigest(),pending_entries=len(a.pending),forward_patterns=len(a.forward),split_patterns=len(a.split),retained_entries=len(b.retained)))
    return records


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);args=p.parse_args();root=args.project/'results'
    oldroot=root/'budgeted_credit_memory/development';runtime=root/'optimized_credit_memory/development';out=root/'retained_credit/diagnostic';refroot=root/'posterior_state_reuse/first_write_reference'
    parent=json.loads((runtime/'protocol.json').read_text());hashes=parent['source_sha256'].copy()
    for name,h in hashes.items():assert sha(Path(__file__).with_name(name))==h,name
    for name in [Path(__file__).name,Path(capture.__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    old=json.loads((oldroot/'protocol.json').read_text());cfgs={c['name']:c for c in old['configs']};seeds=list(range(old['seed0'],old['seed0']+old['count']))
    refs={r['seed']:r for r in json.loads((refroot/'coverage.json').read_text())};baseline={(r['seed'],r['method']):r['after_keys'] for r in json.loads((root/'optimized_branch_dual/diagnostic/rows.json').read_text()) if r['bank']=='native'}
    for row in json.loads((runtime/'episodes.json').read_text()):
        if row['repetition']==0 and row['n_context']==4 and row['method'] in ['pc_residual_opt_full','nodual_residual_opt_full']:
            key='pc_residual_k24' if row['method'].startswith('pc') else 'nodual_residual_k24';baseline[row['seed'],key]=row['post_normal_pattern_keys']
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,seeds=seeds,profiles=PROFILES,verification=capture.verify(),selection='one nonzero actual credit per (pattern,label), max old normalized rough, first tie; original accepted bank unchanged',
        baseline_protocol_sha256=sha(runtime/'protocol.json'),scope='old support-only diagnosis, no query information used in retention; new directions matching only')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];proofs=[];audits=[]
    for seed in seeds:
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24);x=xx[:4];v=(model.base.forward(xx,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24))[:4]
        pending=[]
        for name in PROFILES:
            before=capture.capture(x,v,cfgs[name],False);after=capture.capture(x,v,cfgs[name],True);audit=compare(before,after)
            _,regs,meta,collectors,_=after;assert [r.tobytes().hex() for r in regs]==baseline[seed,name]
            reject,pf,cost=capture.extra_matching(x,v,regs,collectors);pending.append((name,regs,reject,pf,cost,audit))
        item=refs[seed];path=refroot/item['reference_file'];assert sha(path)==item['reference_sha256'];ref=json.loads(path.read_text());positive={r['pattern'] for r in ref['positive_regions']}
        cr=root/'posterior_state_reuse/conditional_risk';ca=json.loads((cr/f'audit_{seed}.json').read_text());cp=cr/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as curve:patterns=curve['patterns'];weights=curve['weights'];means=curve['region_means'];q=curve['q'];full=np.einsum('k,rkq->rq',weights,means)
        for name,regs,reject,pf,cost,audit in pending:
            before=[r.tobytes().hex() for r in regs];after=[k for k,r in zip(before,reject) if not r];assert not {p['pattern'] for p in pf}&positive
            for proof in pf:
                reg=np.frombuffer(bytes.fromhex(proof['pattern']),np.uint8).reshape(4,4);exact=capture.normal.exact_optimum(x,v,reg,np.array(proof['a']));assert exact==proof['exact'] and exact['positive']
                _,_,g,rhs=model.old.old.previous.neighbor.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                proofs.append(dict(seed=seed,method=name,**proof,lp_status=int(lp.status)))
            rp={k:i+1 for i,k in enumerate(before) if k in positive};rq={k:i+1 for i,k in enumerate(after) if k in positive};assert set(rp)==set(rq)
            for k in range(len(before)+1):assert set(before[:k])&positive<=set(after[:k])&positive
            quality={}
            for label,seq in [('before',before),('retained',after)]:
                mask=np.array([k in set(seq[:24]) for k in patterns]);mass=float(weights[mask].sum());risk=None
                if mass:
                    ideal=np.einsum('k,rkq->rq',weights*mask/mass,means);risk=float(np.mean([np.trapezoid((ideal[a]-full[a])*(ideal[b]-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
                quality[label]=dict(mass=mass,ideal_truncation=risk,positive_modes=int(mask.sum()))
            rows.append(dict(seed=seed,method=name,before_calls=len(before),after_calls=len(after),new_strict_rejections=len(pf),nonpositive_old_rough_proofs=sum(p['old_rough']<=0 for p in pf),
                advanced_positive_modes=sum(rp[k]>rq[k] for k in rp),rank_advancement=sum(rp[k]-rq[k] for k in rp),before_keys=before,after_keys=after,k24=quality,**cost))
            audits.append(dict(seed=seed,method=name,bank_and_original_sequence_bitwise=True,collectors=audit,reference_sha256=item['reference_sha256']))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8');(out/'audits.json').write_text(json.dumps(audits,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=seed,completed=len(rows)//len(PROFILES),new_strict_proofs=len(proofs))),flush=True)
    summary=[]
    for name in PROFILES:
        rr=[r for r in rows if r['method']==name]
        summary.append(dict(method=name,**{k:sum(r[k] for r in rr) for k in ['before_calls','after_calls','new_strict_rejections','nonpositive_old_rough_proofs','advanced_positive_modes','rank_advancement','matching_direction_pairs','exact_checks','retained_entries','retention_updates']},
            mean_retention_seconds=float(np.mean([r['retention_seconds'] for r in rr])),mean_matching_seconds=float(np.mean([r['seconds'] for r in rr])),max_retained_numeric_bytes=max(r['retained_numeric_bytes'] for r in rr),
            k24={label:dict(mean_mass=float(np.mean([r['k24'][label]['mass'] for r in rr])),mean_ideal_truncation=None if any(r['k24'][label]['ideal_truncation'] is None for r in rr) else float(np.mean([r['k24'][label]['ideal_truncation'] for r in rr])),positive_modes=sum(r['k24'][label]['positive_modes'] for r in rr)) for label in ['before','retained']},
            changed_k24_tasks=[r['seed'] for r in rr if r['k24']['retained']['positive_modes']>r['k24']['before']['positive_modes']]))
    result=dict(source_hashes=len(hashes),trajectory_sequence_and_pending_replays=len(rows),exact_fraction_and_lp_proofs=len(proofs),summary=summary,scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
