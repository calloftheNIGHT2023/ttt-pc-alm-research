"""Fixed 32-direction cross-mode cut banks; support-only diagnostic."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import split_activity_mode_memory as model
import local_region_screen as screen
from certified_branch_solver import exact_certificate


def bank_from_proofs(records,limit=32):
    bank=[];seen=set()
    for r in sorted(records,key=lambda r:(r['pattern'],r.get('label',''))):
        a=np.array(r['a']);scale=float(np.max(np.abs(a)))
        if scale==0:continue
        a=a/scale;key=a.tobytes()
        if key in seen:continue
        bank.append(a);seen.add(key)
        if len(bank)==limit:break
    return np.array(bank).reshape(-1,4,4)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True);a=parser.parse_args()
    root=a.project/'results';out=root/'cross_mode_credit_bank/diagnostic';refroot=root/'posterior_state_reuse'
    inherited=json.loads((root/'bp_credit_control/diagnostic/protocol.json').read_text());hashes=inherited['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    hashes[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    paths=dict(local=root/'credit_cached_memory/certificate_audit/proofs.json',bp=root/'bp_credit_control/diagnostic/proofs.json')
    evidence={name:json.loads(path.read_text()) for name,path in paths.items()}
    tasks=[dict(name='alm16',generator='alm',sweeps=16),dict(name='adam8',generator='adam',steps=8),dict(name='adam16',generator='adam',steps=16)]
    protocol=dict(seeds=list(range(5900000,5900016)),tasks=tasks,max_directions=32,source_sha256=hashes,
        proof_file_sha256={name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in paths.items()},
        direction_rule='sort accepted proof by (pattern,label); normalize by max abs; deduplicate exact direction bytes; first 32',
        random_rule='32 standard normal directions with rng17731, normalized; shared across contexts, independent of observations',
        predeclared='contract5 first, then max-rough direction per remaining mode, outward validation, exact Fraction and LP audit',
        scope='old support-only fixed-bank diagnostic; collection and certification costs not yet an online speed result')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    random=np.random.default_rng(17731).normal(size=(32,4,4));random/=np.max(np.abs(random),axis=(1,2))[:,None,None]
    refs={r['seed']:r for r in json.loads((refroot/'first_write_reference/coverage.json').read_text())};rows=[];proofs=[]
    for seed in protocol['seeds']:
        path=refroot/'first_write_reference'/refs[seed]['reference_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==refs[seed]['reference_sha256']
        data=json.loads(path.read_text());x=np.array(data['x']);v=np.array(data['v']);positive={r['pattern'] for r in data['positive_regions']}
        for task in tasks:
            _,regs,_=model.discover(x,v,dict(**task,restarts=64));keys=[r.tobytes().hex() for r in regs]
            rejected=screen.contract(x,v,regs,5);contracted={k for k,reject in zip(keys,rejected) if reject}
            assert not positive&contracted
            local=[r for r in evidence['local'] if r['seed']==seed and r['method']=='alm16_all']
            residual=[r for r in evidence['local'] if r['seed']==seed and r['method']=='alm16_residual']
            bp=[r for r in evidence['bp'] if r['seed']==seed and r['method']==task['name']]
            native={r['pattern'] for r in (local if task['generator']=='alm' else bp)}
            assert native<=set(keys)
            banks={'bp32':bank_from_proofs(bp),'random32':random}
            if task['generator']=='alm':banks={'local32':bank_from_proofs(local),'residual32':bank_from_proofs(residual),**banks}
            remaining=np.flatnonzero(~rejected)
            for label,bank in banks.items():
                cross=set();matches=[];tested=len(remaining)*len(bank);temporary=0
                if len(bank) and len(remaining):
                    rr=np.repeat(regs[remaining],len(bank),axis=0);aa=np.tile(bank,(len(remaining),1,1));pp=model.base.SLOPES[rr]*aa
                    rough=screen.float_bound(x,v,rr,pp,aa).reshape(len(remaining),len(bank));scale=(1+np.abs(pp).sum((1,2))+np.abs(aa).sum((1,2))).reshape(rough.shape)
                    selected=np.argmax(rough,axis=1);score=rough[np.arange(len(remaining)),selected];ss=scale[np.arange(len(remaining)),selected]
                    ids=np.flatnonzero(score>1e-10*ss);flat=ids*len(bank)+selected[ids]
                    lower=screen.certified_lower_bound(x,v,rr[flat],pp[flat],aa[flat]) if len(flat) else []
                    temporary=rr.nbytes+aa.nbytes+pp.nbytes+rough.nbytes+scale.nbytes
                    for ii,index,lb in zip(ids,flat,lower):
                        if lb<=0:continue
                        key=keys[remaining[ii]];assert key not in positive
                        with model.old.core.pipeline.discovery_box(.12):exact=exact_certificate(x,v,rr[index],pp[index],aa[index])
                        val=Fraction(int(exact['numerator']),int(exact['denominator']));assert val>0 and Fraction(float(lb))<=val
                        _,_,g,rhs=model.neighbor.pattern_matrix(x,v,rr[index]);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                        cross.add(key);proofs.append(dict(seed=seed,task=task['name'],bank=label,pattern=key,direction=int(selected[ii]),
                            a=aa[index].tolist(),p=pp[index].tolist(),lower=float(lb),exact=exact,lp_status=int(lp.status)))
                rows.append(dict(seed=seed,task=task['name'],bank=label,directions=len(bank),direction_array_bytes=bank.nbytes,
                    patterns=len(keys),contract5_rejects=len(contracted),native_cache_extra=len(native-contracted),
                    cross_certified_after_contract5=len(cross),new_beyond_native_cache=len(cross-native),
                    native_plus_cross_extra=len((native|cross)-contracted),geometry_calls_remaining=len(set(keys)-contracted-native-cross),
                    evaluated_mode_direction_pairs=tested,main_batch_array_bytes_subtotal=temporary))
        (out/'audits.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,rows=len(rows),proofs=len(proofs))),flush=True)
    summary=[]
    for task,label in sorted({(r['task'],r['bank']) for r in rows}):
        rr=[r for r in rows if r['task']==task and r['bank']==label]
        summary.append(dict(task=task,bank=label,mean_directions=float(np.mean([r['directions'] for r in rr])),
            **{k:sum(r[k] for r in rr) for k in ['patterns','contract5_rejects','native_cache_extra','cross_certified_after_contract5','new_beyond_native_cache','native_plus_cross_extra','geometry_calls_remaining']},
            max_direction_array_bytes=max(r['direction_array_bytes'] for r in rr),max_main_batch_array_bytes_subtotal=max(r['main_batch_array_bytes_subtotal'] for r in rr)))
    result=dict(summary=summary,exact_rational_and_lp_proofs=len(proofs),source_hashes=len(hashes),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
