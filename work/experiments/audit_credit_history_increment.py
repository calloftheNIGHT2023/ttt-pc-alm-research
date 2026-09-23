"""Same-trajectory ablation: historical multiplier versus current residual only."""
import argparse,hashlib,json
from pathlib import Path
from fractions import Fraction
import numpy as np
from scipy.optimize import linprog
import audit_reused_local_credit as prior


class ResidualObserver(prior.Observer):
    def activity(self,b,h,u,step,phase):
        # This does not change the actual solver's u or any update.
        super().activity(b,h,np.zeros_like(u),step,phase)


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    root=a.project/'results/reused_local_credit';inp=root/'diagnostic';out=root/'history_increment'
    protocol=json.loads((inp/'protocol.json').read_text());original=json.loads((inp/'proofs.json').read_text());oldrows=json.loads((inp/'audits.json').read_text())
    hashes=protocol['source_sha256'].copy()
    for name,h in hashes.items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    hashes[Path(__file__).name]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    newprotocol=dict(seeds=protocol['seeds'],configs=protocol['configs'],source_sha256=hashes,
        original_proofs_sha256=hashlib.sha256((inp/'proofs.json').read_bytes()).hexdigest(),
        hypothesis='on IDENTICAL trajectories, does historical u add certified rejected cells beyond a=r?',
        predeclared='compare residual-only, raw u, u+r and their union; all certificates exact + LP audited; same contractor 5/20 controls')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(newprotocol,indent=2),encoding='utf-8');rows=[];proofs=[]
    for seed in protocol['seeds']:
        path=a.project/f'results/posterior_state_reuse/first_write_reference/reference_{seed}.json';data=json.loads(path.read_text())
        x=np.array(data['x']);v=np.array(data['v']);anchor=np.zeros(4)
        pool=prior.trace.old.interface.make_pool(4,'prior256');starts,_=prior.trace.old.interface.select_pool(x,v,anchor,pool,64)
        for cfg in protocol['configs']:
            obs=ResidualObserver(x,v)
            with prior.trace.old.core.pipeline.discovery_box(.12):
                actual=prior.trace.refine(starts,x,v,anchor,cfg['method'],cfg['sweeps'],obs)
                expected=prior.trace.original(starts,x,v,anchor,cfg['method'],cfg['sweeps'])
            assert np.array_equal(actual,expected)
            keys=sorted(obs.keys);regs=np.array([np.frombuffer(k,np.uint8).reshape(4,4) for k in keys]);indices={k.hex():i for i,k in enumerate(keys)}
            c5=prior.screen.contract(x,v,regs,5);c20=prior.screen.contract(x,v,regs,20)
            raw={r['pattern'] for r in original if r['seed']==seed and r['method']==cfg['name'] and r['credit']=='raw_dual'}
            augmented={r['pattern'] for r in original if r['seed']==seed and r['method']==cfg['name'] and r['credit']=='augmented_credit'}
            residual=set()
            for (label,key),cut in obs.cuts.items():
                assert label=='augmented_credit';reg=np.frombuffer(key,np.uint8).reshape(4,4)
                with prior.trace.old.core.pipeline.discovery_box(.12):exact=prior.exact_certificate(x,v,reg,cut['p'],cut['a'])
                val=Fraction(int(exact['numerator']),int(exact['denominator']));assert val>0 and Fraction(cut['lower'])<=val
                _,_,g,rhs=prior.model.pattern_matrix(x,v,reg);lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2
                residual.add(key.hex());proofs.append(dict(seed=seed,method=cfg['name'],pattern=key.hex(),lower=cut['lower'],exact=exact,
                    p=cut['p'].tolist(),a=cut['a'].tolist(),lp_status=int(lp.status)))
            history=(raw|augmented)-residual
            rows.append(dict(seed=seed,method=cfg['name'],original_path_bitwise=True,residual_certificates=len(residual),raw_certificates=len(raw),
                augmented_certificates=len(augmented),history_only_certificates=len(history),
                history_only_beyond_contract5=sum(not c5[indices[k]] for k in history),
                history_only_beyond_contract20=sum(not c20[indices[k]] for k in history),
                residual_beyond_contract5=sum(not c5[indices[k]] for k in residual),
                residual_beyond_contract20=sum(not c20[indices[k]] for k in residual)))
            if cfg['method']!='alm':assert residual==augmented and not raw
        (out/'audits.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
        print(json.dumps(dict(completed=seed-protocol['seeds'][0]+1,total=16,proofs=len(proofs))),flush=True)
    summary=[]
    for cfg in protocol['configs']:
        rr=[r for r in rows if r['method']==cfg['name']]
        summary.append(dict(method=cfg['name'],**{k:sum(r[k] for r in rr) for k in rr[0] if k not in ['seed','method','original_path_bitwise']}))
    result=dict(summary=summary,exact_proofs=len(proofs),original_paths_exact=len(rows),source_hashes=len(hashes))
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
