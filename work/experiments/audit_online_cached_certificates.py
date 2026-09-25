"""Reconstruct every online accepted cut and verify exact rational / LP evidence."""
import argparse,hashlib,json
from fractions import Fraction
from pathlib import Path
import numpy as np
from scipy.optimize import linprog
import credit_cached_mode_memory as model
from certified_branch_solver import exact_certificate


def main():
    p=argparse.ArgumentParser();p.add_argument('--project',type=Path,required=True);a=p.parse_args()
    root=a.project/'results/credit_cached_memory';inp=root/'development';out=root/'certificate_audit'
    protocol=json.loads((inp/'protocol.json').read_text());rows=json.loads((inp/'episodes.json').read_text());lookup={(r['seed'],r['method']):r for r in rows if r['repetition']==0 and r['n_context']==4}
    for name,h in protocol['source_sha256'].items():assert hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()==h,name
    proofs=[];matches=[]
    for seed in range(protocol['seed0'],protocol['seed0']+protocol['count']):
        rng=np.random.default_rng(seed);truth=rng.uniform(-.12,.12,4);x=rng.uniform(0,1,24)
        v=model.base.forward(x,truth)+np.random.default_rng(seed+19000000).uniform(-model.base.EPS,model.base.EPS,24)
        for cfg in protocol['configs']:
            if cfg['credit_mode']=='none':continue
            bank,regs,meta,collector=model.discover(x[:4],v[:4],dict(**cfg,capture_proofs=True));row=lookup[seed,cfg['name']]
            assert meta['certified_mode_keys']==row['certified_mode_keys'] and meta['certificate_lower_bounds']==row['certificate_lower_bounds']
            positive=set(row['positive_mode_keys'])
            for key,proof in collector.proofs.items():
                reg=np.frombuffer(key,np.uint8).reshape(4,4)
                with model.old.core.pipeline.discovery_box(.12):exact=exact_certificate(x[:4],v[:4],reg,proof['p'],proof['a'])
                value=Fraction(int(exact['numerator']),int(exact['denominator']));assert value>0 and Fraction(proof['lower'])<=value
                _,_,g,rhs=model.previous.neighbor.pattern_matrix(x[:4],v[:4],reg)
                lp=linprog(np.zeros(4),A_ub=g,b_ub=rhs,bounds=[(-.12,.12)]*4);assert lp.status==2 and key.hex() not in positive
                proofs.append(dict(seed=seed,method=cfg['name'],pattern=key.hex(),p=proof['p'].tolist(),a=proof['a'].tolist(),
                    lower=proof['lower'],exact=exact,label=proof['label'],lp_status=int(lp.status)))
            matches.append(dict(seed=seed,method=cfg['name'],original_cut_keys_and_bounds_exact=True,proofs=len(collector.proofs)))
        print(json.dumps(dict(completed=seed-protocol['seed0']+1,total=protocol['count'],proofs=len(proofs))),flush=True)
    out.mkdir(parents=True,exist_ok=True)
    result=dict(exact_replayed_first_writes=len(matches),exact_rational_and_lp_checks=len(proofs),matches=matches,
        source_sha256=protocol['source_sha256'],audit_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope='same accepted online certificates reconstructed with evidence capture; instrumentation is not timed performance')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');(out/'proofs.json').write_text(json.dumps(proofs,indent=2),encoding='utf-8')
    print(json.dumps(dict(complete=True,first_writes=len(matches),proofs=len(proofs))))


if __name__=='__main__':main()
