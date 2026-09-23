"""Frozen 128-step diagnostic over all original unproved regions.

Does not open joint-LP output values, weights, or success flags.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import finite_credit_game as model
from online_endpoint_h2 import EndpointIntervalBank
from verify_finite_credit_game import guarded_solve


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True); root=ap.parse_args().project.resolve()
    primitive=root/'results/finite_credit_game/primitive/summary.json'; prim=json.loads(primitive.read_text()); assert prim['passed']
    hashes=dict(prim['source_sha256'])
    for name,value in hashes.items(): assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    inp=root/'results/light_h2_credit/full_bank_ceiling'; records=json.loads((inp/'records.json').read_text())
    design=root/'outputs/ttt-pc-alm-research/222_finite_credit_game_design.md'
    p=dict(source_sha256=hashes,primitive_sha256=sha(primitive),records_sha256=sha(inp/'records.json'),design_sha256=sha(design),
        steps=model.STEPS,prefixes=model.PREFIXES,initialization='uniform',eta='sqrt(2*log(K)/128)',payoff_bound='d*n',
        methods=sorted({r['method'] for r in records}),seeds=sorted({r['seed'] for r in records}),
        input_policy='All original unproved regions, no joint-LP outputs opened by this runner',
        stopping='First positive original Fraction credit proof; other regions all 128 steps',
        threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Old development pools; diagnostic not unseen-query improvement or official TTT reproduction')
    assert all(value=='1' for value in p['threads'].values())
    out=root/'results/finite_credit_game/development'; out.mkdir(parents=True,exist_ok=True); assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    banks=[]; counts=dict(banks=0,regions=0,old_proofs=0,attempted=0,new_proofs=0,oracle_pairs=0,exact_calls=0)
    for rec in records:
        source=inp/rec['data_file']; assert sha(source)==rec['data_sha256']
        with np.load(source) as z: x=z['x']; v=z['v']; regs=z['regs']; bank=z['bank']
        start=time.perf_counter(); old=EndpointIntervalBank(x,v,bank); mask,proofs,_=old.screen(regs)
        old_seconds=time.perf_counter()-start
        assert [(q['pattern'],q['direction']) for q in proofs]==[(q['pattern'],q['direction']) for q in rec['proofs']]
        indices=np.flatnonzero(~mask); arrays,meta=guarded_solve(x,v,regs[indices],bank)
        arrays.update(indices=indices,old_positive=mask)
        stem=f'bank_{rec["seed"]}_{rec["method"]}'; apath=out/(stem+'.npz'); mpath=out/(stem+'.json')
        np.savez_compressed(apath,**arrays); mpath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        row=dict(seed=rec['seed'],method=rec['method'],regions=len(regs),directions=len(bank),old_proofs=int(mask.sum()),
            attempted=len(indices),new_proofs=meta['positive'],oracle_pairs=meta['oracle_pairs'],exact_calls=meta['exact_calls'],
            total_seconds=meta['total_seconds'],old_screen_seconds=old_seconds,oracle_seconds=meta['oracle_seconds'],
            update_seconds=meta['update_seconds'],exact_seconds=meta['exact_seconds'],
            named_array_bytes_subtotal_max=meta['named_array_bytes_subtotal_max'],output_array_bytes=meta['output_array_bytes'],
            prefix_proofs={str(q['step']):q['positive'] for q in meta['traces']},
            source_data_file=rec['data_file'],source_data_sha256=rec['data_sha256'],arrays_file=apath.name,arrays_sha256=sha(apath),
            meta_file=mpath.name,meta_sha256=sha(mpath))
        banks.append(row); counts['banks']+=1
        for key in ['regions','old_proofs','attempted','new_proofs','oracle_pairs','exact_calls']: counts[key]+=row[key]
        (out/'banks.json').write_text(json.dumps(banks,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=rec['seed'],method=rec['method'],this_new=row['new_proofs'],**counts)),flush=True)
    summaries=[]
    for method in p['methods']:
        rr=[r for r in banks if r['method']==method]
        summaries.append(dict(method=method,**{key:sum(r[key] for r in rr) for key in ['regions','old_proofs','attempted','new_proofs','oracle_pairs','exact_calls']},
            prefix_proofs={str(step):sum(r['prefix_proofs'][str(step)] for r in rr) for step in model.PREFIXES},
            mean_seconds=float(np.mean([r['total_seconds'] for r in rr])),
            mean_exact_seconds=float(np.mean([r['exact_seconds'] for r in rr])),
            max_named_array_bytes_subtotal=max(r['named_array_bytes_subtotal_max'] for r in rr)))
    result=dict(execution_complete=True,counts=counts,summaries=summaries,protocol_sha256=sha(out/'protocol.json'),banks_sha256=sha(out/'banks.json'),
                source_sha256=sha(Path(__file__)),scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
