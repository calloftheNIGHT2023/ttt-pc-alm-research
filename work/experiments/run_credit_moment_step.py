"""Frozen one-moment-step diagnostic; no oracle LP outputs read."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import credit_moment_step as model
from online_endpoint_h2 import EndpointIntervalBank
from verify_credit_moment_step import guarded_solve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    primitive=root/'results/credit_moment_step/primitive/summary.json';prim=json.loads(primitive.read_text());assert prim['passed']
    hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    inp=root/'results/light_h2_credit/full_bank_ceiling';records=json.loads((inp/'records.json').read_text())
    design=root/'outputs/ttt-pc-alm-research/226_credit_moment_step_design.md'
    p=dict(source_sha256=hashes,primitive_sha256=sha(primitive),records_sha256=sha(inp/'records.json'),design_sha256=sha(design),
        steps=model.STEPS,prefixes=model.PREFIXES,initialization='uniform',delta=model.DELTA,inner_steps=1,
        normalized_step='logaddexp(0, log(target-mean)-log(centered_variance))',
        methods=sorted({r['method'] for r in records}),seeds=sorted({r['seed'] for r in records}),
        input_policy='All original unproved regions; no joint-LP outputs read by this runner',
        stopping='Fraction positive or explicitly unproved numerical/target-limited terminal; otherwise 128 responses',
        threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Old development banks, not independent task or online cost advantage')
    assert all(value=='1' for value in p['threads'].values())
    out=root/'results/credit_moment_step/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    banks=[];keys=['regions','old_proofs','attempted','new_proofs','oracle_pairs','mean_evaluations','variance_evaluations','exact_calls',
        'target_limited','zero_variance','nonfinite_failures','already_at_target']
    counts={key:0 for key in keys};counts['banks']=0
    for rec in records:
        path=inp/rec['data_file'];assert sha(path)==rec['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        start=time.perf_counter();old=EndpointIntervalBank(x,v,bank);mask,proofs,_=old.screen(regs);old_seconds=time.perf_counter()-start
        assert [(q['pattern'],q['direction']) for q in proofs]==[(q['pattern'],q['direction']) for q in rec['proofs']]
        indices=np.flatnonzero(~mask);arrays,meta=guarded_solve(x,v,regs[indices],bank);arrays.update(indices=indices,old_positive=mask)
        stem=f'bank_{rec["seed"]}_{rec["method"]}';apath=out/(stem+'.npz');mpath=out/(stem+'.json')
        np.savez_compressed(apath,**arrays);mpath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
        row=dict(seed=rec['seed'],method=rec['method'],regions=len(regs),directions=len(bank),old_proofs=int(mask.sum()),attempted=len(indices),
            new_proofs=meta['positive'],old_screen_seconds=old_seconds,
            **{key:meta[key] for key in ['oracle_pairs','mean_evaluations','variance_evaluations','exact_calls','target_limited','zero_variance',
                'nonfinite_failures','already_at_target','total_seconds','oracle_seconds','update_seconds','exact_seconds',
                'named_array_bytes_subtotal_max','output_array_bytes']},
            prefix_proofs={str(q['step']):q['positive'] for q in meta['traces']},source_data_file=rec['data_file'],source_data_sha256=rec['data_sha256'],
            arrays_file=apath.name,arrays_sha256=sha(apath),meta_file=mpath.name,meta_sha256=sha(mpath))
        banks.append(row);counts['banks']+=1
        for key in keys:counts[key]+=row[key]
        (out/'banks.json').write_text(json.dumps(banks,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=rec['seed'],method=rec['method'],this_new=row['new_proofs'],**counts)),flush=True)
    summaries=[]
    for method in p['methods']:
        rr=[r for r in banks if r['method']==method]
        summaries.append(dict(method=method,**{key:sum(r[key] for r in rr) for key in keys},
            prefix_proofs={str(step):sum(r['prefix_proofs'][str(step)] for r in rr) for step in model.PREFIXES},
            **{'mean_'+key:float(np.mean([r[key] for r in rr])) for key in ['total_seconds','oracle_seconds','update_seconds','exact_seconds']},
            max_named_array_bytes_subtotal=max(r['named_array_bytes_subtotal_max'] for r in rr)))
    result=dict(execution_complete=True,counts=counts,summaries=summaries,protocol_sha256=sha(out/'protocol.json'),banks_sha256=sha(out/'banks.json'),
        source_sha256=sha(Path(__file__)),scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
