"""Three predeclared forward-credit envelopes on every common old region."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import forward_credit_envelope as model
import exact_credit_hull as hull
from run_credit_wall_budget import load_inputs
from run_baseline_history import inherited_proof

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def summarize(records,alm_records):
    alm={(r['seed'],r['method']):r for r in alm_records};summaries=[];paired=[]
    for method in model.METHODS:
        rr=[r for r in records if r['method']==method];counts=Counter();raw=Counter();pairs=Counter();strict=[]
        for r in rr:
            counts.update(r['counts']);raw.update(r['raw_counts']);reference=alm[r['seed'],'alm_native']['status_by_pattern']
            for key,status in r['status_by_pattern'].items():
                old=reference[key];pairs[old+'__'+status]+=1
                if (old,status) in [('positive','nonpositive'),('nonpositive','positive')]:strict.append(dict(seed=r['seed'],pattern=key,alm=old,envelope=status))
        summaries.append(dict(method=method,counts=dict(counts),raw_counts=dict(raw),directions_min=min(r['directions'] for r in rr),directions_max=max(r['directions'] for r in rr),
            mean_bank_bytes=float(np.mean([r['bank_bytes'] for r in rr])),diagnostic_construction_seconds=sum(r['construction_seconds'] for r in rr),diagnostic_solver_seconds=sum(r['solver_seconds'] for r in rr)))
        paired.append(dict(method=method,status_pairs=dict(pairs),strict_separations=strict))
    return summaries,paired

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/forward_credit_envelope';out=base/'development';src=Path(__file__).parent
    prim=read(base/'primitive/summary.json');assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(src/name)==value,name
    hashes[Path(__file__).name]=sha(Path(__file__));inputs,input_hashes=load_inputs(root);prior=root/'results/exact_credit_hull/development/banks.json';alm_records=read(prior)
    design=root/'outputs/ttt-pc-alm-research/239_forward_credit_envelope_protocol.md';assert sha(design)==prim['design_sha256']
    p=dict(source_sha256=hashes,primitive_sha256=sha(base/'primitive/summary.json'),design_sha256=sha(design),input_hashes=input_hashes,prior_banks_sha256=sha(prior),
        prior_strict_sha256=sha(root/'results/baseline_history/analysis/retained_geometry.json'),seeds=sorted(inputs),methods=model.METHODS,total_regions=700,total_method_regions=2100,
        queries_used=False,eta=model.pack(model.ETA),threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},platform=platform.platform(),
        scope='Outer cone of all single-observation forward credits; positive is not proof of a realizable joint BP trajectory; real arithmetic')
    assert all(v=='1' for v in p['threads'].values());out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    records=[]
    for seed,(x,v,regs,_) in inputs.items():
        banks,metadata,details=model.build(x,v);bankfile=f'envelope_{seed}.npz';detailfile=f'intervals_{seed}.json';np.savez_compressed(out/bankfile,**banks)
        (out/detailfile).write_text(json.dumps(details,indent=2),encoding='utf-8');previous=None;previous_bank=None
        for method in model.METHODS:
            bank=banks[method];started=time.perf_counter();rb=hull.RationalBank(bank);rows=[];counts=Counter();raw=Counter();statuses={}
            for i,reg in enumerate(regs):
                r=hull.solve(x,v,reg,rb);status=r['status'];raw[status]+=1;inherited=None
                if previous is not None and previous[i]['effective_status']=='positive':
                    assert status!='nonpositive',(seed,method,i)
                    if status=='unknown':
                        prev=previous[i];old=prev['result'] if prev['result']['status']=='positive' else dict(prev['inherited'],status='positive')
                        inherited=inherited_proof(x,v,reg,bank,previous_bank,old);status='positive'
                key=reg.tobytes().hex();rows.append(dict(index=i,pattern=key,result=r,inherited=inherited,effective_status=status));counts[status]+=1;statuses[key]=status
            seconds=time.perf_counter()-started;filename=f'bank_{seed}_{method}.json';(out/filename).write_text(json.dumps(rows,indent=2),encoding='utf-8')
            rec=dict(seed=seed,method=method,**metadata[method],solver_seconds=seconds,counts=dict(counts),raw_counts=dict(raw),status_by_pattern=statuses,regions=len(regs),
                bank_file=bankfile,bank_file_sha256=sha(out/bankfile),bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),intervals_file=detailfile,intervals_sha256=sha(out/detailfile),rows_file=filename,rows_sha256=sha(out/filename))
            records.append(rec);previous=rows;previous_bank=bank;print(json.dumps({k:v for k,v in rec.items() if k not in ['status_by_pattern','bank_sha256','bank_file_sha256','intervals_sha256','rows_sha256']}),flush=True)
        (out/'banks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summaries,paired=summarize(records,alm_records);(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,banks=len(records),regions=700,total_method_regions=2100,summaries=summaries,
        comparisons=[{k:v for k,v in r.items() if k!='strict_separations'} for r in paired],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(out/name) for name in ['protocol.json','banks.json','paired.json']},scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
