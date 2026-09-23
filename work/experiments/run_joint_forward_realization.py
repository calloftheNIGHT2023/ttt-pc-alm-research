"""All common regions versus actual shared-parameter probe credits."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
import joint_forward_realization as model
import exact_credit_hull as hull
from run_credit_wall_budget import load_inputs
from run_baseline_history import inherited_proof

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())

def summarize(records,alm_records):
    reference={r['seed']:r['status_by_pattern'] for r in alm_records if r['method']=='alm_native'};summary=[];paired=[]
    for method in model.METHODS:
        rr=[r for r in records if r['method']==method];counts=Counter();raw=Counter();pairs=Counter();strict=[]
        for rec in rr:
            counts.update(rec['counts']);raw.update(rec['raw_counts']);seed=rec['seed']
            for key,status in rec['status_by_pattern'].items():
                alm=reference[seed][key];pairs[alm+'__'+status]+=1
                if (alm,status) in [('positive','nonpositive'),('nonpositive','positive')]:strict.append(dict(seed=seed,pattern=key,alm=alm,comparator=status))
        summary.append(dict(method=method,counts=dict(counts),raw_counts=dict(raw),directions_min=min(r['directions'] for r in rr),directions_max=max(r['directions'] for r in rr),
            mean_bank_bytes=float(np.mean([r['bank_bytes'] for r in rr])),diagnostic_solver_seconds=sum(r['solver_seconds'] for r in rr)))
        paired.append(dict(method=method,status_pairs=dict(pairs),strict_separations=strict))
    return summary,paired

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/joint_forward_realization';out=base/'development';src=Path(__file__).parent
    prim=read(base/'primitive/summary.json');assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(src/name)==value,name
    hashes[Path(__file__).name]=sha(Path(__file__));inputs,input_hashes=load_inputs(root);hist=root/'results/baseline_history/development';captures=read(hist/'captures.json')
    old_manifest=read(hist/'banks.json');alm_path=root/'results/exact_credit_hull/development/banks.json';alm=read(alm_path)
    design=root/'outputs/ttt-pc-alm-research/241_joint_forward_realization_protocol.md';assert sha(design)==prim['design_sha256']
    p=dict(source_sha256=hashes,primitive_sha256=sha(base/'primitive/summary.json'),design_sha256=sha(design),input_hashes=input_hashes,
        full_history_input_sha256={name:sha(hist/name) for name in ['captures.json','banks.json','summary.json']},alm_banks_sha256=sha(alm_path),
        prior_strict_sha256=sha(root/'results/baseline_history/analysis/retained_geometry.json'),seeds=sorted(inputs),methods=model.METHODS,total_regions=700,total_method_regions=1400,
        queries_used=False,threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Actual joint-parameter forward probe credits, not an online adapter or universal forward-credit capacity')
    assert all(v=='1' for v in p['threads'].values());out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    generations=[];records=[]
    for seed,(x,v,regs,_) in inputs.items():
        arrays,generation,meta=model.generate(x,v);afile=f'probes_{seed}.npz';gfile=f'generation_{seed}.json';np.savez_compressed(out/afile,**arrays);(out/gfile).write_text(json.dumps(generation,indent=2),encoding='utf-8')
        cap=next(r for r in captures if r['seed']==seed and r['learner']=='adam60');assert sha(hist/cap['arrays_file'])==cap['arrays_sha256']
        with np.load(hist/cap['arrays_file']) as z:old_bank=z['bank']
        oldrec=next(r for r in old_manifest if r['seed']==seed and r['method']=='adam60_history_full');assert sha(hist/oldrec['rows_file'])==oldrec['rows_sha256'];oldrows=read(hist/oldrec['rows_file'])
        gen=dict(seed=seed,metadata=meta,arrays_file=afile,arrays_sha256=sha(out/afile),generation_file=gfile,generation_sha256=sha(out/gfile),old_bank_source=cap,old_rows=oldrec)
        generations.append(gen);print(json.dumps(dict(seed=seed,**meta)),flush=True)
        banks=dict(constructed_joint=arrays['bank'],old_plus_constructed_joint=np.concatenate([old_bank,arrays['bank']]));previous=None
        for method,bank in banks.items():
            started=time.perf_counter();rb=hull.RationalBank(bank);rows=[];counts=Counter();raw=Counter();statuses={}
            for i,reg in enumerate(regs):
                key=reg.tobytes().hex();assert oldrows[i]['pattern']==key;r=hull.solve(x,v,reg,rb);status=r['status'];raw[status]+=1;inherited=None
                if method=='old_plus_constructed_joint':
                    candidates=[('new_constructed',banks['constructed_joint'],previous[i]),('old_history',old_bank,oldrows[i])]
                    positives=[(name,b,row) for name,b,row in candidates if row['effective_status']=='positive']
                    if positives:assert status!='nonpositive',(seed,method,i)
                    if status=='unknown' and positives:
                        name,b,row=positives[0];old=row['result'] if row['result']['status']=='positive' else dict(row['inherited'],status='positive')
                        inherited=dict(inherited_proof(x,v,reg,bank,b,old),source_library=name);status='positive'
                rows.append(dict(index=i,pattern=key,result=r,inherited=inherited,effective_status=status));counts[status]+=1;statuses[key]=status
            seconds=time.perf_counter()-started;file=f'bank_{seed}_{method}.json';(out/file).write_text(json.dumps(rows,indent=2),encoding='utf-8')
            rec=dict(seed=seed,method=method,regions=len(regs),directions=len(bank),bank_bytes=bank.nbytes,solver_seconds=seconds,counts=dict(counts),raw_counts=dict(raw),
                status_by_pattern=statuses,rows_file=file,rows_sha256=sha(out/file),bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest())
            records.append(rec);previous=rows;print(json.dumps({k:v for k,v in rec.items() if k not in ['status_by_pattern','rows_sha256','bank_sha256']}),flush=True)
        (out/'generations.json').write_text(json.dumps(generations,indent=2),encoding='utf-8');(out/'banks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
    summary,paired=summarize(records,alm);(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,generations=len(generations),banks=len(records),regions=700,total_method_regions=1400,summaries=summary,
        comparisons=[{k:v for k,v in r.items() if k!='strict_separations'} for r in paired],source_sha256=sha(Path(__file__)),
        input_sha256={name:sha(out/name) for name in ['protocol.json','generations.json','banks.json','paired.json']},scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
