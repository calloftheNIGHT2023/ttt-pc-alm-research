"""All 700 common regions, nine unchanged libraries and a strong union."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import exact_credit_hull as model
from run_credit_wall_budget import load_inputs,METHODS as NINE

METHODS=(*NINE,'strong_union')
UNION=('pc_native','adam60_native','nodual_native')


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def prepare(inputs):
    for seed,(x,v,regs,banks) in inputs.items():
        banks['strong_union']=np.concatenate([banks[m] for m in UNION],axis=0)
    return inputs


def summarize(records):
    result=[]
    for method in METHODS:
        rr=[r for r in records if r['method']==method];counts=Counter()
        for r in rr:counts.update(r['counts'])
        result.append(dict(method=method,counts=dict(counts),diagnostic_solver_seconds=sum(r['solver_seconds'] for r in rr),
            preparation_seconds=sum(r['rational_bank_seconds'] for r in rr),directions_min=min(r['directions'] for r in rr),directions_max=max(r['directions'] for r in rr)))
    lookup={(r['seed'],r['method']):r for r in records};paired=[]
    for method in METHODS[1:]:
        counts=Counter();cases=[]
        for seed in sorted({r['seed'] for r in records}):
            a=lookup[seed,'alm_native']['status_by_pattern'];b=lookup[seed,method]['status_by_pattern'];assert a.keys()==b.keys()
            for key,av in a.items():
                bv=b[key];counts[av+'__'+bv]+=1
                if (av,bv) in [('positive','nonpositive'),('nonpositive','positive')]:cases.append(dict(seed=seed,pattern=key,alm=av,comparator=bv))
        paired.append(dict(comparator=method,status_pairs=dict(counts),strict_separations=cases))
    return result,paired


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/exact_credit_hull';out=base/'development';prim=read(base/'primitive/summary.json');assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__));inputs,data_hashes=load_inputs(root);prepare(inputs)
    protocol=dict(source_sha256=hashes,primitive_sha256=sha(base/'primitive/summary.json'),design_sha256=sha(root/'outputs/ttt-pc-alm-research/235_exact_credit_hull_separation_protocol.md'),
        input_hashes=data_hashes,seeds=sorted(inputs),methods=METHODS,union_components=UNION,total_regions=sum(len(t[2]) for t in inputs.values()),
        total_method_regions=sum(len(t[2])*len(METHODS) for t in inputs.values()),classification='exact positive / exact common nonpositive witness / unknown',
        methods_unchanged=True,queries_used=False,threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        platform=platform.platform(),python=platform.python_version(),scope='Offline mathematical diagnostic of fixed credit libraries; no online task/speed claim')
    assert all(v=='1' for v in protocol['threads'].values());assert protocol['total_regions']==700 and protocol['total_method_regions']==7000
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=[]
    for seed,(x,v,regs,banks) in inputs.items():
        for method in METHODS:
            bank=banks[method];start=time.perf_counter();rb=model.RationalBank(bank);preparation=time.perf_counter()-start
            rows=[];counts=Counter();statuses={};start=time.perf_counter()
            for i,reg in enumerate(regs):
                result=model.solve(x,v,reg,rb);counts[result['status']]+=1;pattern=reg.tobytes().hex();statuses[pattern]=result['status']
                rows.append(dict(index=i,pattern=pattern,result=result))
            seconds=time.perf_counter()-start;filename=f'bank_{seed}_{method}.json';target=out/filename;assert not target.exists()
            target.write_text(json.dumps(rows,indent=2),encoding='utf-8')
            rec=dict(seed=seed,method=method,regions=len(regs),directions=len(bank),bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),bank_bytes=bank.nbytes,
                integer_denominator_exponent=rb.exponent,rational_bank_seconds=preparation,solver_seconds=seconds,counts=dict(counts),status_by_pattern=statuses,
                rows_file=filename,rows_sha256=sha(target));records.append(rec)
            print(json.dumps({k:v for k,v in rec.items() if k not in ['status_by_pattern','rows_sha256','bank_sha256']}),flush=True)
    summary,paired=summarize(records);(out/'banks.json').write_text(json.dumps(records,indent=2),encoding='utf-8');(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    result=dict(execution_complete=True,banks=len(records),regions=700,total_method_regions=7000,summaries=summary,comparisons=[{k:v for k,v in p.items() if k!='strict_separations'} for p in paired],
        protocol_sha256=sha(out/'protocol.json'),banks_sha256=sha(out/'banks.json'),paired_sha256=sha(out/'paired.json'),source_sha256=sha(Path(__file__)),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
