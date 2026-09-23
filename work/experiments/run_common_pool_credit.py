"""Full predeclared common-pool comparison; no prior outcome labels used."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import common_pool_credit as model


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    base=root/'results/common_pool_credit';primpath=base/'primitive/summary.json';prim=json.loads(primpath.read_text());assert prim['passed']
    hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    source=root/'results/light_h2_credit/full_bank_ceiling';records=json.loads((source/'records.json').read_text());seeds=sorted({r['seed'] for r in records})
    assert seeds==list(range(5920000,5920016))
    protocol=dict(source_sha256=hashes,primitive_sha256=sha(primpath),design_sha256=sha(root/'outputs/ttt-pc-alm-research/229_common_pool_credit_protocol.md'),
        records_sha256=sha(source/'records.json'),methods=model.METHODS,seeds=seeds,pool='Sorted bytewise union of all six original candidate sets',
        steps=128,delta=.001001,initialization='uniform',prior_outcomes_or_queries_used=False,
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Common-pool credit attribution, precomputed pool and credit generation excluded, not independent online task evidence')
    assert all(v=='1' for v in protocol['threads'].values())
    out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');pools=[];banks=[];paired=[];unique_regions=0
    for seed in seeds:
        rr={r['method']:r for r in records if r['seed']==seed};items={}
        for method in model.METHODS:
            r=rr[method];assert sha(source/r['data_file'])==r['data_sha256']
            with np.load(source/r['data_file']) as z:items[method]={key:z[key] for key in ['x','v','regs','bank']}
        x,v,regs,membership=model.assemble(items);unique_regions+=len(regs)
        poolpath=out/f'pool_{seed}.npz';np.savez_compressed(poolpath,x=x,v=v,regs=regs,membership=membership)
        pools.append(dict(seed=seed,regions=len(regs),arrays_file=poolpath.name,arrays_sha256=sha(poolpath),pattern_sha256=hashlib.sha256(regs.tobytes()).hexdigest(),
            original_region_counts={method:len(items[method]['regs']) for method in model.METHODS}))
        sets={}
        for method in model.METHODS:
            arrays,meta=model.solve(x,v,regs,items[method]['bank']);stem=f'bank_{seed}_{method}';apath=out/f'{stem}.npz';mpath=out/f'{stem}.json'
            np.savez_compressed(apath,**arrays);mpath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
            fields=['directions','bank_bytes','old_count','positive','total_positive','regions','oracle_pairs','mean_evaluations','variance_evaluations',
                'construction_seconds','old_screen_seconds','total_seconds','charged_component_seconds','target_limited','zero_variance','nonfinite_failures','already_at_target']
            r=rr[method];row=dict(seed=seed,method=method,pool_regions=len(regs),pool_file=poolpath.name,pool_sha256=sha(poolpath),
                source_file=r['data_file'],source_sha256=r['data_sha256'],source_capture_seconds_diagnostic=r['capture_seconds_diagnostic'],
                arrays_file=apath.name,arrays_sha256=sha(apath),meta_file=mpath.name,meta_sha256=sha(mpath),**{key:meta[key] for key in fields},
                prefix_total={str(t['step']):meta['old_count']+t['positive'] for t in meta['traces']})
            banks.append(row);sets[method]={reg.tobytes().hex() for reg in regs[arrays['accepted']]}
            (out/'banks.json').write_text(json.dumps(banks,indent=2),encoding='utf-8')
            print(json.dumps(dict(seed=seed,method=method,pool=len(regs),old=meta['old_count'],new=meta['positive'],total=meta['total_positive'],banks=len(banks))),flush=True)
        for i,m1 in enumerate(model.METHODS):
            for m2 in model.METHODS[i+1:]:
                a=sets[m1];b=sets[m2];paired.append(dict(seed=seed,method_a=m1,method_b=m2,common=len(a&b),only_a=len(a-b),only_b=len(b-a),
                    only_a_patterns=sorted(a-b),only_b_patterns=sorted(b-a)))
        (out/'pools.json').write_text(json.dumps(pools,indent=2),encoding='utf-8');(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    summaries=[]
    for method in model.METHODS:
        rr=[r for r in banks if r['method']==method]
        summaries.append(dict(method=method,**{key:sum(r[key] for r in rr) for key in ['pool_regions','old_count','positive','total_positive','regions','oracle_pairs',
            'mean_evaluations','variance_evaluations','target_limited','zero_variance','nonfinite_failures','already_at_target']},
            **{'mean_'+key:float(np.mean([r[key] for r in rr])) for key in ['directions','bank_bytes','construction_seconds','old_screen_seconds','total_seconds','charged_component_seconds','source_capture_seconds_diagnostic']},
            prefix_total={str(t):sum(r['prefix_total'][str(t)] for r in rr) for t in [1,4,16,64,128]}))
    comparisons=[]
    for i,m1 in enumerate(model.METHODS):
        for m2 in model.METHODS[i+1:]:
            rr=[r for r in paired if r['method_a']==m1 and r['method_b']==m2]
            comparisons.append(dict(method_a=m1,method_b=m2,**{key:sum(r[key] for r in rr) for key in ['common','only_a','only_b']}))
    result=dict(execution_complete=True,banks=len(banks),tasks=len(seeds),unique_task_regions=unique_regions,method_region_pairs=unique_regions*len(model.METHODS),
        summaries=summaries,comparisons=comparisons,source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),
        banks_sha256=sha(out/'banks.json'),pools_sha256=sha(out/'pools.json'),paired_sha256=sha(out/'paired.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
