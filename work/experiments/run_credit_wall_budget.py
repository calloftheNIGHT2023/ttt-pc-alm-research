"""Nine frozen credit families at four actual clock budgets, two repeats."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
import credit_wall_budget as model
import common_pool_credit as common
import credit_history_capture as history

BUDGETS=(.05,.1,.2,.4)
METHODS=(*common.METHODS,*history.VARIANTS)


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def load_inputs(root):
    parent=root/'results/common_pool_credit/development';records={(r['seed'],r['method']):r for r in read(parent/'banks.json')}
    hp=root/'results/credit_history/development';caps={r['seed']:r for r in read(hp/'captures.json')};inputs={};hashes={}
    for pool in read(parent/'pools.json'):
        seed=pool['seed'];assert sha(parent/pool['arrays_file'])==pool['arrays_sha256']
        with np.load(parent/pool['arrays_file']) as z:x=z['x'];v=z['v'];regs=z['regs']
        banks={};hashes[seed]=dict(pool=pool['arrays_sha256'],banks={})
        for method in common.METHODS:
            rec=records[seed,method];path=root/'results/light_h2_credit/full_bank_ceiling'/rec['source_file'];assert sha(path)==rec['source_sha256']
            with np.load(path) as z:assert z['x'].tobytes()==x.tobytes() and z['v'].tobytes()==v.tobytes();banks[method]=z['bank']
            hashes[seed]['banks'][method]=dict(file=str(path.relative_to(root)),sha256=sha(path))
        cap=caps[seed];path=hp/cap['history_file'];assert sha(path)==cap['history_sha256']
        with np.load(path) as z:
            for method in history.VARIANTS:banks[method]=z[method];hashes[seed]['banks'][method]=dict(file=str(path.relative_to(root)),sha256=sha(path))
        inputs[seed]=(x,v,regs,banks)
    return inputs,hashes


def summarize(rows,seeds):
    rng=np.random.default_rng(5900001);boot=rng.integers(0,len(seeds),(20000,len(seeds)));lookup={(r['seed'],r['method'],r['budget_seconds'],r['repeat']):r for r in rows}
    result=[]
    for budget in BUDGETS:
        rr=[r for r in rows if r['budget_seconds']==budget];summaries=[];comparisons=[]
        means={m:np.array([np.mean([lookup[seed,m,budget,rep]['total_positive'] for rep in range(2)]) for seed in seeds]) for m in METHODS}
        for method in METHODS:
            sub=[r for r in rr if r['method']==method];over=np.array([r['overrun_seconds'] for r in sub]);external=np.array([r['external_guard_seconds'] for r in sub])
            summaries.append(dict(method=method,count_mean_over_repeats=float(sum(r['total_positive'] for r in sub)/2),
                old_count_mean_over_repeats=float(sum(r['old_count'] for r in sub)/2),new_count_mean_over_repeats=float(sum(r['new_count'] for r in sub)/2),
                mean_actual_seconds=float(np.mean([r['total_seconds'] for r in sub])),mean_external_seconds=float(external.mean()),
                overrun_mean=float(over.mean()),overrun_p95=float(np.quantile(over,.95)),overrun_max=float(over.max()),
                external_overrun_mean=float(np.maximum(external-budget,0).mean()),external_overrun_max=float(np.maximum(external-budget,0).max()),
                late_old=sum(r['late_old_count'] for r in sub),late_new=sum(r['late_new_count'] for r in sub),
                mean_response_batches=float(np.mean([r['response_batches'] for r in sub])),max_response_batches=max(r['response_batches'] for r in sub),
                stop_reasons=dict(Counter(r['stop_reason'] for r in sub)),stop_events=dict(Counter(str(r['stop_event']) for r in sub))))
            if method=='alm_native':continue
            diff=means['alm_native']-means[method];ci=np.quantile(diff[boot].mean(1),[.025,.975]);both=onlya=onlyb=0
            for seed in seeds:
                for rep in range(2):
                    a=set(lookup[seed,'alm_native',budget,rep]['accepted_patterns']);b=set(lookup[seed,method,budget,rep]['accepted_patterns'])
                    both+=len(a&b);onlya+=len(a-b);onlyb+=len(b-a)
            comparisons.append(dict(comparator=method,native_minus_comparator_mean_per_task=float(diff.mean()),paired_bootstrap95=ci.tolist(),
                common_mean_over_repeats=both/2,only_native_mean_over_repeats=onlya/2,only_comparator_mean_over_repeats=onlyb/2))
        result.append(dict(budget_seconds=budget,summaries=summaries,comparisons=comparisons))
    return result


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_wall_budget'
    pp=base/'primitive/summary.json';prim=read(pp);assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__));inputs,data_hashes=load_inputs(root);seeds=sorted(inputs)
    protocol=dict(source_sha256=hashes,primitive_sha256=sha(pp),design_sha256=sha(root/'outputs/ttt-pc-alm-research/233_credit_wall_budget_protocol.md'),
        input_hashes=data_hashes,seeds=seeds,methods=METHODS,budgets_seconds=BUDGETS,repeats=2,max_steps=4096,
        initialization='uniform',delta=model.moment.DELTA,method_order='Base order reversed on repeat 1, then rotated by (seed_index + 3*budget_index + repeat) mod 9',
        bootstrap_samples=20000,bootstrap_seed=5900001,threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        platform=platform.platform(),python=platform.python_version(),input_loading='All inputs loaded before timing; file output excluded',
        scope='Cooperative actual clock budget for credit component only; atomic overruns fully reported; no hard-return or online-risk claim')
    assert all(v=='1' for v in protocol['threads'].values())
    out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists();(out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    rows=[]
    for rep in range(2):
        for bi,budget in enumerate(BUDGETS):
            for si,seed in enumerate(seeds):
                x,v,regs,banks=inputs[seed];order=list(METHODS if rep==0 else reversed(METHODS));shift=(si+3*bi+rep)%len(order);order=order[shift:]+order[:shift]
                for rank,method in enumerate(order):
                    start=time.perf_counter();arrays,meta=model.guarded_solve(x,v,regs,banks[method],budget=budget,max_steps=4096);external=time.perf_counter()-start
                    assert all(pr['completed_seconds']<=budget for pr in meta['proofs'])
                    stem=f'run_{seed}_{method}_{int(budget*1000)}ms_r{rep}';apath=out/f'{stem}.npz';mpath=out/f'{stem}.json'
                    np.savez_compressed(apath,**arrays);mpath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
                    keys=['budget_seconds','old_count','new_count','total_positive','late_old_count','late_new_count','response_batches','update_batches','last_step',
                          'oracle_pairs','mean_evaluations','variance_evaluations','exact_checks','late_exact_checks','stop_reason','stop_event','total_seconds','overrun_seconds',
                          'target_limited','zero_variance','nonfinite_failures','already_at_target']
                    rows.append(dict(seed=seed,method=method,repeat=rep,rank=rank,pool_regions=len(regs),directions=len(banks[method]),
                        external_guard_seconds=external,arrays_file=apath.name,arrays_sha256=sha(apath),meta_file=mpath.name,meta_sha256=sha(mpath),
                        accepted_patterns=[reg.tobytes().hex() for reg in regs[arrays['accepted']]],**{k:meta[k] for k in keys}))
                (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(repeat=rep,budget_ms=int(budget*1000),seed=seed,runs=len(rows))),flush=True)
    result=dict(execution_complete=True,runs=len(rows),summaries=summarize(rows,seeds),source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),
        rows_sha256=sha(out/'rows.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(execution_complete=True,runs=len(rows))),flush=True)


if __name__=='__main__':main()
