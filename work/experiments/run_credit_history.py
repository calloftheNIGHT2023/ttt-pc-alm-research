"""Frozen full-pool history attribution; expansions never enter the solvers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import numpy as np
import credit_history_capture as history
import common_pool_credit as common


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();base=root/'results/credit_history'
    primpath=base/'primitive/summary.json';prim=read(primpath);assert prim['passed'];hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    parent=root/'results/common_pool_credit/development';pools=read(parent/'pools.json');oldbanks={(r['seed'],r['method']):r for r in read(parent/'banks.json')}
    source=root/'results/light_h2_credit/full_bank_ceiling';sources={(r['seed'],r['method']):r for r in read(source/'records.json')}
    methods=['alm_native',*history.VARIANTS];protocol=dict(source_sha256=hashes,primitive_sha256=sha(primpath),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/231_credit_history_attribution_protocol.md'),parent_pools_sha256=sha(parent/'pools.json'),
        parent_banks_sha256=sha(parent/'banks.json'),source_records_sha256=sha(source/'records.json'),methods=methods,
        seeds=[r['seed'] for r in pools],steps=128,delta=.001001,initialization='uniform',archive_steps=history.SELECTED,
        method_order='Rotate fixed four-method order by seed index mod 4',exact_expansion_used_by_solver=False,
        threads={k:os.environ.get(k) for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Same-ALM trajectory credit representation attribution; source history capture separately charged diagnostic; not online speed benchmark')
    assert all(v=='1' for v in protocol['threads'].values())
    out=base/'development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');captures=[];records=[];paired=[]
    for si,pool in enumerate(pools):
        seed=pool['seed'];assert sha(parent/pool['arrays_file'])==pool['arrays_sha256']
        with np.load(parent/pool['arrays_file']) as z:x=z['x'];v=z['v'];regs=z['regs']
        ref=history.capture(x,v,False);actual=history.capture(x,v,True);history.equal_tree(ref[0],actual[0])
        for k in [2,3,4,5]:assert ref[k].tobytes()==actual[k].tobytes()
        for k in ['credit_bank','credit_proofs','positive_mode_keys','discovery_bank_sha256','completion_proposals']:assert ref[1][k]==actual[1][k]
        for k in ['trajectory_sha256','best_sha256','events']:assert ref[7][k]==actual[7][k]
        gold=sources[seed,'alm_native'];assert sha(source/gold['data_file'])==gold['data_sha256']
        with np.load(source/gold['data_file']) as z:
            for key,value in [('x',x),('v',v),('regs',actual[2]),('bank',actual[3]),('labels',actual[4]),('steps',actual[5])]:assert z[key].tobytes()==value.tobytes(),key
        banks,archive_meta=history.build(actual[6]);assert banks['native'].tobytes()==actual[3].tobytes()
        hpath=out/f'history_{seed}.npz';np.savez_compressed(hpath,**actual[6],**banks)
        cap=dict(seed=seed,history_file=hpath.name,history_sha256=sha(hpath),pool_file=pool['arrays_file'],pool_sha256=pool['arrays_sha256'],
            source_file=gold['data_file'],source_sha256=gold['data_sha256'],baseline=ref[7],captured=actual[7],archive=archive_meta,
            diagnostic_history_array_bytes=sum(q.nbytes for q in actual[6].values()),
            unchanged_discovery_bank_sha256=actual[1]['discovery_bank_sha256'],unchanged_positive_modes=len(actual[1]['positive_mode_keys']))
        captures.append(cap);sets={}
        order=methods[si%4:]+methods[:si%4]
        for rank,method in enumerate(order):
            bank=banks['native' if method=='alm_native' else method];arrays,meta=common.solve(x,v,regs,bank)
            stem=f'bank_{seed}_{method}';apath=out/f'{stem}.npz';mpath=out/f'{stem}.json'
            np.savez_compressed(apath,**arrays);mpath.write_text(json.dumps(meta,indent=2),encoding='utf-8')
            keys=['old_count','positive','total_positive','regions','directions','bank_bytes','oracle_pairs','mean_evaluations','variance_evaluations',
                  'construction_seconds','old_screen_seconds','total_seconds','charged_component_seconds','target_limited','zero_variance','nonfinite_failures','already_at_target']
            row=dict(seed=seed,method=method,rank=rank,pool_regions=len(regs),arrays_file=apath.name,arrays_sha256=sha(apath),meta_file=mpath.name,meta_sha256=sha(mpath),
                history_file=hpath.name,history_sha256=sha(hpath),**{k:meta[k] for k in keys},prefix_total={str(t['step']):meta['old_count']+t['positive'] for t in meta['traces']})
            records.append(row);sets[method]={reg.tobytes().hex() for reg in regs[arrays['accepted']]}
            (out/'banks.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps(dict(seed=seed,method=method,old=meta['old_count'],new=meta['positive'],total=meta['total_positive'],banks=len(records))),flush=True)
        for method in history.VARIANTS:
            a=sets['alm_native'];b=sets[method];paired.append(dict(seed=seed,method=method,common=len(a&b),only_native=len(a-b),only_history=len(b-a),
                only_native_patterns=sorted(a-b),only_history_patterns=sorted(b-a)))
        (out/'captures.json').write_text(json.dumps(captures,indent=2),encoding='utf-8');(out/'paired.json').write_text(json.dumps(paired,indent=2),encoding='utf-8')
    summaries=[]
    for method in methods:
        rr=[r for r in records if r['method']==method]
        sums=['pool_regions','old_count','positive','total_positive','regions','oracle_pairs','mean_evaluations','variance_evaluations','target_limited','zero_variance','nonfinite_failures','already_at_target']
        means=['directions','bank_bytes','construction_seconds','old_screen_seconds','total_seconds','charged_component_seconds']
        summaries.append(dict(method=method,**{k:sum(r[k] for r in rr) for k in sums},**{'mean_'+k:float(np.mean([r[k] for r in rr])) for k in means},
            prefix_total={str(t):sum(r['prefix_total'][str(t)] for r in rr) for t in [1,4,16,64,128]}))
    comparisons=[dict(method=m,**{k:sum(r[k] for r in paired if r['method']==m) for k in ['common','only_native','only_history']}) for m in history.VARIANTS]
    result=dict(execution_complete=True,banks=len(records),tasks=len(pools),summaries=summaries,comparisons=comparisons,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),banks_sha256=sha(out/'banks.json'),captures_sha256=sha(out/'captures.json'),
        paired_sha256=sha(out/'paired.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
