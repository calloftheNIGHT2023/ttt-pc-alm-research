"""Two-repeat, six-order, same-machine comparison of three frozen solvers."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time
import numpy as np
from verify_finite_credit_game import guarded_solve as fixed_solve
from verify_credit_halfspace_projection import guarded_solve as projection_solve
from verify_credit_moment_step import guarded_solve as moment_solve


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    source=root/'results/light_h2_credit/full_bank_ceiling';base=root/'results/credit_moment_step'
    audit=base/'audit/summary.json';assert json.loads(audit.read_text())['passed']
    p=json.loads((base/'development/protocol.json').read_text());hashes=dict(p['source_sha256'])
    hashes['audit_credit_moment_step.py']=sha(Path(__file__).with_name('audit_credit_moment_step.py'));hashes[Path(__file__).name]=sha(Path(__file__))
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    config={'fixed':('finite_credit_game',fixed_solve),'projection':('credit_halfspace_projection',projection_solve),'moment':('credit_moment_step',moment_solve)}
    paths={key:root/f'results/{value[0]}/development' for key,value in config.items()}
    records={key:{(r['seed'],r['method']):r for r in json.loads((path/'banks.json').read_text())} for key,path in paths.items()}
    banks=json.loads((base/'development/banks.json').read_text());orders=[['fixed','projection','moment'],['projection','moment','fixed'],['moment','fixed','projection'],
        ['moment','projection','fixed'],['projection','fixed','moment'],['fixed','moment','projection']]
    protocol=dict(source_sha256=hashes,design_sha256=sha(root/'outputs/ttt-pc-alm-research/227_credit_solver_timing_protocol.md'),audit_sha256=sha(audit),
        repeats=2,orders=orders,order_index='(bank_index + 3*repeat) % 6',methods=p['methods'],seeds=p['seeds'],bootstrap_samples=20000,bootstrap_seed=5900001,
        input_hashes={key:sha(path/'banks.json') for key,path in paths.items()},platform=platform.platform(),python=platform.python_version(),
        threads={name:os.environ.get(name) for name in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS']},
        scope='Component timing with own-output replay, not identical-output speedup or actual hard-deadline online experiment')
    assert all(v=='1' for v in protocol['threads'].values())
    out=base/'timing';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[];array_checks=0;proof_checks=0
    for repeat in range(2):
        for bi,rec in enumerate(banks):
            assert sha(source/rec['source_data_file'])==rec['source_data_sha256']
            with np.load(source/rec['source_data_file']) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
            expected={};metas={};indices=None
            for variant,path in paths.items():
                r=records[variant][rec['seed'],rec['method']]
                assert sha(path/r['arrays_file'])==r['arrays_sha256'] and sha(path/r['meta_file'])==r['meta_sha256']
                with np.load(path/r['arrays_file']) as z:expected[variant]={key:z[key] for key in z.files}
                if indices is None:indices=expected[variant]['indices']
                else:assert np.array_equal(indices,expected[variant]['indices'])
                metas[variant]=json.loads((path/r['meta_file']).read_text())
            for rank,variant in enumerate(orders[(bi+3*repeat)%6]):
                start=time.perf_counter();arrays,meta=config[variant][1](x,v,regs[indices],bank);external=time.perf_counter()-start
                for key,value in arrays.items():assert value.tobytes()==expected[variant][key].tobytes();array_checks+=1
                assert meta['proofs']==metas[variant]['proofs'];proof_checks+=len(meta['proofs'])
                rows.append(dict(seed=rec['seed'],method=rec['method'],repeat=repeat,rank=rank,variant=variant,
                    seconds=meta['total_seconds'],external_guard_seconds=external,positive=meta['positive'],oracle_pairs=meta['oracle_pairs'],traces=meta['traces']))
            (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
            print(json.dumps(dict(repeat=repeat,banks=bi+1,solves=len(rows))),flush=True)
    lookup={(r['seed'],r['method'],r['repeat'],r['variant']):r for r in rows};rng=np.random.default_rng(5900001)
    boot=rng.integers(0,len(p['seeds']),(20000,len(p['seeds'])));summaries=[]
    for method in p['methods']:
        values={variant:np.array([np.mean([lookup[seed,method,rep,variant]['seconds'] for rep in range(2)]) for seed in p['seeds']]) for variant in config}
        comparison={}
        for variant in ['fixed','projection']:
            diff=values['moment']-values[variant];ci=np.quantile(diff[boot].mean(1),[.025,.975])
            comparison[variant]=dict(moment_minus_baseline_mean_seconds=float(diff.mean()),paired_bootstrap95=ci.tolist(),
                baseline_mean_div_moment_mean=float(values[variant].mean()/values['moment'].mean()))
        envelope={variant:[] for variant in ['fixed','projection']};moment_total=0
        for seed in p['seeds']:
            for rep in range(2):
                m=lookup[seed,method,rep,'moment'];moment_total+=m['positive']
                for variant in envelope:
                    eligible=[t for t in lookup[seed,method,rep,variant]['traces'] if t['elapsed_seconds']<=m['seconds']]
                    last=eligible[-1] if eligible else dict(step=0,positive=0)
                    envelope[variant].append(dict(seed=seed,repeat=rep,step=last['step'],positive=last['positive']))
        summaries.append(dict(method=method,mean_seconds={key:float(v.mean()) for key,v in values.items()},comparisons=comparison,
            full_128_new={variant:sum(lookup[seed,method,0,variant]['positive'] for seed in p['seeds']) for variant in config},
            prefix_envelope_mean_count_per_repeat={variant:sum(r['positive'] for r in rr)/2 for variant,rr in envelope.items()},
            prefix_envelope=envelope,moment_mean_count_per_repeat=moment_total/2))
    result=dict(passed=True,solves=len(rows),array_checks=array_checks,proof_checks=proof_checks,summaries=summaries,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),scope=protocol['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,solves=len(rows),array_checks=array_checks,proof_checks=proof_checks)),flush=True)


if __name__=='__main__':main()
