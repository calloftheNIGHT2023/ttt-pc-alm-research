"""Cold float-bound benchmark: identical candidates after deduplication.

All exact certificates are replayed outside timing. No new tasks or online
claims; this experiment is forbidden until the online interval run finishes.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import endpoint_factorized_credit_bank as endpoint


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    inp=root/'results/light_h2_credit/full_bank_ceiling';online=root/'results/online_interval_h2/development'
    assert json.loads((online/'run_audit.json').read_text())['execution_complete']
    p=json.loads((online/'protocol.json').read_text());hashes=dict(p['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(endpoint.__file__)]:hashes[path.name]=sha(path)
    out=root/'results/endpoint_factorization/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(online/'protocol.json'),verification=endpoint.verify(),
        variants=['full_candidates','endpoints'],repetitions=3,order_seed=482433,
        scope='cold float bound only on frozen pools; exact proofs checked outside timing; not online efficiency')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=json.loads((inp/'records.json').read_text());rows=[];audits=[];order=np.random.default_rng(protocol['order_seed'])
    for record in records:
        path=inp/record['data_file'];assert sha(path)==record['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        reference=endpoint.parent.Bank(x,v,bank).values(regs);actual=endpoint.Bank(x,v,bank).values(regs)
        assert reference.tobytes()==actual.tobytes()
        solver=endpoint.Bank(x,v,bank);_,proofs,_=solver.screen(regs);assert proofs==record['proofs']
        cache_solves=solver.layer_solves;reverse=solver.values(regs[::-1]);assert reverse.tobytes()==reference[::-1].tobytes() and solver.layer_solves==cache_solves
        jobs=[(variant,rep) for variant in protocol['variants'] for rep in range(3)]
        for index in order.permutation(len(jobs)):
            variant,rep=jobs[index];klass=endpoint.Bank if variant=='endpoints' else endpoint.parent.Bank
            start=time.perf_counter();solver=klass(x,v,bank);values=solver.values(regs);seconds=time.perf_counter()-start
            assert values.tobytes()==reference.tobytes()
            rows.append(dict(seed=record['seed'],method=record['method'],variant=variant,repetition=rep,
                seconds=seconds,layer_solves=solver.layer_solves,value_count=values.size))
        audits.append(dict(seed=record['seed'],method=record['method'],data_sha256=record['data_sha256'],values_bitwise=actual.size,
            proofs=len(proofs),reverse_cache_no_new_solves=True))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(banks=len(audits),total=96,method=record['method'],seed=record['seed'])),flush=True)
    summaries=[]
    for name in dict.fromkeys(r['method'] for r in records):
        means={variant:float(np.mean([r['seconds'] for r in rows if r['method']==name and r['variant']==variant])) for variant in protocol['variants']}
        summaries.append(dict(method=name,mean_seconds=means,full_over_endpoints=means['full_candidates']/means['endpoints']))
    result=dict(passed=True,banks=len(audits),rows=len(rows),float_values_bitwise=sum(r['values_bitwise'] for r in audits),
        exact_proof_replays=sum(r['proofs'] for r in audits),summaries=summaries,protocol_sha256=sha(out/'protocol.json'),
        source_sha256=sha(Path(__file__)),scope=protocol['scope'])
    for name,value in [('summary.json',result),('audits.json',audits)]: (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
