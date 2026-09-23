"""Frozen cold-cache component benchmark on the 96 captured H2 banks."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import factorized_credit_bank as factor
import diagnose_h2_full_bank_ceiling as dense


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/light_h2_credit/full_bank_ceiling';out=root/'results/factorized_credit_bank/development'
    parent=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'summary.json').read_text())['passed']
    hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(factor.__file__)]:hashes[path.name]=sha(path)
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),seeds=parent['seeds'],methods=parent['methods'],
        variants=['dense','factorized'],repetitions=3,order_seed=482123,verification=factor.verify(),
        scope='cold full-bank bound plus exact certificate component only; captured pools, not full online or future-candidate access at deployment')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    records=json.loads((inp/'records.json').read_text());rows=[];audits=[];rng=np.random.default_rng(protocol['order_seed'])
    for record in records:
        path=inp/record['data_file'];assert sha(path)==record['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        bb=factor.Bank(x,v,bank);actual=bb.values(regs);reference=[]
        for first in range(0,len(bank),32):
            aa=bank[first:first+32]
            values=factor.original.float_optimum(x,v,np.repeat(regs,len(aa),axis=0),np.tile(aa,(len(regs),1,1)))
            reference.append(values.reshape(len(regs),len(aa)))
        reference=np.concatenate(reference,axis=1);assert np.array_equal(actual,reference)
        # This independent identity check is outside either measured variant.
        original_layer_solves=bb.layer_solves;reverse=bb.values(regs[::-1]);assert np.array_equal(reverse,reference[::-1]) and bb.layer_solves==original_layer_solves
        jobs=[(variant,rep) for variant in protocol['variants'] for rep in range(3)]
        for index in rng.permutation(len(jobs)):
            variant,rep=jobs[index];start=time.perf_counter()
            if variant=='dense':proofs,meta=dense.full_screen(x,v,regs,bank);cache_bytes=0;layer_solves=4*len(regs)
            else:
                solver=factor.Bank(x,v,bank);_,proofs,meta=solver.screen(regs);cache_bytes=meta['cached_values_numeric_bytes'];layer_solves=solver.layer_solves
            elapsed=time.perf_counter()-start;assert proofs==record['proofs']
            rows.append(dict(seed=record['seed'],method=record['method'],variant=variant,repetition=rep,total_seconds=elapsed,
                exact_seconds=meta['exact_seconds'],float_seconds=meta['float_seconds'],proofs=len(proofs),
                direction_rows=len(bank),region_rows=len(regs),layer_solves=layer_solves,cache_numeric_bytes=cache_bytes))
        audits.append(dict(seed=record['seed'],method=record['method'],data_sha256=record['data_sha256'],values_bitwise=actual.size,
            all_proof_records_bitwise=True,reverse_cache_no_new_solves=True))
        (out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
        print(json.dumps(dict(seed=record['seed'],method=record['method'],completed_banks=len(audits),total_banks=len(records),
            proofs=len(record['proofs']),dense_rows=record['naive_layer_rows'],factor_rows=bb.layer_solves)),flush=True)
    summaries=[]
    for name in protocol['methods']:
        means={variant:float(np.mean([r['total_seconds'] for r in rows if r['method']==name and r['variant']==variant])) for variant in protocol['variants']}
        summaries.append(dict(method=name,mean_seconds=means,dense_over_factorized=means['dense']/means['factorized']))
    result=dict(passed=True,rows=len(rows),banks=len(audits),total_float_values_bitwise=sum(a['values_bitwise'] for a in audits),
        proof_records_per_variant=sum(r['uncompressed_certificates'] for r in records),summaries=summaries,scope=protocol['scope'])
    for name,data in [('audits.json',audits),('summary.json',result)]: (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
