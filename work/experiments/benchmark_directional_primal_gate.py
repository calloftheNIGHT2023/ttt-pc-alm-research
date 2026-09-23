"""Same frozen pools and charged interface, tighter directional upper gate."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import benchmark_primal_upper_gate as first
import directional_primal_upper_gate as gate


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def execute(*args):
    old=first.gate
    try:first.gate=gate;return first.execute(*args)
    finally:first.gate=old


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    primitive=root/'results/directional_primal_gate/primitive/summary.json';prim=json.loads(primitive.read_text());assert prim['passed']
    hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    inp=root/'results/light_h2_credit/full_bank_ceiling';records=json.loads((inp/'records.json').read_text())
    p=dict(source_sha256=hashes,primitive_sha256=sha(primitive),pool_records_sha256=sha(inp/'records.json'),
        variants=['endpoint_interval','directional_gate_endpoint_interval'],repetitions=3,order_seed=483229,
        scope='All 96 frozen C20-surviving pools; charged cold component only, not causal online benefit')
    out=root/'results/directional_primal_gate/component';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    rows=[];audits=[];order=np.random.default_rng(p['order_seed'])
    for record in records:
        path=inp/record['data_file'];assert sha(path)==record['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        ref,rp,rm,_,_=execute(x,v,regs,bank,False);actual,pp,am,skip,upper=execute(x,v,regs,bank,True)
        assert np.array_equal(ref,actual) and rp==pp
        assert [(p['pattern'],p['direction']) for p in pp]==[(p['pattern'],p['direction']) for p in record['proofs']]
        jobs=[(enabled,rep) for enabled in [False,True] for rep in range(p['repetitions'])]
        for index in order.permutation(len(jobs)):
            enabled,rep=jobs[index];mask,proofs,meta,ss,_=execute(x,v,regs,bank,enabled)
            assert np.array_equal(mask,ref) and proofs==rp
            if enabled:assert np.array_equal(ss,skip)
            rows.append(dict(seed=record['seed'],method=record['method'],enabled=enabled,repetition=rep,**meta))
        artifact=out/f'gate_{record["seed"]}_{record["method"]}.npz';np.savez_compressed(artifact,skip=skip,upper=upper)
        audits.append(dict(seed=record['seed'],method=record['method'],source_data_file=record['data_file'],source_data_sha256=record['data_sha256'],
            gate_file=artifact.name,gate_sha256=sha(artifact),regions=len(regs),directions=len(bank),skipped=int(skip.sum()),retained_proofs=len(pp),gate_info=am['gate']))
        print(json.dumps(dict(banks=len(audits),regions=len(regs),skipped=int(skip.sum()),proofs=len(pp))),flush=True)
    summaries=[]
    for name in dict.fromkeys(r['method'] for r in records):
        rr=[r for r in rows if r['method']==name];aa=[a for a in audits if a['method']==name]
        base=float(np.mean([r['seconds'] for r in rr if not r['enabled']]));candidate=float(np.mean([r['seconds'] for r in rr if r['enabled']]))
        summaries.append(dict(method=name,baseline_seconds=base,gated_seconds=candidate,difference_seconds=candidate-base,
            mean_gate_seconds=float(np.mean([r['gate']['seconds'] for r in rr if r['enabled']])),regions=sum(a['regions'] for a in aa),
            skipped=sum(a['skipped'] for a in aa),preserved_proofs=sum(a['retained_proofs'] for a in aa),
            baseline_layer_solves=float(np.mean([r['layer_solves'] for r in rr if not r['enabled']])),
            gated_layer_solves=float(np.mean([r['layer_solves'] for r in rr if r['enabled']]))))
    result=dict(passed=True,banks=len(audits),timings=len(rows),regions=sum(a['regions'] for a in audits),skipped=sum(a['skipped'] for a in audits),
        preserved_proofs=sum(a['retained_proofs'] for a in audits),summaries=summaries,source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),scope=p['scope'])
    for name,value in [('rows.json',rows),('audits.json',audits),('summary.json',result)]:
        (out/name).write_text(json.dumps(value,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
