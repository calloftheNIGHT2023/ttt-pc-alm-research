"""Coverage ceiling of all already-captured checkpoints, before compression.

Diagnostic only: all dense float evaluations and exact checks are recorded,
not presented as the cost of a deployable 32-direction method. No queries.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import light_h2_credit as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(x,v,cfg):
    snapshots=memory.Snapshots;screen=memory.certificate.screen_bank;captured=[];pools=[]
    class Observer(snapshots):
        def directions(self):
            result=super().directions()
            raw=np.concatenate(self.values);labels=np.concatenate([np.full(len(a),e['label']) for a,e in zip(self.values,self.events)])
            steps=np.concatenate([np.full(len(a),e['step'],int) for a,e in zip(self.values,self.events)])
            captured.append((raw,labels,steps));return result
    def watch(x,v,regs,bank):
        pools.append(regs.copy());return screen(x,v,regs,bank)
    try:
        memory.Snapshots=Observer;memory.certificate.screen_bank=watch
        data,meta=memory.prepare(x,v,cfg)
    finally:memory.Snapshots=snapshots;memory.certificate.screen_bank=screen
    assert len(captured)==1
    raw,labels,steps=captured[0];norm=np.linalg.norm(raw,axis=(1,2));keep=norm>1e-14
    raw=raw[keep];labels=labels[keep];steps=steps[keep]
    raw/=np.max(abs(raw),axis=(1,2),keepdims=True)
    seen=set();ids=[]
    for i,row in enumerate(raw):
        if row.tobytes() not in seen:seen.add(row.tobytes());ids.append(i)
    bank=raw[ids];labels=labels[ids];steps=steps[ids]
    for a in np.array(meta['credit_bank']):assert a.tobytes() in seen
    keys=sorted({r.tobytes() for regs in pools for r in regs})
    regs=np.array([np.frombuffer(k,np.uint8).reshape(4,len(x)) for k in keys],np.uint8)
    return data,meta,regs,bank,labels,steps


def full_screen(x,v,regs,bank):
    start=time.perf_counter();best=np.full(len(regs),-np.inf);chosen=np.zeros(len(regs),int);pairs=0
    for first in range(0,len(bank),32):
        bb=bank[first:first+32];rr=np.repeat(regs,len(bb),axis=0);aa=np.tile(bb,(len(regs),1,1))
        values=memory.certificate.float_optimum(x,v,rr,aa).reshape(len(regs),len(bb));ids=values.argmax(1);value=values[np.arange(len(regs)),ids]
        improve=value>best;best[improve]=value[improve];chosen[improve]=first+ids[improve];pairs+=len(rr)
    float_seconds=time.perf_counter()-start;proofs=[];start=time.perf_counter()
    for i in np.flatnonzero(best>1e-10*(1+abs(bank[chosen]).sum((1,2)))):
        exact=memory.certificate.exact_optimum(x,v,regs[i],bank[chosen[i]])
        if exact['positive']:proofs.append(dict(pattern=regs[i].tobytes().hex(),direction=int(chosen[i]),exact=exact))
    return proofs,dict(dense_direction_pairs=pairs,float_seconds=float_seconds,exact_seconds=time.perf_counter()-start)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/light_h2_credit/development';out=root/'results/light_h2_credit/full_bank_ceiling'
    parent=json.loads((inp/'protocol.json').read_text());assert json.loads((inp/'run_audit.json').read_text())['execution_complete']
    hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    names=['alm_native','alm_residual','adam60_native','adam60_residual','pc_native','nodual_native']
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),methods=names,seeds=parent['seeds'],
        scope='post-192 diagnostic of uncompressed fixed snapshots; not new observations, confirmation, or a free dense solver')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');records=[]
    for seed in parent['seeds']:
        for name in names:
            cfg=next(c for c in parent['configs'] if c['name']==name)
            with np.load(inp/f'state_{seed}_{name}_0_4.npz') as z:x=z['x'];v=z['v']
            reference=json.loads((inp/f'detail_{seed}_{name}.json').read_text());start=time.perf_counter()
            data,meta,regs,bank,labels,steps=capture(x,v,cfg);capture_seconds=time.perf_counter()-start
            assert meta['credit_bank']==reference['credit_bank'] and meta['credit_proofs']==reference['credit_proofs']
            assert meta['positive_mode_keys']==reference['positive_mode_keys'] and meta['discovery_bank_sha256']==reference['discovery_bank_sha256']
            proofs,timing=full_screen(x,v,regs,bank);full={p['pattern'] for p in proofs};compressed={p['pattern'] for p in meta['credit_proofs']}
            assert compressed<=full and not full&set(meta['positive_mode_keys'])
            artifact=out/f'bank_{seed}_{name}.npz';np.savez_compressed(artifact,x=x,v=v,regs=regs,bank=bank,labels=labels,steps=steps)
            record=dict(seed=seed,method=name,compressed_certificates=len(compressed),uncompressed_certificates=len(full),
                extra_certificates=len(full-compressed),full_direction_rows=len(bank),h2_c20_cells=len(regs),
                h2_distinct_layer_rows=sum(len({r[j].tobytes() for r in regs}) for j in range(4)),
                naive_layer_rows=4*len(regs),snapshot_bank_bytes=bank.nbytes,capture_seconds_diagnostic=capture_seconds,**timing,
                data_file=artifact.name,data_sha256=sha(artifact),proofs=proofs)
            records.append(record);(out/'records.json').write_text(json.dumps(records,indent=2),encoding='utf-8')
            print(json.dumps({k:v for k,v in record.items() if k not in ['proofs']}),flush=True)
    summaries=[]
    for name in names:
        rr=[r for r in records if r['method']==name]
        summaries.append(dict(method=name,compressed=sum(r['compressed_certificates'] for r in rr),full=sum(r['uncompressed_certificates'] for r in rr),
            extra=sum(r['extra_certificates'] for r in rr),tasks_with_extra=sum(r['extra_certificates']>0 for r in rr),
            mean_full_rows=float(np.mean([r['full_direction_rows'] for r in rr])),
            mean_dense_seconds=float(np.mean([r['float_seconds']+r['exact_seconds'] for r in rr])),
            naive_layer_rows=sum(r['naive_layer_rows'] for r in rr),distinct_layer_rows=sum(r['h2_distinct_layer_rows'] for r in rr)))
    result=dict(passed=True,records=len(records),summaries=summaries,source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),
        scope=protocol['scope']);(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
