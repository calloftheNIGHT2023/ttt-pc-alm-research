"""Support-only full-hull ceiling on all frozen pools, including old skips."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import joint_credit_minimax as model
from online_endpoint_h2 import EndpointIntervalBank


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    primitive=root/'results/joint_credit_minimax/primitive/summary.json';prim=json.loads(primitive.read_text());assert prim['passed']
    hashes=dict(prim['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    hashes[Path(__file__).name]=sha(Path(__file__))
    inp=root/'results/light_h2_credit/full_bank_ceiling';records=json.loads((inp/'records.json').read_text())
    gate=root/'results/bounded_error_primal_gate/component';gates={(r['seed'],r['method']):r for r in json.loads((gate/'audits.json').read_text())}
    p=dict(source_sha256=hashes,primitive_sha256=sha(primitive),records_sha256=sha(inp/'records.json'),gate_audits_sha256=sha(gate/'audits.json'),
        pool_count=len(records),methods=sorted({r['method'] for r in records}),seeds=sorted({r['seed'] for r in records}),
        domain='Same preactivation-consistent boxes as original fixed-credit exact optimum',
        decision='A float LP proposes a credit; Fraction positive lower after exact convex-mixture rounding allowance accepts it',
        scope='All old pools; LP oracle diagnostic, not deployable speedup or independent confirmation',
        negative_policy='Nonpositive LP values are numerical, not exact hull exclusion without an exact common primal witness')
    out=root/'results/joint_credit_minimax/development';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(p,indent=2),encoding='utf-8')
    banks=[];counts=dict(regions=0,old_proofs=0,lp_calls=0,lp_failures=0,new_exact_proofs=0,new_from_old_gate_skip=0)
    for rec in records:
        path=inp/rec['data_file'];assert sha(path)==rec['data_sha256']
        with np.load(path) as z:x=z['x'];v=z['v'];regs=z['regs'];bank=z['bank']
        solver=EndpointIntervalBank(x,v,bank);mask,proofs,_=solver.screen(regs);bounds=solver.values(regs)
        assert [(p['pattern'],p['direction']) for p in proofs]==[(p['pattern'],p['direction']) for p in rec['proofs']]
        gr=gates[rec['seed'],rec['method']];assert sha(gate/gr['gate_file'])==gr['gate_sha256']
        with np.load(gate/gr['gate_file']) as z:oldskip=z['skip']
        assert not np.any(oldskip & mask)
        aout=np.full((len(regs),*bank.shape[1:]),np.nan);wout=np.full((len(regs),len(bank)),np.nan)
        pout=np.full((len(regs),bank.shape[1]+2*regs[0].size+1),np.nan);rows=[];start=time.perf_counter()
        for i,reg in enumerate(regs):
            row=dict(index=i,pattern=reg.tobytes().hex(),old_positive=bool(mask[i]),old_gate_skipped=bool(oldskip[i]),old_float_max=float(bounds[i].max()))
            counts['regions']+=1;counts['old_proofs']+=int(mask[i])
            if not mask[i]:
                data,meta=model.solve(x,v,reg,bank);counts['lp_calls']+=1
                row['joint']=meta
                if data is None:counts['lp_failures']+=1
                else:
                    aout[i]=data['credit'];wout[i]=data['weights'];pout[i]=data['primal']
                    assert meta['numerical_value']>=row['old_float_max']-1e-8
                    counts['new_exact_proofs']+=int(meta['positive']);counts['new_from_old_gate_skip']+=int(meta['positive'] and oldskip[i])
            rows.append(row)
        elapsed=time.perf_counter()-start
        stem=f'bank_{rec["seed"]}_{rec["method"]}';arrays=out/(stem+'.npz');rowfile=out/(stem+'.json')
        np.savez_compressed(arrays,credit=aout,weights=wout,primal=pout,old_positive=mask,old_gate_skipped=oldskip)
        rowfile.write_text(json.dumps(rows,indent=2),encoding='utf-8')
        cases=[r['joint'] for r in rows if 'joint' in r];positive=[r for r in rows if r.get('joint',{}).get('positive')]
        banks.append(dict(seed=rec['seed'],method=rec['method'],regions=len(regs),directions=len(bank),old_proofs=int(mask.sum()),
            lp_calls=len(cases),lp_failures=sum(not c['success'] for c in cases),new_proofs=len(positive),
            new_from_old_gate_skip=sum(r['old_gate_skipped'] for r in positive),remaining=len(regs)-int(mask.sum())-len(positive),
            loop_seconds_diagnostic=elapsed,lp_seconds=sum(c['lp_seconds'] for c in cases),
            certification_seconds=sum(c.get('certification_seconds',0) for c in cases),lp_iterations=sum(c['iterations'] for c in cases),
            max_matrix_numeric_bytes=max([c['numeric_arrays_bytes'] for c in cases],default=0),stored_arrays_bytes=aout.nbytes+wout.nbytes+pout.nbytes,
            source_data_file=rec['data_file'],source_data_sha256=rec['data_sha256'],rows_file=rowfile.name,rows_sha256=sha(rowfile),arrays_file=arrays.name,arrays_sha256=sha(arrays)))
        (out/'banks.json').write_text(json.dumps(banks,indent=2),encoding='utf-8')
        print(json.dumps(dict(banks=len(banks),seed=rec['seed'],method=rec['method'],new_proofs=len(positive),**counts)),flush=True)
    summaries=[]
    for name in p['methods']:
        rr=[r for r in banks if r['method']==name]
        summaries.append(dict(method=name,**{k:sum(r[k] for r in rr) for k in ['regions','old_proofs','lp_calls','lp_failures','new_proofs','new_from_old_gate_skip','remaining']},
            mean_seconds_diagnostic=float(np.mean([r['loop_seconds_diagnostic'] for r in rr])),mean_lp_seconds=float(np.mean([r['lp_seconds'] for r in rr])),
            mean_certification_seconds=float(np.mean([r['certification_seconds'] for r in rr])),max_matrix_numeric_bytes=max(r['max_matrix_numeric_bytes'] for r in rr)))
    result=dict(execution_complete=True,counts=counts,summaries=summaries,source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/'protocol.json'),
        banks_sha256=sha(out/'banks.json'),scope=p['scope'])
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
