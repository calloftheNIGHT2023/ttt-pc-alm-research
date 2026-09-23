"""Frozen charged same-count materialization and held-out prediction test."""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import rejection_memory_materialization as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/rejection_credit/diagnostic';curve=root/'results/posterior_state_reuse/conditional_risk'
    out=root/'results/rejection_materialization/development';parent=json.loads((inp/'protocol.json').read_text())
    hashes=dict(parent['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    for path in [Path(__file__),Path(memory.__file__)]:hashes[path.name]=sha(path)
    with np.load(inp/'proposal_5900001.npz') as z:
        x=z['x'];v=z['v'];keys=z['patterns'].tolist();masks=dict(zip(z['methods'].tolist(),z['rejected']))
    masks['bp_local_union']=masks['bp_residual']|masks['local']
    protocol=dict(source_sha256=hashes,parent_protocol_sha256=sha(inp/'protocol.json'),config=parent['config'],seeds=parent['seeds'],
        methods=list(memory.METHODS),sample_counts=[512,2048],repetitions=2,proposal_batch=32768,maximum_proposals=268435456,
        rng_seed=481763,order_seed=481769,query_points=257,verification=memory.verify(x,v,parent['config'],keys,masks),
        scope='fresh per-method context-to-prediction timing on common original ALM discovery, first write only; not distinct optimizer/full online/official TTT comparison')
    out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    audits={a['seed']:a for a in json.loads((inp/'audits.json').read_text())};rngorder=np.random.default_rng(protocol['order_seed']);rows=[]
    q=np.linspace(0,1,protocol['query_points'])
    for seed in protocol['seeds']:
        prop=inp/f'proposal_{seed}.npz';assert sha(prop)==audits[seed]['proposal_sha256']
        with np.load(prop) as z:x=z['x'];v=z['v'];keys=z['patterns'].tolist();masks=dict(zip(z['methods'].tolist(),z['rejected']))
        masks['bp_local_union']=masks['bp_residual']|masks['local']
        seedrows=[];jobs=[(m,k,r) for m in protocol['methods'] for k in protocol['sample_counts'] for r in range(protocol['repetitions'])]
        for index in rngorder.permutation(len(jobs)):
            method,count,rep=jobs[index];rng=np.random.default_rng(np.random.SeedSequence([protocol['rng_seed'],seed,rep,count]))
            begin=time.perf_counter();data,meta=memory.prepare(x,v,protocol['config'],method)
            points,labels,sampling=memory.sample(data,count,rng,protocol['proposal_batch'],protocol['maximum_proposals'])
            read_begin=time.perf_counter();prediction=memory.predict(points,q);read_seconds=time.perf_counter()-read_begin
            elapsed=time.perf_counter()-begin
            assert [r.tobytes().hex() for r in data['regs']]==keys
            if method!='geometry':assert np.array_equal(data['indices'],np.flatnonzero(~masks[method]))
            codes,h=memory.capture.model.light.forward_many(x,points)
            assert np.array_equal(codes,data['regs'][labels]);assert np.max(np.abs(h[:,-1]-v))<=.001+1e-8
            state=out/f'state_{seed}_{method}_{count}_{rep}.npz'
            np.savez_compressed(state,points=points,labels=labels,q=q,prediction=prediction)
            row=dict(seed=seed,method=method,samples=count,repetition=rep,total_context_to_prediction_seconds=elapsed,
                read_seconds=read_seconds,**sampling,capture_seconds=meta['capture_seconds'],contraction_seconds=meta['contraction_seconds'],
                construction_seconds=meta['construction_seconds'],screen_seconds=meta['screen_seconds'],
                proposal_arrays_subtotal=meta['proposal_arrays_subtotal'],inequality_arrays_subtotal=meta['inequality_arrays_subtotal'],
                geometry_arrays_subtotal=meta['geometry_arrays_subtotal'],particle_state_bytes=points.nbytes,
                pool=meta['pool'],rejected=meta['rejected'],state_file=state.name,state_sha256=sha(state),
                actual_positive_patterns=sorted({keys[int(i)] for i in labels}),max_support_error=float(np.max(np.abs(h[:,-1]-v))))
            seedrows.append(row)
            print(json.dumps(dict(seed=seed,method=method,samples=count,rep=rep,completed=len(rows)+len(seedrows),total=320,seconds=round(elapsed,4))),flush=True)
        # All deployments and prediction files are fixed before revealing any
        # query targets or posterior-reference moments for this context.
        ca=json.loads((curve/f'audit_{seed}.json').read_text());cp=curve/ca['curve_file'];assert sha(cp)==ca['curve_sha256']
        with np.load(cp) as z:
            assert np.array_equal(q,z['q'][::4])
            full=np.einsum('k,rkq->rq',z['weights'],z['region_means'])[:,::4]
        teacher=np.random.default_rng(seed).uniform(-.12,.12,4);targets=memory.geometry.base.forward(q,teacher)
        for row in seedrows:
            with np.load(out/row['state_file']) as z:prediction=z['prediction']
            row['query_mse_grid257']=float(np.trapezoid((prediction-targets)**2,x=q))
            row['conditional_excess_grid257']=float(np.mean([np.trapezoid((prediction-full[a])*(prediction-full[b]),x=q) for a,b in [(0,1),(2,3)]]))
            row['reference_sha256']=ca['curve_sha256']
        rows.extend(seedrows);(out/'rows.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    result=dict(complete=True,rows=len(rows),tasks=len(protocol['seeds']),sources=len(hashes),scope=protocol['scope']);assert len(rows)==320
    (out/'run_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
