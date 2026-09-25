"""All visited activation regions, including individually infeasible iterates.

Reuses frozen support-only trajectories; no online optimizer change or queries.
"""
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
import region_posterior_memory as geometry
import local_region_screen as screen


def unique_modes(x,bank):
    h=np.broadcast_to(x,(len(bank),len(x)));regs=[]
    for j in range(bank.shape[1]):
        z=h+bank[:,j,None];regs.append(np.searchsorted(base.KNOTS,z,side='right').astype(np.uint8));h=base.g(z)
    regs=np.ascontiguousarray(np.stack(regs,axis=1));signature=regs.reshape(len(bank),-1)
    tokens=signature.view(np.dtype((np.void,signature.shape[1]))).reshape(-1)
    _,first=np.unique(tokens,return_index=True);first=np.sort(first)
    return bank[first],regs[first]


def main():
    p=argparse.ArgumentParser();p.add_argument('--trace',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    proto=json.loads((a.trace/'protocol.json').read_text());trace_rows=json.loads((a.trace/'modes.json').read_text())
    assert all(hashlib.sha256(Path(__file__).with_name(n).read_bytes()).hexdigest()==h for n,h in proto['source_sha256'].items())
    a.out.mkdir(parents=True,exist_ok=True);assert not (a.out/'protocol.json').exists()
    protocol=dict(seeds=proto['seeds'],stages=proto['stages'],configs=proto['configs'],source_sha256={**proto['source_sha256'],
        Path(__file__).name:hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),Path(screen.__file__).name:hashlib.sha256(Path(screen.__file__).read_bytes()).hexdigest()},
        scope='support-only all visited cells, finite volume geometry, no query data; repeated old development tasks',contract_rounds=5)
    (a.out/'protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8');rows=[]
    for row in trace_rows:
        start=time.perf_counter();path=a.trace/row['trace_file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['trace_sha256']
        with np.load(path) as data:visited,original,x,v=[data[k] for k in ['visited','bank','x','v']]
        bank,regs=unique_modes(x,visited);assert len(bank)==row['unique_visited_patterns']
        original_keys={base.pattern(x,b).astype(np.uint8).tobytes() for b in original}
        extra=np.array([r.tobytes() not in original_keys for r in regs]);assert np.sum(~extra)==len(original_keys)
        rejected=screen.contract(x,v,regs,rounds=5)
        positive_original_keys={base.pattern(x,np.asarray(p['representative'])).astype(np.uint8).tobytes() for p in row['positive_retained']}
        assert all(not reject for reg,reject in zip(regs,rejected) if reg.tobytes() in positive_original_keys)
        ids=np.flatnonzero(extra&~rejected);positive=[];reasons={}
        for i in ids:
            _,_,g,rhs=base.branch_polytope(x,v,bank[i]);poly,note=geometry.polytope(g,rhs);reason=note['reason'];reasons[reason]=reasons.get(reason,0)+1
            if poly is not None:positive.append(dict(representative=bank[i].tolist(),pattern=regs[i].tobytes().hex(),**note))
        mass=row['retained_absolute_prior_mass'];gain=sum(p['volume'] for p in positive)/.24**4
        rows.append(dict(method=row['method'],seed=row['seed'],n_context=row['n_context'],unique_visited_patterns=len(bank),
                         original_retained_patterns=len(original_keys),extra_patterns=int(extra.sum()),screen_rejected_extra=int(np.sum(extra&rejected)),
                         extra_geometry_calls=len(ids),positive_extra_patterns=len(positive),positive_retained_patterns=row['positive_retained_patterns'],
                         retained_absolute_prior_mass=mass,extra_absolute_prior_mass=gain,retained_fraction_of_observed_union=mass/(mass+gain) if mass+gain else None,
                         compact_bank_and_signature_bytes=bank.nbytes+regs.nbytes,trace_array_bytes=visited.nbytes,positive_extra=positive,geometry_reasons=reasons,
                         diagnostic_seconds=time.perf_counter()-start,screen_keeps_original_positive_regions=True))
        if len(rows)%8==0:
            (a.out/'regions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(completed_rows=len(rows),total=len(trace_rows))),flush=True)
    (a.out/'regions.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');print(json.dumps(dict(complete=True,rows=len(rows))),flush=True)


if __name__=='__main__':main()
