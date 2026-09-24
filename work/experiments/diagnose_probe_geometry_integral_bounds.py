"""Certified simplex integral bounds for the first failed geometry only.

Frozen 257-point grid and 4096 splits/query, fixed before this diagnosis.
Each saved rational proof is independently replayed. A Lipschitz grid-gap
bound can certify the whole query interval [0,1], not only the sampled points.
No query answers, model updates, timing comparison or path-gate replacement.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
from probe_simplex_readout_intervals import integrate
from audit_probe_simplex_readout_intervals import verify
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/'geometry_budget_diagnosis_v1'
    out=base/'simplex_readout_integral_diagnosis_v1';assert not out.exists()
    assert not (base/'evaluation_v3').exists()
    summary=read(inp/'summary.json');assert summary['diagnosis_complete'] and not summary['query_targets_accessed']
    for name,digest in summary['outputs_sha256'].items():assert sha(inp/name)==digest
    hashes=dict(read(inp/'protocol.json')['source_sha256'])
    for directory in ['simplex_interval_selftests_v1','simplex_interval_independent_selftests_v1']:
        tested=read(base/directory/'summary.json');assert tested['passed'] and not tested['query_targets_accessed']
        for name,digest in tested['outputs_sha256'].items():assert sha(base/directory/name)==digest
        hashes.update(tested['source_sha256'])
    hashes[Path(__file__).name]=sha(Path(__file__))
    for name,digest in hashes.items():assert sha(src/name)==digest
    witness=read(inp/'witness.json');certificate=read(inp/'certificate.json')
    with np.load(inp/'geometry.npz') as z:
        raw_facets=z['facets'].copy();raw_weights=z['simplex_probs'].copy()
    raw,inverse=np.unique(raw_facets.reshape(-1,4),axis=0,return_inverse=True)
    assert len(raw)==len(witness['matching'])
    vertices=[tuple(F(t) for t in p) for p in witness['vertices']]
    origin=tuple(F(t) for t in witness['origin'])
    roots=[tuple([origin]+[vertices[witness['matching'][int(i)]] for i in face])
           for face in inverse.reshape(-1,4)]
    p=[F(float(t)) for t in raw_weights];total=sum(p);p=[t/total for t in p]
    r=[F(t)/F(witness['reference_volume']) for t in witness['cone_volumes']]
    assert sum(p)==sum(r)==1 and all(t>=0 for t in p+r)
    delta=[a-b for a,b in zip(p,r)];tv=sum(abs(t) for t in delta)/2
    assert tv==F(certificate['exact_weight_total_variation'])
    position=sum(F(2**(4-j))*F(t) for j,t in enumerate(certificate['exact_maximum_coordinate_displacements']))
    budget=F(1,10**12);grid_intervals=256;max_splits=4096
    query_lipschitz=16*sum(abs(t) for t in delta)
    grid_gap=query_lipschitz/F(2*grid_intervals)
    target=budget-position-grid_gap
    assert target>0
    out.mkdir();(out/'queries').mkdir()
    exclusive_json(out/'inputs.json',dict(
        roots=[[[str(t) for t in vertex] for vertex in simplex] for simplex in roots],
        coefficients=[str(t) for t in delta],query_targets_accessed=False))
    exclusive_json(out/'protocol.json',dict(
        seed=summary['seed'],pattern=summary['pattern'],input_diagnosis_sha256=sha(inp/'summary.json'),
        source_sha256=hashes,query_grid=list(range(grid_intervals+1)),query_denominator=grid_intervals,
        max_splits_per_query=max_splits,exact_nominal_budget=str(budget),
        exact_position_bound=str(position),exact_query_lipschitz=str(query_lipschitz),
        exact_grid_gap_bound=str(grid_gap),exact_point_signed_integral_target=str(target),
        query_targets_accessed=False,audit_gate_passed=False,
        scope='Only first failed geometry; independent exact integral proof replay plus query-interval coverage; all original algorithms, arrays and gates unchanged'))
    counts=Counter();files={};maximum=F(0);worst=[];breaches=[];begin=time.perf_counter()
    for index in range(grid_intervals+1):
        q=F(index,grid_intervals)
        answer=integrate(q,roots,delta,target=target,max_splits=max_splits)
        checked=verify(roots,delta,q,4,answer)
        assert checked['passed']
        lo,hi=F(answer['exact_lower']),F(answer['exact_upper'])
        upper=max(abs(lo),abs(hi));lower=lo if lo>0 else (-hi if hi<0 else F(0))
        actual_lower=max(F(0),lower-position)
        if actual_lower>budget:breaches.append(dict(query_index=index,exact_nominal_lower_bound=str(actual_lower)))
        if upper>maximum:maximum=upper;worst=[index]
        elif upper==maximum:worst.append(index)
        counts[answer['status']]+=1;counts['queries']+=1;counts['splits']+=answer['splits'];counts['leaves']+=answer['leaf_count']
        filename=f'queries/{index:03d}.json'
        exclusive_json(out/filename,dict(query_index=index,proof=answer,independent_check=checked,
            exact_nominal_absolute_upper_bound=str(upper+position),
            exact_nominal_absolute_lower_bound=str(actual_lower),query_targets_accessed=False))
        files[filename]=sha(out/filename)
        print(json.dumps(dict(query=index,total=grid_intervals+1,status=answer['status'],
            splits=answer['splits'],nominal_upper=float(upper+position),
            elapsed=time.perf_counter()-begin,query_targets_accessed=False)),flush=True)
    for name,digest in hashes.items():assert sha(src/name)==digest
    exclusive_json(out/'files.json',files)
    continuous=maximum+position+grid_gap
    result=dict(diagnosis_complete=True,audit_gate_passed=False,query_targets_accessed=False,
        seed=summary['seed'],pattern=summary['pattern'],counts=counts,
        continuous_query_bound_certified=continuous<=budget,
        exact_continuous_absolute_upper_bound=str(continuous),continuous_absolute_upper_bound=float(continuous),
        exact_maximum_point_signed_integral_bound=str(maximum),worst_query_indices=worst,
        grid_gap_bound=float(grid_gap),position_bound=float(position),
        confirmed_nominal_gate_violations=breaches,
        seconds=time.perf_counter()-begin,
        scope='Nominal continuous query interval [0,1], first geometry only; not finite-particle quality or full path acceptance',
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','inputs.json','files.json']})
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
