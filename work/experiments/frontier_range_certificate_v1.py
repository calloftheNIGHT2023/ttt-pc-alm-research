"""414 all-compatible-parameter query ranges from a certified search cover."""
from fractions import Fraction as F
import time
import numpy as np
import partial_support_certificate_v1 as core


def inverse(matrix):
    n=len(matrix);a=[list(map(F,row))+[F(int(i==j)) for j in range(n)] for i,row in enumerate(matrix)]
    assert all(len(row)==2*n for row in a)
    for j in range(n):
        pivot=next((i for i in range(j,n) if a[i][j]),None)
        assert pivot is not None, 'Singular directions cannot bound a full parameter box'
        a[j],a[pivot]=a[pivot],a[j];scale=a[j][j];a[j]=[t/scale for t in a[j]]
        for i in range(n):
            if i==j:continue
            scale=a[i][j];a[i]=[v-scale*w for v,w in zip(a[i],a[j])]
    inv=tuple(tuple(row[n:]) for row in a)
    assert all(sum((F(matrix[i][k])*inv[k][j] for k in range(n)),F(0))==int(i==j) for i in range(n) for j in range(n))
    return inv


def affine_bounds(coeff,offset,intervals):
    lo=hi=F(offset)
    for c,(left,right) in zip(coeff,intervals):
        a,b=c*left,c*right;lo+=min(a,b);hi+=max(a,b)
    return lo,hi


def enclose(q,inv,intervals):
    """Retain affine dependence until crossing; rigorous remainder thereafter."""
    d=len(inv);coef=[F(0)]*d;offset=F(q);error=(F(0),F(0));crossings=0
    for layer in range(d):
        z=[a+b for a,b in zip(coef,inv[layer])]
        low,high=affine_bounds(z,offset,intervals);low+=error[0];high+=error[1]
        if high<=0 or low>=1:
            coef=[F(0)]*d;offset=F(0);error=(F(0),F(0))
        elif 0<=low<=high<=F(1,2):
            coef=[2*t for t in z];offset*=2;error=(2*error[0],2*error[1])
        elif F(1,2)<=low<=high<=1:
            coef=[-2*t for t in z];offset=2-2*offset;error=(-2*error[1],-2*error[0])
        else:
            error=core.mathcore.tent_interval(low,high);coef=[F(0)]*d;offset=F(0);crossings+=1
    low,high=affine_bounds(coef,offset,intervals)
    low,high=max(F(0),low+error[0]),min(F(1),high+error[1]);assert low<=high
    return (low,high),crossings


def fit(sa,sm,q):
    begin=time.perf_counter();upper=core.forest_upper(sa,sm,max_cells=256)
    identity=tuple(tuple(F(int(i==j)) for j in range(4)) for i in range(4))
    prior_box=[(-core.BOUND,core.BOUND)]*4
    proposals=[];skipped=[]
    if not upper['cells'] and upper['reason']!='all_full_and_pending_cells':
        proposals=[dict(token='',inverse=identity,intervals=prior_box)]
    else:
        for record in upper['cells']:
            if record['kind'] in ['negative_constant','inconsistent_parallel_rows']:
                assert record['volume']==0;skipped.append(record['token']);continue
            assert record['kind']=='paired_rows'
            k=record['observations'];region=np.frombuffer(bytes.fromhex(record['token']),np.uint8).reshape(k,4).T.copy()
            a,r=core.constraints(sa['x_observed'][:k],sa['v_observed'][:k],region)
            matrix=[a[i] for i in record['indices']]
            pp=record['parallelotope'];assert not pp['empty']
            inv=inverse(matrix)
            proposals.append(dict(token=record['token'],inverse=inv,intervals=pp['intervals'],zero_volume=pp['volume']==0))
    assert proposals, 'No nonempty cell remains; no compatible-target range asserted'
    prepare_seconds=time.perf_counter()-begin;tick=time.perf_counter()
    ranges=[];per_cell=[];crossings=0
    for proposal in proposals:
        cell=[]
        for point in q:
            bound,count=enclose(F(float(point)),proposal['inverse'],proposal['intervals'])
            cell.append(bound);crossings+=count
        per_cell.append(cell)
    for i,point in enumerate(q):
        cell_union=(min(c[i][0] for c in per_cell),max(c[i][1] for c in per_cell))
        prior,_=enclose(F(float(point)),identity,prior_box)
        ranges.append(core.mathcore.intersect_intervals(cell_union,prior))
    cheap=core.lipschitz_intervals(sa['original_x'],sa['original_v'],q)
    combined=[core.mathcore.intersect_intervals(a,b) for a,b in zip(ranges,cheap)]
    return dict(upper=upper,coordinate_boxes=proposals,empty_cells=skipped,cell_query_ranges=per_cell,
        frontier_intervals=ranges,cheap_intervals=cheap,intersected_intervals=combined,
        crossing_operations=crossings,prepare_seconds=prepare_seconds,reading_seconds=time.perf_counter()-tick,
        total_seconds=time.perf_counter()-begin,query_targets_accessed=False,full_modes_required=False,
        posterior_mass_required=False,global_bp_used=False,new_global_lp_used=False,
        exact_rational_bounds=True,archived_search_component_only=True)
