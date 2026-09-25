"""413 exact inner mass and a complete search-forest mass upper bound.

All numeric geometry is only a proposal. This component has no target labels,
reference completion, or claim that its uncapped runtime fits a deadline.
"""
from fractions import Fraction as F
import time
import numpy as np
import certified_partial_readout_v1 as mathcore
import online_credit_branch_search_v1 as local
import complete_credit_mode_geometry_v1 as classifier
import enumerate_support_modes as exact_geometry

BOUND=F(.12)
EPS=F(.001)
PRIOR=(2*BOUND)**4


def exact_rank(rows):
    if not rows:return 0
    a=[list(map(F,row)) for row in rows];r=0
    for j in range(len(a[0])):
        pivot=next((i for i in range(r,len(a)) if a[i][j]),None)
        if pivot is None:continue
        a[r],a[pivot]=a[pivot],a[r];value=a[r][j]
        a[r]=[t/value for t in a[r]]
        for i in range(r+1,len(a)):
            ratio=a[i][j];a[i]=[v-ratio*w for v,w in zip(a[i],a[r])]
        r+=1
        if r==len(a):break
    return r


def constraints(x,v,region):
    assert region.shape==(4,len(x)) and len(x)==len(v)
    a,r=exact_geometry.constraints(x,v,region,exact=True)
    ba,br=classifier.box_rows(4)
    return a+ba,r+br


def cell_upper(a,rhs):
    """Greedy basis proposes; exact existing paired rows certify the volume."""
    groups={}
    for i,(row,r) in enumerate(zip(a,rhs)):
        if not any(row):
            if r<0:return dict(volume=F(0),kind='negative_constant',row=i)
            continue
        first=next(t for t in row if t)
        key=tuple(t/first for t in row);group=groups.setdefault(key,dict(low=None,high=None,index=i))
        side='high' if first>0 else 'low';value=r/first
        previous=group[side]
        group[side]=value if previous is None else (min if side=='high' else max)(previous,value)
    options=[]
    for key,g in groups.items():
        if g['low'] is None or g['high'] is None:continue
        if g['low']>g['high']:return dict(volume=F(0),kind='inconsistent_parallel_rows',direction=key,low=g['low'],high=g['high'])
        score=float((g['high']-g['low'])**2/mathcore.dot(key,key))
        options.append((score,key,g['index']))
    selected=[];indices=[]
    for _,direction,index in sorted(options):
        if exact_rank(selected+[direction])>len(selected):
            selected.append(direction);indices.append(index)
        if len(selected)==4:break
    assert len(selected)==4, 'Explicit prior box must supply a complete basis'
    bound=mathcore.row_parallelotope_upper(a,rhs,indices)
    assert bound is not None and bound['volume']>=0
    return dict(volume=min(PRIOR,bound['volume']),kind='paired_rows',indices=indices,parallelotope=bound)


def forest_cover(arrays,metadata):
    """Antichain of prefixes covering returned leaves and every pending subtree.

The caller must verify the search's pruning and stack accounting separately.
Byte tokens are observation-major groups of four branch identifiers.
"""
    order=arrays['support_order'];n=len(order);tokens=[]
    for region in arrays['regions']:
        tokens.append(region[:,order].T.copy().tobytes())
    for i,pm in enumerate(metadata['pending']):
        k,start=pm['k'],pm['start'];parents=arrays['pending_'+str(i)]
        count=len(arrays['language_'+str(k-1)])
        assert count>0 and 0<=start<len(parents)*count and parents.shape[1:]==(4,k-1)
        tokens.extend(parent.T.copy().tobytes() for parent in parents[start//count:])
    accepted=set()
    for token in sorted(set(tokens),key=lambda t:(len(t),t)):
        assert len(token)%4==0 and len(token)<=4*n
        if not any(token[:k] in accepted for k in range(0,len(token)+1,4)):
            accepted.add(token)
    return sorted(accepted,key=lambda t:(len(t),t))


def forest_upper(arrays,metadata,max_cells=256):
    begin=time.perf_counter();cover=forest_cover(arrays,metadata)
    if b'' in cover or len(cover)>max_cells:
        return dict(volume=PRIOR,cover_tokens=[t.hex() for t in cover],cells=[],
            reason='root_cover' if b'' in cover else 'safe_cell_cap_fallback',seconds=time.perf_counter()-begin)
    cells=[];total=F(0)
    x,v=arrays['x_observed'],arrays['v_observed']
    for token in cover:
        k=len(token)//4;region=np.frombuffer(token,np.uint8).reshape(k,4).T.copy()
        a,r=constraints(x[:k],v[:k],region);bound=cell_upper(a,r)
        total+=bound['volume'];cells.append(dict(token=token.hex(),observations=k,**bound))
    return dict(volume=min(PRIOR,total),cover_tokens=[t.hex() for t in cover],cells=cells,
        reason='all_full_and_pending_cells',seconds=time.perf_counter()-begin)


def lipschitz_intervals(x,v,q):
    xx,vv=tuple(map(F,x)),tuple(map(F,v));out=[]
    assert len(xx)==len(vv)>0
    for point in map(F,q):
        low=max([F(0)]+[t-EPS-16*abs(point-s) for s,t in zip(xx,vv)])
        high=min([F(1)]+[t+EPS+16*abs(point-s) for s,t in zip(xx,vv)])
        assert low<=high, 'Observed support incompatible with the declared Lipschitz envelope'
        out.append((low,high))
    return out


def cone_proposal(poly,a,r):
    facets=poly['center'][None,None,:]+poly['scale'][None,None,:]*poly['facets']
    vertices,inverse=np.unique(facets.reshape(-1,4),axis=0,return_inverse=True)
    faces=inverse.reshape(-1,4)
    anchor=poly['center']+poly['scale']*poly['interior']
    proof=classifier.point_certificate(a,r,anchor)
    assert proof['strict_interior']
    cones=mathcore.certified_cones(vertices,faces,anchor,a,r)
    return cones,dict(vertices=vertices,faces=faces,anchor=anchor)


def fit(x,v,q,search_arrays,search_metadata,*,max_modes=4,max_cover_cells=256):
    begin=time.perf_counter()
    assert np.array_equal(x,search_arrays['original_x']) and np.array_equal(v,search_arrays['original_v'])
    upper=forest_upper(search_arrays,search_metadata,max_cover_cells)
    zbar=upper['volume'];assert zbar>0, 'No positive posterior mass upper bound'
    tick=time.perf_counter();cheap=lipschitz_intervals(x,v,q);cheap_seconds=time.perf_counter()-tick
    keys=sorted(region.tobytes().hex() for region in search_arrays['regions'])
    assert len(keys)==len(set(keys))
    mass=F(0);lo=[F(0)]*len(q);hi=[F(0)]*len(q)
    saved={};records=[];checkpoints=[];geometry_seconds=cone_seconds=moment_seconds=0.
    previous=[(F(0),F(1)) for _ in q]
    with local.no_bp(True):
        for index,key in enumerate(keys[:max_modes]):
            tick=time.perf_counter();poly,note=local.explicit_geometry(x,v,key)
            geometry_seconds+=time.perf_counter()-tick
            _,a,r,_,_=classifier.matrices(x,v,key)
            note_record=dict(key=key,classification=note)
            if poly is not None:
                tick=time.perf_counter();cone,proposal=cone_proposal(poly,a,r)
                cone_seconds+=time.perf_counter()-tick
                saved.update({f'mode_{index}_{k}':val for k,val in proposal.items()})
                note_record.update(cone=cone,numerical_volume_diagnostic_only=poly['volume'])
                mass+=cone['mass'];assert mass<=zbar
                tick=time.perf_counter()
                for j,point in enumerate(q):
                    for simplex in cone['simplices']:
                        mm=mathcore.moment_bounds(simplex,F(float(point)),splits=0)
                        lo[j]+=mm['lower'];hi[j]+=mm['upper']
                moment_seconds+=time.perf_counter()-tick
            bounds=[mathcore.posterior_interval(mass,l,h,zbar) for l,h in zip(lo,hi)]
            assert all(p[0]<=b[0]<=b[1]<=p[1] for p,b in zip(previous,bounds))
            previous=bounds
            checkpoints.append(dict(mode_index=index,key=key,mass=mass,mass_fraction=mass/zbar,
                lower_moment=lo.copy(),upper_moment=hi.copy(),posterior_intervals=bounds,
                prior_only_intervals=[mathcore.posterior_interval(mass,l,h,PRIOR) for l,h in zip(lo,hi)],
                intersected_intervals=[mathcore.intersect_intervals(b,c) for b,c in zip(bounds,cheap)],
                cumulative_seconds=time.perf_counter()-begin))
            records.append(note_record)
    final=previous
    intersection=[mathcore.intersect_intervals(b,c) for b,c in zip(final,cheap)]
    arrays=dict(x=x.copy(),v=v.copy(),q=q.copy(),**saved)
    metadata=dict(upper=upper,cheap_intervals=cheap,cheap_seconds=cheap_seconds,
        total_inner_mass=mass,total_upper=zbar,covered_fraction=mass/zbar,
        posterior_intervals=final,intersected_intervals=intersection,records=records,checkpoints=checkpoints,
        geometry_seconds=geometry_seconds,cone_seconds=cone_seconds,moment_seconds=moment_seconds,
        max_modes=max_modes,max_cover_cells=max_cover_cells,full_candidate_modes=len(keys),processed_modes=len(records),
        total_seconds=time.perf_counter()-begin,query_targets_accessed=False,
        geometry_global_lp_used=bool(records),global_bp_used=False,
        source_search_seconds_excluded=True,hard_end_to_end_deadline=False,
        named_returned_numeric_array_bytes=sum(value.nbytes for value in arrays.values()),
        exact_objects_state_included_in_numeric_bytes=False)
    return arrays,metadata
