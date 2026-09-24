"""Exact fan certificate permitting redundant saved vertices to collapse.

All exact vertices must still be represented. Both raw and mapped boundary
chains close; every cone is nonnegative, every nonzero outer facet lies on
the true boundary, and total cone volume equals independent true volume.
Those conditions imply unit interior multiplicity; bijection is unnecessary.
"""
from collections import defaultdict
from fractions import Fraction as F
import time
import numpy as np
from probe_exact_polytope_certificate import normalize, all_vertices, reference_volume, orientation, rank, determinant, dot


def certify(poly, rows, rhs):
    started=time.perf_counter(); rows,rhs=normalize(rows,rhs)
    exact, work=all_vertices(rows,rhs)
    refvol, more=reference_volume(exact,rows,rhs); work.update(more)
    raw,inverse=np.unique(poly['facets'].reshape(-1,4),axis=0,return_inverse=True)
    simplices=inverse.reshape(-1,4).tolist(); signs,nridges=orientation(simplices)
    center=[F(float(t)) for t in poly['center']]; scale=[F(float(t)) for t in poly['scale']]
    assert all(t>0 for t in scale)
    physical=[tuple(c+s*F(float(t)) for c,s,t in zip(center,scale,row)) for row in raw]
    origin=tuple(c+s*F(float(t)) for c,s,t in zip(center,scale,poly['interior']))
    assert all(dot(a,origin)<b for a,b in zip(rows,rhs)), 'Cone origin is not strictly inside exact constraints'
    matching=[]; displacement=[F(0)]*4
    for point in physical:
        distances=[max(abs(a-b) for a,b in zip(point,v)) for v in exact]
        idx=min(range(len(exact)),key=lambda i:distances[i]); matching.append(idx)
        for j in range(4): displacement[j]=max(displacement[j],abs(point[j]-exact[idx][j]))
    assert set(matching)==set(range(len(exact))), 'Some exact vertices are not represented'
    # Distinct saved floating vertices may collapse to the same exact vertex.
    # The mapped simplicial chain remains closed, even with such identifications.
    mapped_boundary=defaultdict(int)
    for simplex, sign in zip(simplices, signs):
        for omitted in range(4):
            ridge=[matching[simplex[j]] for j in range(4) if j != omitted]
            if len(set(ridge)) < 3: continue
            inversions=sum(ridge[i]>ridge[j] for i in range(3) for j in range(i+1,3))
            mapped_boundary[tuple(sorted(ridge))]+=sign*(-1)**(omitted+inversions)
    assert all(value==0 for value in mapped_boundary.values()), 'Mapped chain has a nonzero boundary'
    moved=[[exact[matching[i]] for i in s] for s in simplices]
    determinants=[determinant([[a-b for a,b in zip(v,origin)] for v in face]) for face in moved]
    signed=sum(s*d for s,d in zip(signs,determinants))
    if signed < 0: signs=[-s for s in signs]
    volumes=[s*d/24 for s,d in zip(signs,determinants)]
    assert all(v>=0 for v in volumes), 'Exact mapped fan has negative cones'
    supporting=[]
    for face, volume in zip(moved,volumes):
        if not volume:
            assert rank(face)<3, 'Zero cone is not a degenerate boundary face'
            supporting.append(None)
        else:
            hits=[i for i,(a,b) in enumerate(zip(rows,rhs)) if all(dot(a,v)==b for v in face)]
            assert hits, 'Mapped facet is not on a true constraint boundary'
            supporting.append(hits[0])
    assert sum(volumes)==refvol, 'Mapped fan volume differs from independent face-lattice volume'
    reference_weights=[v/refvol for v in volumes]
    stored_weights=[F(float(t)) for t in poly['simplex_probs']]
    assert all(t>=0 for t in stored_weights)
    weight_sum=sum(stored_weights); stored_weights=[t/weight_sum for t in stored_weights]
    weight_tv=sum(abs(a-b) for a,b in zip(stored_weights,reference_weights))/2
    position_bound=sum(F(2**(4-j))*d for j,d in enumerate(displacement))
    error_bound=weight_tv+position_bound
    stored_volume=F(float(poly['volume'])); relative_volume=abs(stored_volume-refvol)/refvol
    result=dict(exact_constraints=len(rows),exact_vertices=len(exact),saved_vertices=len(raw),
        complete_vertex_enumeration=True,exact_boundary_chain_closed=True,connected_boundary=True,
        exact_vertices_all_represented=True,vertex_mapping_bijective=len(set(matching))==len(matching),
        collapsed_saved_vertices=len(matching)-len(exact),exact_mapped_boundary_chain_closed=True,
        exact_origin_strict=True,all_mapped_nonzero_facets_on_true_boundary=True,
        mapped_negative_cones=0,mapped_zero_cones=sum(v==0 for v in volumes),
        mapped_volume_equals_independent_reference=True,
        exact_reference_volume=str(refvol),reference_volume=float(refvol),
        saved_volume_relative_error=float(relative_volume),
        exact_weight_total_variation=str(weight_tv),weight_total_variation=float(weight_tv),
        maximum_coordinate_displacements=[float(t) for t in displacement],
        exact_maximum_coordinate_displacements=[str(t) for t in displacement],
        nominal_readout_expectation_error_bound=float(error_bound),
        exact_nominal_readout_expectation_error_bound=str(error_bound),
        squared_risk_of_mean_difference_bound=float(2*error_bound),
        proof_scope='Exact true-constraint polytope and unchanged saved-simplex nominal continuous mixture; finite RNG/particle/float arithmetic is separately replayed, not certified here',
        ridges=nridges,work=work,seconds=time.perf_counter()-started)
    witness=dict(rows=[list(r) for r in rows],rhs=[str(r) for r in rhs],
        vertices=[[str(t) for t in v] for v in exact],matching=matching,signs=signs,
        supporting_constraints=supporting,cone_volumes=[str(t) for t in volumes],
        origin=[str(t) for t in origin],reference_volume=str(refvol))
    if len(matching)>len(exact):
        witness['mapping_type']='surjective_vertex_collapse'
        witness['raw_vertex_fibers']=[[i for i,t in enumerate(matching) if t==j] for j in range(len(exact))]
    return result,witness
