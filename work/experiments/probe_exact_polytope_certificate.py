"""Offline exact H-polytope certificate for an unchanged floating sampler.

All model inputs/constants are interpreted as binary rationals. Exhaustive
four-plane intersections establish the complete vertex set. An independent
face-lattice barycentric subdivision supplies the reference volume. No teacher
query, fit, parameter update, tolerance relaxation, or online credit is added.
"""
import argparse
from collections import defaultdict, deque
from fractions import Fraction as F
from functools import lru_cache
from itertools import combinations, permutations
import json
import math
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


def determinant(a):
    n = len(a)
    answer = 0
    for p in permutations(range(n)):
        value = (-1)**sum(p[i] > p[j] for i in range(n) for j in range(i+1,n))
        for i in range(n): value *= a[i][p[i]]
        answer += value
    return answer


def dot(a,b): return sum(x*y for x,y in zip(a,b))


def normalize(rows, rhs):
    strongest = {}
    for row, value in zip(rows,rhs):
        row = tuple(int(t) for t in row); value = F(value)
        divisor = math.gcd(*row)
        if not divisor:
            assert value >= 0, 'Exact constant infeasibility'
            continue
        row = tuple(t//divisor for t in row); value /= divisor
        strongest[row] = min(strongest.get(row,value),value)
    order = sorted(strongest)
    return tuple(order), tuple(strongest[row] for row in order)


def observed_constraints(x, v, key, bound=F(.12), noise=F(.001)):
    pattern = np.frombuffer(bytes.fromhex(key),dtype=np.uint8).reshape(4,len(x))
    rows, rhs = [], []
    coeff = [[0]*4 for _ in x]; offsets = [F(float(t)) for t in x]
    knots = [F(0),F(1,2),F(1)]; slopes=[0,2,-2,0]; shifts=[0,0,2,0]
    for j in range(4):
        for i in range(len(x)):
            z = coeff[i].copy(); z[j] += 1; k = int(pattern[j,i]); off=offsets[i]
            if k > 0: rows.append([-t for t in z]); rhs.append(off-knots[k-1])
            if k < 3: rows.append(z.copy()); rhs.append(knots[k]-off)
            coeff[i] = [slopes[k]*t for t in z]; offsets[i] = slopes[k]*off+shifts[k]
    for i in range(len(x)):
        rows.extend([coeff[i],[-t for t in coeff[i]]])
        rhs.extend([F(float(v[i]))+noise-offsets[i],offsets[i]-F(float(v[i]))+noise])
    for j in range(4):
        row=[0]*4; row[j]=1; rows.extend([row,[-t for t in row]]); rhs.extend([bound,bound])
    return normalize(rows,rhs)


@lru_cache(maxsize=65536)
def inverse_numerator(matrix):
    det = determinant(matrix)
    if not det: return None
    adj = tuple(tuple((-1)**(i+j)*determinant([[matrix[r][c] for c in range(4) if c != i]
        for r in range(4) if r != j]) for j in range(4)) for i in range(4))
    return adj, det


def all_vertices(rows, rhs):
    # These explicit two-sided coordinate bounds prove boundedness; enumerating
    # vertices of a merely unbounded polyhedron would not prove completeness.
    for j in range(4):
        for sign in [-1,1]:
            e=tuple(sign if i==j else 0 for i in range(4)); assert e in rows
    denominator=math.lcm(*(r.denominator for r in rhs))
    integers=[int(r*denominator) for r in rhs]
    vertices=set(); nonsingular=0; tested=0
    for indices in combinations(range(len(rows)),4):
        tested += 1; inverse=inverse_numerator(tuple(rows[i] for i in indices))
        if inverse is None: continue
        nonsingular += 1; adj, det=inverse
        numerator=[sum(a*integers[k] for a,k in zip(row,indices)) for row in adj]
        if det < 0: det=-det; numerator=[-t for t in numerator]
        if any(dot(row,numerator) > value*det for row,value in zip(rows,integers)): continue
        vertices.add(tuple(F(t,det*denominator) for t in numerator))
    assert vertices
    return sorted(vertices),dict(intersections_tested=tested,nonsingular_intersections=nonsingular)


def rank(points):
    if not points: return -1
    a=[[x-y for x,y in zip(point,points[0])] for point in points[1:]]; r=0
    for j in range(4):
        pivot=next((i for i in range(r,len(a)) if a[i][j]),None)
        if pivot is None: continue
        a[r],a[pivot]=a[pivot],a[r]; scale=a[r][j]; a[r]=[t/scale for t in a[r]]
        for i in range(r+1,len(a)):
            if a[i][j]:
                scale=a[i][j]; a[i]=[x-scale*y for x,y in zip(a[i],a[r])]
        r+=1
        if r==4: break
    return r


def reference_volume(vertices, rows, rhs):
    """Barycentric flags of the exact face lattice, independent of saved mesh."""
    @lru_cache(None)
    def dimension(face): return rank([vertices[i] for i in face])
    root=tuple(range(len(vertices))); assert dimension(root)==4
    boundary=set()
    for a,b in zip(rows,rhs):
        face=tuple(i for i,v in enumerate(vertices) if dot(a,v)==b)
        if dimension(face)==3: boundary.add(face)
    assert boundary
    @lru_cache(None)
    def center(face): return tuple(sum(vertices[i][j] for i in face)/len(face) for j in range(4))
    @lru_cache(None)
    def children(face):
        d=dimension(face); selected=set(face); ans=set()
        for facet in boundary:
            part=tuple(sorted(selected.intersection(facet)))
            if part != face and dimension(part)==d-1: ans.add(part)
        assert ans
        return tuple(sorted(ans))
    def flags(face):
        if dimension(face)==0:
            yield [center(face)]; return
        for child in children(face):
            for tail in flags(child): yield [center(face)]+tail
    volume=F(0); count=0
    for flag in flags(root):
        assert len(flag)==5
        piece=abs(determinant([[x-y for x,y in zip(row,flag[0])] for row in flag[1:]]))/24
        assert piece > 0
        volume+=piece; count+=1
    return volume,dict(reference_facets=len(boundary),barycentric_four_simplices=count,
        face_sets_checked=dimension.cache_info().currsize)


def orientation(simplices):
    ridges=defaultdict(list)
    assert len({tuple(sorted(s)) for s in simplices})==len(simplices), 'Repeated facet'
    for i,s in enumerate(simplices):
        assert len(set(s))==4
        for j in range(4):
            ridge=tuple(s[k] for k in range(4) if k != j)
            parity=(-1)**sum(ridge[a]>ridge[b] for a in range(3) for b in range(a+1,3))
            ridges[tuple(sorted(ridge))].append((i,(-1)**j*parity))
    assert all(len(v)==2 for v in ridges.values()), 'Boundary has missing/nonmanifold ridges'
    adjacency=defaultdict(list)
    for pairs in ridges.values():
        (a,ca),(b,cb)=pairs; adjacency[a].append((b,-ca*cb)); adjacency[b].append((a,-ca*cb))
    signs={0:1}; queue=deque([0])
    while queue:
        a=queue.popleft()
        for b,relative in adjacency[a]:
            desired=signs[a]*relative
            if b in signs: assert signs[b]==desired, 'Nonorientable facet complex'
            else: signs[b]=desired; queue.append(b)
    assert len(signs)==len(simplices), 'Disconnected or repeated boundary components'
    assert all(sum(signs[i]*c for i,c in pairs)==0 for pairs in ridges.values())
    return [signs[i] for i in range(len(simplices))], len(ridges)


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
    assert len(set(matching))==len(matching)==len(exact), 'No complete bijective rounded-to-exact vertex matching'
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
    return result,witness


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve(); base=root/'results/probe_credit_confirmation'
    inp=base/'boundary_diagnosis_preflight'; out=base/'exact_polytope_certificate_preflight'
    assert not out.exists()
    prior=json.loads((inp/'summary.json').read_text())
    for name,digest in prior['outputs_sha256'].items(): assert sha(inp/name)==digest
    with np.load(inp/'geometry.npz') as z: poly={k:z[k].copy() for k in z.files}
    protocol=json.loads((inp/'protocol.json').read_text())
    rows,rhs=observed_constraints(poly['x_observed'],poly['v_observed'],protocol['pattern'])
    result,witness=certify(poly,rows,rhs)
    out.mkdir(); exclusive_json(out/'witness.json',witness)
    result.update(certificate_completed=True,audit_gate_passed=False,query_targets_accessed=False,
        seed=protocol['seed'],pattern=protocol['pattern'],
        source_sha256=sha(Path(__file__)),geometry_sha256=sha(inp/'geometry.npz'),
        witness_sha256=sha(out/'witness.json'))
    exclusive_json(out/'summary.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__': main()
