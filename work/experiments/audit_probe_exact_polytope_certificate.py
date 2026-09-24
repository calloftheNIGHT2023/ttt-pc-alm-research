"""Independent OLD-case rational elimination and parallelotope-volume audit.

Does not import the certificate's constraints, enumerator, determinant or
volume implementation. It verifies the one diagnosed old geometry, not every
future polytope, and does not release the query-evaluation gate.
"""
import argparse
from collections import defaultdict
from fractions import Fraction as F
from itertools import combinations
import json
import math
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


def gauss(a,b=None):
    n=len(a); mat=[[F(t) for t in row]+([] if b is None else [F(b[i])]) for i,row in enumerate(a)]
    determinant=F(1)
    for j in range(n):
        pivot=next((i for i in range(j,n) if mat[i][j]),None)
        if pivot is None: return F(0),None
        if pivot != j: mat[j],mat[pivot]=mat[pivot],mat[j]; determinant=-determinant
        value=mat[j][j]; determinant*=value; mat[j]=[t/value for t in mat[j]]
        for i in range(n):
            if i != j and mat[i][j]:
                value=mat[i][j]; mat[i]=[x-value*y for x,y in zip(mat[i],mat[j])]
    return determinant,None if b is None else tuple(row[-1] for row in mat)


def independent_constraints(x,v,pattern):
    """Evaluate each prescribed branch at 0 and four basis parameter vectors."""
    coefficients={}; slopes=[0,2,-2,0]; offsets=[0,0,2,0]
    states=[[F(float(t))]*5 for t in x]
    def add(a,b):
        assert all(F(t).denominator==1 for t in a)
        a=tuple(int(t) for t in a); d=math.gcd(*a)
        if not d:
            assert b>=0; return
        a=tuple(t//d for t in a); b=F(b)/d
        coefficients[a]=min(coefficients.get(a,b),b)
    for j in range(4):
        for i in range(len(x)):
            z=[t+int(k==j+1) for k,t in enumerate(states[i])]
            row=[t-z[0] for t in z[1:]]; branch=int(pattern[j,i])
            if branch < 3: add(row,[F(0),F(1,2),F(1)][branch]-z[0])
            if branch > 0: add([-t for t in row],z[0]-[F(0),F(1,2),F(1)][branch-1])
            states[i]=[slopes[branch]*t+offsets[branch] for t in z]
    for i in range(len(x)):
        values=states[i]; row=[t-values[0] for t in values[1:]]
        add(row,F(float(v[i]))+F(.001)-values[0])
        add([-t for t in row],values[0]-F(float(v[i]))+F(.001))
    for j in range(4):
        for s in [-1,1]: add([s if k==j else 0 for k in range(4)],F(.12))
    return sorted(coefficients.items())


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve(); base=root/'results/probe_credit_confirmation'
    inp=base/'exact_polytope_certificate_preflight'; geo=base/'boundary_diagnosis_preflight'
    out=base/'exact_polytope_certificate_independent_audit'; assert not out.exists()
    started=time.perf_counter(); result=json.loads((inp/'summary.json').read_text())
    witness=json.loads((inp/'witness.json').read_text())
    assert sha(inp/'witness.json')==result['witness_sha256'] and sha(geo/'geometry.npz')==result['geometry_sha256']
    with np.load(geo/'geometry.npz') as z: poly={k:z[k].copy() for k in z.files}
    pattern=np.frombuffer(bytes.fromhex(result['pattern']),dtype=np.uint8).reshape(4,4)
    independent=independent_constraints(poly['x_observed'],poly['v_observed'],pattern)
    rows=[tuple(row) for row in witness['rows']]; rhs=[F(t) for t in witness['rhs']]
    assert independent==list(zip(rows,rhs))
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    vertices=set(); combinations_checked=0
    for indices in combinations(range(len(rows)),4):
        combinations_checked+=1
        determinant,point=gauss([rows[i] for i in indices],[rhs[i] for i in indices])
        if not determinant: continue
        if all(dot(a,point)<=b for a,b in zip(rows,rhs)): vertices.add(point)
    recorded=[tuple(F(t) for t in v) for v in witness['vertices']]
    assert set(recorded)==vertices and len(recorded)==len(vertices)==16
    # In this OLD fixture exactly four opposing pairs bound an affine box.
    active=[]
    for a,b in zip(rows,rhs):
        points=[v for v in vertices if dot(a,v)==b]
        if len(points)<4: continue
        edges=[[x-y for x,y in zip(v,points[0])] for v in points[1:]]
        if any(gauss([[edge[j] for j in cols] for edge in selected])[0]
            for selected in combinations(edges,3) for cols in combinations(range(4),3)):
            active.append((a,b))
    assert len(active)==8
    pairs=defaultdict(dict)
    for a,b in active:
        s=1 if next(t for t in a if t)>0 else -1
        canonical=tuple(s*t for t in a); pairs[canonical][s]=b
    assert len(pairs)==4 and all(set(v)=={-1,1} for v in pairs.values())
    matrix=list(sorted(pairs)); widths=[pairs[a][1]+pairs[a][-1] for a in matrix]
    assert all(t>0 for t in widths)
    exact_volume=math.prod(widths)/abs(gauss(matrix)[0])
    assert exact_volume==F(witness['reference_volume'])==F(result['exact_reference_volume'])
    raw,indices=np.unique(poly['facets'].reshape(-1,4),axis=0,return_inverse=True); simplices=indices.reshape(-1,4)
    matching=witness['matching']; assert set(matching)==set(range(16)) and len(matching)==16
    origin=tuple(F(t) for t in witness['origin']); assert all(dot(a,origin)<b for a,b in zip(rows,rhs))
    expected_origin=tuple(F(float(c))+F(float(s))*F(float(y)) for c,s,y in zip(poly['center'],poly['scale'],poly['interior']))
    assert origin==expected_origin
    boundary=defaultdict(int); volumes=[]
    for k,simplex in enumerate(simplices):
        points=[recorded[matching[i]] for i in simplex]
        volume=witness['signs'][k]*gauss([[x-y for x,y in zip(v,origin)] for v in points])[0]/24
        assert volume>=0 and volume==F(witness['cone_volumes'][k]); volumes.append(volume)
        if volume:
            c=witness['supporting_constraints'][k]; assert all(dot(rows[c],v)==rhs[c] for v in points)
        else:
            edges=[[x-y for x,y in zip(v,points[0])] for v in points[1:]]
            assert all(gauss([[edge[j] for j in cols] for edge in edges])[0]==0 for cols in combinations(range(4),3))
        for omit in range(4):
            ridge=[int(simplex[i]) for i in range(4) if i!=omit]
            swaps=sum(ridge[i]>ridge[j] for i in range(3) for j in range(i+1,3))
            boundary[tuple(sorted(ridge))]+=witness['signs'][k]*(-1)**(omit+swaps)
    assert all(v==0 for v in boundary.values()) and sum(volumes)==exact_volume
    physical=[tuple(F(float(c))+F(float(s))*F(float(y)) for c,s,y in zip(poly['center'],poly['scale'],row)) for row in raw]
    shifts=[max(abs(row[j]-recorded[matching[i]][j]) for i,row in enumerate(physical)) for j in range(4)]
    assert [str(t) for t in shifts]==result['exact_maximum_coordinate_displacements']
    stored=[F(float(t)) for t in poly['simplex_probs']]; assert len(stored)==len(volumes)
    total=sum(stored); stored=[t/total for t in stored]
    variation=sum(abs(a-b/exact_volume) for a,b in zip(stored,volumes))/2
    error=variation+sum(F(s)*d for s,d in zip([16,8,4,2],shifts))
    assert variation==F(result['exact_weight_total_variation']) and error==F(result['exact_nominal_readout_expectation_error_bound'])
    ans=dict(passed=True,query_targets_accessed=False,audit_gate_passed=False,
        scope='Independent one-old-fixture constraint basis evaluations, rational Gauss-Jordan enumeration, affine-box analytic volume, mesh boundary and coupling bound',
        equation_subsets_checked=combinations_checked,vertices=16,active_facets=8,
        independently_derived_volume=str(exact_volume),mesh_cones=len(volumes),
        exact_zero_cones=sum(v==0 for v in volumes),nominal_readout_bound=float(error),
        seconds=time.perf_counter()-started,
        certificate_summary_sha256=sha(inp/'summary.json'),witness_sha256=sha(inp/'witness.json'),
        source_sha256=sha(Path(__file__)))
    out.mkdir(); exclusive_json(out/'summary.json',ans); print(json.dumps(ans),flush=True)


if __name__=='__main__': main()
