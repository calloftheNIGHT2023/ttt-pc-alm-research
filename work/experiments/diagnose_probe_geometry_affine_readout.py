"""Exact nominal-mean differences on certified common affine query branches.

Only query coordinates are used, never teacher answers. At each layer the
preactivation is affine throughout the convex hull of both vertex sets. An
exact min/max test at every vertex certifies a single tent branch over that
hull; otherwise the query is explicitly left uncertified, not extrapolated.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def affine_on_hull(q, points):
    coefficient = [F(0)]*4
    offset = q
    layers = []
    for layer in range(4):
        coefficient[layer] += 1
        pre = [dot(coefficient, point)+offset for point in points]
        lo, hi = min(pre), max(pre)
        if hi <= 0:
            branch, slope, intercept = 0, 0, 0
        elif lo >= 1:
            branch, slope, intercept = 3, 0, 0
        elif lo >= 0 and hi <= F(1,2):
            branch, slope, intercept = 1, 2, 0
        elif lo >= F(1,2) and hi <= 1:
            branch, slope, intercept = 2, -2, 2
        else:
            return None
        layers.append(dict(layer=layer, exact_min=str(lo), exact_max=str(hi), branch=branch))
        coefficient = [slope*t for t in coefficient]
        offset = slope*offset+intercept
    # A separate direct pointwise recurrence must agree at all hull generators.
    for point in points:
        value = q
        for bias in point:
            z = value+bias
            value = 2*max(F(0), z)-4*max(F(0), z-F(1,2))+2*max(F(0), z-1)
        assert value == dot(coefficient, point)+offset
    return coefficient, offset, layers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    base = root/'results/probe_credit_confirmation'
    inp = base/'geometry_budget_diagnosis_v1'
    out = base/'geometry_affine_readout_diagnosis_v1'
    assert not out.exists()
    diagnosis = read(inp/'summary.json')
    for name, digest in diagnosis['outputs_sha256'].items():
        assert sha(inp/name) == digest
    witness = read(inp/'witness.json')
    with np.load(inp/'geometry.npz') as z:
        poly = {k: z[k].copy() for k in z.files}
    raw, inverse = np.unique(poly['facets'].reshape(-1,4), axis=0, return_inverse=True)
    faces = inverse.reshape(-1,4).tolist()
    center = [F(float(t)) for t in poly['center']]
    scale = [F(float(t)) for t in poly['scale']]
    physical = [tuple(c+s*F(float(t)) for c,s,t in zip(center,scale,row)) for row in raw]
    origin = tuple(F(t) for t in witness['origin'])
    assert origin == tuple(c+s*F(float(t)) for c,s,t in zip(center,scale,poly['interior']))
    exact = [tuple(F(t) for t in row) for row in witness['vertices']]
    mapped = [exact[i] for i in witness['matching']]
    weights = [F(float(t)) for t in poly['simplex_probs']]
    weight_sum = sum(weights)
    weights = [t/weight_sum for t in weights]
    reference = [F(v)/F(witness['reference_volume']) for v in witness['cone_volumes']]
    assert sum(weights) == sum(reference) == 1
    def mean(vertices, probs):
        return [sum((w*(origin[j]+sum(vertices[i][j] for i in face))/5
                     for face,w in zip(faces,probs)), F(0)) for j in range(4)]
    physical_mean = mean(physical, weights)
    reference_mean = mean(mapped, reference)
    shift = [a-b for a,b in zip(physical_mean,reference_mean)]
    hull = sorted(set(physical+exact+[origin]))
    rows = []
    skipped = []
    for index in range(257):
        q = F(index,256)
        answer = affine_on_hull(q,hull)
        if answer is None:
            skipped.append(index)
            continue
        coefficient, offset, layers = answer
        def integrated(vertices, probs):
            return sum((w*(dot(coefficient,origin)+offset+
                sum(dot(coefficient,vertices[i])+offset for i in face))/5
                for face,w in zip(faces,probs)), F(0))
        left, right = integrated(physical,weights), integrated(mapped,reference)
        difference = dot(coefficient,shift)
        assert left-right == difference and 0 <= left <= 1 and 0 <= right <= 1
        rows.append(dict(query_index=index, query_coordinate=str(q),
            coefficients=[str(t) for t in coefficient], offset=str(offset), layers=layers,
            exact_physical_nominal_mean=str(left), exact_true_polytope_mean=str(right),
            exact_signed_difference=str(difference), absolute_difference=float(abs(difference)),
            exceeds_existing_budget=abs(difference)>F(1,10**12)))
    assert rows
    largest = max(rows,key=lambda r:abs(F(r['exact_signed_difference'])))
    out.mkdir()
    exclusive_json(out/'witnesses.json',rows)
    exclusive_json(out/'protocol.json',dict(input_summary_sha256=sha(inp/'summary.json'),
        source_sha256=sha(Path(__file__)),query_indices=list(range(257)),
        query_targets_accessed=False,scope='Exact nominal means only where every layer is affine on the full convex hull; all other queries are left undecided'))
    summary = dict(diagnosis_complete=True,audit_gate_passed=False,query_targets_accessed=False,
        seed=diagnosis['seed'],pattern=diagnosis['pattern'],
        certified_affine_queries=len(rows),uncertified_queries=len(skipped),
        exact_mean_parameter_shift=[str(t) for t in shift],
        mean_parameter_shift=[float(t) for t in shift],
        maximum_certified_absolute_difference=largest['absolute_difference'],
        maximum_witness_query_index=largest['query_index'],
        certified_queries_exceeding_budget=sum(r['exceeds_existing_budget'] for r in rows),
        nominal_mean_only_not_finite_particle_difference=True,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','witnesses.json']})
    exclusive_json(out/'summary.json',summary)
    print(json.dumps(summary),flush=True)


if __name__=='__main__':
    main()
