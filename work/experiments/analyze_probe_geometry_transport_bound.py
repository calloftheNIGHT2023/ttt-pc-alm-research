"""Exact constructive simplex-mixture coupling, diagnosis only.

For a fixed query, the four-fold clipped tent readout has coordinate Lipschitz
constants (16,8,4,2) and range [0,1]. Pairing uniform simplex barycentrics with
a permutation gives E|f(X)-f(Y)| <= min(1, sum_k weighted_l1(v_k,w_pi(k))/5).
The common cone origin is paired to itself. Exact rational residual transport
between component weights then supplies a valid upper bound, not a lower bound
or an estimate of actual query error. No gate or frozen source is changed.
"""
import argparse
from fractions import Fraction as F
from itertools import permutations
import json
from pathlib import Path
import time
import numpy as np
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


LIPSCHITZ = (16, 8, 4, 2)
PERMUTATIONS = tuple(permutations(range(4)))


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def transport(weights, reference, faces, vertices):
    assert len(weights) == len(reference) == len(faces)
    assert sum(weights) == sum(reference) == 1
    assert all(w >= 0 for w in weights + reference)
    delta = [a-b for a, b in zip(weights, reference)]
    supply = {i: d for i, d in enumerate(delta) if d > 0}
    demand = {i: -d for i, d in enumerate(delta) if d < 0}
    mass = sum(supply.values(), F(0))
    assert mass == sum(demand.values(), F(0))
    distances = [[sum(F(l)*abs(x-y) for l, x, y in zip(LIPSCHITZ, a, b))
                  for b in vertices] for a in vertices]
    edges = []
    for i in supply:
        for j in demand:
            costs = [(sum(distances[a][faces[j][perm[k]]] for k, a in enumerate(faces[i]))/5, perm)
                     for perm in PERMUTATIONS]
            distance, perm = min(costs)
            edges.append((min(F(1), distance), i, j, perm, distance))
    edges.sort()
    plan = []
    bound = F(0)
    used_supply = [F(0)]*len(weights)
    used_demand = [F(0)]*len(weights)
    for cost, i, j, perm, distance in edges:
        amount = min(supply[i], demand[j])
        if amount == 0:
            continue
        supply[i] -= amount
        demand[j] -= amount
        used_supply[i] += amount
        used_demand[j] += amount
        bound += amount*cost
        plan.append(dict(source=i, target=j, mass=str(amount), permutation=list(perm),
                         weighted_vertex_distance_over_five=str(distance), cost=str(cost)))
    assert not any(supply.values()) and not any(demand.values())
    assert all(used_supply[i]-used_demand[i] == delta[i] for i in range(len(weights)))
    assert sum(F(p['mass']) for p in plan) == mass
    assert bound <= mass
    return dict(exact_transport_cost=str(bound), transport_cost=float(bound),
                exact_moved_mass=str(mass), moved_mass=float(mass),
                matched_component_mass=str(1-mass), exact_marginals_checked=True,
                exact_weights=[str(w) for w in weights],
                exact_reference_weights=[str(w) for w in reference], plan=plan,
                coupling_type='Exact feasible greedy plan; no optimality claim')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    base = root/'results/probe_credit_confirmation'
    inp = base/'geometry_budget_diagnosis_v1'
    out = base/'geometry_transport_diagnosis_v1'
    assert not out.exists()
    diagnosis = read(inp/'summary.json')
    assert diagnosis['diagnosis_complete'] and not diagnosis['query_targets_accessed']
    for name, digest in diagnosis['outputs_sha256'].items():
        assert sha(inp/name) == digest
    certificate = read(inp/'certificate.json')
    witness = read(inp/'witness.json')
    with np.load(inp/'geometry.npz') as z:
        poly = {k: z[k].copy() for k in z.files}
    raw, inverse = np.unique(poly['facets'].reshape(-1, 4), axis=0, return_inverse=True)
    assert len(raw) == len(witness['matching'])
    vertices = [[F(t) for t in p] for p in witness['vertices']]
    faces = [[witness['matching'][int(i)] for i in face] for face in inverse.reshape(-1, 4)]
    weights = [F(float(t)) for t in poly['simplex_probs']]
    weight_sum = sum(weights)
    weights = [w/weight_sum for w in weights]
    reference_volume = F(witness['reference_volume'])
    reference = [F(v)/reference_volume for v in witness['cone_volumes']]
    begin = time.perf_counter()
    answer = transport(weights, reference, faces, vertices)
    assert F(answer['exact_moved_mass']) == F(certificate['exact_weight_total_variation'])
    position = sum(F(l)*F(d) for l, d in zip(LIPSCHITZ, certificate['exact_maximum_coordinate_displacements']))
    bound = position + F(answer['exact_transport_cost'])
    out.mkdir()
    exclusive_json(out/'plan.json', answer)
    exclusive_json(out/'protocol.json', dict(
        input_summary_sha256=sha(inp/'summary.json'), source_sha256=sha(Path(__file__)),
        seed=diagnosis['seed'], pattern=diagnosis['pattern'], query_targets_accessed=False,
        scope='Exact feasible coupling under uniform simplex barycentrics; diagnostic, not a replacement audit gate'))
    summary = dict(diagnosis_complete=True, audit_gate_passed=False, query_targets_accessed=False,
        seed=diagnosis['seed'], pattern=diagnosis['pattern'],
        exact_transport_bound=str(bound), transport_bound=float(bound),
        previous_bound=diagnosis['total_bound'], position_bound=float(position),
        weight_transport_cost=answer['transport_cost'], transported_mass=answer['moved_mass'],
        plan_edges=len(answer['plan']), exact_marginals_checked=True,
        below_unchanged_budget=bound <= F(diagnosis['exact_budget']),
        seconds=time.perf_counter()-begin,
        outputs_sha256={n: sha(out/n) for n in ['protocol.json', 'plan.json']})
    exclusive_json(out/'summary.json', summary)
    print(json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
