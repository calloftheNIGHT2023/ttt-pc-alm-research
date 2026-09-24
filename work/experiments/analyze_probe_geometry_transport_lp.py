"""LP-guided, exactly feasible simplex coupling; offline diagnosis only.

The floating LP only proposes flows. Every flow is converted to an exact
rational, capped by remaining exact supplies/demands, and completed by an exact
greedy plan. Hence the final upper bound needs no LP optimality or numerical
feasibility assumption. This does not replace the frozen audit gate.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import linprog
from analyze_probe_geometry_transport_bound import LIPSCHITZ, PERMUTATIONS, read
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


def coupling(weights, reference, faces, vertices):
    assert len(weights) == len(reference) == len(faces)
    assert sum(weights) == sum(reference) == 1
    assert all(w >= 0 for w in weights+reference)
    delta = [a-b for a, b in zip(weights, reference)]
    sources = [i for i, d in enumerate(delta) if d > 0]
    targets = [i for i, d in enumerate(delta) if d < 0]
    supply = {i: delta[i] for i in sources}
    demand = {i: -delta[i] for i in targets}
    total = sum(supply.values(), F(0))
    assert total == sum(demand.values(), F(0))
    if total == 0:
        return dict(exact_transport_cost='0', transport_cost=0., plan=[],
                    exact_moved_mass='0', exact_marginals_checked=True,
                    lp_needed=False, lp_guidance_has_no_certifying_role=True)
    distance = [[sum(F(l)*abs(a-b) for l, a, b in zip(LIPSCHITZ, p, q))
                 for q in vertices] for p in vertices]
    edges = []
    for i in sources:
        for j in targets:
            choices = [(sum(distance[a][faces[j][perm[k]]] for k, a in enumerate(faces[i]))/5, perm)
                       for perm in PERMUTATIONS]
            cost, perm = min(choices)
            edges.append((min(F(1), cost), i, j, perm, cost))
    aeq = np.zeros((len(sources)+len(targets), len(edges)))
    source_index = {i: k for k, i in enumerate(sources)}
    target_index = {j: k+len(sources) for k, j in enumerate(targets)}
    for k, (_, i, j, _, _) in enumerate(edges):
        aeq[source_index[i], k] = 1.
        aeq[target_index[j], k] = 1.
    beq = np.array([float(supply[i]/total) for i in sources] + [float(demand[j]/total) for j in targets])
    result = linprog(np.array([float(e[0]) for e in edges]), A_eq=aeq, b_eq=beq,
                     bounds=(0, None), method='highs',
                     options={'primal_feasibility_tolerance': 1e-10,
                              'dual_feasibility_tolerance': 1e-10})
    assert result.success and np.isfinite(result.x).all()
    plan = []
    outflow = [F(0)]*len(weights)
    inflow = [F(0)]*len(weights)
    bound = F(0)

    def take(edge, proposed, kind):
        nonlocal bound
        cost, i, j, perm, raw_cost = edge
        amount = min(supply[i], demand[j], max(F(0), proposed))
        if amount == 0:
            return
        supply[i] -= amount
        demand[j] -= amount
        outflow[i] += amount
        inflow[j] += amount
        bound += amount*cost
        plan.append(dict(source=i, target=j, mass=str(amount), permutation=list(perm),
                         cost=str(cost), weighted_vertex_distance_over_five=str(raw_cost), kind=kind))

    for k in sorted(range(len(edges)), key=lambda k: (-float(result.x[k]), k)):
        take(edges[k], F(float(result.x[k]))*total, 'capped_lp_proposal')
    remainder = sum(supply.values(), F(0))
    assert remainder == sum(demand.values(), F(0))
    for edge in sorted(edges):
        take(edge, min(supply[edge[1]], demand[edge[2]]), 'exact_residual_completion')
    assert not any(supply.values()) and not any(demand.values())
    assert all(outflow[i]-inflow[i] == delta[i] for i in range(len(weights)))
    assert sum(F(p['mass']) for p in plan) == total
    assert 0 <= bound <= total
    return dict(exact_transport_cost=str(bound), transport_cost=float(bound),
                exact_moved_mass=str(total), exact_marginals_checked=True, plan=plan,
                lp_needed=True, lp_success=bool(result.success), lp_message=result.message,
                lp_normalized_objective=float(result.fun),
                exact_residual_after_capped_proposal=str(remainder),
                residual_after_capped_proposal=float(remainder),
                exact_weights=[str(w) for w in weights],
                exact_reference_weights=[str(w) for w in reference],
                lp_guidance_has_no_certifying_role=True,
                scope='Certified feasible plan only; no exact LP optimality claim')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    base = root/'results/probe_credit_confirmation'
    inp = base/'geometry_budget_diagnosis_v1'
    out = base/'geometry_transport_lp_diagnosis_v1'
    assert not out.exists()
    summary = read(inp/'summary.json')
    for name, digest in summary['outputs_sha256'].items():
        assert sha(inp/name) == digest
    assert summary['diagnosis_complete'] and not summary['query_targets_accessed']
    certificate = read(inp/'certificate.json')
    witness = read(inp/'witness.json')
    with np.load(inp/'geometry.npz') as z:
        poly = {k: z[k].copy() for k in z.files}
    raw, inverse = np.unique(poly['facets'].reshape(-1, 4), axis=0, return_inverse=True)
    assert len(raw) == len(witness['matching'])
    vertices = [[F(t) for t in v] for v in witness['vertices']]
    faces = [[witness['matching'][int(i)] for i in face] for face in inverse.reshape(-1, 4)]
    weights = [F(float(w)) for w in poly['simplex_probs']]
    total = sum(weights)
    weights = [w/total for w in weights]
    reference = [F(v)/F(witness['reference_volume']) for v in witness['cone_volumes']]
    begin = time.perf_counter()
    answer = coupling(weights, reference, faces, vertices)
    position = sum(F(l)*F(d) for l, d in zip(LIPSCHITZ, certificate['exact_maximum_coordinate_displacements']))
    bound = position+F(answer['exact_transport_cost'])
    out.mkdir()
    exclusive_json(out/'plan.json', answer)
    exclusive_json(out/'protocol.json', dict(
        input_summary_sha256=sha(inp/'summary.json'),
        source_sha256={p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name('analyze_probe_geometry_transport_bound.py')]},
        seed=summary['seed'], pattern=summary['pattern'], query_targets_accessed=False,
        scope='LP guidance plus exact feasible coupling; diagnosis only, not a gate'))
    report = dict(diagnosis_complete=True, audit_gate_passed=False, query_targets_accessed=False,
        seed=summary['seed'], pattern=summary['pattern'],
        exact_transport_bound=str(bound), transport_bound=float(bound),
        previous_tv_bound=summary['total_bound'], position_bound=float(position),
        weight_transport_cost=answer['transport_cost'], plan_edges=len(answer['plan']),
        exact_marginals_checked=True, lp_guidance_has_no_certifying_role=True,
        below_unchanged_budget=bound <= F(summary['exact_budget']),
        seconds=time.perf_counter()-begin,
        outputs_sha256={n: sha(out/n) for n in ['protocol.json', 'plan.json']})
    exclusive_json(out/'summary.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
