"""Independently check saved rational coupling witnesses and analytic fixtures.

The checker does not call the producing cost/flow routines. Tests may call them
to generate witnesses, then check every mass, permutation and cost separately.
This verifies a nominal-distribution bound, not query quality or the path gate.
"""
import argparse
from copy import deepcopy
from fractions import Fraction
import json
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha
from run_probe_credit_confirmation_v2 import exclusive_json


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def check(weights, reference, faces, vertices, record):
    assert len(weights) == len(reference) == len(faces)
    assert all(t >= 0 for t in weights+reference)
    assert sum(weights) == sum(reference) == 1
    n = len(faces)
    outgoing = [Fraction(0) for _ in faces]
    incoming = [Fraction(0) for _ in faces]
    total_mass = Fraction(0)
    total_cost = Fraction(0)
    for entry in record['plan']:
        source, target = entry['source'], entry['target']
        assert type(source) is int and type(target) is int and 0 <= source < n and 0 <= target < n
        permutation = entry['permutation']
        assert len(permutation) == 4 and sorted(permutation) == list(range(4))
        amount = Fraction(entry['mass'])
        assert amount > 0
        distance = Fraction(0)
        for k in range(4):
            left = vertices[faces[source][k]]
            right = vertices[faces[target][permutation[k]]]
            assert len(left) == len(right) == 4
            for coordinate in range(4):
                distance += (2**(4-coordinate))*abs(left[coordinate]-right[coordinate])
        distance /= 5
        cost = min(Fraction(1), distance)
        assert Fraction(entry['weighted_vertex_distance_over_five']) == distance
        assert Fraction(entry['cost']) == cost
        outgoing[source] += amount
        incoming[target] += amount
        total_mass += amount
        total_cost += amount*cost
    for i, (p, q) in enumerate(zip(weights, reference)):
        assert outgoing[i] == max(p-q, Fraction(0)), ('Supply differs', i)
        assert incoming[i] == max(q-p, Fraction(0)), ('Demand differs', i)
    tv = sum(abs(p-q) for p, q in zip(weights, reference))/2
    assert total_mass == tv and total_mass+sum(min(p, q) for p, q in zip(weights, reference)) == 1
    assert Fraction(record['exact_transport_cost']) == total_cost
    assert Fraction(record['exact_moved_mass']) == total_mass
    assert 0 <= total_cost <= total_mass
    return dict(passed=True, exact_mass=str(total_mass), exact_cost=str(total_cost),
                plan_edges=len(record['plan']), exact_marginals_independently_checked=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root/'results/probe_credit_confirmation'
    out = base/'geometry_transport_independent_audit_v1'
    assert not out.exists()
    # Import producers only to generate analytic test witnesses, never in check().
    from analyze_probe_geometry_transport_bound import transport
    from analyze_probe_geometry_transport_lp import coupling
    producers = {'greedy': transport, 'lp_guided': coupling}
    eps = Fraction(1, 1024)
    vertices = [[eps if i == j else Fraction(0) for j in range(4)] for i in range(4)]
    vertices += [[value+(eps if j == 0 else 0) for j, value in enumerate(p)] for p in vertices[:4]]
    faces = [list(range(4)), list(range(4, 8))]
    half, quarter = Fraction(1, 2), Fraction(1, 4)
    tests = {}
    cases = [
        ('equal_weights', [half, half], [half, half], faces, Fraction(0)),
        ('half_mass_translation', [3*quarter, quarter], [quarter, 3*quarter], faces, 32*eps/5),
        ('full_mass_translation', [Fraction(1), Fraction(0)], [Fraction(0), Fraction(1)], faces, 64*eps/5),
        ('reverse_translation', [quarter, 3*quarter], [3*quarter, quarter], faces, 32*eps/5),
        ('same_simplex_components', [3*quarter, quarter], [quarter, 3*quarter], [faces[0], faces[0]], Fraction(0)),
        ('permuted_vertices', [3*quarter, quarter], [quarter, 3*quarter], [faces[0][::-1], [6,4,7,5]], 32*eps/5),
    ]
    for name, p, q, ff, expected in cases:
        results = {}
        for label, producer in producers.items():
            record = producer(p, q, ff, vertices)
            checked = check(p, q, ff, vertices, record)
            assert Fraction(checked['exact_cost']) == expected
            results[label] = checked
        tests[name] = dict(passed=True, exact_known_cost=str(expected), producers=results,
            analytic_scope='Translated simplex affine readout has exact mean shift 4*16*eps/5; it remains in [0,1] on these small fixtures')
    p, q = [3*quarter, quarter], [quarter, 3*quarter]
    good = transport(p, q, faces, vertices)

    def rejected(name, change):
        bad = deepcopy(good)
        change(bad)
        try:
            check(p, q, faces, vertices, bad)
        except (AssertionError, ValueError, IndexError):
            tests[name] = dict(passed=True, deliberate_corruption_rejected=True)
            return
        raise AssertionError('Corruption accepted: '+name)

    rejected('missing_flow_rejected', lambda r: r.update(plan=[]))
    rejected('negative_flow_rejected', lambda r: r['plan'][0].update(mass='-1/2'))
    rejected('excess_flow_rejected', lambda r: r['plan'][0].update(mass='1'))
    rejected('invalid_permutation_rejected', lambda r: r['plan'][0].update(permutation=[0,0,2,3]))
    rejected('understated_edge_cost_rejected', lambda r: r['plan'][0].update(cost='0'))
    rejected('understated_total_cost_rejected', lambda r: r.update(exact_transport_cost='0'))
    rejected('invalid_source_rejected', lambda r: r['plan'][0].update(source=2))
    inp = base/'geometry_budget_diagnosis_v1'
    diagnosis = read(inp/'summary.json')
    for name, digest in diagnosis['outputs_sha256'].items():
        assert sha(inp/name) == digest
    witness = read(inp/'witness.json')
    certificate = read(inp/'certificate.json')
    with np.load(inp/'geometry.npz') as z:
        raw_facets = z['facets'].copy()
        raw_probs = z['simplex_probs'].copy()
    unique, indices = np.unique(raw_facets.reshape(-1, 4), axis=0, return_inverse=True)
    assert len(unique) == len(witness['matching'])
    exact_vertices = [[Fraction(t) for t in p] for p in witness['vertices']]
    exact_faces = [[witness['matching'][int(i)] for i in f] for f in indices.reshape(-1,4)]
    p = [Fraction(float(t)) for t in raw_probs]
    total = sum(p)
    p = [t/total for t in p]
    q = [Fraction(t)/Fraction(witness['reference_volume']) for t in witness['cone_volumes']]
    actual = {}
    for label, folder in [('greedy', 'geometry_transport_diagnosis_v1'), ('lp_guided', 'geometry_transport_lp_diagnosis_v1')]:
        directory = base/folder
        saved = read(directory/'summary.json')
        for name, digest in saved['outputs_sha256'].items():
            assert sha(directory/name) == digest
        protocol = read(directory/'protocol.json')
        assert protocol['input_summary_sha256'] == sha(inp/'summary.json')
        if isinstance(protocol['source_sha256'], str):
            assert sha(src/'analyze_probe_geometry_transport_bound.py') == protocol['source_sha256']
        else:
            for name, digest in protocol['source_sha256'].items():
                assert sha(src/name) == digest
        record = read(directory/'plan.json')
        checked = check(p, q, exact_faces, exact_vertices, record)
        position = sum(Fraction(2**(4-j))*Fraction(t) for j, t in enumerate(certificate['exact_maximum_coordinate_displacements']))
        bound = position+Fraction(checked['exact_cost'])
        assert bound == Fraction(saved['exact_transport_bound'])
        checked.update(exact_full_bound=str(bound), full_bound=float(bound),
                       below_unchanged_budget=bound <= Fraction(1,10**12))
        assert not checked['below_unchanged_budget']
        actual[label] = checked
    out.mkdir()
    exclusive_json(out/'tests.json', tests)
    exclusive_json(out/'actual_plans.json', actual)
    sources = [Path(__file__), src/'analyze_probe_geometry_transport_bound.py', src/'analyze_probe_geometry_transport_lp.py']
    result = dict(passed=True, analytic_tests=6, corruption_rejections=7,
        actual_saved_plans_independently_checked=2, total_checks=15,
        audit_gate_passed=False, query_targets_accessed=False,
        scope='Independent exact coupling-witness arithmetic; no independent true-polytope reconstruction, no query score, no threshold waiver',
        source_sha256={p.name: sha(p) for p in sources},
        input_diagnosis_sha256=sha(inp/'summary.json'),
        outputs_sha256={n:sha(out/n) for n in ['tests.json','actual_plans.json']})
    exclusive_json(out/'summary.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
