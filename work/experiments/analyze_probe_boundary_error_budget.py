"""Exact finite-mixture error accounting for the OLD boundary diagnosis.

This does not certify the true convex hull, finite RNG, or the whole study.
No original predictions, geometry, thresholds, or gates are changed.
"""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from audit_confirmation_geometry_determinants import det
from diagnose_probe_confirmation_boundary import inspect
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def tv(a, b):
    assert sum(a) == sum(b) == 1
    return sum(abs(x-y) for x,y in zip(a,b))/2


def remove_mass(weights, bad):
    mass = sum(weights[i] for i in bad)
    assert 0 <= mass < 1
    retained = [F(0) if i in bad else p/(1-mass) for i,p in enumerate(weights)]
    assert tv(weights, retained) == mass
    return mass, retained


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); base = root/'results/probe_credit_confirmation'
    inp = base/'boundary_diagnosis_preflight'; out = base/'boundary_error_budget_preflight'
    assert not out.exists() and not (base/'predictions/RUNNING.lock').exists()
    summary = json.loads((inp/'summary.json').read_text())
    assert summary['diagnosis_completed'] and not summary['query_targets_accessed']
    for name, digest in summary['outputs_sha256'].items(): assert sha(inp/name) == digest
    toy = [F(1,4),F(1,4),F(1,2)]
    assert remove_mass(toy,{0})[0] == F(1,4)
    assert remove_mass(toy,set())[0] == 0
    with np.load(inp/'geometry.npz') as z: poly = {k:z[k].copy() for k in z.files}
    diagnosis = inspect(poly)
    for field in ['combinatorial_boundary_closed','consistently_oriented_normal_closed','negative_cones','orientation_disagreements']:
        assert diagnosis[field] == summary[field]
    assert diagnosis['components'] == 1 and not diagnosis['orientation_conflicts']
    origin = [F(float(x)) for x in poly['interior']]
    exact = [abs(det([[F(float(x))-o for x,o in zip(row,origin)] for row in facet]))/24 for facet in poly['facets']]
    exact = [p/sum(exact) for p in exact]
    nominal = [F(float(p)) for p in poly['simplex_probs']]
    nominal_sum = sum(nominal); nominal = [p/nominal_sum for p in nominal]
    bad = set(diagnosis['orientation_disagreements'])
    assert len(bad) == diagnosis['negative_cones']
    exact_bad, positive = remove_mass(exact,bad)
    nominal_bad, _ = remove_mass(nominal,bad)
    rounding = tv(nominal,exact)
    actual_tv = tv(nominal,positive)
    assert actual_tv <= rounding+exact_bad
    result = dict(completed=True, audit_gate_passed=False, query_targets_accessed=False,
        scope='One old mode; nominal continuous mixtures on identical saved simplices only',
        seed=5910000, pattern='01020100020202010202010102010201',
        exact_nominal_weight_sum=str(nominal_sum),
        exact_negative_cone_mass=str(exact_bad), negative_cone_mass=float(exact_bad),
        nominal_negative_cone_mass=float(nominal_bad),
        weight_rounding_total_variation=float(rounding),
        exact_weight_rounding_total_variation=str(rounding),
        total_variation_to_positive_cones=float(actual_tv),
        exact_total_variation_to_positive_cones=str(actual_tv),
        triangle_upper_bound=float(rounding+exact_bad),
        bounded_readout_expectation_difference_upper=float(actual_tv),
        squared_risk_of_mean_readout_difference_upper=float(2*actual_tv),
        identities='TV(p, p conditioned on positive cones)=negative-cone mass; TV triangle inequality; |f-fprime|<=TV for f in [0,1]; squared loss is 2-Lipschitz on [0,1]',
        scope_limits=['Not a uniform-true-polytope certificate',
            'Not a guarantee for the fixed finite-precision RNG or finite particle draw',
            'Not evidence about every old or new geometry',
            'Not permission to weaken or remove the existing audit gate'],
        selftests=2,
        source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'diagnose_probe_confirmation_boundary.py','audit_confirmation_geometry_determinants.py']},
        diagnosis_summary_sha256=sha(inp/'summary.json'))
    out.mkdir(); exclusive_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
