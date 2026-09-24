"""Analytic fixtures for boundary diagnostics; not a path-audit replacement."""
import argparse
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import numpy as np
from diagnose_probe_confirmation_boundary import inspect
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    out = root/'results/probe_credit_confirmation/boundary_diagnostic_selftests'
    assert not out.exists()
    vertices = np.vstack([np.zeros((1,4)), np.eye(4)])
    facets = vertices[np.array(list(combinations(range(5),4)))]
    interior = np.full(4,.1)
    fixtures = {}
    correct = inspect(dict(facets=facets, interior=interior))
    assert correct['combinatorial_boundary_closed'] and correct['consistently_oriented_normal_closed']
    assert correct['negative_cones'] == 0 and F(correct['original']['exact_volume_fraction']) == F(1,24)
    assert correct['original']['exact_boundary_closed']
    fixtures['exact_standard_4_simplex'] = correct

    permuted = facets.copy()
    for i in range(len(permuted)): permuted[i] = permuted[i][np.roll(np.arange(4), i%4)]
    mixed = inspect(dict(facets=permuted[::-1], interior=interior))
    assert mixed['combinatorial_boundary_closed'] and mixed['negative_cones'] == 0
    assert F(mixed['exact_signed_volume']) == F(1,24)
    fixtures['arbitrary_vertex_order_and_face_order'] = mixed

    scale = np.array([3.,2.,4.,5.])
    affine = inspect(dict(facets=facets*scale, interior=interior*scale))
    assert affine['combinatorial_boundary_closed'] and affine['negative_cones'] == 0
    assert F(affine['exact_signed_volume']) == 5
    fixtures['integer_affine_volume_5'] = affine

    missing = inspect(dict(facets=facets[:-1], interior=interior))
    assert not missing['combinatorial_orientation_attempted'] and 1 in missing['ridge_incidence']
    fixtures['missing_facet_rejected'] = missing

    duplicate = inspect(dict(facets=np.concatenate([facets, facets[:1]]), interior=interior))
    assert duplicate['duplicate_facets'] == 1 and not duplicate['combinatorial_orientation_attempted']
    assert 3 in duplicate['ridge_incidence']
    fixtures['duplicate_facet_rejected'] = duplicate

    outside = inspect(dict(facets=facets, interior=np.full(4,2.)))
    assert outside['combinatorial_boundary_closed'] and outside['consistently_oriented_normal_closed']
    assert outside['negative_cones'] > 0 and outside['negative_volume_fraction'] > .01
    assert F(outside['exact_signed_volume']) == F(1,24)
    fixtures['closed_boundary_does_not_certify_valid_fan'] = outside

    out.mkdir()
    exclusive_json(out/'fixtures.json',fixtures)
    result = dict(passed=True, tests=len(fixtures), query_targets_accessed=False,
        audit_gate_passed=False,
        scope='Analytically known fixtures for the diagnostic; no original prediction or audit modification',
        key_counterexample='A closed, exactly cancelling boundary with an exterior cone origin still has negative cones',
        source_sha256={n:sha(Path(__file__).parent/n) for n in [Path(__file__).name,'diagnose_probe_confirmation_boundary.py','audit_confirmation_geometry_determinants.py']},
        fixtures_sha256=sha(out/'fixtures.json'))
    exclusive_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
