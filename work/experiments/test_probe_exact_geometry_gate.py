"""Analytic positives and deliberate corruptions for exact true-H certification."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
from itertools import product
import json
import math
from pathlib import Path
import time
import numpy as np
from scipy.spatial import ConvexHull
from probe_exact_polytope_certificate import normalize, observed_constraints
from probe_exact_geometry_gate import checked_certificate
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def box_rows(lower, upper):
    rows, rhs = [], []
    for j in range(4):
        e = [int(k == j) for k in range(4)]
        rows.extend([e, [-t for t in e]]); rhs.extend([F(upper[j]), -F(lower[j])])
    return rows, rhs


def fixture(vertices, volume):
    vertices = np.asarray(vertices, dtype=float); hull = ConvexHull(vertices)
    facets = vertices[hull.simplices]; origin = vertices.mean(0)
    volumes = np.abs(np.linalg.det(facets-origin))/24
    return dict(center=np.zeros(4), scale=np.ones(4), interior=origin,
        facets=facets, simplex_probs=volumes/volumes.sum(), volume=np.array(float(volume)))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); src = Path(__file__).parent
    base = root/'results/probe_credit_confirmation'; out = base/'exact_geometry_gate_selftests'
    assert not out.exists(); started = time.perf_counter(); tests = {}
    cube = list(product([0, 1], repeat=4)); rows, rhs = box_rows([0]*4, [1]*4)
    simplex = [[0]*4] + np.eye(4, dtype=int).tolist()
    triangle = [(a,b,c,d) for a,b in [(0,0),(1,0),(0,1)] for c,d in product([0,1], repeat=2)]
    cases = [
        ('unit_cube', cube, rows, rhs, F(1)),
        ('standard_simplex', simplex, rows+[[1,1,1,1]], rhs+[F(1)], F(1,24)),
        ('triangle_times_square', triangle, rows+[[1,1,0,0]], rhs+[F(1)], F(1,2)),
    ]
    # An independent, analytically known affine-volume transformation.
    scale = [2,3,4,5]; shift = [-1,2,-2,3]
    transformed = [[scale[j]*p[j]+shift[j] for j in range(4)] for p in triangle]
    ar, ab = box_rows(shift, [a+b for a,b in zip(shift,scale)])
    cases.append(('translated_scaled_triangle_product', transformed,
        ar+[[3,2,0,0]], ab+[F(6+3*shift[0]+2*shift[1])], F(math.prod(scale),2)))
    for name, vertices, rr, bb, volume in cases:
        p = fixture(vertices, volume); result, witness = checked_certificate(p, rr, bb)
        assert F(result['exact_reference_volume']) == volume
        assert result['exact_vertices'] == len(vertices)
        tests[name] = dict(passed=True, known_volume=str(volume), certificate=result)
    good = fixture(cube, F(1)); simplex_poly = fixture(simplex, F(1,24))
    # Permutations and a different internal affine coordinate system preserve P.
    permuted = deepcopy(good); permuted['facets'] = permuted['facets'][::-1, ::-1]
    permuted['simplex_probs'] = permuted['simplex_probs'][::-1]
    result, _ = checked_certificate(permuted, rows, rhs)
    tests['ordering_invariance'] = dict(passed=True, certificate=result)
    affine = deepcopy(good); affine['center'] = np.array([2.,-1.,3.,-4.]); affine['scale'] = np.array([2.,.5,4.,.25])
    affine['facets'] = (good['facets']-affine['center'])/affine['scale']
    affine['interior'] = (good['interior']-affine['center'])/affine['scale']
    result, _ = checked_certificate(affine, rows, rhs)
    tests['saved_coordinate_invariance'] = dict(passed=True, certificate=result)

    def reject(name, p, rr=rows, bb=rhs):
        try: checked_certificate(p, rr, bb)
        except (AssertionError, ValueError) as exc:
            tests[name] = dict(passed=True, rejected=True, reason=str(exc)); return
        raise AssertionError('Corruption accepted: '+name)

    bad = deepcopy(good); bad['facets'] = bad['facets'][:-1]; bad['simplex_probs'] = np.ones(len(bad['facets']))/len(bad['facets'])
    reject('missing_facet', bad)
    bad = deepcopy(good); bad['facets'] = np.concatenate([bad['facets'],bad['facets'][:1]]); bad['simplex_probs'] = np.ones(len(bad['facets']))/len(bad['facets'])
    reject('duplicate_facet', bad)
    bad = deepcopy(good); bad['interior'][:] = 2
    reject('exterior_origin', bad)
    bad = deepcopy(good); bad['simplex_probs'] = bad['simplex_probs'][:-1]
    reject('truncated_probability_vector', bad)
    bad = deepcopy(good); bad['simplex_probs'][:] = 0
    reject('zero_probability_vector', bad)
    bad = deepcopy(good); bad['simplex_probs'] *= 2
    reject('unnormalized_probabilities', bad)
    bad = deepcopy(good); bad['simplex_probs'][0] = -1
    reject('negative_probability', bad)
    bad = deepcopy(good); bad['facets'][0,0,0] = np.nan
    reject('nonfinite_vertex', bad)
    bad = deepcopy(good); bad['center'] = np.zeros(3)
    reject('wrong_affine_shape', bad)
    bad = deepcopy(good); bad['scale'][0] = -1
    reject('negative_scale', bad)
    bad = deepcopy(good); bad['volume'] *= 2
    reject('wrong_stored_volume', bad)
    bad = deepcopy(good); bad['simplex_probs'][:] = 0; bad['simplex_probs'][0] = 1
    reject('biased_simplex_weights', bad)
    reject('incomplete_vertex_set', simplex_poly)
    bad_rows = deepcopy(rows); bad_rows[0][0] = F(1,2)
    reject('fractional_normal_not_silently_truncated', good, bad_rows, rhs)
    bad = deepcopy(simplex_poly); bad['facets'][0,0] = .9*bad['facets'][0,0] + .1*bad['interior']
    reject('broken_vertex_incidence', bad, rows+[[1,1,1,1]], rhs+[F(1)])

    # The actual old fixture is bound to the earlier independently checked proof.
    old = base/'boundary_diagnosis_preflight'; old_summary = json.loads((old/'summary.json').read_text())
    for name, digest in old_summary['outputs_sha256'].items(): assert sha(old/name) == digest
    p = json.loads((old/'protocol.json').read_text())
    with np.load(old/'geometry.npz') as z: poly = {k:z[k].copy() for k in z.files}
    rr, bb = observed_constraints(poly['x_observed'], poly['v_observed'], p['pattern'])
    result, witness = checked_certificate(poly, rr, bb)
    prior = json.loads((base/'exact_polytope_certificate_preflight/witness.json').read_text())
    assert witness == prior
    tests['old_fixture_unchanged_witness'] = dict(passed=True, certificate=result)
    out.mkdir(); exclusive_json(out/'tests.json', tests)
    answer = dict(passed=True, tests=len(tests), analytic_positive_cases=6, deliberate_rejections=15,
        old_fixture_replay=True, query_targets_accessed=False, audit_gate_passed=False,
        source_sha256={n:sha(src/n) for n in [Path(__file__).name, 'probe_exact_geometry_gate.py','probe_exact_polytope_certificate.py']},
        tests_sha256=sha(out/'tests.json'), seconds=time.perf_counter()-started)
    assert len(tests) == 22
    exclusive_json(out/'summary.json', answer); print(json.dumps(answer), flush=True)


if __name__ == '__main__': main()
