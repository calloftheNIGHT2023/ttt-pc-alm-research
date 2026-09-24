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
from probe_pool_geometry_gate_v2 import checked_component, pool_coupling
from probe_pool_geometry_gate_v2 import exact_pool_bound, independent_pool_bound_check


def checked_certificate(poly, rows, rhs):
    result, witness = checked_component(poly, rows, rhs)
    pool_coupling(['only'], {'only':poly}, {'only':result})
    return result, witness
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
    base = root/'results/probe_credit_confirmation'; out = base/'pool_geometry_gate_selftests_v2'
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
    # Large component errors may be harmless at small actual mixture weight.
    cases = [
        ('rare_component_large_bound', [F(1,10**12),1-F(1,10**12)], [F(1,10**12),1-F(1,10**12)], [F(1,10),0], True),
        ('dominant_component_rejected', [F(1,2),F(1,2)], [F(1,2),F(1,2)], [F(1,10),0], False),
        ('weight_tv_alone_rejected', [F(1,2),F(1,2)], [F(1,2)+F(1,10**9),F(1,2)-F(1,10**9)], [0,0], False),
        ('exact_budget_boundary', [F(1)], [F(1)], [F(1,10**12)], True),
        ('zero_error', [F(1,3),F(2,3)], [F(1,3),F(2,3)], [0,0], True),
    ]
    for name, pp, rr, ee, allowed in cases:
        result = exact_pool_bound(pp,rr,ee)
        checked = independent_pool_bound_check(pp,rr,ee,result)
        assert result['full_pool_bound_within_original_budget'] == allowed
        tests[name] = dict(passed=True, certificate=result, check=checked)
    pp,rr,ee = cases[0][1:4]
    good_record = exact_pool_bound(pp,rr,ee)
    mutations = [
        ('forged_zero_bound', lambda row:row.update(exact_nominal_readout_error_bound='0')),
        ('wrong_weighted_sum', lambda row:row.update(exact_weighted_component_bound='0')),
        ('wrong_budget', lambda row:row.update(budget=1.)),
        ('hidden_component_exceedance', lambda row:row.update(original_component_threshold_exceedances=0)),
    ]
    for name, change in mutations:
        bad = deepcopy(good_record); change(bad)
        try: independent_pool_bound_check(pp,rr,ee,bad)
        except AssertionError: tests[name] = dict(passed=True, rejected=True)
        else: raise AssertionError('Forged pool bound accepted: '+name)
    # Recheck all first-task certificates, not merely the census success flag.
    census = base/'first_task_geometry_census_v2'
    census_summary = json.loads((census/'summary.json').read_text())
    assert census_summary['diagnosis_complete'] and not census_summary['query_targets_accessed']
    for name,digest in census_summary['outputs_sha256'].items():assert sha(census/name)==digest
    for name,digest in json.loads((census/'files.json').read_text()).items():assert sha(census/name)==digest
    cache,certificates = {},{}
    pools = json.loads((census/'pools.json').read_text())
    for key in sorted({key for row in pools for key in row.get('patterns',[])}):
        record = json.loads((census/'geometries'/str(key+'.json')).read_text())
        assert record['certificate_valid']
        with np.load(census/record['geometry_file']) as arrays:
            cache[key] = {name:arrays[name].copy() for name in arrays.files}
        certificates[key] = record['certificate']
    for row in pools:
        if row.get('empty_pool'): continue
        result = pool_coupling(row['patterns'],cache,certificates)
        assert F(result['exact_nominal_readout_error_bound']) == F(row['exact_old_nominal_bound'])
        tests['first_task_'+row['method']] = dict(passed=True,certificate=result)
    # Subdivide an existing mesh edge at an exactly representable near-endpoint.
    # The saved mesh has a redundant vertex, while the true cube still has16.
    from audit_probe_collapsed_vertices import verify_mesh
    def subdivide(poly, fraction):
        answer=deepcopy(poly); edge=poly['facets'][0,:2].copy(); middle=(1-fraction)*edge[0]+fraction*edge[1]
        new=[]
        for face in poly['facets']:
            first=next((i for i in range(4) if np.array_equal(face[i],edge[0])),None)
            second=next((i for i in range(4) if np.array_equal(face[i],edge[1])),None)
            if first is not None and second is not None:
                left=face.copy();right=face.copy();left[first]=middle;right[second]=middle
                new.extend([left,right])
            else:new.append(face.copy())
        answer['facets']=np.array(new)
        volumes=np.abs(np.linalg.det(answer['facets']-answer['interior']))/24
        answer['simplex_probs']=volumes/volumes.sum()
        return answer
    redundant=good
    for count in [1,2]:
        redundant=subdivide(redundant,2.**-48)
        result,witness=checked_certificate(redundant,rows,rhs)
        assert result['collapsed_saved_vertices']==count and not result['vertex_mapping_bijective']
        check=verify_mesh(redundant,[tuple(r) for r in rows],rhs,result,witness,[tuple(F(t) for t in p) for p in cube],F(1))
        tests['subdivided_cube_'+str(count)]=dict(passed=True,certificate=result,independent_check=check)
    coarse=subdivide(good,.25); result,witness=checked_component(coarse,rows,rhs)
    assert not result['original_component_budget_diagnostic_passed']
    try:pool_coupling(['only'],{'only':coarse},{'only':result})
    except AssertionError:tests['large_collapse_error_not_free']=dict(passed=True,rejected=True)
    else:raise AssertionError('Large displacement accepted without its error charge')
    sixth=base/'task_5500005_geometry_census_v1'
    ss=json.loads((sixth/'summary.json').read_text());assert ss['diagnosis_complete'] and not ss['query_targets_accessed']
    for name,digest in ss['outputs_sha256'].items():assert sha(sixth/name)==digest
    for name,digest in json.loads((sixth/'files.json').read_text()).items():assert sha(sixth/name)==digest
    sixthpools=json.loads((sixth/'pools.json').read_text());cache2={};cert2={};collapse_matches=0
    for key in sorted({key for row in sixthpools for key in row.get('patterns',[])}):
        record=json.loads((sixth/'geometries'/str(key+'.json')).read_text())
        with np.load(sixth/record['geometry_file']) as arrays:poly={name:arrays[name].copy() for name in arrays.files}
        prior=record.get('witness') if record['certificate_valid'] else record['matching_diagnosis']
        rr=prior.get('rows',prior.get('exact_rows')); bb=[F(t) for t in prior.get('rhs',prior.get('exact_rhs'))]
        result,witness=checked_component(poly,rr,bb);cache2[key]=poly;cert2[key]=result
        if not record['certificate_valid']:
            assert witness==json.loads((base/'collapsed_vertex_diagnosis_v1/witness.json').read_text())
            collapse_matches+=1
    assert collapse_matches==1
    tests['actual_collapse_same_witness']=dict(passed=True)
    for row in sixthpools:
        if row.get('empty_pool'):continue
        result=pool_coupling(row['patterns'],cache2,cert2)
        tests['sixth_task_'+row['method']]=dict(passed=True,certificate=result)
    independent=base/'collapsed_vertex_independent_audit_v1'; independent_summary=json.loads((independent/'summary.json').read_text())
    assert independent_summary['passed'] and independent_summary['corruption_rejections']==9
    for name,digest in independent_summary['source_sha256'].items():assert sha(src/name)==digest
    for name,digest in independent_summary['outputs_sha256'].items():assert sha(independent/name)==digest
    out.mkdir(); exclusive_json(out/'tests.json', tests)
    answer = dict(passed=True, tests=len(tests), analytic_positive_cases=6, deliberate_rejections=19, pool_analytic_cases=5, first_task_pool_checks=16, sixth_task_pool_checks=16, redundant_mesh_positive_cases=2, large_collapse_rejection=True,
        old_fixture_replay=True, query_targets_accessed=False, audit_gate_passed=False,
        source_sha256={n:sha(src/n) for n in [Path(__file__).name, 'probe_exact_geometry_gate.py','probe_exact_polytope_certificate.py','probe_pool_geometry_gate.py','census_probe_first_task_geometry_v2.py','probe_pool_geometry_gate_v2.py','probe_exact_polytope_certificate_v2.py','audit_probe_collapsed_vertices.py','audit_probe_exact_polytope_certificate.py','diagnose_probe_collapsed_vertices.py','census_probe_task_5500005_geometry.py']},
        tests_sha256=sha(out/'tests.json'), seconds=time.perf_counter()-started)
    assert len(tests) == 67
    answer['collapse_independent_summary_sha256']=sha(independent/'summary.json')
    answer['sixth_census_summary_sha256']=sha(sixth/'summary.json')
    answer['census_summary_sha256'] = sha(census/'summary.json')
    exclusive_json(out/'summary.json', answer); print(json.dumps(answer), flush=True)


if __name__ == '__main__': main()
