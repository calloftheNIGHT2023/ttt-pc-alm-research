"""Analytic/corruption tests and a full task38 mean-coupling diagnosis."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
from probe_barycentric_position_bound import position_bound, independently_check
from probe_pool_geometry_gate_v2 import checked_component, exact_pool_bound, independent_pool_bound_check
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    src = Path(__file__).parent
    base = root/'results/probe_credit_confirmation'
    inp = base/'task_5500037_geometry_census_v1'
    out = base/'barycentric_position_diagnosis_v1'
    assert not out.exists() and not (base/'evaluation_v5').exists()
    summary = read(inp/'summary.json')
    assert summary['diagnosis_complete'] and not summary['query_targets_accessed']
    for name,h in summary['outputs_sha256'].items(): assert sha(inp/name) == h
    files = read(inp/'files.json')
    for name,h in files.items(): assert sha(inp/name) == h
    hashes = dict(read(inp/'protocol.json')['source_sha256'])
    for name in [Path(__file__).name,'probe_barycentric_position_bound.py']: hashes[name] = sha(src/name)
    for name,h in hashes.items(): assert sha(src/name) == h
    out.mkdir()
    exclusive_json(out/'protocol.json', dict(source_sha256=hashes, census_summary_sha256=sha(inp/'summary.json'),
        query_targets_accessed=False, audit_gate_passed=False, budget='1/1000000000000',
        scope='Exact barycentric position cost plus unchanged TV bounds; diagnostics only'))
    tests = {}
    start = time.perf_counter()
    # Abstract coupling fixtures deliberately need no true-polytope assertion.
    # They establish exact displacement costs and catch an omitted origin move.
    facet = np.eye(4, dtype=float)/64
    poly = dict(facets=facet[None], center=np.zeros(4), scale=np.ones(4),
        interior=np.zeros(4), simplex_probs=np.ones(1))
    sorted_vertices = np.unique(facet, axis=0)
    witness = dict(vertices=[[str(F(float(t))) for t in row] for row in sorted_vertices],
        matching=list(range(4)), origin=['0']*4)
    cases = []
    cases.append(('identical', deepcopy(poly), deepcopy(witness), F(0)))
    moved = deepcopy(witness)
    for vertex in moved['vertices']: vertex[0] = str(F(vertex[0])+F(1,1024))
    cases.append(('all_four_facet_vertices_translated', deepcopy(poly), moved, F(4,5)*16/F(1024)))
    one = deepcopy(witness)
    one['vertices'][0][0] = str(F(one['vertices'][0][0])+F(1,1024))
    cases.append(('one_vertex_translated', deepcopy(poly), one, 16/F(5120)))
    # Two cones share the origin, but only the first has a shifted vertex.
    two = deepcopy(poly)
    two['facets'] = np.array([facet, facet*2])
    two['simplex_probs'] = np.array([.25,.75])
    vertices = np.unique(two['facets'].reshape(-1,4),axis=0)
    tw = dict(vertices=[[str(F(float(t))) for t in row] for row in vertices], matching=list(range(8)), origin=['0']*4)
    ix = next(i for i,row in enumerate(vertices) if np.array_equal(row,facet[0]))
    tw['vertices'][ix][0] = str(F(tw['vertices'][ix][0])+F(1,1024))
    cases.append(('unequal_cone_weights', two, tw, F(1,4)*16/F(5120)))
    for name,pp,ww,expected in cases:
        result = position_bound(pp,ww)
        check = independently_check(pp,ww,result)
        assert F(result['exact_barycentric_position_bound']) == expected
        tests[name] = dict(passed=True, exact_known_position_cost=str(expected), certificate=result, check=check)
    good = position_bound(cases[1][1],cases[1][2])
    mutations = [
        ('forged_zero_cost',lambda r:r.update(exact_barycentric_position_bound='0')),
        ('forged_float_cost',lambda r:r.update(barycentric_position_bound=0.)),
        ('missing_cone',lambda r:r.update(exact_cone_position_bounds=[])),
        ('forged_cone_cost',lambda r:r.update(exact_cone_position_bounds=['0'])),
        ('wrong_barycentric_mean',lambda r:r.update(exact_barycentric_coordinate_mean='1/4')),
        ('false_origin_claim',lambda r:r.update(origin_displacement_exactly_zero=False)),
        ('forged_maximum',lambda r:r.update(exact_worst_point_position_bound='0')),
    ]
    for name,change in mutations:
        record = deepcopy(good)
        change(record)
        try: independently_check(cases[1][1],cases[1][2],record)
        except AssertionError: tests[name] = dict(passed=True, deliberate_corruption_rejected=True)
        else: raise AssertionError(name)
    moved_origin = deepcopy(witness)
    moved_origin['origin'][0] = '1/1024'
    for name,call in [('producer_origin_shift',lambda:position_bound(poly,moved_origin)),
                      ('checker_origin_shift',lambda:independently_check(poly,moved_origin,position_bound(poly,witness)))]:
        try: call()
        except AssertionError: tests[name] = dict(passed=True, deliberate_corruption_rejected=True)
        else: raise AssertionError(name)
    pools = read(inp/'pools.json')
    cache,records = {},{}
    for key in sorted({key for row in pools for key in row['patterns']}):
        prior = read(inp/'geometries'/f'{key}.json')
        with np.load(inp/prior['geometry_file']) as data: pp = {name:data[name].copy() for name in data.files}
        ww = prior['witness']
        cert,witness_now = checked_component(pp,ww['rows'],[F(t) for t in ww['rhs']])
        assert witness_now == ww
        result = position_bound(pp,ww)
        check = independently_check(pp,ww,result)
        old_position = sum(F(2**(4-j))*F(t) for j,t in enumerate(cert['exact_maximum_coordinate_displacements']))
        assert old_position == F(result['exact_worst_point_position_bound'])
        improved = F(cert['exact_weight_total_variation'])+F(result['exact_barycentric_position_bound'])
        assert improved <= F(cert['exact_nominal_readout_expectation_error_bound'])
        records[key] = dict(certificate=cert, barycentric=result, check=check,
            exact_refined_component_bound=str(improved), refined_component_bound=float(improved))
        cache[key] = pp
    answer = []
    for row in pools:
        keys = row['patterns']
        assert keys
        raw = np.array([cache[key]['volume'] for key in keys]); raw /= raw.sum()
        p = [F(float(t)) for t in raw]; total=sum(p); p=[t/total for t in p]
        r = [F(records[key]['certificate']['exact_reference_volume']) for key in keys]; total=sum(r); r=[t/total for t in r]
        e = [F(records[key]['exact_refined_component_bound']) for key in keys]
        result = exact_pool_bound(p,r,e)
        check = independent_pool_bound_check(p,r,e,result)
        answer.append(dict(method=row['method'], patterns=keys, previous_bound=row['nominal_readout_error_bound'], result=result, check=check))
    for name,h in hashes.items(): assert sha(src/name) == h
    exclusive_json(out/'tests.json',tests)
    exclusive_json(out/'components.json',records)
    exclusive_json(out/'pools.json',answer)
    result = dict(diagnosis_complete=True, analytic_cases=len(cases), corruption_rejections=len(mutations)+2,
        task38_geometry_certificates=len(records), task38_pools=len(answer),
        all_original_budget_bounds_pass=all(row['result']['full_pool_bound_within_original_budget'] for row in answer),
        maximum_refined_bound=max(row['result']['nominal_readout_error_bound'] for row in answer),
        query_targets_accessed=False, audit_gate_passed=False, seconds=time.perf_counter()-start,
        outputs_sha256={name:sha(out/name) for name in ['protocol.json','tests.json','components.json','pools.json']})
    exclusive_json(out/'summary.json',result)
    print(json.dumps(result),flush=True)


if __name__ == '__main__': main()
