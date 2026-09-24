"""Reproduce the OLD preflight boundary failure; no query teacher access.

Do not change or waive the path-audit gate. Inspect its exact orientations and
the combinatorial boundary of the same saved floating-point facet geometry.
"""
import argparse
from collections import Counter, defaultdict, deque
from fractions import Fraction as F
from itertools import combinations
import json
from pathlib import Path
import numpy as np
import torch
import probe_credit_confirmation_suite as suite
import probe_confirmation_path_reference as reference
import conditioned_mode_geometry as conditioned
import shared_mode_readout as shared
from audit_confirmation_geometry_determinants import check, det
from posterior_confirmation_pipeline import discovery_box
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def parity(items):
    return (-1)**sum(items[i] > items[j] for i in range(len(items)) for j in range(i+1, len(items)))


def inspect(poly):
    facets = poly['facets']
    vertices, inverse = np.unique(facets.reshape(-1, 4), axis=0, return_inverse=True)
    simplices = inverse.reshape(-1, 4)
    oo = [F(float(t)) for t in poly['interior']]
    normals, outward_signs, products, ridges = [], [], [], defaultdict(list)
    for fi, ids in enumerate(simplices):
        vv = [[F(float(t)) for t in vertices[i]] for i in ids]
        edges = [[a-b for a,b in zip(row, vv[0])] for row in vv[1:]]
        normal = [F((-1)**j)*det([[row[k] for k in range(4) if k != j] for row in edges]) for j in range(4)]
        product = sum(a*(b-c) for a,b,c in zip(normal, vv[0], oo))
        normals.append(normal); products.append(product)
        outward_signs.append(-1 if product < 0 else 1)
        for j in range(4):
            ridge = tuple(int(ids[k]) for k in range(4) if k != j)
            ridges[tuple(sorted(ridge))].append((fi, (-1)**j*parity(ridge)))
    incidence = Counter(len(v) for v in ridges.values())
    original_residuals = {str(k): sum(outward_signs[i]*c for i,c in v) for k,v in ridges.items()}
    original_residuals = {k:v for k,v in original_residuals.items() if v}
    original = check(facets.reshape(-1,4), np.arange(len(facets)*4).reshape(-1,4), poly['interior'])
    report = dict(vertices=len(vertices), facets=len(facets), ridge_incidence=dict(incidence),
        duplicate_facets=len(simplices)-len({tuple(sorted(s)) for s in simplices}),
        original=original, original_nonzero_ridge_boundary=original_residuals,
        exact_zero_cones=sum(t == 0 for t in products))
    if set(incidence) != {2}:
        report['combinatorial_orientation_attempted'] = False
        return report
    adjacent = defaultdict(list)
    for pairs in ridges.values():
        (a, ca), (b, cb) = pairs
        adjacent[a].append((b, -ca*cb)); adjacent[b].append((a, -ca*cb))
    signs, components, conflicts = {}, [], []
    for initial in range(len(facets)):
        if initial in signs: continue
        signs[initial] = outward_signs[initial]; queue = deque([initial]); component = []
        while queue:
            a = queue.popleft(); component.append(a)
            for b, relative in adjacent[a]:
                desired = signs[a]*relative
                if b in signs:
                    if signs[b] != desired: conflicts.append([a,b])
                else:
                    signs[b] = desired; queue.append(b)
        if sum(signs[i]*products[i] for i in component) < 0:
            for i in component: signs[i] *= -1
        components.append(component)
    residuals = [sum(signs[i]*c for i,c in pairs) for pairs in ridges.values()]
    normalsum = [sum(signs[i]*normals[i][j] for i in signs) for j in range(4)]
    oriented = [signs[i]*products[i]/24 for i in range(len(facets))]
    total = sum(abs(v) for v in oriented)
    negative = -sum(v for v in oriented if v < 0)
    report.update(combinatorial_orientation_attempted=True, components=len(components),
        orientation_conflicts=conflicts,
        combinatorial_boundary_closed=all(v == 0 for v in residuals),
        consistently_oriented_exact_normal_sum=[str(v) for v in normalsum],
        consistently_oriented_normal_closed=all(v == 0 for v in normalsum),
        negative_cones=sum(v < 0 for v in oriented),
        exact_negative_volume=str(negative), exact_absolute_volume=str(total),
        negative_volume_fraction=float(negative/total),
        orientation_disagreements=[i for i in signs if signs[i] != outward_signs[i]],
        exact_signed_volume=str(sum(oriented)),
        min_oriented_cone=float(min(oriented)), max_oriented_cone=float(max(oriented)))
    return report


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); base = root/'results/probe_credit_confirmation'
    out = base/'boundary_diagnosis_preflight'; assert not out.exists()
    assert not (base/'predictions/RUNNING.lock').exists()
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    inp = base/'runner_preflight_v2'; seed = 5910000
    key = '01020100020202010202010102010201'
    protocol = json.loads((base/'path_audit_preflight/protocol.json').read_text())
    hashes = dict(protocol['source_sha256']); hashes[Path(__file__).name] = sha(Path(__file__))
    for name, digest in hashes.items(): assert sha(Path(__file__).parent/name) == digest, name
    commit = json.loads((inp/'tasks'/str(seed)/'commit.json').read_text())
    rows = {r['method']:r for r in json.loads((inp/commit['rows_file']).read_text())}
    cfg = suite.catalogue(root)[0]; row = rows[cfg['name']]
    assert sha(inp/row['file']) == row['sha256']
    with np.load(inp/row['file']) as z: saved = {k:z[k].copy() for k in z.files}
    x, v, q = (saved[k] for k in ['x_observed','v_observed','q_observed'])
    loaded, _ = suite.resources.legacy.oldfit.meta.load(root)
    with discovery_box(.12):
        def traced(c, xx, vv, qq, ss, ll): return suite.resources.fit(c, xx, vv, qq, ss, ll, trace=True)
        with conditioned.geometry_scope([]):
            fresh, meta = suite.resources.legacy.guarded_fit(cfg, x, v, q, seed, loaded, runner=traced)
        for name, value in saved.items():
            if name not in ['x_observed','v_observed','q_observed']: assert value.tobytes() == fresh[name].tobytes(), name
        seen, counters = reference.reference(cfg, x, v, fresh, meta)
        repairs = []
        with conditioned.geometry_scope(repairs): poly, proof = shared.build(x, v, seen[key])
    assert poly is not None and proof['proof']['accepted']
    diagnosis = inspect(poly)
    assert diagnosis['original']['exact_boundary_closed'] is False
    out.mkdir()
    exclusive_json(out/'protocol.json', dict(seed=seed, pattern=key, method=cfg['name'],
        source_sha256=hashes, path_preflight_protocol_sha256=sha(base/'path_audit_preflight/protocol.json'),
        prediction_sha256=row['sha256'], query_targets_accessed=False,
        scope='Old preflight failure reproduction and exact geometry diagnosis only; no gate waiver'))
    with (out/'geometry.npz').open('xb') as f: np.savez_compressed(f, **poly, x_observed=x, v_observed=v, representative_b=seen[key])
    exclusive_json(out/'diagnosis.json', dict(**diagnosis, repair_log=repairs, proof=proof['proof'], reference_counts=counters))
    result = dict(diagnosis_completed=True, query_targets_accessed=False, audit_gate_passed=False,
        outputs_sha256={n:sha(out/n) for n in ['protocol.json','geometry.npz','diagnosis.json']},
        **{k:v for k,v in diagnosis.items() if k not in ['original_nonzero_ridge_boundary']})
    exclusive_json(out/'summary.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__': main()
