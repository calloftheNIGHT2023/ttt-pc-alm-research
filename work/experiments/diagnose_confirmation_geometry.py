"""Observation-only diagnosis of the frozen confirmation's shared geometry error.

No query target/evaluation import, no replacement of a frozen prediction.
All alternative volumes below are diagnostics, not a revised predictor.
"""
import argparse
import json
import math
import os
from pathlib import Path
import time
import numpy as np
import torch
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection
import matched_budget_suite as suite
import region_posterior_memory as geo
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha, dump


def inspect(g, rhs):
    d = g.shape[1]
    a = np.r_[g, np.eye(d), -np.eye(d)]
    r = np.r_[rhs, np.full(2*d, geo.PRIOR_BOUND)]
    norm = np.linalg.norm(a, axis=1)
    keep = norm >= 1e-14
    a, r = a[keep]/norm[keep, None], r[keep]/norm[keep]
    lows, highs = [], []
    for e in np.eye(d):
        lo = linprog(e, A_ub=a, b_ub=r, bounds=[(None, None)]*d,
                     options={'primal_feasibility_tolerance': 1e-9})
        hi = linprog(-e, A_ub=a, b_ub=r, bounds=[(None, None)]*d,
                     options={'primal_feasibility_tolerance': 1e-9})
        assert lo.success and hi.success
        lows.append(e@lo.x)
        highs.append(e@hi.x)
    center = (np.array(lows)+highs)/2
    scale = (np.array(highs)-lows)/2
    ag, rr = a*scale[None], r-a@center
    cc = linprog(np.r_[np.zeros(d), -1.],
                 A_ub=np.c_[ag, np.linalg.norm(ag, axis=1)], b_ub=rr,
                 bounds=[(None, None)]*d+[(0, None)],
                 options={'primal_feasibility_tolerance': 1e-9})
    assert cc.success and cc.x[-1] > 1e-10
    interior = cc.x[:-1]
    vertices = HalfspaceIntersection(np.c_[ag, -rr], interior).intersections
    hull = ConvexHull(vertices)
    centroid = vertices.mean(0)
    _, singular, vt = np.linalg.svd(vertices-centroid, full_matrices=False)
    transform = vt.T*(singular/np.sqrt(len(vertices)))[None]
    rows = []

    def record(name, verts, hu, origin, mat, bounds, jacobian):
        facets = verts[hu.simplices]
        cones = np.abs(np.linalg.det(facets-origin))/math.factorial(d)
        # A boundary simplex of the H-polytope has a common tight input face.
        residuals = np.einsum('fvj,cj->fvc', facets, mat)-bounds
        face_errors = np.min(np.max(np.abs(residuals), axis=1), axis=1)
        ans = dict(name=name, vertices=len(verts), facets=len(facets),
            hull_volume=float(hu.volume*jacobian), cone_volume=float(cones.sum()*jacobian),
            relative_volume_gap=float(abs(cones.sum()-hu.volume)/hu.volume),
            max_vertex_violation=float(np.max(verts@mat.T-bounds)),
            max_origin_hull_violation=float(np.max(hu.equations[:, :-1]@origin+hu.equations[:, -1])),
            max_facet_not_on_input_face=float(face_errors.max()),
            max_origin_input_violation=float(np.max(mat@origin-bounds)))
        rows.append(ans)
        return facets, cones

    record('original_chebyshev', vertices, hull, interior, ag, rr, 1.)
    record('same_facets_vertex_centroid', vertices, hull, centroid, ag, rr, 1.)
    white = np.linalg.solve(transform, (vertices-centroid).T).T
    aw, rw = ag@transform, rr-ag@centroid
    jacobian = abs(float(np.linalg.det(transform)))
    wh = ConvexHull(white)
    record('whiten_existing_vertices', white, wh, np.zeros(d), aw, rw, jacobian)
    fresh = HalfspaceIntersection(np.c_[aw, -rw], np.zeros(d)).intersections
    fh = ConvexHull(fresh)
    record('whiten_halfspaces_and_recompute_vertices', fresh, fh, np.zeros(d), aw, rw, jacobian)
    # Public convex-hull input is saved in full so a separate implementation can audit.
    arrays = dict(g=g, rhs=rhs, a=a, r=r, center=center, scale=scale,
                  ag=ag, rr=rr, interior=interior, vertices=vertices,
                  old_simplices=hull.simplices, centroid=centroid,
                  transform=transform, fresh_white_vertices=fresh,
                  fresh_white_simplices=fh.simplices)
    return arrays, dict(chebyshev_radius=float(cc.x[-1]), singular_values=singular.tolist(),
                       condition=float(singular[0]/singular[-1]), axis_scale=scale.tolist(),
                       physical_jacobian=float(np.prod(scale)), comparisons=rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    src = Path(__file__).parent
    inp = root/'results/matched_budget_confirmation/confirmation'
    out = root/'results/matched_budget_confirmation/geometry_diagnosis'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    summary = json.loads((inp/'summary.json').read_text())
    assert summary['passed'] and not summary['query_targets_accessed']
    p = json.loads((inp/'protocol.json').read_text())
    hashes = dict(p['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    for n, h in hashes.items():
        assert sha(src/n) == h, n
    failures = [r for r in json.loads((inp/'rows.json').read_text()) if r['metadata']['execution_failed']]
    dump(out/'protocol.json', dict(source_sha256=hashes, confirmation_summary_sha256=sha(inp/'summary.json'),
        observations_only=True, seed=5910048, representative_method='cold__alm16',
        scope='diagnostic affine-coordinate changes only; no revised predictor or query truth'))
    dump(out/'failures.json', failures)
    saved = geo.polytope
    caught = []
    def capture(g, rhs):
        try:
            return saved(g, rhs)
        except AssertionError:
            caught.append((g.copy(), rhs.copy()))
            raise
    loaded, manifest = suite.oldfit.meta.load(root)
    assert manifest == p['checkpoint_manifest']
    cfg = next(c for c in p['configs'] if c['name'] == 'cold__alm16')
    x, v = observations(5910048)
    begin = time.perf_counter()
    try:
        geo.polytope = capture
        with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
            _, meta = suite.guarded_fit(cfg, x[:4], v[:4], np.linspace(0, 1, 257), 5910048, loaded)
    finally:
        geo.polytope = saved
    assert meta['execution_failed'] and len(caught) == 1
    arrays, diagnosis = inspect(*caught[0])
    np.savez_compressed(out/'geometry.npz', x_observed=x[:4], v_observed=v[:4], **arrays)
    dump(out/'diagnosis.json', diagnosis)
    ans = dict(passed=True, recorded_failures=len(failures), failing_seeds=sorted(set(r['seed'] for r in failures)),
        replayed_failure=meta['failure_message'], diagnosis=diagnosis, seconds=time.perf_counter()-begin,
        protocol_sha256=sha(out/'protocol.json'), geometry_sha256=sha(out/'geometry.npz'),
        diagnosis_sha256=sha(out/'diagnosis.json'), failures_sha256=sha(out/'failures.json'), query_targets_accessed=False)
    dump(out/'summary.json', ans)
    print(json.dumps(ans), flush=True)


if __name__ == '__main__':
    main()
