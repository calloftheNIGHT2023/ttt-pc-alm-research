"""Exact binary-rational determinant check, without truth or model fitting."""
import argparse
from fractions import Fraction as F
from itertools import permutations
import json
import math
from pathlib import Path
import numpy as np
from scipy.spatial import ConvexHull
from run_multiplier_fixed_point_screen import sha, dump


def det(matrix):
    n = len(matrix)
    ans = F(0)
    for p in permutations(range(n)):
        inversions = sum(p[i] > p[j] for i in range(n) for j in range(i+1, n))
        term = F((-1)**inversions)
        for i in range(n):
            term *= matrix[i][p[i]]
        ans += term
    return ans


def check(vertices, simplices, origin):
    total = F(0)
    gaps = []
    normal_sum = [F(0)]*4
    exact_volumes = []
    for simplex in simplices:
        vv = [[F(float(t)) for t in vertices[i]] for i in simplex]
        oo = [F(float(t)) for t in origin]
        matrix = [[v-o for v, o in zip(row, oo)] for row in vv]
        determinant = det(matrix)
        volume = abs(determinant)/24
        exact_volumes.append(float(volume))
        total += volume
        numerical = abs(float(np.linalg.det(vertices[simplex]-origin)))/24
        gaps.append(abs(numerical-float(volume)))
        edges = [[v-o for v, o in zip(row, vv[0])] for row in vv[1:]]
        normal = [F((-1)**j)*det([[row[k] for k in range(4) if k != j] for row in edges]) for j in range(4)]
        outward = sum(n*t for n, t in zip(normal, matrix[0]))
        if outward < 0:
            normal = [-n for n in normal]
        normal_sum = [a+b for a, b in zip(normal_sum, normal)]
    # Repeat the batched operation used by the original implementation.
    batched = np.abs(np.linalg.det(vertices[simplices]-origin))/24
    return dict(exact_volume=float(total), exact_volume_fraction=str(total),
        scalar_max_volume_error=max(gaps),
        batched_max_volume_error=float(np.max(abs(batched-exact_volumes))),
        batched_sum_volume=float(batched.sum()),
        oriented_boundary_normal_sum=[float(t) for t in normal_sum],
        exact_boundary_closed=all(t == 0 for t in normal_sum))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve()
    base = root/'results/matched_budget_confirmation'
    inp = base/'geometry_diagnosis'
    out = base/'geometry_determinant_audit'
    out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    summary = json.loads((inp/'summary.json').read_text())
    assert summary['passed'] and not summary['query_targets_accessed']
    assert sha(inp/'geometry.npz') == summary['geometry_sha256']
    hashes = dict(json.loads((inp/'protocol.json').read_text())['source_sha256'])
    hashes[Path(__file__).name] = sha(Path(__file__))
    for n, h in hashes.items():
        assert sha(Path(__file__).parent/n) == h, n
    dump(out/'protocol.json', dict(source_sha256=hashes,
        diagnosis_sha256=sha(inp/'summary.json'), query_targets_accessed=False,
        scope='independent exact determinant permutations and boundary normals'))
    with np.load(inp/'geometry.npz') as z:
        comparisons = {}
        for name, origin in [('original_chebyshev', z['interior']), ('original_centroid', z['centroid'])]:
            comparisons[name] = check(z['vertices'], z['old_simplices'], origin)
        fresh = z['fresh_white_vertices']
        comparisons['fresh_white'] = check(fresh, z['fresh_white_simplices'], np.zeros(4))
        comparisons['fresh_white']['hull_volume'] = float(ConvexHull(fresh).volume)
        comparisons['fresh_white']['old_coordinate_volume'] = comparisons['fresh_white']['exact_volume']*abs(float(np.linalg.det(z['transform'])))
    ans = dict(passed=True, comparisons=comparisons, query_targets_accessed=False,
               protocol_sha256=sha(out/'protocol.json'))
    dump(out/'summary.json', ans)
    print(json.dumps(ans), flush=True)


if __name__ == '__main__':
    main()
