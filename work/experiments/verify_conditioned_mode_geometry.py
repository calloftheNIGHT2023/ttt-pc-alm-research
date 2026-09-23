"""264 observation-only preflight for the common geometry correction."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import time
import numpy as np
import torch
import conditioned_mode_geometry as corrected
import matched_budget_suite as suite
import shared_mode_readout as shared
from audit_confirmation_geometry_determinants import check
from audit_shared_mode_readout import check_cube
from analyze_recovered_online_comparison import forward
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha, dump


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); src = Path(__file__).parent
    base = root/'results/matched_budget_confirmation'; inp = base/'confirmation'
    out = base/'geometry_preflight'; out.mkdir(parents=True, exist_ok=True)
    assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    hashes = {}
    for directory in ['prediction_audit', 'geometry_determinant_audit']:
        assert json.loads((base/directory/'summary.json').read_text())['passed']
        hashes.update(json.loads((base/directory/'protocol.json').read_text())['source_sha256'])
    hashes.update({n: sha(src/n) for n in ['conditioned_mode_geometry.py', Path(__file__).name]})
    for n, h in hashes.items(): assert sha(src/n) == h, n
    p = json.loads((inp/'protocol.json').read_text()); loaded, manifest = suite.oldfit.meta.load(root)
    assert manifest == p['checkpoint_manifest']
    design = root/'outputs/ttt-pc-alm-research/264_common_geometry_prequery_protocol.md'
    dump(out/'protocol.json', dict(source_sha256=hashes, design_sha256=sha(design),
        original_before_query_manifest_sha256=sha(inp/'before_query_manifest.json'),
        query_targets_accessed=False, new_task_seed=5910048, old_task_seed=5900001,
        scope='known volumes, exact determinants and interiors, all46 old and affected-task fits'))
    begin = time.perf_counter(); counts = Counter(); boxes = []
    matrices = [np.diag([.08, .05, .03, .02]),
        np.array([[.05, .02, .01, 0], [0, .03, .01, .01], [0, 0, .025, .005], [0, 0, 0, .02]])]
    thin = matrices[-1].copy(); thin[1, 1] = 1e-5; matrices.append(thin)
    for i, transform in enumerate(matrices):
        inv = np.linalg.inv(transform); g = np.r_[inv, -inv]; rhs = np.ones(8)
        poly, note = corrected.conditioned_polytope(g, rhs)
        expected = 16*abs(float(np.linalg.det(transform)))
        assert np.isclose(poly['volume'], expected, rtol=1e-8, atol=1e-22)
        samples = shared.geometry.sample(poly, 4096, np.random.default_rng(264700+i))
        assert np.max(samples@g.T-rhs) < 1e-7
        boxes.append(dict(index=i, expected_volume=expected, measured_volume=poly['volume'], note=note))
        counts['known_volume_polytopes'] += 1; counts['known_polytope_particles'] += len(samples)
    with np.load(base/'geometry_diagnosis/geometry.npz') as z:
        poly, note = corrected.conditioned_polytope(z['g'], z['rhs'])
        point = poly['center']+poly['scale']*poly['interior']
        pattern = suite.cold.base.pattern(z['x_observed'], point)
        proof = shared.strict_interior(z['x_observed'], z['v_observed'], pattern, point)
        assert proof['accepted']; cube = check_cube(z['x_observed'], z['v_observed'], pattern.astype(np.uint8).tobytes().hex(), proof)
        samples = shared.geometry.sample(poly, 8192, np.random.default_rng(264731))
        maxsupport = float(np.max(abs(forward(z['x_observed'], samples)-z['v_observed'])))
        assert maxsupport <= .001+1e-7
        exact = check(poly['facets'].reshape(-1, 4), np.arange(len(poly['facets'])*4).reshape(-1, 4), poly['interior'])
        assert np.isclose(exact['exact_volume']*np.prod(poly['scale']), poly['volume'], rtol=1e-8, atol=1e-22)
        np.savez_compressed(out/'repaired_polytope.npz', **poly)
        dump(out/'polytope_checks.json', dict(note=note, exact=exact, proof=proof, cube=cube, max_support=maxsupport))
        counts['exact_interior_corners'] += cube['corners']; counts['independent_support_particles'] += len(samples)
    q = np.linspace(0, 1, 257); rows = []
    refs = {r['method']: r for r in json.loads((inp/'rows.json').read_text()) if r['seed'] == 5910048}
    with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
        x, v = observations(5900001)
        for cfg in p['configs']:
            aa, mm = suite.guarded_fit(cfg, x[:4], v[:4], q, 5900001, loaded)
            a, m = corrected.fit(cfg, x[:4], v[:4], q, 5900001, loaded)
            assert not mm['execution_failed'] and not m['execution_failed'] and m['geometry_repair_count'] == 0
            for k, value in a.items(): assert value.tobytes() == aa[k].tobytes(); counts['old_task_identical_arrays'] += 1
            counts['old_task_identical_predictors'] += 1
        x, v = observations(5910048)
        for cfg in p['configs']:
            a, m = corrected.fit(cfg, x[:4], v[:4], q, 5910048, loaded); ref = refs[cfg['name']]
            with np.load(inp/ref['file']) as original:
                if not ref['metadata']['execution_failed']:
                    assert not m['execution_failed'] and m['geometry_repair_count'] == 0
                    for k, value in a.items(): assert value.tobytes() == original[k].tobytes(); counts['affected_task_unchanged_arrays'] += 1
                    counts['affected_task_unchanged_successes'] += 1
                else:
                    assert m['geometry_repair_count'] > 0
                    counts['affected_task_original_failures'] += 1
                    counts['affected_task_repaired_successes'] += not m['execution_failed']
            path = out/f"5910048_{cfg['name']}.npz"; np.savez_compressed(path, **a)
            rows.append(dict(method=cfg['name'], metadata=m, file=path.name, sha256=sha(path)))
            print(json.dumps(dict(preflight_done=len(rows), total=46, corrected_failures=sum(r['metadata']['execution_failed'] for r in rows))), flush=True)
    assert counts['old_task_identical_predictors'] == 46 and counts['affected_task_unchanged_successes'] == 14
    assert counts['affected_task_original_failures'] == counts['affected_task_repaired_successes'] == 32
    dump(out/'boxes.json', boxes); dump(out/'rows.json', rows)
    ans = dict(passed=True, counts=counts, seconds=time.perf_counter()-begin, query_targets_accessed=False,
        protocol_sha256=sha(out/'protocol.json'), rows_sha256=sha(out/'rows.json'), boxes_sha256=sha(out/'boxes.json'),
        repaired_polytope_sha256=sha(out/'repaired_polytope.npz'), polytope_checks_sha256=sha(out/'polytope_checks.json'))
    dump(out/'summary.json', ans); print(json.dumps(ans), flush=True)


if __name__ == '__main__': main()
