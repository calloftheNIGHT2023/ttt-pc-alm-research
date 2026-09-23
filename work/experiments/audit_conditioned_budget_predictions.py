"""264 all-file pre-query audit, independent geometry and forward checks."""
import argparse
from collections import Counter
import hashlib
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
from audit_local_dual_jump_modes import modes
from analyze_recovered_online_comparison import forward
from run_independent_hybrid_memory import observations
from run_multiplier_fixed_point_screen import sha, dump


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--project', type=Path, required=True)
    root = ap.parse_args().project.resolve(); src = Path(__file__).parent
    base = root/'results/matched_budget_confirmation'; inp = base/'conditioned_confirmation'
    old = base/'confirmation'; out = base/'conditioned_prediction_audit'
    out.mkdir(parents=True, exist_ok=True); assert not (out/'protocol.json').exists()
    assert os.environ.get('OPENBLAS_NUM_THREADS') == os.environ.get('OMP_NUM_THREADS') == '1'
    torch.set_num_threads(1); torch.set_num_interop_threads(1)
    p = json.loads((inp/'protocol.json').read_text()); summary = json.loads((inp/'summary.json').read_text())
    assert summary['passed'] and summary['predictions_complete'] and not summary['query_targets_accessed']
    hashes = dict(p['source_sha256']); hashes[Path(__file__).name] = sha(Path(__file__))
    for n, h in hashes.items(): assert sha(src/n) == h, n
    before = json.loads((inp/'before_query_manifest.json').read_text())
    assert sha(inp/'before_query_manifest.json') == summary['before_query_manifest_sha256']
    for n in ['protocol', 'rows', 'memory']: assert sha(inp/f'{n}.json') == before[f'{n}_sha256']
    assert sha(old/'before_query_manifest.json') == p['original_before_query_manifest_sha256']
    oldbefore = json.loads((old/'before_query_manifest.json').read_text())
    assert sha(old/'rows.json') == oldbefore['rows_sha256']
    references = {(r['seed'], r['method']): r for r in json.loads((old/'rows.json').read_text())}
    dump(out/'protocol.json', dict(source_sha256=hashes, prediction_summary_sha256=sha(inp/'summary.json'),
        query_targets_accessed=False, scope='all2944 files, all independent forwards and particles; original2912 bytewise; primary64 and repaired32 fresh pipelines, distinct repair geometry exact checks'))
    rows = json.loads((inp/'rows.json').read_text()); cfgs = {c['name']: c for c in p['configs']}
    assert len(rows) == 2944 and len({(r['seed'], r['method']) for r in rows}) == 2944
    assert {(r['seed'], r['method']) for r in rows} == {(s, c) for s in p['seeds'] for c in cfgs}
    loaded, manifest = suite.oldfit.meta.load(root); assert manifest == p['checkpoint_manifest']
    counts = Counter(); gaps = []; geometries = {}; maxgap = 0.; maxsupport = 0.; begin = time.perf_counter(); q = np.linspace(0, 1, 257)
    saved = corrected.conditioned_polytope
    context = {}
    def capture(g, rhs):
        poly, note = saved(g, rhs); key = hashlib.sha256(g.tobytes()+rhs.tobytes()).hexdigest()
        if key not in geometries:
            point = poly['center']+poly['scale']*poly['interior']
            pattern = suite.cold.base.pattern(context['x'], point)
            proof = shared.strict_interior(context['x'], context['v'], pattern, point); assert proof['accepted']
            cube = check_cube(context['x'], context['v'], pattern.astype(np.uint8).tobytes().hex(), proof)
            exact = check(poly['facets'].reshape(-1, 4), np.arange(len(poly['facets'])*4).reshape(-1, 4), poly['interior'])
            assert np.isclose(exact['exact_volume']*np.prod(poly['scale']), poly['volume'], rtol=1e-8, atol=1e-22)
            assert note['boundary_relative_residual'] < 1e-10
            filename = f'geometry_{key}.npz'; np.savez_compressed(out/filename, g=g, original_rhs=rhs, **poly)
            geometries[key] = dict(file=filename, sha256=sha(out/filename), note=note, proof=proof, cube=cube, exact=exact)
            counts['distinct_exact_geometry_checks'] += 1; counts['exact_geometry_cube_corners'] += 16
        return poly, note
    corrected.conditioned_polytope = capture
    try:
        with suite.cold.frozen.original.old.core.pipeline.discovery_box(.12):
            for row in rows:
                path = inp/row['file']; assert sha(path) == row['sha256'] == before['prediction_files'][row['file']]
                x, v = observations(row['seed']); x = x[:4]; v = v[:4]; context.update(x=x, v=v)
                m = row['metadata']; cfg = cfgs[row['method']]; ref = references[row['seed'], row['method']]
                assert row['original_execution_failed'] == ref['metadata']['execution_failed']
                assert sha(old/ref['file']) == ref['sha256'] == oldbefore['prediction_files'][ref['file']]
                with np.load(path) as a:
                    assert x.tobytes() == a['x_observed'].tobytes() and v.tobytes() == a['v_observed'].tobytes() and q.tobytes() == a['q_observed'].tobytes()
                    assert row['seconds'] == m['charged_complete_seconds'] > 0; counts['input_files'] += 1
                    for k in a.files:
                        if np.issubdtype(a[k].dtype, np.number): assert np.isfinite(a[k]).all()
                    if not row['original_execution_failed']:
                        assert m['geometry_repair_count'] == 0 and not m['execution_failed']
                        with np.load(old/ref['file']) as z:
                            for k in a.files: assert a[k].tobytes() == z[k].tobytes(), (row['seed'], row['method'], k)
                        counts['original_successes_bytewise'] += 1
                    if m['execution_failed']:
                        expected = forward(q, np.zeros((1, 4)))[0]; counts['fixed_fallbacks'] += 1
                    elif row['readout'] == 'mode':
                        points = a['points']; expected = forward(q, points).mean(0); counts['independent_mode_predictions'] += 1
                        if m['positive_modes']:
                            assert len(points) == 2048; support = float(np.max(abs(forward(x, points)-v)))
                            assert support <= .001+1e-7; maxsupport = max(maxsupport, support)
                            assert set(modes(x, points)) <= set(m['positive_modes']) <= set(m['feasible_modes'])
                            if 'visited_modes' in m: assert set(m['feasible_modes']) <= set(m['visited_modes']) and m['lp_calls'] == len(m['visited_modes'])
                            else: assert len(m['feasible_modes']) <= m['observed_modes'] == m['lp_calls']
                            counts['support_and_membership_particles'] += len(points)
                        else:
                            assert len(points) == 1 and points[0].tobytes() == a['selected_b'].tobytes(); counts['empty_pool_fallbacks'] += 1
                        assert np.max(abs(forward(q, a['selected_b'][None])[0]-a['point_prediction'])) < 1e-12
                    else:
                        assert not row['original_execution_failed']
                        with np.load(old/ref['file']) as z: expected = z['prediction']
                        counts['heads_inherit_independent_original_audit'] += 1
                    gap = float(np.max(abs(expected-a['prediction']))); assert gap < 1e-12; maxgap = max(maxgap, gap)
                    gaps.append(dict(seed=row['seed'], method=row['method'], gap=gap))
                    if row['method'] == p['primary'] or row['original_execution_failed']:
                        aa, mm = corrected.fit(cfg, x, v, q, row['seed'], loaded)
                        assert mm['execution_failed'] == m['execution_failed'] and mm['geometry_repair_log'] == m['geometry_repair_log']
                        for k, value in aa.items(): assert value.tobytes() == a[k].tobytes(); counts['fresh_replay_arrays'] += 1
                        counts['fresh_primary_pipelines'] += row['method'] == p['primary']
                        counts['fresh_original_failure_pipelines'] += row['original_execution_failed']
                if counts['input_files'] % len(cfgs) == 0:
                    print(json.dumps(dict(audited=counts['input_files'], total=len(rows), seconds=time.perf_counter()-begin)), flush=True)
    finally:
        corrected.conditioned_polytope = saved
    assert counts['fresh_primary_pipelines'] == 64 and counts['fresh_original_failure_pipelines'] == 32
    assert counts['original_successes_bytewise'] == 2912 and counts['fixed_fallbacks'] == summary['failures']
    memory = json.loads((inp/'memory.json').read_text()); assert len(memory) == 138
    assert {(r['seed'], r['method']) for r in memory} == {(s, c) for s in p['memory_seeds'] for c in cfgs}
    for r in memory: assert r['traced_peak_bytes'] >= r['traced_current_bytes'] >= 0
    dump(out/'prediction_gaps.json', gaps); dump(out/'geometries.json', geometries)
    ans = dict(passed=True, counts=counts, max_independent_prediction_gap=maxgap, max_support_particle_error=maxsupport,
        memory_rows=len(memory), seconds=time.perf_counter()-begin, protocol_sha256=sha(out/'protocol.json'),
        prediction_gaps_sha256=sha(out/'prediction_gaps.json'), geometries_sha256=sha(out/'geometries.json'), query_targets_accessed=False)
    dump(out/'summary.json', ans); print(json.dumps(ans), flush=True)


if __name__ == '__main__': main()
