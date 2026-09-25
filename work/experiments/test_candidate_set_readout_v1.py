"""382 prototype-only common geometry invariants; no queried teacher targets."""
from collections import Counter
from pathlib import Path
import time
import traceback
import numpy as np
import candidate_set_readout_v1 as model
import complete_credit_mode_geometry_v1 as classifier
import run_retired_region_join_v1 as prior


def main(root, out):
    begin = time.perf_counter(); counts = Counter(); files = {}
    sources = ['candidate_set_readout_v1.py', 'test_candidate_set_readout_v1.py',
               'online_credit_branch_search_v1.py', 'shared_mode_readout.py', 'complete_credit_mode_geometry_v1.py',
               'conditioned_mode_geometry.py', 'region_posterior_memory.py', 'verify_known_range_projection_v1.py']
    hashes = prior.hashes(root)
    for p in sources: hashes['work/experiments/'+p] = prior.sha(root/'work/experiments'/p)
    base = model.shared.geometry.base
    for n in [4, 24]:
        rng = np.random.default_rng(382071+n); x = rng.uniform(.05, .95, n); b = rng.uniform(-.1, .1, 4)
        v = base.forward(x, b); q = np.linspace(0., 1., 257)
        good = base.pattern(x, b)[None].astype(np.uint8); bad = np.zeros_like(good)
        assert not np.array_equal(good, bad) and np.any(v > .01)
        regions = np.concatenate([good, bad]); states = []
        for label, rs, completed in [('single', good, True), ('mixed', regions, True),
                                      ('permuted', regions[::-1].copy(), True), ('incomplete', regions, False),
                                      ('empty', np.empty((0, 4, n), np.uint8), False)]:
            a, m = model.fit(x, v, q, rs, seed=382071+n, search_completed=completed)
            assert not m['query_targets_accessed'] and not m['global_bp_used']
            assert not m['hidden_recovery_used'] and not m['cross_method_cache_used']
            assert m['returned_array_bytes'] == sum(t.nbytes for t in a.values())
            assert m['classified_modes'] == sorted(r.tobytes().hex() for r in rs)
            for key, note in m['classifications_detail'].items():
                _, matrix, rhs, _, _ = classifier.matrices(x, v, key)
                counts['exact_geometry_certificates'] += classifier.verify_certificates(matrix, rhs, note)
            if label != 'empty':
                assert m['readout_available'] and not m['unresolved_modes']
                assert m['positive_modes'] == [good[0].tobytes().hex()]
                assert a['points'].shape == (2048, 4) and a['prediction'].shape == q.shape
                assert np.all(np.isfinite(a['prediction'])) and np.all((a['prediction'] >= 0) & (a['prediction'] <= 1))
                for point in a['points']:
                    assert np.max(abs(base.forward(x, point)-v)) <= .001+1e-7
                    counts['posterior_support_checks'] += 1
                assert m['conditional_on_output_union'] == (not completed)
                assert m['full_candidate_set_resolved'] == completed
            else:
                assert not m['readout_available'] and a['prediction'].size == a['points'].size == 0
                assert m['conditional_on_output_union'] and not m['full_candidate_set_resolved']
                counts['explicit_empty_readout'] += 1
            states.append(a)
            directory = out/f'n{n}_{label}'; directory.mkdir()
            np.savez_compressed(directory/'arrays.npz', **a); prior.save(directory/'metadata.json', m)
            for f in ['arrays.npz', 'metadata.json']: files[f'n{n}_{label}/{f}'] = prior.sha(directory/f)
            counts['prototype_calls'] += 1
        for other in states[1:4]:
            keys = ['points', 'allocation', 'positive_volumes', 'prediction']
            counts['readout_invariance_arrays'] += prior.same({k: states[0][k] for k in keys}, {k: other[k] for k in keys})
    assert all(prior.sha(root/p) == h for p, h in hashes.items())
    result = dict(passed=True, checks=dict(counts), source_sha256=hashes, seconds=time.perf_counter()-begin,
        query_targets_accessed=False, actual_task_search_tested=False, online_cost_advantage_tested=False,
        outputs_sha256=files)
    prior.save(out/'summary.json', result)
    print(dict(passed=True, checks=dict(counts), seconds=result['seconds']), flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/candidate_set_readout/prototype_preflight_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: main(root, out)
    except Exception:
        prior.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
