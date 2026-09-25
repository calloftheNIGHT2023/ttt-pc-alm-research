"""347 old full-trial path/outputs plus watch and six credit controls, no targets."""
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import strong_pool_credit_bridge_v1 as bridge
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from audit_search_radius_development_v1 import same, load
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates

DESIGN = 'outputs/ttt-pc-alm-research/347_strong_pool_credit_bridge_protocol_v1.md'


def run(root, out):
    begin = time.perf_counter(); previous = root/'results/support_language_online'
    io.complete(previous/'development_predictions_v1'); io.complete(previous/'development_audit_v1')
    hashes = io.read(previous/'development_predictions_v1/protocol.json')['source_sha256'].copy()
    for name in [DESIGN, 'work/experiments/'+Path(__file__).name, 'work/experiments/strong_pool_credit_bridge_v1.py']:
        hashes[name] = io.sha(root/name)
    for name, digest in hashes.items(): assert io.sha(root/name) == digest
    old = root/'results/online_credit_fresh_pilot/pilot_predictions_v2'
    lookup = {(r['seed'], r['method']): r for r in io.read(old/'rows.json')}
    base_cfg = next(c for c in io.budget.resources.catalogue(root) if c['name'] == 'probe_all_alm64') if hasattr(io, 'budget') else next(c for c in io.resources.catalogue(root) if c['name'] == 'probe_all_alm64')
    configs = [('native', dict()), ('watch', dict(watch=True))]+[(ch, dict(channel=ch)) for ch in bridge.CHANNELS]
    io.save(out/'protocol.json', dict(source_sha256=hashes, seeds=list(range(328000000, 328000008)),
        configurations=[name for name, _ in configs], query_targets_accessed=False, posterior_reference_accessed=False))
    counts = dict(calls=0, baseline_output_arrays=0, trigger_arrays=0, trajectory_states=0,
                  support_particles=0, independent_certificates=0, original_reference_arrays=0)
    rows = []; files = {}
    with discovery_box(.12):
        for seed in range(328000000, 328000008):
            oldrow = lookup[seed, 'probe_all_alm64']
            assert io.sha(old/oldrow['file']) == oldrow['sha256'] and io.sha(old/oldrow['metadata_file']) == oldrow['metadata_sha256']
            source = load(old/oldrow['file']); oldmeta = io.read(old/oldrow['metadata_file'])['metadata']
            x, v, q = [source[n] for n in ['x_observed', 'v_observed', 'q_observed']]
            native = watch = native_meta = None
            for name, kwargs in configs:
                a, m = bridge.fit(x, v, q, seed, trace_hash=True, **kwargs)
                assert not m['execution_failed'] and not m['query_targets_accessed']
                assert m['visited_modes'] == oldmeta['visited_modes'] and m['feasible_modes'] == oldmeta['feasible_modes']
                assert m['original_positive_modes'] == oldmeta['positive_modes']
                counts['baseline_output_arrays'] += same(a, source, ['selected_b', 'best_bank', 'point_prediction'])
                if name in ['native', 'watch'] or m['positive_modes'] == m['original_positive_modes']:
                    counts['baseline_output_arrays'] += same(a, source, ['points', 'allocation', 'prediction'])
                assert set(oldmeta['positive_modes']) <= set(m['positive_modes'])
                if name == 'native': native, native_meta = a, m
                else:
                    assert m['trajectory_state_sha256'] == native_meta['trajectory_state_sha256']; counts['trajectory_states'] += len(m['trajectory_state_sha256'])
                    if name == 'watch': watch = a; watch_meta = m
                    else:
                        assert m['selected_state'] == watch_meta['selected_state']
                        counts['trigger_arrays'] += same(a, watch, ['trigger_b', 'trigger_h', 'trigger_u', 'trigger_location'])
                if m['positive_modes']:
                    y = np.broadcast_to(x, (len(a['points']), len(x)))
                    for j in range(4): y = np.maximum(0., 1.-abs(2*(y+a['points'][:, j, None])-1.))
                    assert float(np.max(abs(y-v))) <= .001+1e-7; counts['support_particles'] += len(a['points'])
                for key, note in m['new_mode_classifications_detail'].items():
                    aa, rhs = inequalities(x, v, key); counts['independent_certificates'] += certificates(aa, rhs, note)
                path = out/f'{seed}_{name}.npz'; mp = path.with_suffix('.json')
                with path.open('xb') as f: np.savez_compressed(f, **a)
                io.save(mp, m); files[path.name] = io.sha(path); files[mp.name] = io.sha(mp)
                rows.append(dict(seed=seed, method=name, file=path.name, metadata_file=mp.name,
                    seconds=m['charged_complete_seconds'], new_positive_modes=len(m['new_positive_modes']),
                    query_targets_accessed=False)); counts['calls'] += 1
            if seed == 328000000:
                # Independently run the old implementation, observing original steps only.
                old_step = cold.Local.step; fingerprints = []
                def observe(state):
                    if state.method == 'alm' and len(state.b) == 629:
                        if not fingerprints: fingerprints.append(bridge.state_digest(state))
                        result = old_step(state); fingerprints.append(bridge.state_digest(state)); return result
                    return old_step(state)
                try:
                    cold.Local.step = observe
                    a, m, _ = io.resources.invoke(base_cfg, x, v, q, seed, None)
                finally: cold.Local.step = old_step
                assert not m['execution_failed'] and fingerprints == native_meta['trajectory_state_sha256']
                counts['original_reference_arrays'] += same(a, native, list(native)); counts['trajectory_states'] += len(fingerprints)
            print(dict(tasks=seed-328000000+1, total=8, calls=counts['calls'], seconds=time.perf_counter()-begin), flush=True)
    for name, digest in hashes.items(): assert io.sha(root/name) == digest
    io.save(out/'rows.json', rows); io.save(out/'files.json', files)
    summary = dict(passed=True, counts=counts, seconds=time.perf_counter()-begin,
        query_targets_accessed=False, posterior_reference_accessed=False, core_research_goal_complete=False,
        outputs_sha256={name: io.sha(out/name) for name in ['protocol.json', 'rows.json', 'files.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/strong_pool_credit_bridge/preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
