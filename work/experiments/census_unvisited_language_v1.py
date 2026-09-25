"""350 support-only all-task/six-credit census, not an online cost benchmark."""
from collections import Counter
from contextlib import contextmanager
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import scipy.optimize as opt
import budget_reinvestment_suite_v1 as io
import cold_stagnation_switch as cold
import support_language_chain_v1 as old
import unvisited_language_chain_v1 as new
import branch_image_chain_v1 as reference
import online_credit_branch_search_v1 as geometry
from test_support_language_chain_v1 import possible
from test_unvisited_language_chain_v1 import source_hashes
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates
from posterior_confirmation_pipeline import discovery_box

CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']


@contextmanager
def guarded_selector():
    saved = []
    def forbidden(*args, **kwargs): raise AssertionError('Global solver entered local branch selector')
    try:
        for obj, name in [(cold.bp, 'evaluate'), (cold.bp, 'refine'), (opt, 'linprog'), (opt, 'minimize')]:
            saved.append((obj, name, getattr(obj, name))); setattr(obj, name, forbidden)
        yield
    finally:
        for obj, name, method in saved: setattr(obj, name, method)


def run(root, out):
    begin = time.perf_counter(); hashes = source_hashes(root)
    pre = root/'results/unvisited_language/preflight_v1'; io.complete(pre)
    assert io.read(pre/'protocol.json')['source_sha256'] == hashes
    base = root/'results/strong_pool_online/development_predictions_v1'; io.complete(base)
    source_rows = io.read(base/'rows.json')
    lookup = {(r['seed'], r['method']): r for r in source_rows if r['method'].startswith('strong_language_')}
    assert len(lookup) == 32*6
    protocol = dict(source_sha256=hashes, preflight_summary_sha256=io.sha(pre/'summary.json'),
        parent_summary_sha256=io.sha(base/'summary.json'), seeds=list(range(328000000, 328000032)), channels=CHANNELS,
        k=8, query_targets_accessed=False, posterior_reference_accessed=False, offline_saved_state_census=True,
        cost_scope='Loaded saved triggers, shared geometry; not end-to-end online cost or fair benchmark')
    io.save(out/'protocol.json', protocol); counts = Counter(); records = []
    aggregate = {ch: Counter() for ch in CHANNELS}; times = {ch: dict(old=0., new=0.) for ch in CHANNELS}
    with discovery_box(.12):
        for seed in protocol['seeds']:
            directory = out/str(seed); directory.mkdir(); calls = {}; original = {}; task_x = task_v = None
            for ch in CHANNELS:
                r = lookup[seed, 'strong_language_'+ch]
                assert io.sha(root/r['file']) == r['sha256'] and io.sha(root/r['metadata_file']) == r['metadata_sha256']
                meta = io.read(root/r['metadata_file'])['metadata']
                assert not meta['query_targets_accessed'] and not meta['execution_failed']
                with np.load(root/r['file'], allow_pickle=False) as z:
                    x, v, credit = z['x_observed'], z['v_observed'], z['trigger_credit']
                if task_x is None: task_x, task_v = x.copy(), v.copy()
                assert x.tobytes() == task_x.tobytes() and v.tobytes() == task_v.tobytes()
                pattern = np.array(list(bytes.fromhex(meta['selected_state']['original_mode']))).reshape(4, 4)
                visited = set(meta['visited_modes']); original[ch] = meta
                with guarded_selector():
                    old_result = old.propose(x, v, credit, pattern, k=8)
                    proposed = new.propose(x, v, credit, pattern, k=8, forbidden=sorted(visited))
                assert {k: val for k, val in old_result.items() if k != 'meta'} == {k: val for k, val in meta['proposal'].items() if k != 'meta'}
                counts['original_proposal_replays'] += 1
                oldkeys = {p['mode'] for p in old_result['proposals']}; newkeys = {p['mode'] for p in proposed['proposals']}
                assert not newkeys & visited and oldkeys-visited <= newkeys
                for p in proposed['proposals']:
                    pat = np.array(list(bytes.fromhex(p['mode']))).reshape(4, 4)
                    assert all(possible(x[i], v[i], pat[:, i]) for i in range(4))
                    assert reference.fixed_value(x, v, credit, pat) == F(p['lower']) <= 0
                    counts['independent_candidate_bounds'] += 1
                summary = dict(old_candidates=len(oldkeys), old_visited_slots=len(oldkeys & visited),
                    old_unvisited_candidates=len(oldkeys-visited), new_candidates=len(newkeys),
                    additional_candidates=len(newkeys-oldkeys), max_dp_entries=proposed['meta']['max_retained_dp_entries'],
                    trie_nodes=proposed['meta']['trie_nodes'])
                calls[ch] = dict(seed=seed, channel=ch, source_file=r['file'], source_sha256=r['sha256'],
                    source_metadata_file=r['metadata_file'], source_metadata_sha256=r['metadata_sha256'],
                    summary=summary, old=old_result, new=proposed)
                times[ch]['old'] += old_result['meta']['total_seconds']; times[ch]['new'] += proposed['meta']['total_seconds']
            common_visited = original['dual']['visited_modes']; common_parent = original['dual']['original_positive_modes']
            assert all(m['visited_modes'] == common_visited and m['original_positive_modes'] == common_parent for m in original.values())
            keys = sorted({p['mode'] for c in calls.values() for p in c['new']['proposals']}); geo = {}
            tick = time.perf_counter()
            for key in keys:
                poly, note = geometry.explicit_geometry(task_x, task_v, key)
                aa, rhs = inequalities(task_x, task_v, key); counts['independent_certificates'] += certificates(aa, rhs, note)
                assert (poly is not None) == note['numerical_volume_available']; geo[key] = note
            geometry_seconds = time.perf_counter()-tick; counts['unique_geometry_calls'] += len(keys)
            for ch, call in calls.items():
                positive = {p['mode'] for p in call['new']['proposals'] if geo[p['mode']]['numerical_volume_available']}
                oldpositive = set(original[ch]['positive_modes']); combined = set(common_parent) | positive
                assert oldpositive <= combined
                call['new_positive_modes'] = sorted(positive)
                call['new_positive_beyond_old_method'] = sorted(combined-oldpositive)
                call['classifications'] = dict(Counter(geo[p['mode']]['classification'] for p in call['new']['proposals']))
                for key, val in call['summary'].items():
                    if key not in ['max_dp_entries', 'trie_nodes']: aggregate[ch][key] += val
                aggregate[ch]['new_positive_pairs'] += len(positive)
                aggregate[ch]['extra_positive_beyond_old_method'] += len(combined-oldpositive)
                aggregate[ch]['tasks_with_extra_positive'] += bool(combined-oldpositive)
                aggregate[ch]['tasks_with_new_positive'] += bool(positive)
                aggregate[ch]['maximum_dp_entries'] = max(aggregate[ch]['maximum_dp_entries'], call['summary']['max_dp_entries'])
                aggregate[ch]['maximum_trie_nodes'] = max(aggregate[ch]['maximum_trie_nodes'], call['summary']['trie_nodes'])
            io.save(directory/'calls.json', calls); io.save(directory/'geometry.json', geo)
            row = dict(seed=seed, files={n: io.sha(directory/n) for n in ['calls.json', 'geometry.json']},
                       shared_diagnostic_geometry_seconds=geometry_seconds)
            io.save(directory/'commit.json', row); records.append(row)
            print(dict(task=len(records), total=32, unique_geometry_calls=counts['unique_geometry_calls'],
                       seconds=time.perf_counter()-begin, query_targets_accessed=False), flush=True)
    for path, digest in hashes.items(): assert io.sha(root/path) == digest
    io.save(out/'tasks.json', records)
    summary = dict(passed=True, tasks=32, calls=192, counts=dict(counts), aggregate={c: dict(a) for c, a in aggregate.items()},
        kernel_seconds_not_online_benchmark=times, seconds=time.perf_counter()-begin, query_targets_accessed=False,
        posterior_reference_accessed=False, offline_saved_state_census=True, core_research_goal_complete=False,
        outputs_sha256={n: io.sha(out/n) for n in ['protocol.json', 'tasks.json']})
    io.save(out/'summary.json', summary); print(dict(passed=True, tasks=32, aggregate=summary['aggregate'], counts=dict(counts)), flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/unvisited_language/census_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
