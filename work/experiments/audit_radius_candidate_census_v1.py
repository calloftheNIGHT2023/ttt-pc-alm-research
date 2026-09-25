"""343 sealed support-only proposal census and exact finite-K exclusion reasons."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_dyadic_v1 as integer
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates

DESIGN = 'outputs/ttt-pc-alm-research/343_radius_candidate_census_protocol_v1.md'
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
STRONG = ['probe_all_alm64', 'probe_alm64_33', 'probe_then_adam1920_33', 'probe_then_adam3840_33']
SEEDS = list(range(328000000, 328000032))


def arrays(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def distance(a, b):
    aa, bb = bytes.fromhex(a), bytes.fromhex(b)
    assert len(aa) == len(bb) == 16
    return sum(x != y for x, y in zip(aa, bb))


def run(root, out):
    start = time.perf_counter()
    folder = root/'results/search_radius_development/development_predictions_v1'
    sealed = io.complete(folder)
    assert sealed['tasks'] == 32 and sealed['checks']['failures'] == 0
    reachable = root/'results/search_radius_reachability/audit_v1'
    io.complete(reachable)
    reach_files = io.read(reachable/'files.json')
    rows = io.read(folder/'rows.json')
    lookup = {(r['seed'], r['method']): r for r in rows}
    source_hashes = io.read(folder/'protocol.json')['source_sha256'].copy()
    for name in [DESIGN, 'work/experiments/'+Path(__file__).name,
                 'work/experiments/audit_post_escape_geometry_v1.py',
                 'work/experiments/audit_stasis_escape_geometry_v1.py']:
        digest = io.sha(root/name)
        if name in source_hashes:
            assert digest == source_hashes[name]
        source_hashes[name] = digest
    for name, digest in source_hashes.items():
        assert io.sha(root/name) == digest
    io.save(out/'protocol.json', dict(seeds=SEEDS, channels=CHANNELS, strong_methods=STRONG,
        source_sha256=source_hashes, prediction_summary_sha256=io.sha(folder/'summary.json'),
        rows_sha256=io.sha(folder/'rows.json'), reachability_summary_sha256=io.sha(reachable/'summary.json'),
        query_targets_accessed=False, posterior_reference_accessed=False, new_blind_tasks=False))
    counts = Counter(); tasks = []; input_files = {}; task_files = {}
    def load(seed, method):
        r = lookup[seed, method]
        for name, field in [('file', 'sha256'), ('metadata_file', 'metadata_sha256')]:
            assert io.sha(root/r[name]) == r[field]
            input_files[r[name]] = r[field]
        a = arrays(root/r['file']); m = io.read(root/r['metadata_file'])['metadata']
        assert not m['execution_failed'] and not m['query_targets_accessed']
        return a, m
    for seed in SEEDS:
        pairs = []; primary = None
        for channel in CHANNELS:
            old_a, old = load(seed, 'online_first_fit_'+channel)
            new_a, new = load(seed, 'radius_first_fit_'+channel+'__k8__sall')
            assert old_a.keys() == new_a.keys()
            assert old['original_positive_modes'] == new['original_positive_modes']
            assert old['selected_state'] == new['selected_state']
            p_old = {p['mode']: p for p in old['proposal']['proposals']}
            p_new = {p['mode']: p for p in new['proposal']['proposals']}
            assert len(p_old) == len(old['proposal']['proposals'])
            assert len(p_new) == len(new['proposal']['proposals'])
            assert p_old.keys() <= p_new.keys()
            assert all(p == p_new[k] for k, p in p_old.items())
            assert set(old['positive_modes']) <= set(new['positive_modes'])
            equal = {}
            for name, a in old_a.items():
                b = new_a[name]
                equal[name] = a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
                counts['bitwise_array_comparisons'] += 1
            extra = []
            for key in sorted(p_new.keys()-p_old.keys()):
                entry = dict(p_new[key], in_old_final_pool=key in old['positive_modes'],
                             in_original_trajectory_pool=key in new['original_positive_modes'])
                if entry['in_original_trajectory_pool']:
                    entry['classification'] = 'original_positive_pool_cache_hit'
                else:
                    note = new['new_mode_classifications_detail'][key]
                    a, rhs = inequalities(new_a['x_observed'], new_a['v_observed'], key)
                    counts['independent_certificates'] += certificates(a, rhs, note)
                    entry.update(classification=note['classification'],
                        positive_volume_certified=note['positive_volume_certified'],
                        numerical_volume_available=note['numerical_volume_available'])
                extra.append(entry)
            record = dict(channel=channel, old_proposals=len(p_old), all_proposals=len(p_new),
                added_proposals=len(extra), added_proposal_classifications=dict(Counter(e['classification'] for e in extra)),
                extra_proposals=extra, arrays_bitwise_equal=equal,
                pool_identical=old['positive_modes'] == new['positive_modes'],
                new_positive_modes_vs_three_shell=sorted(set(new['positive_modes'])-set(old['positive_modes'])))
            pairs.append(record)
            if channel == 'dual':
                primary = (new_a, new, p_new)
        data, main, proposals = primary
        old_reach_file = f'task_{seed}.json'
        assert io.sha(reachable/old_reach_file) == reach_files[old_reach_file]
        input_files[str((reachable/old_reach_file).relative_to(root))] = reach_files[old_reach_file]
        reach = io.read(reachable/old_reach_file)
        strong_modes = {}
        for method in STRONG:
            _, meta = load(seed, method)
            strong_modes[method] = set(meta['positive_modes'])
        missing = set().union(*strong_modes.values())-set(main['positive_modes'])
        problem = integer.IntegerProblem(data['x_observed'], data['v_observed'], data['trigger_credit'])
        ranked = {}
        for p in proposals.values():
            ranked.setdefault(p['hamming'], []).append((Fraction(p['lower']), p['mode']))
        ranked = {h: sorted(r) for h, r in ranked.items()}
        entries = []
        for key in sorted(missing):
            note = reach['certificates'][key]
            a, rhs = inequalities(data['x_observed'], data['v_observed'], key)
            counts['independent_certificates'] += certificates(a, rhs, note)
            assert note['positive_volume_certified']
            pattern = np.array(list(bytes.fromhex(key))).reshape(4, 4)
            bound = problem.fixed(pattern)
            assert bound is not None and bound <= 0, 'A feasible path was incorrectly excluded by the relaxation'
            h = distance(main['selected_state']['original_mode'], key)
            score = Fraction(bound, problem.denominator)
            edge = ranked.get(h, [])
            if h == 0:
                reason = 'original_pattern_not_requested'
            elif not edge:
                raise AssertionError('A positive feasible path has no extracted shell')
            else:
                assert len(edge) == 8 and (score, key) > edge[-1], 'Missing feasible path should have been in exact K-best'
                reason = 'beyond_top8_in_its_shell'
            entries.append(dict(mode=key, hamming=h, lower=str(score), reason=reason,
                reference_methods=[m for m in STRONG if key in strong_modes[m]],
                best_returned_lower=str(edge[0][0]) if edge else None,
                kth_returned_lower=str(edge[-1][0]) if edge else None,
                strictly_larger_lower_than_kth=score > edge[-1][0] if edge else None))
        task = dict(seed=seed, channel_pairs=pairs, primary_missing_positive_modes=entries,
                    support_fit=Fraction(main['selected_state']['exact_support_loss']) == 0)
        name = f'task_{seed}.json'; io.save(out/name, task); task_files[name] = io.sha(out/name)
        tasks.append(task); counts['tasks'] += 1
        if len(tasks) % 8 == 0:
            print(dict(tasks=len(tasks), total=32, seconds=time.perf_counter()-start), flush=True)
    aggregates = []
    for channel in CHANNELS:
        rr = [next(p for p in t['channel_pairs'] if p['channel'] == channel) for t in tasks]
        cc = Counter()
        for p in rr: cc.update(p['added_proposal_classifications'])
        aggregates.append(dict(channel=channel, tasks=len(rr), added_proposals=sum(p['added_proposals'] for p in rr),
            added_classifications=dict(cc), identical_pool_tasks=sum(p['pool_identical'] for p in rr),
            new_positive_pairs=sum(len(p['new_positive_modes_vs_three_shell']) for p in rr),
            identical_all_array_tasks=sum(all(p['arrays_bitwise_equal'].values()) for p in rr),
            identical_prediction_tasks=sum(p['arrays_bitwise_equal']['prediction'] for p in rr),
            changed_array_tasks=dict(Counter(k for p in rr for k, equal in p['arrays_bitwise_equal'].items() if not equal))))
    missing = [e for t in tasks for e in t['primary_missing_positive_modes']]
    missed = dict(pairs=len(missing), tasks=sum(bool(t['primary_missing_positive_modes']) for t in tasks),
        reasons=dict(Counter(e['reason'] for e in missing)),
        strictly_lower_rank_not_ties=sum(e['strictly_larger_lower_than_kth'] is True for e in missing),
        hamming_histogram=dict(Counter(str(e['hamming']) for e in missing)))
    for name, digest in source_hashes.items(): assert io.sha(root/name) == digest
    for name, value in [('channels.json', aggregates), ('missing.json', missed),
                        ('files.json', task_files), ('inputs.json', input_files)]:
        io.save(out/name, value)
    summary = dict(passed=True, counts=dict(counts), channels=aggregates, primary_missing=missed,
        seconds=time.perf_counter()-start, query_targets_accessed=False, posterior_reference_accessed=False,
        independent_task_gain_established=False, core_research_goal_complete=False,
        outputs_sha256={n: io.sha(out/n) for n in ['protocol.json', 'channels.json', 'missing.json', 'files.json', 'inputs.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]
    out = root/'results/radius_candidate_census/audit_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
