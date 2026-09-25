"""351 exhaustive full-pattern ranking with independent rational layer costs."""
from bisect import insort
from collections import Counter
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
from math import lcm
from pathlib import Path
import time
import traceback
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as reference
from test_support_language_chain_v1 import possible
from audit_stasis_escape_geometry_v1 import inequalities
from audit_post_escape_geometry_v1 import certificates

CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
DESIGN = 'outputs/ttt-pc-alm-research/351_unvisited_rank_audit_protocol_v1.md'


def enumerate_rank(x, v, credit, original, visited, languages):
    n, d = len(x), len(credit)
    xx = tuple(F(float(t)) for t in x); aa = tuple(tuple(F(float(t)) for t in row) for row in credit)
    scale_x = lcm(*(F(float(t)).denominator for t in [*x, *v, .12, .001, .5]))
    scale_a = lcm(*(a.denominator for row in aa for a in row)); scale = scale_x*scale_a
    zero = (F(0),)*n; counts = Counter(); pool = {}
    @lru_cache(None)
    def value(j, incoming, row):
        low = xx if j == 0 else zero
        high = low if j == 0 else tuple(F(int(bool(incoming & (1 << i)))) for i in range(n))
        val = reference.layer_value(low, high, zero if j == 0 else aa[j-1], aa[j], row)
        if val is None: return None
        if j == d-1:
            terminal = reference.terminal(v, aa[-1], sum(1 << i for i, r in enumerate(row) if r in (1, 2)))
            if terminal is None: return None
            val += terminal
        assert scale % val.denominator == 0
        return val.numerator*(scale//val.denominator)
    flat_original = bytes(int(r) for row in original for r in row)
    forbidden = {bytes.fromhex(key) for key in visited}
    for columns in product(*languages):
        rows = tuple(zip(*columns)); flat = bytes(r for row in rows for r in row)
        counts['enumerated_patterns'] += 1
        if flat in forbidden:
            counts['forbidden_patterns'] += 1; continue
        cost = 0; incoming = -1
        for j, row in enumerate(rows):
            term = value(j, incoming, row)
            if term is None: break
            cost += term; incoming = sum(1 << i for i, r in enumerate(row) if r in (1, 2))
        else:
            distance = sum(a != b for a, b in zip(flat, flat_original))
            values = pool.setdefault(distance, []); insort(values, (cost, flat))
            if len(values) > 8: values.pop()
            counts['admissible_patterns'] += 1
    counts['independent_fraction_transitions'] = value.cache_info().currsize
    minima = {str(h): str(F(values[0][0], scale)) for h, values in sorted(pool.items())}
    proposals = [dict(mode=flat.hex(), hamming=h, rank=rank, lower=str(F(cost, scale)))
                 for h, values in sorted(pool.items()) if h >= 1 for rank, (cost, flat) in enumerate(values) if cost <= 0]
    return minima, proposals, counts


def run(root, out):
    start = time.perf_counter(); census = root/'results/unvisited_language/census_v1'; io.complete(census)
    original_summary = io.read(census/'summary.json'); cp = io.read(census/'protocol.json')
    hashes = cp['source_sha256'].copy()
    for p in [DESIGN, 'work/experiments/'+Path(__file__).name]: hashes[p] = io.sha(root/p)
    for p, digest in hashes.items(): assert io.sha(root/p) == digest
    io.save(out/'protocol.json', dict(source_sha256=hashes, census_summary_sha256=io.sha(census/'summary.json'),
        all_real_patterns_enumerated=True, query_targets_accessed=False, posterior_reference_accessed=False))
    counts = Counter(); aggregate = {ch: Counter() for ch in CHANNELS}; witnesses = []; exclusive = {ch: [] for ch in CHANNELS}
    for record in io.read(census/'tasks.json'):
        seed = record['seed']; directory = census/str(seed)
        assert io.read(directory/'commit.json') == record
        for name, digest in record['files'].items(): assert io.sha(directory/name) == digest
        calls = io.read(directory/'calls.json'); geo = io.read(directory/'geometry.json')
        first = calls['dual']; assert io.sha(root/first['source_file']) == first['source_sha256']
        with np.load(root/first['source_file'], allow_pickle=False) as z: x, v = z['x_observed'], z['v_observed']
        languages = [[p for p in product(range(4), repeat=4) if possible(x[i], v[i], p)] for i in range(4)]
        counts['independent_single_observation_paths'] += 4*4**4
        for key, note in geo.items():
            aa, rhs = inequalities(x, v, key); counts['independent_certificates'] += certificates(aa, rhs, note)
        positive_by = {}
        for ch in CHANNELS:
            call = calls[ch]
            assert io.sha(root/call['source_file']) == call['source_sha256']
            assert io.sha(root/call['source_metadata_file']) == call['source_metadata_sha256']
            meta = io.read(root/call['source_metadata_file'])['metadata']
            with np.load(root/call['source_file'], allow_pickle=False) as z:
                assert z['x_observed'].tobytes() == x.tobytes() and z['v_observed'].tobytes() == v.tobytes()
                credit = z['trigger_credit']
            original = np.array(list(bytes.fromhex(meta['selected_state']['original_mode']))).reshape(4, 4)
            minima, proposed, checks = enumerate_rank(x, v, credit, original, meta['visited_modes'], languages)
            assert minima == call['new']['shell_minima'] and proposed == call['new']['proposals']
            counts.update(checks); counts['exact_complete_rankings'] += 1
            oldkeys = {p['mode'] for p in call['old']['proposals']}; newkeys = {p['mode'] for p in proposed}
            visited = set(meta['visited_modes']); assert not newkeys & visited and oldkeys-visited <= newkeys
            positive = {key for key in newkeys if geo[key]['numerical_volume_available']}; positive_by[ch] = positive
            oldpositive = set(meta['positive_modes']); combined = set(meta['original_positive_modes']) | positive
            assert oldpositive <= combined
            assert sorted(positive) == call['new_positive_modes'] and sorted(combined-oldpositive) == call['new_positive_beyond_old_method']
            assert dict(Counter(geo[key]['classification'] for key in newkeys)) == call['classifications']
            expected = dict(old_candidates=len(oldkeys), old_visited_slots=len(oldkeys & visited),
                old_unvisited_candidates=len(oldkeys-visited), new_candidates=len(newkeys), additional_candidates=len(newkeys-oldkeys),
                max_dp_entries=call['new']['meta']['max_retained_dp_entries'], trie_nodes=call['new']['meta']['trie_nodes'])
            assert expected == call['summary']
            for key, val in expected.items():
                if key not in ['max_dp_entries', 'trie_nodes']: aggregate[ch][key] += val
            aggregate[ch]['new_positive_pairs'] += len(positive)
            aggregate[ch]['extra_positive_beyond_old_method'] += len(combined-oldpositive)
            aggregate[ch]['tasks_with_extra_positive'] += bool(combined-oldpositive)
            aggregate[ch]['tasks_with_new_positive'] += bool(positive)
            aggregate[ch]['maximum_dp_entries'] = max(aggregate[ch]['maximum_dp_entries'], expected['max_dp_entries'])
            aggregate[ch]['maximum_trie_nodes'] = max(aggregate[ch]['maximum_trie_nodes'], expected['trie_nodes'])
            witnesses.append(dict(seed=seed, channel=ch, counts=dict(checks), new_positive=sorted(positive)))
        for ch in CHANNELS:
            others = set().union(*(p for c, p in positive_by.items() if c != ch))
            exclusive[ch].extend(dict(seed=seed, mode=key) for key in sorted(positive_by[ch]-others))
        print(dict(audited_tasks=len(witnesses)//6, complete_rankings=counts['exact_complete_rankings'],
                   patterns=counts['enumerated_patterns'], seconds=time.perf_counter()-start), flush=True)
    assert {c: dict(a) for c, a in aggregate.items()} == original_summary['aggregate']
    assert counts['exact_complete_rankings'] == 192
    assert counts['independent_certificates'] == original_summary['counts']['independent_certificates']
    for p, digest in hashes.items(): assert io.sha(root/p) == digest
    io.save(out/'witnesses.json', witnesses); io.save(out/'exclusive_positive.json', exclusive)
    summary = dict(passed=True, counts=dict(counts), exclusive_positive_counts={c: len(v) for c, v in exclusive.items()},
        census_summary_sha256=io.sha(census/'summary.json'), seconds=time.perf_counter()-start,
        query_targets_accessed=False, posterior_reference_accessed=False, core_research_goal_complete=False,
        outputs_sha256={n: io.sha(out/n) for n in ['protocol.json', 'witnesses.json', 'exclusive_positive.json']})
    io.save(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/unvisited_language/audit_v1'
    out.mkdir(parents=True, exist_ok=False)
    try: run(root, out)
    except Exception:
        io.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
