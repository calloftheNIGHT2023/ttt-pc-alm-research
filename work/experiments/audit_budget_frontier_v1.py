"""Independent scalar ordering and Fraction proof audit for 360."""
from collections import Counter
from fractions import Fraction
from pathlib import Path
import time
import numpy as np
import budget_reinvestment_suite_v1 as io
import branch_image_chain_v1 as exact


def take(first, budget):
    result = []
    for index, step in enumerate(first):
        if int(step) == 0 and len(result) < budget:
            result.append(index)
    return result


def run(root, out):
    begin = time.perf_counter(); src = root/'results/budget_frontier/development_v1'; io.complete(src)
    old = root/'results/cross_region_credit/development_v1'; geo = root/'results/cross_region_credit/audit_v1'
    p = io.read(src/'protocol.json'); counts = Counter()
    for f, digest in p['source_sha256'].items():
        assert io.sha(root/f) == digest
    assert io.sha(old/'protocol.json') == p['source_protocol_sha256']
    assert io.sha(old/'tasks.json') == p['source_tasks_sha256']
    assert io.sha(geo/'summary.json') == p['geometry_summary_sha256']
    configs = {c['name']: c for c in io.read(old/'protocol.json')['configs']}
    for seal in io.read(src/'input_seals.json'):
        seed = seal['seed']
        for f, digest in seal['source_files'].items():
            assert io.sha(old/str(seed)/f) == digest
        assert io.sha(geo/(str(seed)+'_geometry.json')) == seal['geometry_sha256']
        assert io.sha(src/str(seed)/'commit.json') == seal['commit_sha256']
    aggregates = {name: Counter() for name in configs if not configs[name]['reuse']}
    for row in io.read(src/'rows.json'):
        seed, name = row['seed'], row['method']; directory = src/str(seed)
        for ext, digest in row['files'].items():
            assert io.sha(directory/(name+ext)) == digest
        inp = io.read(old/str(seed)/'input.json'); meta = io.read(directory/(name+'.json'))
        x, v = np.array(inp['x_observed']), np.array(inp['v_observed'])
        with np.load(old/str(seed)/'pool.npz', allow_pickle=False) as z:
            regs = z['regions']
        with np.load(old/str(seed)/(name+'.npz'), allow_pickle=False) as z:
            reference = {k:z[k] for k in z.files}
        with np.load(directory/(name+'.npz'), allow_pickle=False) as z:
            a = {k:z[k] for k in z.files}
        expected = take(reference['first_step'], 8); end = expected[-1]+1 if len(expected) == 8 else len(regs)
        assert expected == row['selected_indices'] == a['selected_indices'].tolist()
        assert end == row['processed'] == meta['processed']
        for key in reference:
            assert a[key].tobytes() == reference[key][:end].tobytes(); counts['bitwise_prefix_arrays'] += 1
        prior = 0; gathered = []; work = 0
        for call in meta['calls']:
            start, stop = call['begin'], call['end']; m = call['metadata']
            assert start == prior and stop == min(len(regs), start+8-len(gathered)); prior = stop
            gathered.extend(i for i in range(start, stop) if int(a['first_step'][i]) == 0)
            assert not a['disabled'][start:stop].any()
            pair_count = sum(int(t) if t else configs[name]['steps'] for t in a['first_step'][start:stop])
            assert pair_count == m['response_pairs']; work += pair_count
            proof_indices = set()
            for proof in m['proofs']:
                i = start+proof['index']; proof_indices.add(i)
                assert int(a['first_step'][i]) == proof['step'] and proof['mode'] == regs[i].tobytes().hex()
                bound = exact.fixed_value(x, v, a['proof_credit'][i], regs[i])
                assert (proof['structural_empty'] and bound is None) or bound == Fraction(proof['lower']) > 0
                counts['independent_fraction_proofs'] += 1
            assert proof_indices == {i for i in range(start, stop) if a['first_step'][i] > 0}
            counts['batch_checks'] += 1
        assert gathered == expected and prior == end and work == row['response_pairs'] == meta['response_pairs']
        assert row['full_response_pairs'] == sum(int(t) if t else configs[name]['steps'] for t in reference['first_step'])
        assert row['response_pairs'] <= row['full_response_pairs']
        counts['exact_selections'] += 1
        for k in ['processed', 'candidates', 'response_pairs', 'full_response_pairs', 'batches', 'seconds']:
            aggregates[name][k] += row[k]
    summary = io.read(src/'summary.json')
    assert {k:dict(v) for k,v in aggregates.items()} == summary['totals']
    frontier_index = {}
    for row in io.read(src/'frontiers.json'):
        seed, name, budget = row['seed'], row['method'], row['budget']
        with np.load(old/str(seed)/(name+'.npz'), allow_pickle=False) as z:
            first = z['first_step']
        with np.load(old/str(seed)/'pool.npz', allow_pickle=False) as z:
            regs = z['regions']
        geometry = io.read(geo/(str(seed)+'_geometry.json'))
        labels = [geometry[r.tobytes().hex()]['classification'] for r in regs]
        chosen = take(first, budget); end = chosen[-1]+1 if len(chosen) == budget else len(regs)
        assert row['selected'] == chosen and row['cutoff'] == end
        assert row['selected_positive'] == [i for i in chosen if labels[i] == 'positive_volume']
        assert row['positive_total'] == labels.count('positive_volume')
        ideal = [i for i, label in enumerate(labels) if label != 'infeasible'][:budget]
        assert row['ideal_positive'] == [i for i in ideal if labels[i] == 'positive_volume']
        assert row['cuts_prefix'] == sum(bool(t) for t in first[:end])
        assert row['cuts_tail'] == sum(bool(t) for t in first[end:])
        assert row['cuts_total'] == row['cuts_prefix']+row['cuts_tail']
        for j in range(len(first)):
            count = sum(bool(t) for t in first[:j+1])
            assert (j in chosen) == (first[j] == 0 and j+1-count <= budget)
            counts['scalar_rank_identities'] += 1
        if not configs[name]['reuse']:
            assert row['prefix_response_pairs'] == sum(int(t) if t else configs[name]['steps'] for t in first[:end])
        frontier_index[seed, name, budget] = row
        counts['frontier_rows'] += 1
    for row in io.read(src/'positive_frontiers.json'):
        f = frontier_index[row['seed'], row['method'], 8]
        with np.load(old/str(row['seed'])/(row['method']+'.npz'), allow_pickle=False) as z:
            first = z['first_step']
        j = row['index']; actual = sum(bool(t) for t in first[:j+1]); required = max(0, j+1-8)
        assert row['rank'] == j+1 and row['required_prefix_cuts'] == required
        assert row['actual_prefix_cuts'] == actual and row['deficit'] == max(0, required-actual)
        assert row['selected'] == (actual >= required) == (j in f['selected_positive'])
        counts['positive_rank_conditions'] += 1
    result = dict(passed=True, counts=dict(counts), seconds=time.perf_counter()-begin,
        source_sha256=io.sha(Path(__file__)), component_summary_sha256=io.sha(src/'summary.json'),
        query_targets_accessed=False, query_risk_evaluated=False, core_research_goal_complete=False)
    io.save(out/'summary.json', result); print(result, flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/'results/budget_frontier/audit_v1'
    out.mkdir(parents=True, exist_ok=False); run(root, out)
