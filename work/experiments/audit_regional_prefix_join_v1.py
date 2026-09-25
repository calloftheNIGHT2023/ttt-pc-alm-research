"""378 independent search replay, exact cuts and unresolved-subtree coverage."""
from collections import Counter, defaultdict
from fractions import Fraction as F
from pathlib import Path
import math
import time
import traceback
import numpy as np
import run_regional_prefix_join_v1 as run
import prefix_obstruction_v1 as proof
import local_region_screen as screen


def survivors(a, i):
    prefix = f'batch_{i}_'; mask = ~(a[prefix+'c5'] | a[prefix+'c20'])
    mask[a[prefix+'local_ids']] = False
    return a[prefix+'regs'][mask].copy()


def audit_case(a, m, positive, *, exact):
    counts = Counter(); x, v = a['x_observed'], a['v_observed']; d, n = 4, len(x)
    assert np.array_equal(x, a['original_x'][a['support_order']]) and np.array_equal(v, a['original_v'][a['support_order']])
    assert sorted(m['order']) == list(range(n)) and m['order'] == a['support_order'].tolist()
    languages = [np.array(run.old_join.singleton(xx, vv, d), np.uint8).reshape(-1, d) for xx, vv in zip(x, v)]
    for i, language in enumerate(languages):
        assert np.array_equal(language, a['language_'+str(i)]); counts['fraction_single_languages'] += 1
    expanded = solver_rows = units = rounded = proposals = local_rejected = 0
    generated_outputs = []; stack = [(1, np.empty((1, d, 0), np.uint8), 0)]
    bfs_parent = np.empty((1, d, 0), np.uint8); bfs_children = []; prior_k = 1; stage_done = 0
    for i, row in enumerate(m['batches']):
        k, start, take = row['k'], row['start'], row['processed']; prefix = f'batch_{i}_'
        if m['schedule'] == 'dfs':
            while stack and stack[-1][2] >= len(stack[-1][1])*len(languages[stack[-1][0]-1]): stack.pop()
            kk, parent, ss = stack.pop(); assert (k, start) == (kk, ss)
            possible = len(parent)*len(languages[k-1]); expected_take = min(1024, possible-start, m['max_expanded']-expanded)
            assert take == expected_take
            if start+take < possible: stack.append((k, parent, start+take))
        else:
            if k != prior_k:
                assert k == prior_k+1
                assert stage_done == len(bfs_parent)*len(languages[prior_k-1])
                bfs_parent = np.concatenate(bfs_children) if bfs_children else np.empty((0, d, prior_k), np.uint8)
                assert np.array_equal(bfs_parent, a['prefix_'+str(prior_k)])
                bfs_children = []; stage_done = 0; prior_k = k
            parent = bfs_parent; assert start == stage_done
            possible = len(parent)*len(languages[k-1]); assert take == min(1024, possible-start)
            stage_done += take
        ids = np.arange(start, start+take)
        expected = np.concatenate([parent[ids//len(languages[k-1])], languages[k-1][ids % len(languages[k-1]), :, None]], axis=2)
        regs = a[prefix+'regs']; assert np.array_equal(expected, regs); counts['enumerated_combinations'] += take
        c5 = screen.contract(x[:k], v[:k], regs, 5); assert np.array_equal(c5, a[prefix+'c5'])
        c20 = np.zeros(take, bool); eligible = np.flatnonzero(~c5)
        if len(eligible): c20[eligible] = screen.contract(x[:k], v[:k], regs[eligible], 20)
        assert np.array_equal(c20, a[prefix+'c20']); counts['contractor_mask_replays'] += 2
        eligible = np.flatnonzero(~(c5 | c20)); reject = a[prefix+'local_ids']; sm = row['solver']
        assert set(reject) <= set(eligible) and len(set(reject)) == len(reject)
        family, steps = next((f, s) for name, f, s in run.model.CONFIGS if name == m['method'])
        assert (sm is not None) == (family is not None and len(eligible) >= 64)
        if sm is not None:
            assert sm['family'] == family and sm['steps'] == steps and sm['checkpoints'][-1]['step'] == steps
            assert sm['proposal_rows'] == 3*len(eligible)*(1+steps//32)
            assert sm['checkpoints'][-1]['certified'] == len(reject)
            solver_rows += len(eligible); units += len(eligible)*steps*d*k
            rounded += sm['rounded_checks']; proposals += sm['proposal_rows']
            for j, index in enumerate(reject):
                assert a[prefix+'proof_lower'][j] > 0
                assert 0 <= a[prefix+'first_step'][j] <= steps
                if exact:
                    bound = proof.exact_box_bound(x[:k], v[:k], regs[index], a[prefix+'proof_p'][j], a[prefix+'proof_a'][j])
                    assert bound > 0 and F(float(a[prefix+'proof_lower'][j])) <= bound
                    counts['exact_wide_domain_cuts'] += 1
        child = survivors(a, i)
        assert len(child) == row['retained'] and row['rejected_c5'] == int(c5.sum()) and row['rejected_c20'] == int(c20.sum())
        assert row['rejected_local'] == len(reject) and len(child)+int(c5.sum())+int(c20.sum())+len(reject) == take
        if m['schedule'] == 'dfs':
            if len(child):
                if k == n: generated_outputs.append(child)
                else: stack.append((k+1, child, 0))
        else:
            if len(child): bfs_children.append(child)
        expanded += take; local_rejected += len(reject); counts['batch_accounting'] += 1
    if m['schedule'] == 'dfs':
        stack = [(k, parents, start) for k, parents, start in stack if start < len(parents)*len(languages[k-1])]
        assert len(stack) == len(m['pending'])
        for i, (k, parents, start) in enumerate(stack):
            pm = m['pending'][i]
            assert pm['k'] == k and pm['start'] == start and np.array_equal(parents, a['pending_'+str(i)])
        full = np.concatenate(generated_outputs) if generated_outputs else np.empty((0, d, n), np.uint8)
    else:
        for s in m['stages']:
            relevant = [r for r in m['batches'] if r['k'] == s['k']]
            assert sum(r['processed'] for r in relevant) == s['processed']
            assert sum(r['retained'] for r in relevant) == s['retained']
            assert len(a['prefix_'+str(s['k'])]) == s['retained']
            if s['complete']: assert s['processed'] == s['possible']
        full = a.get('prefix_'+str(n), np.empty((0, d, n), np.uint8))
        # Rebuild the exact unresolved subtrees at an interrupted stage.
        expected_pending = []
        if not m['completed']:
            completed_stages = [s['k'] for s in m['stages'] if s['complete']]
            next_k = max(completed_stages, default=0)+1
            parents = a['prefix_'+str(next_k-1)] if next_k > 1 else np.empty((1, d, 0), np.uint8)
            stage = next((s for s in m['stages'] if s['k'] == next_k), None)
            offset = stage['processed'] if stage else 0
            if offset < len(parents)*len(languages[next_k-1]): expected_pending.append((next_k, parents, offset))
            if stage and next_k < n and len(a['prefix_'+str(next_k)]): expected_pending.append((next_k+1, a['prefix_'+str(next_k)], 0))
            assert len(expected_pending) == len(m['pending'])
            for i, (k, parents, offset) in enumerate(expected_pending):
                assert m['pending'][i]['k'] == k and m['pending'][i]['start'] == offset
                assert np.array_equal(a['pending_'+str(i)], parents)
    expected_set = {r[:, np.argsort(a['support_order'])].copy().tobytes() for r in full}
    actual_set = {r.tobytes() for r in a['regions']}
    assert expected_set == actual_set and len(actual_set) == len(a['regions'])
    assert m['completed'] == (not m['pending']) and m['expanded'] == expanded <= m['max_expanded']
    assert m['rejected_local'] == local_rejected and m['solver_rows'] == solver_rows
    assert m['solver_coordinate_steps'] == units and m['rounded_checks'] == rounded and m['proposal_rows'] == proposals
    assert m['returned_array_bytes'] == sum(t.nbytes for t in a.values())
    assert not any(m[k] for k in ['query_targets_accessed', 'geometry_accessed', 'global_bp_used', 'lp_used'])
    for key in positive:
        pattern = np.frombuffer(bytes.fromhex(key), np.uint8).reshape(d, n)
        covered = pattern.tobytes() in actual_set
        reordered = pattern[:, a['support_order']]
        for i, pm in enumerate(m['pending']):
            k = pm['k']; parents = a['pending_'+str(i)]
            matching = np.flatnonzero(np.all(parents == reordered[:, :k-1], axis=(1, 2)))
            paths = np.flatnonzero(np.all(languages[k-1] == reordered[:, k-1], axis=1))
            covered |= any(int(pi)*len(languages[k-1])+int(qi) >= pm['start'] for pi in matching for qi in paths)
        assert covered, ('Lost feasible reference', key, m['method'], m['schedule'])
        counts['positive_output_or_pending_coverage'] += 1
    counts['search_accounting'] += 1
    return counts


def audit(root, out):
    begin = time.perf_counter(); source = root/run.BASE/'development_v1'
    summary = run.complete(source); protocol = run.read(source/'protocol.json')
    assert summary['all_methods_sealed_before_external_labels'] and protocol['source_sha256'] == run.hashes(root)
    rows = run.read(source/'rows.json'); assert len(rows) == 576
    old_rows = run.read(root/'results/context_block_scaling/development/rows.json')
    positives = {seed: sorted(set().union(*(set(r.get('positive_mode_keys') or []) for r in old_rows if r['seed'] == seed and r['n'] == 24 and r['complete']))) for seed in range(5920000, 5920016)}
    checks = Counter(); groups = defaultdict(list); all_records = []
    for i, row in enumerate(rows):
        directory = root/row['directory']
        for f, h in row['files'].items(): assert run.sha(directory/f) == h
        a = run.load(directory/'arrays.npz'); m = run.read(directory/'metadata.json')
        one = audit_case(a, m, positives[row['seed']], exact=True); checks.update(one)
        assert row['expanded'] == m['expanded'] and row['completed'] == m['completed'] and row['seconds'] == m['total_seconds']
        all_records.append(dict(directory=row['directory'], checks=dict(one), reference_modes=len(positives[row['seed']])))
        groups[row['method'], row['schedule'], row['ordering']].append(row)
        if (i+1) % 36 == 0: print(dict(stage='audit', calls=i+1, checks=dict(checks), seconds=time.perf_counter()-begin), flush=True)
    results = []
    for (method, schedule, order), rr in groups.items():
        results.append(dict(method=method, schedule=schedule, ordering=order, calls=len(rr), completed=sum(r['completed'] for r in rr),
                            mean_seconds=math.fsum(r['seconds'] for r in rr)/len(rr), expanded=sum(r['expanded'] for r in rr),
                            local_rejected=sum(r['local_rejected'] for r in rr), final_candidates=sum(r['remaining'] for r in rr),
                            solver_coordinate_steps=sum(r['solver_coordinate_steps'] for r in rr),
                            mean_returned_bytes=math.fsum(r['returned_array_bytes'] for r in rr)/len(rr)))
    run.save(out/'groups.json', results); run.save(out/'checks.json', all_records)
    result = dict(passed=True, calls=len(rows), checks=dict(checks), seconds=time.perf_counter()-begin,
                  development_summary_sha256=run.sha(source/'summary.json'), source_sha256=run.sha(Path(__file__)),
                  query_targets_accessed=False, core_research_goal_complete=False,
                  outputs_sha256={f: run.sha(out/f) for f in ['groups.json', 'checks.json']})
    run.save(out/'summary.json', result); print(dict(stage='audit_complete', **result), flush=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/run.BASE/'audit_v1'; out.mkdir(parents=True, exist_ok=False)
    try: audit(root, out)
    except Exception:
        run.save(out/'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False)); raise
