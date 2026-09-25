"""380 independently reconstruct actual live-row work; reuse exact search-proof replay."""
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import math
import time
import traceback
import numpy as np
import run_retired_region_join_v1 as run
import audit_regional_prefix_join_v1 as previous


def audit_solver(first, m, d, n):
    r = len(first); cap = m['steps']; counts = Counter()
    assert np.all((first == -1) | ((first >= 0) & (first <= cap) & ((first % 32 == 0) | (first == cap))))
    stop = np.where(first >= 0, first, cap)
    expected = [0]+list(range(32, cap+1, 32))
    if expected[-1] != cap: expected.append(cap)
    assert [cp['step'] for cp in m['checkpoints']] == expected
    proposals = 0; rounded = 0
    for cp in m['checkpoints']:
        t = cp['step']; before = int((~((first >= 0) & (first < t))).sum()) if t else r
        after = int((~((first >= 0) & (first <= t))).sum())
        proposals += 3*before
        assert cp['active_before'] == before and cp['active_after'] == after
        assert cp['certified'] == r-after and cp['proposal_rows'] == proposals
        assert cp['cumulative_row_steps'] == int(np.minimum(stop, t).sum())
        assert rounded <= cp['rounded_checks'] <= proposals; rounded = cp['rounded_checks']
        counts['independent_live_checkpoints'] += 1
    assert m['row_steps'] == int(stop.sum()) and m['coordinate_steps'] == int(stop.sum())*d*n
    assert m['full_row_steps_upper_bound'] == r*cap and m['full_coordinate_steps_upper_bound'] == r*cap*d*n
    assert m['executed_sweeps'] == int(stop.max()) and m['proposal_rows'] == proposals and m['rounded_checks'] == rounded
    assert m['strict_certificate_retirement'] and not any(m[k] for k in ['query_targets_accessed', 'suffix_accessed', 'global_bp_used', 'lp_used'])
    counts['independent_solver_work_checks'] += 1
    return counts


def audit_case(a, m, positive, *, exact):
    normalized = deepcopy(m); counts = Counter(); actual = nominal = proposals = 0
    for i, row in enumerate(m['batches']):
        sm = row['solver']
        if sm is None: continue
        prefix = f'batch_{i}_'; eligible = np.flatnonzero(~(a[prefix+'c5'] | a[prefix+'c20']))
        rejected = a[prefix+'local_ids']; positions = np.searchsorted(eligible, rejected)
        assert np.array_equal(eligible[positions], rejected)
        first = np.full(len(eligible), -1, np.int32); first[positions] = a[prefix+'first_step']
        counts.update(audit_solver(first, sm, a['regions'].shape[1], row['k']))
        actual += sm['coordinate_steps']; nominal += sm['full_coordinate_steps_upper_bound']; proposals += sm['proposal_rows']
        # The old verifier explicitly expects the full-row upper bound. Normalize a
        # separate in-memory copy only; the original archive remains actual work.
        normalized['batches'][i]['solver']['proposal_rows'] = 3*len(eligible)*(1+sm['steps']//32)
    assert m['solver_coordinate_steps'] == actual <= nominal == m['solver_coordinate_steps_upper_bound']
    assert m['proposal_rows'] == proposals and m['strict_certificate_retirement']
    normalized['solver_coordinate_steps'] = nominal
    normalized['proposal_rows'] = sum(row['solver']['proposal_rows'] for row in normalized['batches'] if row['solver'] is not None)
    with run.model.configuration(retire=False): counts.update(previous.audit_case(a, normalized, positive, exact=exact))
    counts['actual_search_resource_checks'] += 1
    return counts


def audit(root, out):
    begin = time.perf_counter(); source = root/run.BASE/'development_v1'
    summary = run.complete(source); protocol = run.read(source/'protocol.json')
    assert summary['all_methods_sealed_before_external_labels'] and protocol['source_sha256'] == run.hashes(root)
    rows = run.read(source/'rows.json'); assert len(rows) == 544
    old_rows = run.read(root/'results/context_block_scaling/development/rows.json')
    positives = {seed: sorted(set().union(*(set(r.get('positive_mode_keys') or []) for r in old_rows
        if r['seed'] == seed and r['n'] == 24 and r['complete']))) for seed in range(5920000, 5920016)}
    run.complete(root/run.previous.BASE/'development_v1')
    legacy = {(r['seed'], r['method'], r['schedule'], r['ordering']): r
              for r in run.read(root/run.previous.BASE/'development_v1/rows.json')}
    checks = Counter(); groups = defaultdict(list); records = []
    for i, row in enumerate(rows):
        directory = root/row['directory']
        for f, value in row['files'].items(): assert run.sha(directory/f) == value
        a = run.load(directory/'arrays.npz'); m = run.read(directory/'metadata.json')
        one = audit_case(a, m, positives[row['seed']], exact=True)
        for key in ['expanded', 'completed', 'solver_coordinate_steps', 'solver_coordinate_steps_upper_bound', 'proposal_rows', 'rounded_checks']:
            assert row[key] == m[key], key
        assert row['seconds'] == m['total_seconds'] and row['remaining'] == len(a['regions'])
        if row['method'] in run.LEGACY_NAMES:
            old = legacy[row['seed'], row['method'], row['schedule'], row['ordering']]
            previous_dir = root/old['directory']
            for f, value in old['files'].items(): assert run.sha(previous_dir/f) == value
            if not any('seconds' in (r['stop_reason'] or '') or 'state_limit' in (r['stop_reason'] or '') for r in [row, old]):
                one['published_legacy_array_replays'] += run.same(run.load(previous_dir/'arrays.npz'), a)
                one['published_legacy_search_replays'] += 1
            else: one['legacy_resource_bound_noncomparisons'] += 1
        checks.update(one); groups[row['method'], row['schedule'], row['ordering']].append(row)
        records.append(dict(directory=row['directory'], checks=dict(one), reference_modes=len(positives[row['seed']])))
        if (i+1) % 34 == 0:
            print(dict(stage='audit', calls=i+1, exact_cuts=checks['exact_wide_domain_cuts'], seconds=time.perf_counter()-begin), flush=True)
    results = []
    for (method, schedule, order), rr in groups.items():
        results.append(dict(method=method, schedule=schedule, ordering=order, calls=len(rr), completed=sum(r['completed'] for r in rr),
            mean_seconds=math.fsum(r['seconds'] for r in rr)/len(rr), expanded=sum(r['expanded'] for r in rr),
            local_rejected=sum(r['local_rejected'] for r in rr), final_candidates=sum(r['remaining'] for r in rr),
            solver_coordinate_steps=sum(r['solver_coordinate_steps'] for r in rr),
            solver_coordinate_steps_upper_bound=sum(r['solver_coordinate_steps_upper_bound'] for r in rr),
            mean_returned_bytes=math.fsum(r['returned_array_bytes'] for r in rr)/len(rr),
            mean_named_bytes=math.fsum(r['named_bytes'] for r in rr)/len(rr),
            mean_solver_seconds=math.fsum(r['solver_seconds'] for r in rr)/len(rr)))
    run.save(out/'groups.json', results); run.save(out/'checks.json', records)
    assert run.hashes(root) == protocol['source_sha256']
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
