"""378 certified local pruning in BFS and anytime batch-DFS prefix joins."""
from math import prod
import time
import numpy as np
from branch_image_chain_dyadic_v1 import IntegerProblem
from support_language_chain_v1 import accepted_paths
from local_region_screen import contract
import regional_alm_explicit_v1 as regional

CONFIGS = [('none', None, 0), ('regional_active128', 'regional_active', 128),
           ('regional_passive128', 'regional_passive', 128),
           ('pdhg_cold128', 'pdhg_cold', 128), ('pdhg_cold256', 'pdhg_cold', 256),
           ('pdhg_cold512', 'pdhg_cold', 512), ('pdhg_box128', 'pdhg_box', 128),
           ('pdhg_box256', 'pdhg_box', 256), ('pdhg_box512', 'pdhg_box', 512)]
PRIMARY = 'regional_active128'


def ordering(x, name):
    if name == 'observed':
        return np.arange(len(x))
    assert name == 'farthest_x'
    selected = [int(np.argmin(x))]
    while len(selected) < len(x):
        remaining = set(range(len(x)))-set(selected)
        selected.append(min(remaining, key=lambda i: (-min(abs(x[i]-x[j]) for j in selected), i)))
    return np.array(selected)


def join(x, v, *, name, schedule, order_name='observed', depth=4,
         max_expanded=65536, max_states=20000, max_seconds=8.):
    begin = time.perf_counter()
    assert schedule in ['bfs', 'dfs'] and x.shape == v.shape and len(x) > 0
    cfg = next(c for c in CONFIGS if c[0] == name)
    with regional.guarded():
        a, m = _join(x, v, cfg, schedule, order_name, depth, max_expanded, max_states, max_seconds, begin)
    m['total_seconds'] = time.perf_counter()-begin
    return a, m


def _join(x, v, cfg, schedule, order_name, depth, max_expanded, max_states, max_seconds, begin):
    name, family, solver_steps = cfg; n = len(x)
    rank = ordering(x, order_name); xx, vv = x[rank].copy(), v[rank].copy()
    p = IntegerProblem(xx, vv, np.zeros((depth, n)))
    languages = [np.array(accepted_paths(p, i), np.uint8).reshape(-1, depth) for i in range(n)]
    counts = list(map(len, languages)); language_seconds = time.perf_counter()-begin
    arrays = dict(x_observed=xx, v_observed=vv, support_order=rank, original_x=x.copy(), original_v=v.copy())
    arrays.update({'language_'+str(i): paths.copy() for i, paths in enumerate(languages)})
    batches = []; stages = []; pending = []; outputs = []
    expanded = 0; peak_states = 1; peak_work_bytes = 0; solver_rows = 0; work_units = 0
    solver_seconds = 0.; proof_checks = 0; proposal_rows = 0; rejected_local = 0
    reason = None; first_full_seconds = None; archive_bytes = sum(a.nbytes for a in arrays.values())

    def batch(parents, k, start, take):
        nonlocal expanded, peak_work_bytes, solver_rows, work_units, solver_seconds, proof_checks
        nonlocal proposal_rows, rejected_local, archive_bytes
        ids = np.arange(start, start+take)
        regs = np.concatenate([parents[ids//counts[k-1]], languages[k-1][ids % counts[k-1], :, None]], axis=2)
        prefix = f'batch_{len(batches)}_'
        original = regs.copy(); c5 = contract(xx[:k], vv[:k], regs, 5)
        eligible = np.flatnonzero(~c5); c20 = np.zeros(take, bool)
        if len(eligible):
            c20[eligible] = contract(xx[:k], vv[:k], regs[eligible], 20)
        eligible = np.flatnonzero(~(c5 | c20)); local_ids = np.empty(0, int)
        sm = None; local_bytes = 0
        if family is not None and len(eligible) >= 64:
            a, sm = regional.solve(xx[:k], vv[:k], regs[eligible], family=family, steps=solver_steps)
            local_ids = eligible[a['first_step'] >= 0]
            mask = a['first_step'] >= 0
            for field in ['proof_p', 'proof_a', 'proof_lower', 'proof_kind', 'first_step']:
                arrays[prefix+field] = a[field][mask].copy()
            solver_rows += len(eligible); work_units += len(eligible)*solver_steps*depth*k
            solver_seconds += sm['total_seconds']; proof_checks += sm['rounded_checks']; proposal_rows += sm['proposal_rows']
            local_bytes = sm['live_named_array_bytes']+sm['initial_archive_bytes']
        keep = ~(c5 | c20); keep[local_ids] = False; result = regs[keep].copy()
        arrays.update({prefix+'regs': original, prefix+'c5': c5, prefix+'c20': c20, prefix+'local_ids': local_ids})
        added_bytes = sum(value.nbytes for key, value in arrays.items() if key.startswith(prefix))
        archive_bytes += added_bytes
        peak_work_bytes = max(peak_work_bytes, archive_bytes+parents.nbytes+regs.nbytes+result.nbytes+local_bytes)
        expanded += take; rejected_local += len(local_ids)
        batches.append(dict(k=k, start=start, processed=take, parent_count=len(parents), retained=len(result),
                            rejected_c5=int(c5.sum()), rejected_c20=int(c20.sum()), rejected_local=len(local_ids),
                            solver=sm, cumulative_seconds=time.perf_counter()-begin))
        return result

    if schedule == 'bfs':
        partial = np.empty((1, depth, 0), np.uint8)
        for k in range(1, n+1):
            possible = len(partial)*counts[k-1]
            if expanded+possible > max_expanded:
                reason = 'expanded_limit_before_stage'; pending = [(k, partial, 0)]; break
            blocks = []; done = 0; interrupted = False
            for start in range(0, possible, 1024):
                if time.perf_counter()-begin >= max_seconds:
                    reason = 'seconds_at_batch_boundary'; interrupted = True
                    pending = [(k, partial, start)]; break
                child = batch(partial, k, start, min(1024, possible-start)); done += min(1024, possible-start)
                if len(child): blocks.append(child)
                retained = sum(len(b) for b in blocks); peak_states = max(peak_states, len(partial)+retained)
                if retained > max_states:
                    reason = 'state_limit_at_batch_boundary'; interrupted = True
                    pending = [(k, partial, start+min(1024, possible-start))]; break
            stages.append(dict(k=k, possible=possible, processed=done, retained=sum(len(b) for b in blocks), complete=not interrupted))
            combined = np.concatenate(blocks) if blocks else np.empty((0, depth, k), np.uint8)
            arrays['prefix_'+str(k)] = combined; archive_bytes += combined.nbytes
            if interrupted:
                if k == n:
                    outputs = [combined]
                elif len(combined):
                    pending.append((k+1, combined, 0))
                break
            partial = combined
        else:
            outputs = [partial]
    else:
        stack = [(1, np.empty((1, depth, 0), np.uint8), 0)]
        while stack:
            if time.perf_counter()-begin >= max_seconds:
                reason = 'seconds_at_batch_boundary'; break
            if expanded >= max_expanded:
                reason = 'expanded_limit_at_batch_boundary'; break
            k, parents, start = stack.pop(); possible = len(parents)*counts[k-1]
            if start >= possible:
                continue
            take = min(1024, possible-start, max_expanded-expanded)
            if start+take < possible: stack.append((k, parents, start+take))
            child = batch(parents, k, start, take)
            if len(child):
                if k == n:
                    outputs.append(child)
                    if first_full_seconds is None: first_full_seconds = time.perf_counter()-begin
                else:
                    stack.append((k+1, child, 0))
            current_states = sum(len(parents) for _, parents, _ in stack)+sum(len(b) for b in outputs)
            peak_states = max(peak_states, current_states)
            if current_states > max_states:
                reason = 'state_limit_at_batch_boundary'; break
        pending = stack
    # Empty tasks on the stack carry no unvisited combination and do not imply an incomplete search.
    pending = [(k, parents, start) for k, parents, start in pending if start < len(parents)*counts[k-1]]
    complete = not pending
    if complete: reason = None
    regions = np.concatenate(outputs) if outputs else np.empty((0, depth, n), np.uint8)
    if len(regions):
        regions = regions[:, :, np.argsort(rank)].copy()
        regions = np.array(sorted(regions, key=lambda r: r.tobytes()), np.uint8).reshape(-1, depth, n)
        if first_full_seconds is None: first_full_seconds = time.perf_counter()-begin
    arrays['regions'] = regions
    pending_meta = []
    for i, (k, parents, start) in enumerate(pending):
        arrays['pending_'+str(i)] = parents.copy()
        pending_meta.append(dict(k=k, start=start, parent_count=len(parents), combinations=len(parents)*counts[k-1]-start))
    returned = sum(a.nbytes for a in arrays.values())
    meta = dict(method=name, schedule=schedule, order_name=order_name, order=rank.tolist(), single_counts=counts,
                completed=complete, stop_reason=reason, full_product=prod(counts), expanded=expanded,
                final_candidates=len(regions), first_full_seconds=first_full_seconds, stages=stages, batches=batches,
                pending=pending_meta, peak_search_states=peak_states, live_named_array_bytes_lower_bound=max(peak_work_bytes, returned),
                returned_array_bytes=returned, language_seconds=language_seconds, solver_rows=solver_rows,
                solver_coordinate_steps=work_units, solver_seconds=solver_seconds, rounded_checks=proof_checks,
                proposal_rows=proposal_rows, rejected_local=rejected_local,
                max_expanded=max_expanded, max_states=max_states, max_seconds=max_seconds, gate=64, batch_size=1024,
                query_targets_accessed=False, geometry_accessed=False, global_bp_used=False, lp_used=False,
                complete_search_not_prediction=True, compressed_io_excluded=True,
                resource_scope='Full search including guard and proof archive copies; excludes final geometry/readout, disk compression and independent audit; named arrays not RSS peak')
    return arrays, meta
