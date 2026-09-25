"""Exact lazy prefix for independent region solves, not cross-region reuse."""
import time
import numpy as np
import region_conditioned_credit_v1 as independent


def selected(first, budget):
    if budget < 1:
        raise ValueError('Positive geometry budget required')
    return np.flatnonzero(np.asarray(first) == 0)[:budget]


def solve(x, v, regs, credit, *, steps=128, budget=8):
    started = time.perf_counter()
    if budget < 1 or steps < 1:
        raise ValueError('Positive budgets required')
    total = len(regs); cursor = 0; chosen = []; parts = []; calls = []
    while cursor < total and len(chosen) < budget:
        size = min(budget-len(chosen), total-cursor)
        arrays, meta = independent.solve(x, v, regs[cursor:cursor+size], credit, steps)
        chosen.extend((cursor+np.flatnonzero(arrays['first_step'] == 0)).tolist())
        parts.append(arrays)
        calls.append(dict(begin=cursor, end=cursor+size, metadata=meta))
        cursor += size
    empty = dict(first_step=np.zeros(0, np.int32), proof_credit=np.zeros((0, *credit.shape)),
                 final_credit=np.zeros((0, *credit.shape)), disabled=np.zeros(0, bool))
    arrays = {k: np.concatenate([p[k] for p in parts]) if parts else a for k, a in empty.items()}
    arrays['selected_indices'] = np.asarray(chosen, dtype=np.int64)
    return arrays, dict(processed=cursor, candidates=total, budget=budget, steps=steps, calls=calls,
        response_pairs=sum(c['metadata']['response_pairs'] for c in calls),
        exact_calls=sum(c['metadata']['exact_calls'] for c in calls),
        maximum_batch=max((c['end']-c['begin'] for c in calls), default=0),
        total_seconds=time.perf_counter()-started, query_targets_accessed=False, geometry_accessed=False,
        historical_credit_and_pool_generation_excluded=True,
        named_batch_array_subtotal_max=max((c['metadata']['named_array_bytes_subtotal_max'] for c in calls), default=0),
        output_array_bytes=sum(a.nbytes for a in arrays.values()),
        state_scope='Batch subtotal and output bytes separately; retained proof objects and allocator peak not measured')
