"""335 scoped arithmetic substitution; frozen solver and random stream unchanged."""
from contextlib import contextmanager
import time
import online_credit_branch_search_v1 as original
import branch_image_chain_dyadic_v1 as integer_chain
import support_consistency_trigger_dyadic_v1 as integer_trigger
from verify_known_range_projection_v1 import project

IMPLEMENTATIONS = ['fraction', 'integer_dp', 'integer_dp_trigger']


@contextmanager
def arithmetic(implementation):
    assert implementation in IMPLEMENTATIONS
    saved = []
    try:
        if implementation != 'fraction':
            saved.append((original.chain, 'propose', original.chain.propose))
            original.chain.propose = integer_chain.propose
        if implementation == 'integer_dp_trigger':
            for name in ['support_loss', 'exact_forward']:
                saved.append((original, name, getattr(original, name)))
                setattr(original, name, getattr(integer_trigger, name))
        yield
    finally:
        for obj, name, value in reversed(saved):
            setattr(obj, name, value)


def fit(x, v, q, seed, *, policy, channel, implementation):
    tick = time.perf_counter()
    with arithmetic(implementation):
        arrays, meta = original.fit(x, v, q, seed, policy=policy, channel=channel, trace=False)
        # Same frozen range projection, charged for every implementation.
        for key in ['prediction', 'point_prediction']:
            arrays[key] = project(arrays[key])
    seconds = time.perf_counter()-tick
    meta.update(arithmetic_implementation=implementation, charged_complete_seconds=seconds)
    return arrays, meta, seconds
