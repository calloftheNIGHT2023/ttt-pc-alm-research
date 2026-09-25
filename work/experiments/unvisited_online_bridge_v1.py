"""353 observe only this live parent trajectory; no archived state input."""
import time
import strong_pool_credit_bridge_v1 as bridge
import support_language_chain_v1 as language
import unvisited_language_chain_v1 as complement


def fit(x, v, q, seed, *, channel, exclude=True, trace_hash=True):
    begin = time.perf_counter(); visited = set(); captures = 0; proposals = 0
    original_modes = bridge.modes; original_propose = language.propose
    def collect(xx, bank):
        nonlocal captures
        result = original_modes(xx, bank); visited.update(result); captures += 1
        return result
    def propose(xx, vv, credit, pattern, k=8):
        nonlocal proposals
        proposals += 1
        return complement.propose(xx, vv, credit, pattern, k=k, forbidden=sorted(visited) if exclude else ())
    try:
        bridge.modes = collect; language.propose = propose
        arrays, meta = bridge.fit(x, v, q, seed, channel=channel, trace_hash=trace_hash)
    finally:
        bridge.modes = original_modes; language.propose = original_propose
    assert proposals == 1 and sorted(visited) == meta['visited_modes']
    assert meta['proposal']['meta']['forbidden_modes'] == (len(visited) if exclude else 0)
    meta.update(visited_excluded_before_k=exclude, visited_collector_calls=captures,
        visited_set_numeric_key_bytes_subtotal=sum(len(bytes.fromhex(key)) for key in visited),
        visited_source='Own current live trajectory, not old files or another solver',
        charged_complete_seconds=time.perf_counter()-begin)
    return arrays, meta
