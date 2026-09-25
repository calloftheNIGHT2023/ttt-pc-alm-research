"""350 exact K-best support-language paths outside a supplied visited set.

The visited set must be constructed by the caller's own online trajectory.
This module only receives support, a local credit, a pattern and mode keys.
"""
from itertools import product
import time
import branch_image_chain_dyadic_v1 as integer
import support_language_chain_v1 as language


def trie(keys, n, d):
    children = [{}]; unique = set()
    for key in keys:
        if not isinstance(key, str): raise ValueError('Mode keys must be canonical hex strings')
        try: raw = bytes.fromhex(key)
        except ValueError: raise ValueError('Invalid hexadecimal mode') from None
        if key != raw.hex() or len(raw) != n*d or any(r > 3 for r in raw):
            raise ValueError('Mode shape, symbols or canonical encoding invalid')
        unique.add(raw)
    for raw in sorted(unique):
        node = 0
        for j in range(d):
            row = tuple(raw[j*n:(j+1)*n])
            if row not in children[node]:
                children[node][row] = len(children); children.append({})
            node = children[node][row]
    return children, len(unique)


def shells(problem, original, k, forbidden=()):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1: raise ValueError('k must be positive integer')
    begin = time.perf_counter(); n, d = problem.n, problem.d
    assert len(original) == d and all(len(row) == n for row in original)
    assert all(0 <= int(r) <= 3 and int(r) == r for row in original for r in row)
    assert all(0 <= x <= problem.X for x in problem.x)
    tree, size = trie(forbidden, n, d); tree_seconds = time.perf_counter()-begin
    tick = time.perf_counter(); languages = [language.accepted_paths(problem, i) for i in range(n)]
    machines = [language.automaton(paths, d) for paths in languages]; initial = tuple(m[0] for m in machines)
    language_seconds = time.perf_counter()-tick
    dp = {(0, -1, initial, 0 if size else -1): [(0, ())]} if all(s is not None for s in initial) else {}
    counts = dict(forbidden_modes=size, trie_nodes=len(tree), trie_seconds=tree_seconds,
        singleton_paths_enumerated=n*4**d, singleton_accepted_counts=list(map(len, languages)),
        automaton_states_by_observation=[[len(layer) for layer in m[1]] for m in machines],
        singleton_language_seconds=language_seconds, layer_rows_evaluated=0, empty_transitions=0,
        product_state_transitions=0, candidate_prefixes=0, max_retained_dp_entries=0,
        max_product_states=0, transition_cache_entries=0, forbidden_leaf_transitions=0,
        max_matching_trie_prefixes=0, no_global_solver=True)
    terminal = {m: problem.terminal(m) for m in range(1 << n)}
    for j in range(d):
        cache = {}; nextdp = {}; option_cache = {}
        for (distance, incoming, states, node), prefixes in sorted(dp.items()):
            if states not in option_cache:
                choices = [sorted(machines[i][1][j][s].items()) for i, s in enumerate(states)]
                option_cache[states] = [(tuple(p[0] for p in comb), tuple(p[1] for p in comb)) for comb in product(*choices)]
            low = problem.x if j == 0 else [0]*n
            high = low if j == 0 else [problem.X if incoming & (1 << i) else 0 for i in range(n)]
            previous = [0]*n if j == 0 else problem.aa[j-1]
            for row, state_next in option_cache[states]:
                counts['product_state_transitions'] += 1
                node_next = -1 if node == -1 else tree[node].get(row, -1)
                if j == d-1 and node_next != -1:
                    counts['forbidden_leaf_transitions'] += 1; continue
                outgoing = integer.mask(row); key = (incoming, row)
                if key not in cache:
                    last = terminal[outgoing] if j == d-1 else 0
                    value = None if last is None else problem.layer(low, high, previous, problem.aa[j], row)
                    cache[key] = None if value is None else value+last
                    counts['layer_rows_evaluated'] += 1
                    if cache[key] is None: counts['empty_transitions'] += 1
                value = cache[key]
                if value is None: continue
                delta = sum(a != int(b) for a, b in zip(row, original[j]))
                key_next = (distance+delta, outgoing, state_next, node_next)
                candidates = [(cost+value, path+row) for cost, path in prefixes]
                counts['candidate_prefixes'] += len(candidates)
                nextdp[key_next] = sorted(nextdp.get(key_next, [])+candidates)[:k]
        dp = nextdp
        assert all(len(pool) == 1 for key, pool in dp.items() if key[-1] != -1)
        counts['max_matching_trie_prefixes'] = max(counts['max_matching_trie_prefixes'], sum(key[-1] != -1 for key in dp))
        counts['transition_cache_entries'] += len(cache)
        counts['max_retained_dp_entries'] = max(counts['max_retained_dp_entries'], sum(map(len, dp.values())))
        counts['max_product_states'] = max(counts['max_product_states'], len(dp))
    result = {}
    for (hamming, outgoing, states, node), pool in dp.items():
        assert node == -1 and all(s == 0 for s in states)
        result.setdefault(hamming, []).extend(pool)
    result = {h: sorted(pool)[:k] for h, pool in result.items()}
    counts['total_shell_seconds'] = time.perf_counter()-begin
    counts['state_scope'] = 'Includes trie and product-DP counts, not process peak memory'
    current_accepted = all(tuple(int(original[j][i]) for j in range(d)) in languages[i] for i in range(n))
    return result, counts, current_accepted


def propose(x, v, credit, original, k=8, forbidden=()):
    begin = time.perf_counter(); problem = integer.IntegerProblem(x, v, credit)
    dp, meta, accepted = shells(problem, original, k, forbidden)
    current = problem.fixed(original) if accepted else None
    eligible = sorted(h for h, pool in dp.items() if h >= 1 and pool[0][0] <= 0)
    minimum = eligible[0] if eligible else None
    proposals = [dict(mode=bytes(path).hex(), hamming=h, rank=rank, lower=problem.string(value))
        for h, pool in sorted(dp.items()) if h >= 1 for rank, (value, path) in enumerate(pool) if value <= 0]
    meta.update(total_seconds=time.perf_counter()-begin, exact_common_scale=True,
        coordinate_scale_bits=problem.X.bit_length(), credit_scale_bits=problem.A.bit_length(),
        extraction_span='all', support_language_constrained=True, current_marginally_reachable=accepted,
        visited_excluded_before_k=True)
    # This minimum refers to the complement, not a lower bound for all feasible modes.
    return dict(current_lower=None if current is None else problem.string(current),
        current_structurally_infeasible=current is None, current_certified_infeasible=current is None or current > 0,
        minimum_nonexcluded_hamming=minimum,
        necessary_hamming_lower_bound=minimum if not meta['forbidden_modes'] and (current is None or current > 0) else None,
        minimum_unvisited_hamming=minimum,
        shell_minima={str(h): problem.string(pool[0][0]) for h, pool in sorted(dp.items())},
        proposals=proposals, meta=meta)
