"""344 exact support-conditioned marginal languages inside the K-best DP.

Necessary single-observation consistency, not joint feasibility or posterior mass.
No BP, LP, teacher, reference pool or external state is used by this module.
"""
from itertools import product
import time
import branch_image_chain_dyadic_v1 as integer


def accepted_paths(problem, index):
    allowed = []
    target_lo = max(0, problem.v[index]-problem.E)
    target_hi = min(problem.X, problem.v[index]+problem.E)
    if target_lo > target_hi:
        return allowed
    for path in product(range(4), repeat=problem.d):
        low = high = problem.x[index]
        for branch in path:
            left = max(low-problem.B, problem.zlo[branch])
            right = min(high+problem.B, problem.zhi[branch])
            if left > right:
                break
            a = integer.S[branch]*left+integer.C[branch]*problem.X
            b = integer.S[branch]*right+integer.C[branch]*problem.X
            low, high = min(a, b), max(a, b)
        else:
            if max(low, target_lo) <= min(high, target_hi):
                allowed.append(path)
    return allowed


def automaton(paths, depth):
    """Merge prefixes iff their remaining finite right languages are identical."""
    if not paths:
        return None, [{} for _ in range(depth)]
    below = {p: 0 for p in paths}
    layers = [{} for _ in range(depth)]
    for j in range(depth-1, -1, -1):
        grouped = {}
        for prefix, state in below.items():
            grouped.setdefault(prefix[:-1], {})[prefix[-1]] = state
        signatures = {p: tuple(sorted(edges.items())) for p, edges in grouped.items()}
        ids = {signature: i for i, signature in enumerate(sorted(set(signatures.values())))}
        layers[j] = {i: dict(signature) for signature, i in ids.items()}
        below = {p: ids[signature] for p, signature in signatures.items()}
    return below[()], layers


def shells(problem, original, k):
    begin = time.perf_counter()
    n, d = problem.n, problem.d
    assert len(original) == d and all(len(row) == n for row in original)
    assert all(0 <= int(r) <= 3 and int(r) == r for row in original for r in row)
    assert all(0 <= x <= problem.X for x in problem.x)
    languages = [accepted_paths(problem, i) for i in range(n)]
    machines = [automaton(paths, d) for paths in languages]
    initial = tuple(m[0] for m in machines)
    language_seconds = time.perf_counter()-begin
    dp = {(0, -1, initial): [(0, ())]} if all(s is not None for s in initial) else {}
    counts = dict(singleton_paths_enumerated=n*4**d,
        singleton_accepted_counts=list(map(len, languages)),
        automaton_states_by_observation=[[len(layer) for layer in m[1]] for m in machines],
        singleton_language_seconds=language_seconds,
        layer_rows_evaluated=0, empty_transitions=0, product_state_transitions=0,
        candidate_prefixes=0, max_retained_dp_entries=0, max_product_states=0,
        transition_cache_entries=0, no_global_solver=True)
    terminal = {m: problem.terminal(m) for m in range(1 << n)}
    for j in range(d):
        cache = {}; nextdp = {}; option_cache = {}
        for (distance, incoming, states), prefixes in sorted(dp.items()):
            if states not in option_cache:
                choices = [sorted(machines[i][1][j][s].items()) for i, s in enumerate(states)]
                option_cache[states] = [(tuple(p[0] for p in combination), tuple(p[1] for p in combination))
                                       for combination in product(*choices)]
            low = problem.x if j == 0 else [0]*n
            high = low if j == 0 else [problem.X if incoming & (1 << i) else 0 for i in range(n)]
            previous = [0]*n if j == 0 else problem.aa[j-1]
            for row, state_next in option_cache[states]:
                counts['product_state_transitions'] += 1
                outgoing = integer.mask(row)
                key = (incoming, row)
                if key not in cache:
                    last = terminal[outgoing] if j == d-1 else 0
                    value = None if last is None else problem.layer(low, high, previous, problem.aa[j], row)
                    cache[key] = None if value is None else value+last
                    counts['layer_rows_evaluated'] += 1
                    if cache[key] is None: counts['empty_transitions'] += 1
                value = cache[key]
                if value is None: continue
                delta = sum(a != int(b) for a, b in zip(row, original[j]))
                key_next = (distance+delta, outgoing, state_next)
                candidates = [(cost+value, path+row) for cost, path in prefixes]
                counts['candidate_prefixes'] += len(candidates)
                nextdp[key_next] = sorted(nextdp.get(key_next, [])+candidates)[:k]
        dp = nextdp
        counts['transition_cache_entries'] += len(cache)
        counts['max_retained_dp_entries'] = max(counts['max_retained_dp_entries'], sum(map(len, dp.values())))
        counts['max_product_states'] = max(counts['max_product_states'], len(dp))
    result = {}
    for (hamming, outgoing, states), pool in dp.items():
        assert all(s == 0 for s in states)
        result.setdefault(hamming, []).extend(pool)
    result = {h: sorted(pool)[:k] for h, pool in result.items()}
    counts['total_shell_seconds'] = time.perf_counter()-begin
    counts['state_scope'] = 'Counts include marginal automata and product DP, not process peak memory'
    current_accepted = all(tuple(int(original[j][i]) for j in range(d)) in languages[i] for i in range(n))
    return result, counts, current_accepted


def propose(x, v, credit, original, k=8):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    begin = time.perf_counter()
    problem = integer.IntegerProblem(x, v, credit)
    dp, meta, current_accepted = shells(problem, original, k)
    current = problem.fixed(original) if current_accepted else None
    eligible = sorted(h for h, pool in dp.items() if h >= 1 and pool[0][0] <= 0)
    minimum = eligible[0] if eligible else None
    proposals = []
    if minimum is not None:
        for h in range(minimum, problem.n*problem.d+1):
            for rank, (value, path) in enumerate(dp.get(h, [])):
                if value <= 0:
                    proposals.append(dict(mode=bytes(path).hex(), hamming=h, rank=rank, lower=problem.string(value)))
    meta.update(total_seconds=time.perf_counter()-begin, exact_common_scale=True,
        coordinate_scale_bits=problem.X.bit_length(), credit_scale_bits=problem.A.bit_length(),
        extraction_span='all', support_language_constrained=True, current_marginally_reachable=current_accepted)
    return dict(current_lower=None if current is None else problem.string(current),
        current_structurally_infeasible=current is None, current_certified_infeasible=current is None or current > 0,
        minimum_nonexcluded_hamming=minimum,
        necessary_hamming_lower_bound=minimum if current is None or current > 0 else None,
        shell_minima={str(h): problem.string(pool[0][0]) for h, pool in sorted(dp.items())},
        proposals=proposals, meta=meta)
