"""339 radius-only extraction from the frozen 334 exact all-shell dynamic program.

This changes which already computed shells are returned, not the relaxation,
credit, parameters or solver. Wider radius is not a guarantee of lower query risk.
"""
import time
import branch_image_chain_dyadic_v1 as integer


def propose(x, v, credit, original, k=8, *, span=3):
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError('k must be a positive integer')
    if span != 'all' and (isinstance(span, bool) or not isinstance(span, int) or span < 1):
        raise ValueError('span must be a positive integer or all')
    begin = time.perf_counter()
    problem = integer.IntegerProblem(x, v, credit)
    dp, meta = integer.integer_shells(problem, original, k)
    current = problem.fixed(original)
    admissible = sorted(m for m, pool in dp.items() if m >= 1 and pool[0][0] <= 0)
    minimum = admissible[0] if admissible else None
    proposals = []
    maximum = None
    if minimum is not None:
        maximum = problem.n*problem.d if span == 'all' else min(problem.n*problem.d, minimum+span-1)
        for m in range(minimum, maximum+1):
            for rank, (value, path) in enumerate(dp.get(m, [])):
                if value <= 0:
                    proposals.append(dict(mode=bytes(path).hex(), hamming=m,
                                          rank=rank, lower=problem.string(value)))
    meta.update(total_seconds=time.perf_counter()-begin,
                coordinate_scale_bits=problem.X.bit_length(), credit_scale_bits=problem.A.bit_length(),
                exact_common_scale=True, extraction_span=span, maximum_extracted_hamming=maximum,
                radius_changes_dp=False)
    return dict(current_lower=None if current is None else problem.string(current),
        current_structurally_infeasible=current is None,
        current_certified_infeasible=current is None or current > 0,
        minimum_nonexcluded_hamming=minimum,
        necessary_hamming_lower_bound=minimum if current is None or current > 0 else None,
        shell_minima={str(m): problem.string(pool[0][0]) for m, pool in sorted(dp.items())},
        proposals=proposals, meta=meta)
