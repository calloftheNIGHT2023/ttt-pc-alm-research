"""360 preflight includes empty/all/partial pools and exact rowwise replay."""
from itertools import product
from pathlib import Path
import numpy as np
import budget_frontier_v1 as lazy
import region_conditioned_credit_v1 as full
import budget_reinvestment_suite_v1 as io
from test_region_conditioned_credit_v1 import forward, guarded
from run_budget_frontier_v1 import hashes, BASE


def test():
    identities = 0; cases = 0; arrays = 0
    for size in range(10):
        for bits in product([0, 1], repeat=size):
            for budget in [1, 4, 8, 16]:
                selected = lazy.selected(np.array(bits), budget).tolist()
                formula = [j for j in range(size) if not bits[j] and j+1-sum(bits[:j+1]) <= budget]
                assert selected == formula; identities += size
    rng = np.random.default_rng(360071)
    for d, n, count in [(1, 2, 0), (2, 4, 3), (4, 4, 29)]:
        x = rng.uniform(0, 1, n); v, truth = forward(x, rng.uniform(-.12, .12, d))
        regs = np.array([forward(x, b)[1] for b in rng.uniform(-.12, .12, (count, d))], np.uint8).reshape(count, d, n)
        for credit in [np.zeros((d, n)), rng.normal(size=(d, n))]:
            for budget in [1, 8, 32]:
                with guarded():
                    ref, _ = full.solve(x, v, regs, credit, 16)
                    a, meta = lazy.solve(x, v, regs, credit, steps=16, budget=budget)
                chosen = lazy.selected(ref['first_step'], budget)
                end = int(chosen[-1])+1 if len(chosen) == budget else count
                assert chosen.tolist() == a['selected_indices'].tolist() and meta['processed'] == end
                for key in ref:
                    assert ref[key][:end].tobytes() == a[key].tobytes(); arrays += 1
                assert meta['maximum_batch'] <= budget
                cases += 1
    return dict(passed=True, scalar_identities=identities, cases=cases, bitwise_prefix_arrays=arrays,
                query_targets_accessed=False, no_global_solver_guard_passed=True)


if __name__ == '__main__':
    root = Path(__file__).resolve().parents[2]; out = root/BASE/'preflight_v1'; out.mkdir(parents=True, exist_ok=False)
    frozen = hashes(root); result = test(); assert hashes(root) == frozen
    io.save(out/'summary.json', dict(**result, source_sha256=frozen)); print(result, flush=True)
