"""292 chargeable dual-erased proposals alongside an unchanged ALM trajectory.

Single-thread use only. Frozen modules are restored even on exceptions.
"""
import time
import numpy as np
import certificate_activity_attribution as original
from diagnose_gradient_flat_split_states_v1 import mode_list


def fit(x, v, q, seed, rule='changed_forward_mode', trace=False):
    assert rule in ['none', 'changed_forward_mode']
    assert original.cold.base.BOUND == .12
    cfg = next(dict(c) for c in original.CONFIGS if c['name']=='credit_control_probe33')
    local_class, original_modes = original.cold.Local, original.modes
    pending = []
    proposals, locations = [], []
    count = 0
    observer_seconds = 0.
    shadow_seconds = 0.
    max_extra_state = 0
    created = 0

    class Observed(local_class):
        def __init__(self, *args, **kwargs):
            nonlocal created
            super().__init__(*args, **kwargs)
            self.observer_phase = created
            created += 1

        def step(self):
            nonlocal count, shadow_seconds, observer_seconds, max_extra_state
            if self.observer_phase == 0 or rule == 'none':
                return super().step()
            assert self.observer_phase in [1, 2] and self.method == 'alm'
            assert not pending
            start = time.perf_counter()
            oldb, oldh, oldbest = self.b.copy(), self.h.copy(), self.best.copy()
            before = mode_list(oldb, self.x)
            observer_seconds += time.perf_counter()-start
            super().step()
            start = time.perf_counter()
            after = mode_list(self.b, self.x)
            indices = [i for i, (a, b) in enumerate(zip(before, after)) if a != b]
            cached_bytes = oldb.nbytes+oldh.nbytes+oldbest.nbytes
            observer_seconds += time.perf_counter()-start
            max_extra_state = max(max_extra_state, cached_bytes)
            if not indices:
                return
            start = time.perf_counter()
            shadow = local_class(oldb[indices], self.x, self.v, 'alm')
            shadow.h = oldh[:, indices].copy()
            shadow.best = oldbest[indices].copy()
            shadow.errors, shadow.moves = original.cold.base.score(shadow.best, self.x, self.v, np.zeros(4))
            assert not shadow.u.any()
            shadow.step()
            pending.append(shadow.b.copy())
            count += len(indices)
            max_extra_state = max(max_extra_state, cached_bytes+shadow.numeric_state_bytes()+pending[0].nbytes)
            if trace:
                proposals.extend(shadow.b.copy())
                locations.extend([[self.observer_phase-1, self.step_count, i] for i in indices])
            shadow_seconds += time.perf_counter()-start

    def augmented_modes(xx, bank):
        ans = original_modes(xx, bank)
        for extra in pending:
            for key, b in original_modes(xx, extra).items():
                ans.setdefault(key, b)
        pending.clear()
        return ans

    begin = time.perf_counter()
    try:
        original.cold.Local, original.modes = Observed, augmented_modes
        arrays, metadata = original.fit(cfg, x, v, q, seed, trace=trace)
        assert created == 3 and not pending
    finally:
        original.cold.Local, original.modes = local_class, original_modes
    metadata.update(counterfactual_rule=rule, counterfactual_state_steps=count,
        counterfactual_seconds=shadow_seconds, mode_observer_seconds=observer_seconds,
        extra_live_numeric_bytes_subtotal=max_extra_state,
        extra_state_scope='Saved pre-step b/h/best plus shadow named state and proposal bank; excludes Python objects, workspaces, trace',
        wrapped_complete_seconds=time.perf_counter()-begin, diagnostic_trace_enabled=trace,
        global_bp_used=False, query_targets_accessed=False)
    if trace:
        arrays['counterfactual_b'] = np.array(proposals, dtype=np.float64).reshape(-1, 4)
        arrays['counterfactual_locations'] = np.array(locations, dtype=np.int64).reshape(-1, 3)
    return arrays, metadata
