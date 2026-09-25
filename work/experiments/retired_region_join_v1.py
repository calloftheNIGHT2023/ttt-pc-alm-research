"""380 same frozen search, explicit extended budgets, shared strict-row retirement."""
from contextlib import contextmanager
import time
import regional_prefix_join_v1 as original
import certified_row_retirement_v1 as retirement

CONFIGS = [('none', None, 0)]+[(f'{family}{steps}', family, steps)
    for family in ['regional_active', 'regional_passive', 'pdhg_cold', 'pdhg_box'] for steps in [128, 256, 512, 1024]]
PRIMARY = 'regional_active256'
SETTINGS = [('bfs', 'observed'), ('dfs', 'farthest_x')]


@contextmanager
def configuration(*, retire=True):
    before_configs = original.CONFIGS; before_solver = original.regional.solve
    assert before_solver is retirement.ORIGINAL_SOLVE, 'Nested solver replacement is forbidden'
    original.CONFIGS = CONFIGS
    if retire: original.regional.solve = retirement.solve
    try:
        yield
    finally:
        original.CONFIGS = before_configs; original.regional.solve = before_solver


def join(x, v, *, name, schedule, order_name='observed', retire=True, **kwargs):
    begin = time.perf_counter()
    with configuration(retire=retire):
        a, m = original.join(x, v, name=name, schedule=schedule, order_name=order_name, **kwargs)
    m['solver_coordinate_steps_upper_bound'] = m['solver_coordinate_steps']
    m['solver_coordinate_steps'] = sum(row['solver']['coordinate_steps'] if retire else
                                     row['solver']['steps']*(row['processed']-row['rejected_c5']-row['rejected_c20'])*a['regions'].shape[1]*row['k']
                                     for row in m['batches'] if row['solver'] is not None)
    m['strict_certificate_retirement'] = retire
    m['total_seconds'] = time.perf_counter()-begin
    assert m['solver_coordinate_steps'] <= m['solver_coordinate_steps_upper_bound']
    return a, m
