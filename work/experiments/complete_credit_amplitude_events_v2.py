"""308 v2: correct observed-noise domain; the v1 event engine is unchanged.

Only the input validator differs. An observed noisy target may be outside
[0,1] while its likelihood band still intersects the model output domain.
No observation, clipping rule, polynomial, root, or update is changed.
"""
from fractions import Fraction as F
from complete_credit_amplitude_events_v1 import Trace, partition, sweep, sweep_partition


def rational_state(b, h, direction, x, v, *, bound=.12, trust=.01, eps=.001):
    state = dict(b=[F(q) for q in b], h=[[F(q) for q in row] for row in h],
                 direction=[[F(q) for q in row] for row in direction],
                 x=[F(q) for q in x], v=[F(q) for q in v],
                 bound=F(bound), trust=F(trust), eps=F(eps))
    d, n = len(b), len(x)
    assert d > 0 and n > 0 and len(h) == len(direction) == d and len(v) == n
    assert all(len(row) == n for row in state['h'] + state['direction'])
    assert state['trust'] > 0 and state['bound'] > 0 and state['eps'] >= 0
    assert all(abs(q) <= state['bound'] for q in state['b'])
    assert all(0 <= q <= 1 for row in state['h'] for q in row)
    assert all(0 <= q <= 1 for q in state['x'])
    assert all(max(F(0), q - state['eps']) <= min(F(1), q + state['eps']) for q in state['v']), \
        'The observed noise band must intersect the [0,1] output domain'
    return state
