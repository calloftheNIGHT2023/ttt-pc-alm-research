"""316 candidate domain and independent full exact local-solver checks."""
from fractions import Fraction as F
from complete_credit_rational_reference_v1 import direct_step, tent


def state(point, d, x, v):
    n = len(x)
    assert len(point) == d+2*d*n
    return dict(b=list(point[:d]), h=[list(point[d+j*n:d+(j+1)*n]) for j in range(d)],
                direction=[list(point[d+d*n+j*n:d+d*n+(j+1)*n]) for j in range(d)],
                x=list(map(F, x)), v=list(map(F, v)),
                bound=F(.12), trust=F(.01), eps=F(.001))


def domain(s):
    failures = []
    for j, b in enumerate(s['b']):
        if not -s['bound'] <= b <= s['bound']:
            failures.append(f'b{j}:box')
    for j, row in enumerate(s['h']):
        for i, h in enumerate(row):
            lo, hi = F(0), F(1)
            if j == len(s['h'])-1:
                lo, hi = max(lo, s['v'][i]-s['eps']), min(hi, s['v'][i]+s['eps'])
            if not lo <= h <= hi:
                failures.append(f'h{j},{i}:domain')
    return failures


def forward_metrics(s):
    previous, codes = s['x'][:], []
    residual = []
    for j, b in enumerate(s['b']):
        codes.extend(sum(z+b >= k for k in [F(0), F(1,2), F(1)]) for z in previous)
        previous = [tent(z+b) for z in previous]
        inputs = s['x'] if j == 0 else s['h'][j-1]
        residual.extend(h-tent(z+b) for h, z in zip(s['h'][j], inputs))
    err = [p-v for p, v in zip(previous, s['v'])]
    return dict(forward=previous, mode=''.join(map(str,codes)), residual_max=max(map(abs,residual)),
                support_mse=sum((e*e for e in err), F(0))/len(err),
                support_max_error=max(map(abs,err)),
                support_band_feasible=all(abs(e) <= s['eps'] for e in err))


def exact_step(s, method):
    rr = direct_step(s, 1)
    rate = F(1,2) if method == 'alm' else F(0)
    newu = []
    for j, b in enumerate(rr['b']):
        previous = s['x'] if j == 0 else rr['h'][j-1]
        newu.extend(u+rate*(h-tent(z+b)) for u,h,z in zip(s['direction'][j],rr['h'][j],previous))
    return rr['b'] + [h for row in rr['h'] for h in row] + newu, rr


def validate(point, d, x, v, method):
    s = state(point, d, x, v)
    failures = domain(s)
    result = dict(domain_violations=failures, domain_valid=not failures,
                  actual_fixed_point=False, metrics=forward_metrics(s))
    if not failures:
        actual, rr = exact_step(s, method)
        result.update(actual_step=actual, exact_solver_gap=max(abs(a-b) for a,b in zip(actual,point)),
                      actual_fixed_point=actual == point,
                      activity_ties=rr['activity_ties'], bias_ties=rr['bias_ties'])
        if result['actual_fixed_point'] and method == 'alm':
            assert result['metrics']['residual_max'] == 0
            assert result['metrics']['support_band_feasible']
    return result
