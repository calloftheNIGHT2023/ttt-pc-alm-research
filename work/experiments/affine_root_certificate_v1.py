"""316 exact fixed-point proposal or left-null drift certificate.

The certificate concerns a selected update formula, never task infeasibility.
Free variables retain the supplied state; no least-norm or domain claim.
"""
from fractions import Fraction as F


def dot(a, b):
    return sum((x*y for x, y in zip(a, b)), F(0))


def solve(a, rhs, reference):
    a = [list(map(F, row)) for row in a]
    rhs, reference = list(map(F, rhs)), list(map(F, reference))
    m, n = len(a), len(reference)
    assert m and n and len(rhs) == m and all(len(row) == n for row in a)
    work = [row + [value] + [F(i == j) for j in range(m)]
            for i, (row, value) in enumerate(zip(a, rhs))]
    pivots, rank = [], 0
    for col in range(n):
        pivot = next((i for i in range(rank, m) if work[i][col]), None)
        if pivot is None:
            continue
        work[rank], work[pivot] = work[pivot], work[rank]
        scale = work[rank][col]
        work[rank] = [v/scale for v in work[rank]]
        for i in range(m):
            if i != rank and work[i][col]:
                scale = work[i][col]
                work[i] = [x-scale*y for x, y in zip(work[i], work[rank])]
        pivots.append(col)
        rank += 1
        if rank == m:
            break
    free = [j for j in range(n) if j not in pivots]
    bad = next((i for i in range(rank, m) if work[i][n]), None)
    stats = dict(rank=rank, dimension=n, equations=m, pivot_columns=pivots,
                 free_columns=free,
                 max_rational_bits=max(max(abs(v.numerator).bit_length(), v.denominator.bit_length())
                                       for row in work for v in row),
                 elimination_table_elements=m*(n+1+m))
    if bad is not None:
        weights = [v/work[bad][n] for v in work[bad][n+1:]]
        result = dict(consistent=False, certificate=weights, drift=F(1), **stats)
    else:
        root = reference[:]
        for i, j in enumerate(pivots):
            root[j] = work[i][n] - sum((work[i][k]*root[k] for k in free), F(0))
        basis = []
        for k in free:
            vector = [F(j == k) for j in range(n)]
            for i, j in enumerate(pivots):
                vector[j] = -work[i][k]
            basis.append(vector)
        result = dict(consistent=True, root=root, nullspace=basis, **stats)
    verify(a, rhs, result)
    return result


def verify(a, rhs, result):
    n = len(a[0])
    if result['consistent']:
        assert [dot(row, result['root']) for row in a] == rhs
        for vector in result['nullspace']:
            assert all(dot(row, vector) == 0 for row in a)
        free = result['free_columns']
        assert len(free) == len(result['nullspace']) == n-result['rank']
        assert [[v[j] for j in free] for v in result['nullspace']] == [
            [F(i == j) for j in range(len(free))] for i in range(len(free))]
    else:
        weights = result['certificate']
        assert len(weights) == len(a)
        assert all(sum((weights[i]*a[i][j] for i in range(len(a))), F(0)) == 0
                   for j in range(n))
        assert dot(weights, rhs) == result['drift'] == 1


def system(rows, active=None, fixed=None):
    """Restrict eliminated coordinates, including nodual u=0, explicitly."""
    size = len(rows)
    active = list(range(size)) if active is None else list(active)
    fixed = {} if fixed is None else {i:F(v) for i, v in fixed.items()}
    assert len(set(active)) == len(active)
    assert set(active).isdisjoint(fixed) and set(active) | set(fixed) == set(range(size))
    a, rhs = [], []
    for i in active:
        terms = rows[i].terms
        a.append([F(i == j)-terms.get(j, F(0)) for j in active])
        rhs.append(terms.get(-1, F(0)) + sum((terms.get(j, F(0))*v for j, v in fixed.items()), F(0)))
    return a, rhs


def propose(rows, point, active=None, fixed=None):
    active = list(range(len(rows))) if active is None else list(active)
    fixed = {} if fixed is None else fixed
    a, rhs = system(rows, active, fixed)
    result = solve(a, rhs, [point[i] for i in active])
    result['active'] = active
    if result['consistent']:
        full = [F(fixed.get(i, 0)) for i in range(len(rows))]
        for i, v in zip(active, result['root']):
            full[i] = v
        result['full_root'] = full
    return result
