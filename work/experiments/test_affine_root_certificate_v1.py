"""316 synthetic algebra, independent old RREF, solver and boundary tests."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import traceback
import numpy as np
from affine_root_certificate_v1 import solve, system, propose, dot
from affine_root_validation_v1 import validate
from effective_affine_map_v1 import Affine, build, pack, evaluate
from multiplier_fixed_point_exact import rref_solve
from posterior_confirmation_pipeline import discovery_box
from solver_policy_trace_v1 import step
from evaluate_complete_credit_mode_geometry_v1 import save, sha


def run(root, out):
    counts = Counter()
    cases = [([[2,1],[1,1]],[1,0],[0,0]),
             ([[1,2],[2,4]],[3,6],[F(1,7),F(1,3)]),
             ([[1,2],[2,4]],[3,7],[0,0]),
             ([[1,0],[0,F(1,10**100)]],[1,1],[0,0]),
             ([[0,0],[0,0]],[0,0],[2,3]),
             ([[0]],[1],[0])]
    rng = np.random.default_rng(316001)
    for n in range(1,9):
        for m in [n,max(1,n-1),n+1]:
            for i in range(12):
                a = rng.integers(-4,5,(m,n)).tolist()
                if i%3 == 0 and m > 1:
                    a[-1] = [2*v for v in a[0]]
                reference = [F(int(v),11) for v in rng.integers(-5,6,n)]
                truth = [F(int(v),7) for v in rng.integers(-5,6,n)]
                rhs = [dot(row,truth) for row in a]
                if i%2:
                    rhs[-1] += F(1,13)
                cases.append((a,rhs,reference))
    for a,rhs,reference in cases:
        result = solve(a,rhs,reference)
        other,meta = rref_solve(a,rhs,reference)
        assert result['rank'] == meta['rank']
        assert result['consistent'] == (other is not None)
        if other is not None:
            assert result['root'] == other
            counts['consistent'] += 1
        else:
            w = result['certificate']
            # Independent projected drift test at multiple unrelated points.
            for k in range(3):
                p = [F(k-j,17) for j in range(len(reference))]
                residual = [F(c)-dot(row,p) for row,c in zip(a,rhs)]
                assert dot(w,residual) == 1
            counts['inconsistent'] += 1
        counts['independent_rref_systems'] += 1
    # No-root is not a finite-exit theorem on an unbounded state domain.
    drift = propose([Affine({0:F(1),-1:F(1)})],[F(0)])
    assert not drift['consistent']
    value = F(0)
    for t in range(20):
        assert value == t
        value += 1
    counts['unbounded_drift_counterexamples'] += 1
    # A valid algebraic root may lie outside the selected formula's region.
    rootcase = propose([Affine({0:F(1,2),-1:F(3,4)})],[F(0)])
    assert rootcase['full_root'] == [F(3,2)] and rootcase['full_root'][0] > 1
    counts['off_region_root_counterexamples'] += 1
    # Eliminated u is fixed at zero, not a free solver coordinate.
    rows = [Affine({0:F(1),1:F(1),-1:F(1)}),Affine({1:F(1)})]
    assert propose(rows,[F(0),F(0)])['consistent']
    assert not propose(rows,[F(0),F(0)],active=[0],fixed={1:F(0)})['consistent']
    counts['nodual_restriction_counterexamples'] += 1
    with discovery_box(.12):
        for d,n in [(1,1),(2,3),(4,4)]:
            for method in ['alm','nodual']:
                for i in range(8):
                    x=rng.uniform(0,1,n);v=rng.uniform(.05,.95,n)
                    b=rng.uniform(-.12,.12,d);h=rng.uniform(0,1,(d,n));h[-1]=v
                    u=rng.uniform(-.2,.2,(d,n)) if method=='alm' else np.zeros((d,n))
                    rr=step(b[None],h[:,None],u[:,None],x,v,method)
                    rows=build({g:a[0] for g,a in rr['policies'].items()},rr['schema'],x,v,method)
                    point=pack(b,h,u);dim=d+d*n
                    active=None if method=='alm' else list(range(dim))
                    fixed=None if method=='alm' else {j:F(0) for j in range(dim,len(rows))}
                    ans=propose(rows,point,active,fixed)
                    if ans['consistent']:
                        assert evaluate(rows,ans['full_root']) == ans['full_root']
                        val=validate(ans['full_root'],d,x,v,method)
                        counts['solver_root_domain_valid'] += val['domain_valid']
                        counts['solver_actual_fixed'] += val['actual_fixed_point']
                    counts['synthetic_solver_cases'] += 1
        x=np.array([.25]);v=np.array([.625]);b=np.array([.0625]);h=np.array([[.625]]);u=np.zeros((1,1))
        rr=step(b[None],h[:,None],u[:,None],x,v,'alm')
        rows=build({g:a[0] for g,a in rr['policies'].items()},rr['schema'],x,v,'alm')
        point=pack(b,h,u);ans=propose(rows,point)
        assert ans['consistent'] and ans['full_root']==point
        checked=validate(point,1,x,v,'alm')
        assert checked['actual_fixed_point'] and checked['metrics']['support_band_feasible']
        counts['known_exact_alm_fixed_points'] += 1
    names=[Path(__file__).name,'affine_root_certificate_v1.py','affine_root_validation_v1.py',
           'effective_affine_map_v1.py','complete_credit_rational_reference_v1.py',
           'multiplier_fixed_point_exact.py','solver_policy_trace_v1.py','posterior_confirmation_pipeline.py']
    final=dict(passed=True,counts=dict(counts),real_task_data_accessed=False,
               source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
