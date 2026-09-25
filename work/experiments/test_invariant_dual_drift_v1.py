"""317 synthetic exact identities, Jordan counterexample and phase boundary."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import traceback
import numpy as np
from invariant_dual_drift_v1 import propose_line
from effective_affine_map_v1 import Affine,evaluate
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(root,out):
    counts=Counter()
    rows=[Affine({0:F(1,2),-1:F(1)}),Affine({0:F(1),1:F(1)})]
    answer=propose_line(rows,[F(0),F(0)],1)
    assert answer['consistent'] and answer['q']==[F(2),F(0)] and answer['d']==[F(0),F(2)]
    assert answer['q']!=[F(0),F(0)] # A proposal is not exact acceleration from that start.
    counts['known_constant_primal_drift_line']+=1
    jordan=[Affine({0:F(1),1:F(1)}),Affine({1:F(1),-1:F(1)})]
    assert not propose_line(jordan,[F(0),F(0)],1)['consistent']
    value=[F(0),F(0)]
    for t in range(20):
        assert value==[F(t*(t-1),2),F(t)]
        value=evaluate(jordan,value)
    counts['quadratic_drift_no_invariant_line_counterexample']+=1
    rng=np.random.default_rng(317001)
    for p in [1,2,4]:
        for k in [1,3]:
            n=p+k
            for case in range(12):
                # Construct a map with a known invariant line, then leave its
                # free phase for the deterministic solver to select.
                q=[F(int(v),11) for v in rng.integers(-4,5,n)]
                d=[F(0)]*p+[F(int(v),7) for v in rng.integers(-4,5,k)]
                matrix=[]
                for i in range(n):
                    row=[F(int(v),5) for v in rng.integers(-3,4,p)]+[F(i==j) for j in range(p,n)]
                    matrix.append(row)
                rows=[Affine({**{j:v for j,v in enumerate(row) if v},
                              -1:q[i]+d[i]-sum((v*q[j] for j,v in enumerate(row)),F(0))})
                      for i,row in enumerate(matrix)]
                proposed=propose_line(rows,[F(0)]*n,p);assert proposed['consistent']
                qq,dd=proposed['q'],proposed['d']
                for t in [F(-3),F(0),F(1,3),F(1),F(1000)]:
                    got=evaluate(rows,[a+t*b for a,b in zip(qq,dd)])
                    assert got==[a+(t+1)*b for a,b in zip(qq,dd)]
                    counts['exact_line_vectors']+=1
                counts['constructed_maps']+=1
    names=[Path(__file__).name,'invariant_dual_drift_v1.py','affine_root_certificate_v1.py','effective_affine_map_v1.py']
    final=dict(passed=True,counts=dict(counts),real_task_data_accessed=False,
               actual_solver_guards_established=False,source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
