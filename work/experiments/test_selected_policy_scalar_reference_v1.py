"""Independent scalar formula evaluator preflight, synthetic states only."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from solver_policy_trace_v1 import step
from posterior_confirmation_pipeline import discovery_box
from effective_affine_map_v1 import build,pack,evaluate
from selected_policy_scalar_reference_v1 import scalar
from evaluate_complete_credit_mode_geometry_v1 import save,sha


def run(root,out):
    rng=np.random.default_rng(315043);counts=Counter()
    with discovery_box(.12):
        for d,n in [(1,1),(2,3),(4,4),(4,7)]:
            for method in ['alm','nodual']:
                for _ in range(12):
                    b=rng.uniform(-.12,.12,d);h=rng.uniform(0,1,(d,n));u=rng.normal(size=(d,n)) if method=='alm' else np.zeros((d,n))
                    x=rng.uniform(0,1,n);v=rng.uniform(-.0005,1.0005,n)
                    traced=step(b[None],h[:,None],u[:,None],x,v,method)
                    policies={g:a[0] for g,a in traced['policies'].items()};schema=traced['schema']
                    rows=build(policies,schema,x,v,method);p=pack(b,h,u)
                    for probe in [p]+[[q+F(int(rng.integers(-100,101)),127) for q in p] for _ in range(2)]:
                        assert evaluate(rows,probe)==scalar(policies,schema,probe,x,v,method)
                        counts['exact_vector_comparisons']+=1;counts['exact_scalar_rows']+=len(p)
    names=[Path(__file__).name,'selected_policy_scalar_reference_v1.py','effective_affine_map_v1.py','solver_policy_trace_v1.py']
    final=dict(passed=True,counts=dict(counts),real_task_data_accessed=False,off_policy_probes_are_algebra_only=True,
        source_sha256={n:sha(root/'work/experiments'/n) for n in names})
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
