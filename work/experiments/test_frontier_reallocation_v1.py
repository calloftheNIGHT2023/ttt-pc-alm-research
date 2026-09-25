"""362 primitive trajectory/budget tests, no query scoring."""
from pathlib import Path
from fractions import Fraction
import numpy as np
import budget_reinvestment_suite_v1 as io
import frontier_reallocation_v1 as model
import region_conditioned_credit_v1 as original
import branch_image_chain_v1 as exact
from test_region_conditioned_credit_v1 import guarded,forward

DESIGN='outputs/ttt-pc-alm-research/362_frontier_reallocation_protocol_v1.md'
SOURCES=['frontier_reallocation_v1.py','test_frontier_reallocation_v1.py','run_frontier_reallocation_v1.py',
         'audit_frontier_reallocation_v1.py','budget_frontier_v1.py','region_conditioned_credit_v1.py',
         'branch_image_chain_dyadic_v1.py','branch_image_chain_v1.py','test_region_conditioned_credit_v1.py']


def hashes(root):return {p:io.sha(root/p) for p in [DESIGN]+['work/experiments/'+n for n in SOURCES]}


def test():
    rng=np.random.default_rng(362071);counts=dict(cases=0,rowwise_arrays=0,repeat_arrays=0,proofs=0,base_selections=0)
    for d,n,count in [(1,2,0),(2,4,3),(4,4,29)]:
        x=rng.uniform(0,1,n);v,truth=forward(x,rng.uniform(-.12,.12,d))
        regs=np.array([forward(x,b)[1] for b in rng.uniform(-.12,.12,(count,d))],np.uint8).reshape(count,d,n)
        regs=np.unique(regs,axis=0)
        for credit in [np.zeros((d,n)),rng.normal(size=(d,n))]:
            with guarded():ref,_=original.solve(x,v,regs,credit,steps=16)
            for strategy in ['frontier','uniform']:
                with guarded():
                    a,m=model.solve(x,v,regs,credit,strategy=strategy,steps=16,geometry_budget=3)
                    repeat,again=model.solve(x,v,regs,credit,strategy=strategy,steps=16,geometry_budget=3)
                assert int(a['response_counts'].sum())==m['total_response_pairs']<=len(regs)*16
                assert a['base_selected'].tolist()==np.flatnonzero(ref['first_step']==0)[:3].tolist();counts['base_selections']+=1
                for key in a:
                    assert a[key].tobytes()==repeat[key].tobytes();counts['repeat_arrays']+=1
                for i,t in enumerate(a['response_counts']):
                    if not t:continue
                    with guarded():row,_=original.solve(x,v,regs[i:i+1],credit,steps=int(t))
                    for key in ['first_step','proof_credit','final_credit','disabled']:
                        assert row[key].tobytes()==a[key][i:i+1].tobytes();counts['rowwise_arrays']+=1
                for p in m['proofs']:
                    bound=exact.fixed_value(x,v,a['proof_credit'][p['index']],regs[p['index']])
                    assert (p['structural_empty'] and bound is None) or bound==Fraction(p['lower'])>0;counts['proofs']+=1
                counts['cases']+=1
    return dict(passed=True,counts=counts,query_targets_accessed=False,no_global_solver_guard_passed=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/frontier_reallocation/preflight_v1';out.mkdir(parents=True,exist_ok=False)
    frozen=hashes(root);result=test();assert hashes(root)==frozen;io.save(out/'summary.json',dict(**result,source_sha256=frozen));print(result,flush=True)
