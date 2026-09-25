"""357 preflight: frozen-original equivalence, exact transfer, dominance."""
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import budget_reinvestment_suite_v1 as io
import region_conditioned_credit_v1 as original
import cross_region_credit_v1 as model
import branch_image_chain_v1 as exact
from test_region_conditioned_credit_v1 import guarded,forward

DESIGN='outputs/ttt-pc-alm-research/357_cross_region_credit_protocol_v1.md'
SOURCES=['cross_region_credit_v1.py','test_cross_region_credit_v1.py','run_cross_region_credit_v1.py',
    'audit_cross_region_credit_v1.py','region_conditioned_credit_v1.py','branch_image_chain_dyadic_v1.py',
    'branch_image_chain_v1.py','factorized_dual_branch_search_v1.py','support_language_chain_v1.py',
    'local_region_screen.py','test_region_conditioned_credit_v1.py','budget_reinvestment_suite_v1.py']


def hashes(root):return {p:io.sha(root/p) for p in [DESIGN]+['work/experiments/'+s for s in SOURCES]}


def test():
    rng=np.random.default_rng(357071);counts=dict(cases=0,original_arrays=0,repeat_arrays=0,proofs=0,transfers=0,dominance_regions=0,feasible_retained=0)
    for d,n in [(1,2),(2,4),(4,4),(4,8)]:
        x=rng.uniform(0,1,n);b=rng.uniform(-.12,.12,d);v,truth=forward(x,b)
        regs=np.array([truth]+[forward(x,q)[1] for q in rng.uniform(-.12,.12,(47,d))])
        for credit in [np.zeros((d,n)),rng.normal(size=(d,n)),np.ones((d,n))]:
            with guarded():
                ref,_=original.solve(x,v,regs,credit)
                off,om=model.solve(x,v,regs,credit,propagate=False)
                on,meta=model.solve(x,v,regs,credit,propagate=True)
                repeat,rm=model.solve(x,v,regs,credit,propagate=True)
            for k in ref:
                assert ref[k].tobytes()==off[k].tobytes();counts['original_arrays']+=1
                assert on[k].tobytes()==repeat[k].tobytes();counts['repeat_arrays']+=1
            assert meta['proofs']==rm['proofs'] and meta['directions']==rm['directions']
            mask=off['first_step']>0
            assert np.all((on['first_step'][mask]>0)&(on['first_step'][mask]<=off['first_step'][mask]))
            counts['dominance_regions']+=int(mask.sum());assert on['first_step'][0]==off['first_step'][0]==0
            counts['feasible_retained']+=1
            by={p['index']:p for p in meta['proofs']}
            for p in meta['proofs']:
                i=p['index'];value=exact.fixed_value(x,v,on['proof_credit'][i],regs[i])
                assert (value is None and p['structural_empty']) or value==F(p['lower'])>0
                if p['kind']=='transfer':
                    origin=by[p['source_index']];assert origin['kind']=='local' and origin['step']<=p['step']
                    assert p['source_index']!=i
                    assert on['proof_credit'][i].tobytes()==on['proof_credit'][p['source_index']].tobytes()
                    counts['transfers']+=1
                counts['proofs']+=1
            counts['cases']+=1
    # The random low-dimensional cases finish too early to exercise propagation.
    # A fixed old primitive, not a formal-development seed, covers that branch.
    from run_cross_region_credit_v1 import pool
    old_rng=np.random.default_rng(5900001);teacher=old_rng.uniform(-.12,.12,4)
    x=old_rng.uniform(0,1,24)[:4];v,truth=forward(x,teacher)
    regs,_=pool(x,v,truth,[]);credit=np.zeros((4,4))
    with guarded():
        off,_=model.solve(x,v,regs,credit,propagate=False)
        on,meta=model.solve(x,v,regs,credit,propagate=True)
        repeat,again=model.solve(x,v,regs,credit,propagate=True)
    for k in on:assert on[k].tobytes()==repeat[k].tobytes();counts['repeat_arrays']+=1
    mask=off['first_step']>0;assert np.all((on['first_step'][mask]>0)&(on['first_step'][mask]<=off['first_step'][mask]))
    counts['dominance_regions']+=int(mask.sum());assert meta['proofs']==again['proofs']
    truth_index=next(i for i,r in enumerate(regs) if np.array_equal(r,truth))
    assert on['first_step'][truth_index]==0;counts['feasible_retained']+=1
    by={p['index']:p for p in meta['proofs']}
    for p in meta['proofs']:
        i=p['index'];value=exact.fixed_value(x,v,on['proof_credit'][i],regs[i]);assert value==F(p['lower'])>0
        if p['kind']=='transfer':
            source=by[p['source_index']];assert source['kind']=='local' and source['step']<=p['step'] and source['index']!=i
            assert on['proof_credit'][i].tobytes()==on['proof_credit'][source['index']].tobytes();counts['transfers']+=1
        counts['proofs']+=1
    counts['cases']+=1
    assert counts['transfers']>0
    return dict(passed=True,counts=counts,query_targets_accessed=False,no_global_solver_guard_passed=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/'results/cross_region_credit/preflight_v1'
    out.mkdir(parents=True,exist_ok=False);frozen=hashes(root);result=test();assert hashes(root)==frozen
    io.save(out/'summary.json',dict(**result,source_sha256=frozen));print(result,flush=True)
