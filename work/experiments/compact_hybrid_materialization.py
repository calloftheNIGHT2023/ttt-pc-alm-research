"""Stronger BP control: reuse its already-retained residual credit entries."""
from types import SimpleNamespace
import numpy as np
import hybrid_rejection_materialization as original

METHODS=original.METHODS
capture=original.capture
geometry=original.geometry
sample=original.sample
predict=original.predict


def compact_capture(x,v,cfg,include_bp=True):
    if not include_bp:raise ValueError('Compact observer is only a BP comparator')
    saved=capture.model.Collector
    class Observer(saved):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs);self.bp_observer=capture.WatchCollector(self.x,self.v,'bp')
        def parameter(self,b,*args,**kwargs):
            super().parameter(b,*args,**kwargs);self.bp_observer.parameter(b,*args,**kwargs)
        def activity(self,b,h,u=None,step=0,phase=''):
            super().activity(b,h,u,step,phase);self.bp_observer.activity(b,h,u,step,phase)
    try:capture.model.Collector=Observer;result=capture.model.prepare(x,v,cfg)
    finally:capture.model.Collector=saved
    bp=[c.bp_observer for c in result[3]]
    # A shallow dictionary view shares numeric credit arrays; constructing and
    # screening it are still charged. Residual bank keeps its own cap rule.
    residual=[SimpleNamespace(kind='residual',retained={k:v for k,v in c.retained.items() if k[1]=='residual'}) for c in bp]
    return result,dict(local=result[3],bp=bp,residual=residual)


def prepare(x,v,cfg,method):
    saved=capture.capture
    try:
        capture.capture=compact_capture
        return original.prepare(x,v,cfg,method)
    finally:capture.capture=saved


def verify(x,v,cfg,expected_patterns,expected_masks):
    old,old_roles=capture.capture(x,v,cfg,True);new,new_roles=compact_capture(x,v,cfg)
    assert np.array_equal(old[0],new[0]) and np.array_equal(old[1],new[1])
    retained=0
    for kind in ['local','bp','residual']:
        for a,b in zip(old_roles[kind],new_roles[kind]):
            assert list(a.retained)==list(b.retained),(kind,'insertion order')
            for key in a.retained:
                aa=a.retained[key];bb=b.retained[key];assert np.array_equal(aa['a'],bb['a'])
                assert {k:v for k,v in aa.items() if k!='a'}=={k:v for k,v in bb.items() if k!='a'};retained+=1
    cases=0
    for method in METHODS:
        data,_=prepare(x,v,cfg,method);assert [r.tobytes().hex() for r in data['regs']]==expected_patterns
        if method!='geometry':assert np.array_equal(data['indices'],np.flatnonzero(~expected_masks[method]))
        olddata,_=original.prepare(x,v,cfg,method)
        a,la,_=sample(data,64,np.random.default_rng(481807));b,lb,_=original.sample(olddata,64,np.random.default_rng(481807))
        assert np.array_equal(a,b) and np.array_equal(la,lb);cases+=1
    jac=capture.model.base.forward_jacobian;bp_module=capture.model.old.old.old.old.old.core.batched;refine=bp_module.refine;bp_capture=capture.capture
    def forbidden(*a,**k):raise AssertionError('BP entered compact local path')
    try:
        capture.model.base.forward_jacobian=forbidden;bp_module.refine=forbidden;capture.capture=forbidden
        # prepare() only swaps the optional callable; local method does not
        # invoke it. Global BP entry points remain forbidden through fallback.
        data,_=prepare(x,v,cfg,'local');sample(data,64,np.random.default_rng(481807),proposal_budget=0)
    finally:capture.model.base.forward_jacobian=jac;bp_module.refine=refine;capture.capture=bp_capture
    return dict(passed=True,retained_credit_entries_bitwise=retained,all_methods_state_bitwise=cases,
                local_forced_fallback_no_global_bp=True,hybrid_distribution=original.verify_distribution())
