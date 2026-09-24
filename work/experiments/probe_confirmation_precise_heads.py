"""Preserve original head tolerances; resolve ill-conditioned NumPy references.

Original predictions and adaptation are NEVER changed. Only a failed meta-
ridge independent comparison invokes a fixed 70/110-digit independent solve.
Both precisions must round identically, and all ORIGINAL tolerance checks still
apply. Keep the rejected double-precision discrepancy in diagnostic metadata.
"""
import numpy as np
import probe_confirmation_independent_heads as original
from diagnose_probe_meta_ridge_arithmetic import high_precision


ATOL = 1e-9
RTOL = 1e-10
PRECISIONS = (70, 110)


def replay(cfg,x,v,q,a,meta,loaded):
    prediction,states=original.replay(cfg,x,v,q,a,meta,loaded)
    finite=np.isfinite(prediction).all() and all(np.isfinite(z).all() for z in states.values())
    passed=finite and np.allclose(prediction,a['prediction'],rtol=RTOL,atol=ATOL)
    passed=passed and all(np.allclose(z,a[k],rtol=RTOL,atol=ATOL) for k,z in states.items())
    record=dict(reference='original_numpy',original_numpy_prediction_gap=float(np.max(abs(prediction-a['prediction']))),
        high_precision_used=False,original_tolerances=dict(rtol=RTOL,atol=ATOL))
    name=cfg['name'].removeprefix('cold__')
    if passed or not name.startswith('meta_ridge'):return prediction,states,record
    model=loaded[name]
    p0,b0,n0=high_precision(model,x,v,q,PRECISIONS[0])
    p1,b1,n1=high_precision(model,x,v,q,PRECISIONS[1])
    assert p0.tobytes()==p1.tobytes() and b0.tobytes()==b1.tobytes(), 'High-precision reference is not stable'
    precise_states={'fast_0':b1[None,:,None]}
    np.testing.assert_allclose(p1,a['prediction'],rtol=RTOL,atol=ATOL)
    for k,z in precise_states.items():np.testing.assert_allclose(z,a[k],rtol=RTOL,atol=ATOL)
    record.update(reference='independent_high_precision',high_precision_used=True,precision_runs=[n0,n1],
        high_precision_prediction_gap=float(np.max(abs(p1-a['prediction']))),
        original_numpy_state_gaps={k:float(np.max(abs(z-a[k]))) for k,z in states.items()},
        high_precision_state_gaps={k:float(np.max(abs(z-a[k]))) for k,z in precise_states.items()},
        original_tolerances_passed=True)
    return p1,precise_states,record
