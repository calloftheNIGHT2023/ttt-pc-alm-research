"""365 no-mother structured inference, with all joint geometry explicitly charged."""
from contextlib import contextmanager
import time
import numpy as np
import strong_pool_credit_bridge_v1 as mother
import online_credit_branch_search_v1 as local
import shared_mode_readout as shared
from run_cross_region_credit_v1 import pool
from test_region_conditioned_credit_v1 import guarded
from verify_known_range_projection_v1 import project


@contextmanager
def no_mother():
    saved=[]
    def forbidden(*args,**kwargs):
        raise AssertionError('Direct baseline entered a mother trajectory or local optimizer')
    try:
        for obj,name in [(mother,'fit'),(mother.cold,'Local'),(mother.transfer,'run')]:
            saved.append((obj,name,getattr(obj,name)));setattr(obj,name,forbidden)
        with local.no_bp(True):yield
    finally:
        for obj,name,value in saved:setattr(obj,name,value)


def fit(x,v,q,seed):
    begin=time.perf_counter()
    assert x.shape==v.shape==(4,) and mother.cold.base.BOUND==.12
    with no_mother():
        # All rows are consumed; the arbitrary anchor only orders the enumeration.
        with guarded():regions,poolmeta=pool(x,v,np.zeros((4,4),np.uint8),[])
        tick=time.perf_counter();polys={};notes={}
        for key in sorted(r.tobytes().hex() for r in regions):
            poly,note=local.explicit_geometry(x,v,key);notes[key]=note
            if poly is not None:polys[key]=poly
        geometry_seconds=time.perf_counter()-tick;keys=sorted(polys);tick=time.perf_counter()
        if keys:
            points,allocation=shared.draw(polys,keys,2048,np.random.default_rng(np.random.SeedSequence([249911,seed,2048])))
        else:points=np.zeros((1,4));allocation=np.empty(0,int)
        sampling_seconds=time.perf_counter()-tick;tick=time.perf_counter()
        prediction=project(shared.geometry.make_predict(points)(q))
        # Diagnostic point is the barycenter of sampled parameters, not a fitted optimizer.
        selected=points.mean(axis=0);point_prediction=project(mother.cold.base.forward(q,selected))
        reading_seconds=time.perf_counter()-tick
    arrays=dict(points=points,allocation=allocation,prediction=prediction,point_prediction=point_prediction,
                selected_b=selected,best_bank=selected[None],search_regions=regions,
                positive_volumes=np.array([polys[k]['volume'] for k in keys]))
    meta=dict(execution_failed=False,query_targets_accessed=False,solver='direct_necessary_language',
        mother_trajectory_called=False,global_bp_called=False,archived_candidate_state_accessed=False,
        geometry_is_global_lp_not_local_pc=True,positive_modes=keys,pool=poolmeta,
        new_mode_classifications_detail=notes,geometry_seconds=geometry_seconds,
        sampling_seconds=sampling_seconds,reading_seconds=reading_seconds,
        charged_complete_seconds=time.perf_counter()-begin,
        geometry_numeric_bytes_subtotal=sum(a.nbytes for p in polys.values() for a in p.values() if isinstance(a,np.ndarray)),
        returned_array_bytes=sum(a.nbytes for a in arrays.values()),
        memory_scope='Named arrays only; enumeration blocks, Python containers, geometry temporaries and allocator peak excluded',
        exact_complete_posterior_claim=False,resources_matched=False)
    return arrays,meta
