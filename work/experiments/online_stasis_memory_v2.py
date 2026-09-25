"""304 real API using the previously frozen common conditioned-geometry runtime."""
import time
import conditioned_mode_geometry as conditioned
import online_stasis_memory_v1 as original


def fit(x,v,q,seed,rule='forward_stasis',trace=False):
    start=time.perf_counter();repairs=[]
    with conditioned.geometry_scope(repairs):
        arrays,metadata=original.fit(x,v,q,seed,rule=rule,trace=trace)
    metadata.update(geometry_repair_count=len(repairs),geometry_repair_log=repairs,
        charged_complete_seconds=time.perf_counter()-start,
        common_conditioned_geometry_runtime=True)
    return arrays,metadata
