"""Isolated-process memory audit; not used for benchmark latency.

Reports absolute process peak RSS AND peak traced allocations. Neither is a
GPU-memory result. Includes imports in absolute RSS, disclosed explicitly.
"""
from __future__ import annotations
import argparse
import ctypes
import json
import subprocess
import sys
import tracemalloc
from pathlib import Path


def rss():
    class Counters(ctypes.Structure):
        _fields_=[("cb",ctypes.c_uint32),("PageFaultCount",ctypes.c_uint32),
                  ("PeakWorkingSetSize",ctypes.c_size_t),("WorkingSetSize",ctypes.c_size_t),
                  ("QuotaPeakPagedPoolUsage",ctypes.c_size_t),("QuotaPagedPoolUsage",ctypes.c_size_t),
                  ("QuotaPeakNonPagedPoolUsage",ctypes.c_size_t),("QuotaNonPagedPoolUsage",ctypes.c_size_t),
                  ("PagefileUsage",ctypes.c_size_t),("PeakPagefileUsage",ctypes.c_size_t)]
    api=ctypes.WinDLL("kernel32"); api.GetCurrentProcess.restype=ctypes.c_void_p
    psapi=ctypes.WinDLL("psapi")
    psapi.GetProcessMemoryInfo.argtypes=[ctypes.c_void_p,ctypes.POINTER(Counters),ctypes.c_uint32]
    psapi.GetProcessMemoryInfo.restype=ctypes.c_int
    counters=Counters(); counters.cb=ctypes.sizeof(counters)
    if not psapi.GetProcessMemoryInfo(api.GetCurrentProcess(),ctypes.byref(counters),counters.cb):
        raise ctypes.WinError()
    return {"working_set_bytes":int(counters.WorkingSetSize),"peak_working_set_bytes":int(counters.PeakWorkingSetSize)}


def worker(cfg):
    import numpy as np
    import certified_branch_search as runner
    from run_certified_search import global_bank
    import streaming_branch_projection as base
    import streaming_projection_refinement as refine
    from local_branch_memory import fit_regression, fit_shallow
    rng=np.random.default_rng(5600000)
    truth=rng.uniform(-.12,.12,4); x=rng.uniform(0,1,24); v=base.forward(x,truth)
    guard=False
    if cfg["kind"]=="local":
        def forbidden(*args,**kwargs):
            raise AssertionError("Global derivative/optimizer called inside local candidate")
        base.forward_jacobian=forbidden
        refine.minimize=forbidden
        refine.linprog=forbidden
        base.minimize=forbidden
        base.linprog=forbidden
        base.branch_polytope=forbidden
        guard=True
    before=rss(); tracemalloc.start()
    if cfg["kind"]=="local":
        _,meta=runner.fit(x,v,np.zeros(4),cfg)
    elif cfg["kind"]=="restored_slsqp":
        _,meta=refine.fit_restored_slsqp(x,v,np.zeros(4),restarts=cfg.get("restarts",64))
    elif cfg["kind"]=="global_bank":
        _,meta=global_bank(x,v,np.zeros(4),cfg)
    elif cfg["kind"]=="regression":
        _,meta=fit_regression(x,v,4,family=cfg["family"],features=cfg.get("features",256))
    elif cfg["kind"]=="shallow":
        _,meta=fit_shallow(x,v,4,width=cfg.get("width",64),steps=1000,restarts=4)
    else:
        w=np.zeros(2); inv=np.eye(2)/1e-4
        for xx,y in zip(x,v):
            phi=np.array([1.,xx]); gain=inv@phi/(1+phi@inv@phi)
            w+=gain*(y-phi@w); inv-=np.outer(gain,phi@inv)
        meta={"persistent_scalars":6}
    current,peak=tracemalloc.get_traced_memory(); tracemalloc.stop()
    after=rss()
    return {"method":cfg["name"],"traced_peak_bytes":peak,"traced_retained_bytes":current,
            "process_before":before,"process_after":after,
            "process_peak_growth_bytes":after["peak_working_set_bytes"]-before["peak_working_set_bytes"],
            "global_optimizer_and_jacobian_guard_passed":guard,"metadata":meta}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",type=Path,required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--worker",type=int)
    args=p.parse_args()
    cfgs=json.loads(args.config.read_text(encoding="utf-8"))["configs"]
    if args.worker is not None:
        print(json.dumps(worker(cfgs[args.worker]))); return
    rows=[]
    for i in range(len(cfgs)):
        result=subprocess.run([sys.executable,str(Path(__file__)),"--config",str(args.config.resolve()),"--worker",str(i)],
                              capture_output=True,text=True,check=True,creationflags=subprocess.CREATE_NO_WINDOW)
        row=json.loads(result.stdout); rows.append(row)
        print(json.dumps({k:row[k] for k in ["method","traced_peak_bytes","process_peak_growth_bytes"]}),flush=True)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps({"audit_stage":"one 24-context write from zero; fixed development seed5600000",
                                   "latency_note":"profiling runs are NOT latency results",
                                   "memory_note":"absolute Windows RSS includes common imports; growth can undercount allocations below an earlier import peak; tracemalloc can miss untracked native library allocations",
                                   "rows":rows},indent=2),encoding="utf-8")


if __name__=="__main__": main()
