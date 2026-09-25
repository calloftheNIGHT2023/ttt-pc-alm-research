"""Runtime guard against invoking BP discovery/credit on the ALM candidate path.

This does not claim the regional geometry is local: it explicitly uses global
affine constraints, LP, and convex QP as disclosed in the frozen protocol.
"""
import argparse,json
from pathlib import Path
from unittest.mock import patch
import numpy as np
import streaming_branch_projection as base
import matched_discovery_baselines as bp
import posterior_confirmation_pipeline as pipeline


def main():
    p=argparse.ArgumentParser(); p.add_argument("--out",type=Path); args=p.parse_args()
    rng=np.random.default_rng(65511); x=rng.uniform(0,1,8); b=rng.uniform(-.12,.12,4)
    v=base.forward(x,b)+rng.uniform(-base.EPS,base.EPS,len(x))
    def prohibited(*args,**kwargs): raise RuntimeError("forbidden global BP credit/discovery")
    cfg={"generator":"alm","discovery_bound":.15,"restarts":64,"sweeps":120,"posterior_samples":512}
    with patch.object(base,"forward_jacobian",prohibited),patch.object(bp,"loss_gradient",prohibited),patch.object(pipeline,"bp_discover",prohibited):
        predict,point,meta=pipeline.fit(x,v,np.zeros(4),cfg)
        support=np.max(np.abs(predict(x)-v))
        blocked=False
        try: pipeline.discover(x,v,np.zeros(4),{**cfg,"generator":"trf"})
        except RuntimeError: blocked=True
        assert blocked
    result={"passed":True,"candidate_runs_with_bp_hooks_forbidden":True,"bp_control_guard_triggered":blocked,
        "positive_volume_regions":meta["positive_volume_regions"],"support_max_error":float(support),
        "boundary":"No global BP credit in candidate discovery; global affine propagation, LP/QP and volume calculation remain part of the hybrid method."}
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True); args.out.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(json.dumps(result))


if __name__=="__main__": main()
