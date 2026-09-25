import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
from joint_activity_bias_memory import profile,joint_block

rng=np.random.default_rng(562911); maxgap=-np.inf; maxold=-np.inf; maxsep=-np.inf
for _ in range(40):
    a=rng.normal(.5,1,(3,5)); target=rng.normal(.5,1,(3,5)); oldh=rng.uniform(0,1,(3,5)); oldb=rng.uniform(-.15,.15,3)
    hh,bb,audit=joint_block(a,target,oldh,oldb)
    maxold=max(maxold,audit["max_increase_over_old"]); maxsep=max(maxsep,audit["max_increase_over_separate"])
    exact,_=profile(a,target,oldh,oldb,oldb[:,None]); h=exact[:,0,:]
    grid=np.linspace(0,1,50001)
    for r in range(3):
        for i in range(5):
            energy=lambda z:(z-a[r,i])**2+(base.g(z+oldb[r])-target[r,i])**2+.01*(z-oldh[r,i])**2
            gap=float(energy(h[r,i])-energy(grid).min()); maxgap=max(maxgap,gap)
            assert gap<=1e-10
result={"passed":True,"joint_cases":40,"dense_scalar_checks":600,"max_gap_to_grid":maxgap,"max_increase_over_old":maxold,"max_increase_over_separate":maxsep}
out=Path("outputs/ttt-pc-alm-research/results/joint_activity_bias_memory/block_audit.json")
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
