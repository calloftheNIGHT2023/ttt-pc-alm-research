import json
from pathlib import Path
import numpy as np
import streaming_branch_projection as base
from joint_activity_bias_memory import joint_block
from analytic_joint_blocks import analytic_block,coordinate_block,energy

c=.15; A=1.01; beta=.01; kap=A*beta/(A+beta)
T=lambda k:c/2*(k+np.sqrt(k*(k+4)))
t=(T(kap)+T(beta))/2
a=np.zeros((1,1)); tar=np.array([[t]]); h0=np.zeros((1,1)); b0=np.array([-c])
hc,bc,_=coordinate_block(a,tar,h0,b0,passes=100)
hg,bg,_=joint_block(a,tar,h0,b0)
hj,bj,_=analytic_block(a,tar,h0,b0)
z=(2*t-kap*c)/(4+kap); ht=beta/(A+beta)*(z+c); bt=z-ht
assert np.max(np.abs(hc-h0))<1e-12 and np.max(np.abs(bc-b0))<1e-12
assert 0<ht<1 and abs(bt)<base.BOUND and 0<z<.5
assert np.max(np.abs(hj-ht))<1e-10 and np.max(np.abs(bj-bt))<1e-10
assert energy(a,tar,h0,b0,hj,bj)[0]<t*t
toy={"target":t,"threshold_joint":T(kap),"threshold_coordinate_bias":T(beta),
     "old_energy":t*t,"coordinate100_energy":float(energy(a,tar,h0,b0,hc,bc)[0]),
     "grid_energy":float(energy(a,tar,h0,b0,hg,bg)[0]),"analytic_energy":float(energy(a,tar,h0,b0,hj,bj)[0]),
     "analytic_h":float(hj[0,0]),"analytic_b":float(bj[0])}
rng=np.random.default_rng(562914); gaps=[]
for _ in range(50):
    a=rng.normal(.5,1,(3,7)); tar=rng.normal(.5,1,(3,7)); h0=rng.uniform(0,1,(3,7)); b0=rng.uniform(-.15,.15,3)
    hg,bg,_=joint_block(a,tar,h0,b0); hj,bj,_=analytic_block(a,tar,h0,b0)
    gap=energy(a,tar,h0,b0,hj,bj)-energy(a,tar,h0,b0,hg,bg)
    assert np.max(gap)<1e-10; gaps.extend(gap.tolist())
result={"passed":True,"coordinate_fixed_point_example":toy,"random_rows":len(gaps),"max_analytic_minus_grid":max(gaps),"strict_improvements":sum(v<-1e-10 for v in gaps)}
out=Path("outputs/ttt-pc-alm-research/results/analytic_joint_memory/block_audit.json")
out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,indent=2),encoding="utf-8")
print(json.dumps(result))
