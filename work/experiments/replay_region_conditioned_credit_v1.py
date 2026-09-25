"""Replay solely from packaged support inputs, without parent trajectories.

This verifies component outputs, not original candidate-pool generation or
end-to-end query predictions. No LP geometry labels are read.
"""
import argparse
from contextlib import ExitStack
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np
import scipy.optimize as opt
import region_conditioned_credit_v1 as model
import branch_image_chain_v1 as independent
import local_region_screen as controls
from certified_branch_solver import exact_certificate


def read(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def forbid(*args,**kwargs):raise AssertionError('Global solver/BP forbidden during component replay')


def replay(root):
    started=time.perf_counter();folder=root/'results/region_conditioned_credit/development_v1'
    summary=read(folder/'summary.json');assert summary['passed']
    for n,h in summary['outputs_sha256'].items():assert sha(folder/n)==h
    channels=['dual','dual_plus_residual','residual','bp','random_sign','zero']
    counts=dict(tasks=0,regions=0,arrays=0,exact_proofs=0,pdhg_proofs=0,control_masks=0)
    targets=[opt.linprog,opt.minimize,controls.base.forward_jacobian]
    with ExitStack() as guard:
        for module in list(sys.modules.values()):
            if module is None:continue
            for name,value in list(vars(module).items()):
                if any(value is target for target in targets):guard.enter_context(patch.object(module,name,forbid))
        guard.enter_context(patch.object(controls.base,'BOUND',.12))
        for task in read(folder/'tasks.json'):
            directory=folder/str(task['seed'])
            for n,h in task['files'].items():assert sha(directory/n)==h
            inp=read(directory/'input.json');x=np.array(inp['x_observed']);v=np.array(inp['v_observed'])
            regs=np.array([list(bytes.fromhex(m)) for m in inp['modes']],np.uint8).reshape(-1,4,4)
            for c in channels:
                credit=np.array(inp['credits'][c]);arrays,meta=model.solve(x,v,regs,credit)
                saved=read(directory/(c+'.json'));assert meta['proofs']==saved['proofs']
                with np.load(directory/(c+'.npz'),allow_pickle=False) as z:
                    for key,arr in arrays.items():assert arr.tobytes()==z[key].tobytes();counts['arrays']+=1
                for proof in meta['proofs']:
                    i=proof['index'];value=independent.fixed_value(x,v,arrays['proof_credit'][i],regs[i])
                    assert (proof['structural_empty'] and value is None) or value==Fraction(proof['lower'])>0
                    counts['exact_proofs']+=1
            with np.load(directory/'controls.npz',allow_pickle=False) as z:
                for rounds in [5,20]:
                    assert np.array_equal(controls.contract(x,v,regs,rounds),z['c'+str(rounds)])
                    counts['control_masks']+=1
                b=np.broadcast_to(np.array(inp['trigger_b']),(len(regs),4)).copy()
                mask,note=controls.pdhg(x,v,b,regs,60,20)
                assert np.array_equal(mask,z['pdhg60']);counts['control_masks']+=1
            for index,proof in note['certificates'].items():
                check=exact_certificate(x,v,regs[index],proof['p'],proof['a']);assert check['positive']
                counts['pdhg_proofs']+=1
            counts['tasks']+=1;counts['regions']+=len(regs)
            if counts['tasks']%8==0:print(dict(tasks=counts['tasks'],seconds=time.perf_counter()-started),flush=True)
    return dict(passed=True,counts=counts,seconds=time.perf_counter()-started,
                query_targets_accessed=False,parent_trajectories_accessed=False,geometry_accessed=False,
                component_reproduction=True,original_pool_generation_reproduced=False,
                source_sha256=sha(Path(__file__)),input_summary_sha256=sha(folder/'summary.json'))


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2]
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);args=ap.parse_args()
    assert not args.out.exists();result=replay(root);args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(result,flush=True)
