"""Fraction audit of the cheaper, upward-only upper and layer cache."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import factorized_primal_upper_gate as gate
from verify_directional_primal_gate import exact_upper


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/directional_primal_gate/component/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    rng=np.random.default_rng(483241);checks=dict(regions=0,exact_upper_enclosures=0,empty=0,skipped=0,reverse_cache=0,zero_and_signed_credits=0)
    for d,n in [(1,2),(2,3),(4,4),(4,8),(4,24),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);bs=rng.uniform(-.12,.12,(16,d))
        regs=np.r_[np.array([gate.original.base.pattern(x,b) for b in bs],np.uint8),rng.integers(0,4,(16,d,n),dtype=np.uint8)]
        bank=rng.normal(size=(9,d,n));bank[0]=0.;bank[1]=1.;bank[2]=-1.
        solver=gate.Bank(x,v,bank);upper=solver.values(regs);before=solver.layer_solves
        assert solver.values(regs[::-1]).tobytes()==upper[::-1].tobytes() and solver.layer_solves==before;checks['reverse_cache']+=1
        for r,reg in enumerate(regs):
            checks['regions']+=1;checks['skipped']+=int(np.all(upper[r]<=0))
            for k,a in enumerate(bank):
                exact=exact_upper(x,v,reg,a)
                if exact is None:assert not np.isfinite(upper[r,k]);checks['empty']+=1;continue
                if np.isfinite(upper[r,k]):assert exact<=F(float(upper[r,k]));checks['exact_upper_enclosures']+=1
                if upper[r,k]<=0:assert exact<=0
                if k<3:checks['zero_and_signed_credits']+=1
    for path in [Path(__file__),Path(gate.__file__)]:hashes[path.name]=sha(path)
    out=root/'results/factorized_primal_gate/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    result=dict(passed=True,checks=checks,source_sha256=hashes,parent_protocol_sha256=sha(parent),scope='Strict one-sided midpoint upper and cache; zero, positive, negative coefficient cases')
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks,sources=len(hashes))),flush=True)


if __name__=='__main__':main()
