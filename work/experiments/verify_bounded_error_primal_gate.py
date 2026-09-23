"""Exact audit of bounded-error and fallback upper construction."""
import argparse
import hashlib
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import bounded_error_primal_gate as gate
from verify_directional_primal_gate import exact_upper


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/factorized_primal_gate/component/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    rng=np.random.default_rng(483271);checks=dict(regions=0,enclosures=0,empty=0,cache_replays=0,fallback_banks=0,fast_banks=0)
    for d,n in [(1,2),(2,3),(4,4),(4,8),(4,24),(6,5)]:
        x=rng.uniform(0,1,n);v=rng.uniform(0,1,n);bs=rng.uniform(-.12,.12,(16,d))
        regs=np.r_[np.array([gate.original.base.pattern(x,b) for b in bs],np.uint8),rng.integers(0,4,(16,d,n),dtype=np.uint8)]
        raw=rng.normal(size=(9,d,n));raw/=np.max(abs(raw),axis=(1,2),keepdims=True)
        raw[0]=0;raw[1]=1;raw[2]=-1;raw[3]=np.ldexp(raw[3],-1020)
        for bank in [raw,2*raw]:
            solver=gate.Bank(x,v,bank);upper=solver.values(regs);solves=solver.layer_solves
            assert solver.values(regs[::-1]).tobytes()==upper[::-1].tobytes() and solver.layer_solves==solves;checks['cache_replays']+=1
            checks['fast_banks' if solver.fast else 'fallback_banks']+=1
            for r,reg in enumerate(regs):
                checks['regions']+=1
                for k,a in enumerate(bank):
                    value=exact_upper(x,v,reg,a)
                    if value is None:assert not np.isfinite(upper[r,k]);checks['empty']+=1;continue
                    if np.isfinite(upper[r,k]):assert value<=F(float(upper[r,k]));checks['enclosures']+=1
    for path in [Path(__file__),Path(gate.__file__)]:hashes[path.name]=sha(path)
    result=dict(passed=True,checks=checks,source_sha256=hashes,parent_protocol_sha256=sha(parent),
        scope='Exact generic enclosure tests, including unsupported credit norm fallback and near-subnormal credits')
    out=root/'results/bounded_error_primal_gate/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(dict(passed=True,checks=checks,sources=len(hashes))),flush=True)


if __name__=='__main__':main()
