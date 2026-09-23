"""Old-seed preflight for common-pool assembly and all six credit families."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import common_pool_credit as model
import diagnose_h2_full_bank_ceiling as dense
import light_h2_credit as light
import optimized_branch_dual as cert


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/credit_moment_step/development/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_228_audit.json';assert json.loads(audit.read_text())['passed']
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    items={}
    for method in model.METHODS:
        learner,mode=method.rsplit('_',1);cfg=light.config(learner,mode)
        _,_,regs,bank,_,_=dense.capture(x,v,cfg);items[method]=dict(x=x,v=v,regs=regs,bank=bank)
    xx,vv,regs,membership=model.assemble(items)
    assert xx.tobytes()==x.tobytes() and vv.tobytes()==v.tobytes()
    reverse={k:items[k] for k in reversed(model.METHODS)}
    for a,b in zip(model.assemble(reverse),model.assemble(items)):assert a.tobytes()==b.tobytes()
    checks=dict(common_regions=len(regs),original_regions=sum(len(q['regs']) for q in items.values()),methods=0,replayed_arrays=0,old_exact=0,new_exact=0)
    for method in model.METHODS:
        bank=items[method]['bank'];arrays,meta=model.solve(x,v,regs,bank);again,repeated=model.solve(x,v,regs,bank)
        for key,value in arrays.items():assert value.tobytes()==again[key].tobytes();checks['replayed_arrays']+=1
        assert meta['old_proofs']==repeated['old_proofs'] and meta['proofs']==repeated['proofs']
        lookup={r.tobytes().hex():i for i,r in enumerate(regs)}
        for proof in meta['old_proofs']:
            exact=cert.exact_optimum(x,v,regs[lookup[proof['pattern']]],bank[proof['direction']]);assert exact['positive'];checks['old_exact']+=1
        for proof in meta['proofs']:
            index=int(arrays['indices'][proof['index']]);exact=cert.exact_optimum(x,v,regs[index],arrays['credit'][proof['index']])
            assert exact==proof['exact'] and exact['positive'];checks['new_exact']+=1
        checks['methods']+=1
    extra=['audit_credit_moment_step.py','benchmark_credit_solvers.py','audit_credit_moment_resources.py','plot_credit_moment_step.py',
        'audit_research_round_228.py','common_pool_credit.py',Path(__file__).name]
    for name in extra:hashes[name]=sha(Path(__file__).with_name(name))
    result=dict(passed=True,checks=checks,source_sha256=hashes,parent_protocol_sha256=sha(parent),parent_audit_sha256=sha(audit),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/229_common_pool_credit_protocol.md'),
        scope='Old 5900001 preflight, no query; each new solve inherits LP/BP runtime guard; discovery unchanged')
    out=root/'results/common_pool_credit/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
