"""Bitwise trajectory and exact expansion preflight on old seed 5900001."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import credit_history_capture as history
import credit_history_expansion as expansion
import common_pool_credit as common


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve()
    parent=root/'results/common_pool_credit/development/protocol.json';hashes=dict(json.loads(parent.read_text())['source_sha256'])
    for name,value in hashes.items():assert sha(Path(__file__).with_name(name))==value,name
    audit=root/'results/round_230_audit.json';assert json.loads(audit.read_text())['passed']
    rng=np.random.default_rng(5900001);teacher=rng.uniform(-.12,.12,4);xx=rng.uniform(0,1,24)
    vv=history.light.base.forward(xx,teacher)+np.random.default_rng(24900001).uniform(-.001,.001,24);x=xx[:4];v=vv[:4]
    reference=history.capture(x,v,False);actual=history.capture(x,v,True)
    history.equal_tree(reference[0],actual[0])
    for k in [2,3,4,5]:assert reference[k].tobytes()==actual[k].tobytes()
    for k in ['credit_bank','credit_proofs','positive_mode_keys','discovery_bank_sha256','completion_proposals']:assert reference[1][k]==actual[1][k]
    for k in ['trajectory_sha256','best_sha256','events']:assert reference[7][k]==actual[7][k]
    rawhistory=actual[6];banks,metadata=history.build(rawhistory);assert banks['native'].tobytes()==actual[3].tobytes()
    raw=expansion.rational_history(rawhistory);dual=expansion.shadow_error(rawhistory,raw)
    checks=dict(trajectory_events=actual[7]['events'],shadow_steps=actual[7]['shadow_checks'],methods=0,replayed_arrays=0,expanded=0,expanded_positive=0,expanded_lower_positive=0)
    for name in ['native',*history.VARIANTS]:
        arrays,meta=common.solve(x,v,actual[2],banks[name]);again,mm=common.solve(x,v,actual[2],banks[name])
        for key,value in arrays.items():assert value.tobytes()==again[key].tobytes();checks['replayed_arrays']+=1
        assert meta['old_proofs']==mm['old_proofs'] and meta['proofs']==mm['proofs'];checks['methods']+=1
        if name=='native':
            lookup={r.tobytes().hex():i for i,r in enumerate(actual[2])}
            for proof in meta['old_proofs']:
                weights=np.zeros(len(banks[name]));weights[proof['direction']]=1
                q=expansion.expand(x,v,actual[2][lookup[proof['pattern']]],banks[name][proof['direction']],weights,rawhistory,banks,raw)
                checks['expanded']+=1;checks['expanded_positive']+=q['positive'];checks['expanded_lower_positive']+=q['lower_positive']
            for proof in meta['proofs']:
                index=proof['index'];reg=actual[2][arrays['indices'][index]]
                q=expansion.expand(x,v,reg,arrays['credit'][index],arrays['weights'][index],rawhistory,banks,raw)
                checks['expanded']+=1;checks['expanded_positive']+=q['positive'];checks['expanded_lower_positive']+=q['lower_positive']
    assert checks['expanded']==checks['expanded_positive']==checks['expanded_lower_positive']
    for name in ['audit_common_pool_credit.py','audit_common_pool_resources.py','plot_common_pool_credit.py','audit_research_round_230.py',
                 'credit_history_capture.py','credit_history_expansion.py',Path(__file__).name]:hashes[name]=sha(Path(__file__).with_name(name))
    result=dict(passed=True,checks=checks,dual_rounding=dual,metadata=metadata,source_sha256=hashes,parent_protocol_sha256=sha(parent),
        parent_audit_sha256=sha(audit),design_sha256=sha(root/'outputs/ttt-pc-alm-research/231_credit_history_attribution_protocol.md'))
    out=root/'results/credit_history/primitive';out.mkdir(parents=True,exist_ok=True);assert not (out/'summary.json').exists()
    (out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
