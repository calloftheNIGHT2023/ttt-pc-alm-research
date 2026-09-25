"""Validate the predeclared finite-batch cost law on independent trajectories.

Geometry volumes are evaluator-only here, after all outputs are committed.
They were not available to, or used to tune, hybrid deployment decisions.
"""
import argparse
import hashlib
import itertools
import json
from fractions import Fraction as F
from pathlib import Path
import numpy as np
from scipy.stats import binom
import independent_hybrid_memory as memory


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_cost(p,count,maximum,batch):
    expected=F(0);fallback=F(0)
    for bits in itertools.product([0,1],repeat=maximum):
        probability=p**sum(bits)*(1-p)**(maximum-sum(bits));attempts=0;success=0
        while attempts<maximum and success<count:
            n=min(batch,maximum-attempts);success+=sum(bits[attempts:attempts+n]);attempts+=n
        expected+=probability*attempts;fallback+=probability*int(success<count)
    return expected,fallback


def finite_binomial_cdf(k,n,p):
    import math
    return sum((F(math.comb(n,i))*p**i*(1-p)**(n-i) for i in range(min(k,n)+1)),F(0))


def verify():
    checks=0;monotone=0
    for count in [1,2,3]:
        for maximum in [0,1,4,7]:
            for batch in [1,2,3]:
                previous=None
                for p in [F(1,8),F(1,3),F(2,3),F(7,8)]:
                    expected,fallback=exact_cost(p,count,maximum,batch)
                    law=sum((min(batch,maximum-m)*finite_binomial_cdf(count-1,m,p) for m in range(0,maximum,batch)),F(0))
                    tail=finite_binomial_cdf(count-1,maximum,p)
                    assert (expected,fallback)==(law,tail);checks+=1
                    if previous is not None:assert law<=previous[0] and tail<=previous[1];monotone+=1
                    previous=(law,tail)
    return dict(passed=True,exact_enumeration_cost_and_tail_checks=checks,monotonicity_checks=monotone)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);args=ap.parse_args();root=args.project.resolve()
    inp=root/'results/independent_hybrid/development';out=root/'results/independent_hybrid/cost_model'
    p=json.loads((inp/'protocol.json').read_text());run=json.loads((inp/'run_audit.json').read_text());assert run['execution_complete']
    for name,value in p['source_sha256'].items():assert sha(Path(__file__).with_name(name))==value,name
    rows=json.loads((inp/'rows.json').read_text());count=p['posterior_samples'];maximum=128*count;batch=32768;records=[]
    for seed in p['seeds']:
        for family in ['alm','adam60','pc','nodual']:
            gpath=inp/f'detail_{seed}_{family}_geometry.json';hpath=inp/f'detail_{seed}_{family}_hybrid.json'
            geo=json.loads(gpath.read_text());hybrid=json.loads(hpath.read_text())
            with np.load(inp/f'state_{seed}_{family}_hybrid_0_4.npz') as z:x=z['x'];v=z['v']
            positive=set(geo['positive_mode_keys']);assert positive<=set(hybrid['surviving_pattern_keys'])
            valid_volume=sum(n['volume'] for n in geo['geometry_trace'] if n['reason']=='positive_volume')
            vols={}
            for key in geo['surviving_pattern_keys']:
                reg=np.frombuffer(bytes.fromhex(key),np.uint8).reshape(4,len(x));a,c,_,_=memory.neighbor.pattern_matrix(x,v,reg)
                vols[key]=memory.proposal.make_box(a,c,v)['volume']
            before=sum(vols.values());after=sum(vols[k] for k in hybrid['surviving_pattern_keys'])
            assert after<=before and valid_volume>0 and valid_volume<=after*(1+1e-10)
            predictions=[]
            for volume in [before,after]:
                success=valid_volume/volume;assert 0<success<=1
                attempts=sum(min(batch,maximum-m)*binom.cdf(count-1,m,success) for m in range(0,maximum,batch))
                fallback=binom.cdf(count-1,maximum,success)
                predictions.append(dict(success_probability=float(success),expected_physical_proposals=float(attempts),fallback_probability=float(fallback)))
            assert predictions[1]['expected_physical_proposals']<=predictions[0]['expected_physical_proposals']+1e-7
            assert predictions[1]['fallback_probability']<=predictions[0]['fallback_probability']+1e-14
            rr=[r for r in rows if r['seed']==seed and r['method']==family+'_hybrid' and r['n_context']==4]
            # A later failure caused the frozen runner not to append that
            # entire trial's rows, but its first-write detail is committed.
            # Use exactly that rep-0 record; never invent the lost timing.
            if not rr:rr=[dict(sampling=hybrid['sampling'],post_credit_seconds=hybrid['post_credit_seconds'])]
            records.append(dict(seed=seed,family=family,geometry_sha256=sha(gpath),hybrid_sha256=sha(hpath),
                valid_volume=valid_volume,proposal_volume_before=before,proposal_volume_after=after,
                proposal_volume_removed_fraction=1-after/before,without_credit=predictions[0],with_credit=predictions[1],
                actual_repetition_records=len(rr),actual_mean_physical_proposals=float(np.mean([r['sampling']['physical_proposals'] for r in rr])),
                actual_fallback_fraction=float(np.mean([r['sampling']['fallback'] for r in rr])),
                post_credit_seconds=float(np.mean([r['post_credit_seconds'] for r in rr]))))
    summaries=[]
    for family in ['alm','adam60','pc','nodual']:
        rr=[r for r in records if r['family']==family]
        summaries.append(dict(family=family,tasks=len(rr),
            mean_removed_proposal_fraction=float(np.mean([r['proposal_volume_removed_fraction'] for r in rr])),
            predicted_mean_proposals_before=float(np.mean([r['without_credit']['expected_physical_proposals'] for r in rr])),
            predicted_mean_proposals_after=float(np.mean([r['with_credit']['expected_physical_proposals'] for r in rr])),
            predicted_mean_fallback_before=float(np.mean([r['without_credit']['fallback_probability'] for r in rr])),
            predicted_mean_fallback_after=float(np.mean([r['with_credit']['fallback_probability'] for r in rr])),
            actual_mean_fallback=float(np.mean([r['actual_fallback_fraction'] for r in rr])),
            mean_post_credit_seconds=float(np.mean([r['post_credit_seconds'] for r in rr]))))
    result=dict(passed=True,primitive=verify(),records=len(records),summaries=summaries,
        source_sha256=sha(Path(__file__)),protocol_sha256=sha(inp/'protocol.json'),
        scope='conditional finite-batch prediction from numerical geometry; observed binomial outcomes not forced to expectation; this is not a proof of floating-point geometry completeness or net wall-time superiority')
    out.mkdir(parents=True,exist_ok=True)
    for name,data in [('summary.json',result),('records.json',records)]:
        (out/name).write_text(json.dumps(data,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
