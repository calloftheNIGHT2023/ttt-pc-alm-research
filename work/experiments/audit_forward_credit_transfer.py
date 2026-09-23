"""Independent scalar rational branch and unrolled residual-bound audit for277."""
import argparse,json,time
from pathlib import Path
from fractions import Fraction as F
from collections import Counter
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump


def fold(z):
    if z<F(0) or z>F(1):return F(0)
    return 2*z if z<F(1,2) else 2-2*z


def code(z):
    if z<0:return 0
    if z<F(1,2):return 1
    if z<1:return 2
    return 3


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/forward_credit_transfer';inp=base/'diagnosis';out=base/'audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed']
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,diagnosis_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,scope='Scalar piecewise rational teacher, unrolled weighted residual sums, strict interval endpoint implications and all counts'))
    rows=json.loads((inp/'rows.json').read_text());files=json.loads((inp/'files.json').read_text());reference=root/'results/confirmation_conditional_risk/reference';refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())}
    counts=Counter();groups=Counter();cases=Counter();checks=Counter();begin=time.perf_counter()
    for seed in p['seeds']:
        filename=f'{seed}_atomic_states.npz';assert sha(inp/filename)==files[filename]
        ref=refs[seed];assert sha(reference/ref['file'])==ref['sha256'];positive={r['pattern'] for r in json.loads((reference/ref['file']).read_text())['reference']['positive_regions']}
        with np.load(inp/filename) as z:
            x=list(map(lambda t:F(float(t)),z['x']));v=list(map(lambda t:F(float(t)),z['v']));n=len(x)
            for row in [r for r in rows if r['seed']==seed]:
                r=row['restart'];event=row['event'];outcomes={}
                for action in ['D','A']:
                    result=row[action];b=[F(float(t)) for t in z[action+'_b'][r]];h=[[F(float(t)) for t in layer] for layer in z[action+'_h'][:,r]]
                    residual=[[F(0)]*n for _ in range(4)];forward=[[F(0)]*n for _ in range(4)];preact=[[F(0)]*n for _ in range(4)];bounds=[[F(0)]*n for _ in range(4)]
                    for i,xx in enumerate(x):
                        y=xx
                        for layer in range(4):
                            preact[layer][i]=y+b[layer];y=fold(preact[layer][i]);forward[layer][i]=y
                            prev=xx if layer==0 else h[layer-1][i];residual[layer][i]=h[layer][i]-fold(prev+b[layer])
                            e=sum((2**(layer-j))*abs(residual[j][i]) for j in range(layer+1));bounds[layer][i]=e
                            error=abs(y-h[layer][i]);assert error<=e;assert F(result['residuals'][layer][i])==residual[layer][i]
                            assert F(result['actual_errors'][layer][i])==error and F(result['error_bounds'][layer][i])==e;checks['exact_scalar_bounds']+=1
                    key=bytes(code(t) for layer in preact for t in layer).hex();assert key==result['exact_mode']
                    support_error=max(abs(a-c) for a,c in zip(forward[3],v));assert support_error==F(result['atomic_support_max_error'])
                    assert result['positive_mode']==(key in positive) and result['float_positive_mode']==(result['float_mode'] in positive)
                    assert result['atomic_support_feasible']==(support_error<=F(float(.001))) and result['exact_float_mode_equal']==(key==result['float_mode'])
                    for field in ['positive_mode','float_positive_mode','atomic_support_feasible','exact_float_mode_equal']:counts[action+'_'+field]+=int(result[field])
                    if event is not None:
                        j,i,k=event['j'],event['i'],event['k'];center=h[j][i]+b[j+1];e=bounds[j][i];actual=preact[j+1][i];rr=result['branch_event'];lo=center-e;hi=center+e
                        certified=[hi<0,lo>0 and hi<F(1,2),lo>F(1,2) and hi<1,lo>1][k]
                        assert rr['center']==str(center) and rr['radius']==str(e) and rr['interval']==[str(lo),str(hi)] and rr['true_preactivation']==str(actual)
                        assert rr['true_branch']==code(actual) and rr['split_branch']==code(center)
                        hit=code(actual)==k;split=code(center)==k;assert rr['actual_target_hit']==hit and rr['split_target_hit']==split and rr['strict_certificate']==certified
                        assert not certified or hit;checks['exact_event_certificates']+=1
                        for field in ['split_target_hit','actual_target_hit','strict_certificate']:counts[action+'_'+field]+=int(rr[field])
                        category=('success_certified' if certified else 'success_bound_loose') if hit else ('lost_after_bias_write' if not split else 'lost_from_activity_residual')
                        cases[action+'_'+category]+=1
                    else:assert result['branch_event'] is None
                    outcomes[action]=result;counts['exact_forward_states']+=1;counts['layer_observation_bounds']+=4*n
                if event is None:counts['states_without_event']+=1
                else:
                    counts['states_with_event']+=1
                    for field in ['actual_target_hit','strict_certificate']:
                        left=outcomes['D']['branch_event'][field];right=outcomes['A']['branch_event'][field];groups[field+'_'+('both' if left and right else 'only_D' if left else 'only_A' if right else 'neither')]+=1
                for field in ['positive_mode','atomic_support_feasible']:
                    left=outcomes['D'][field];right=outcomes['A'][field];groups[field+'_'+('both' if left and right else 'only_D' if left else 'only_A' if right else 'neither')]+=1
    assert counts==Counter({k:v for k,v in ss['counts'].items() if k!='exact_atomic_arrays'})
    assert groups==Counter(ss['groups']) and counts['exact_forward_states']==4224
    ans=dict(passed=True,counts=counts,groups=groups,transfer_categories=cases,checks=checks,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False,
        scope='Support-side mechanism diagnosis only; positive_mode means a feasible parameter exists in that branch, not that the current atom satisfies observations')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
