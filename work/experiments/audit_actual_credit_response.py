"""Independent direct-cost/KKT and interval-forward proof audit for279."""
import argparse,json,time
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import actual_credit_response as algebra
from run_multiplier_fixed_point_screen import sha,dump


def fold(z):
    if z<0 or z>1:return F(0)
    return 2*z if z<F(1,2) else 2-2*z


def region(z):
    if z<0:return 0
    if z<F(1,2):return 1
    if z<1:return 2
    return 3


def evaluate(q,t):return sum(q[j]*t**j for j in range(3))


def minimum_nonnegative(q,left,right):
    # Reduce a quadratic at an algebraic endpoint modulo its defining
    # polynomial; only exact root ordering is shared with the generator.
    def endpoint_sign(r):
        if r.value is not None:return algebra.sign(evaluate(q,r.value))
        aa,bb,cc=r.abc;slope=q[1]-q[2]*F(bb,aa);offset=q[0]-q[2]*F(cc,aa)
        return algebra.sign(offset) if not slope else algebra.sign(slope)*r.compare_rational(-offset/slope)
    assert endpoint_sign(left)>=0 and endpoint_sign(right)>=0
    if q[2]>0:
        vertex=-q[1]/(2*q[2])
        if left.compare_rational(vertex)<0 and right.compare_rational(vertex)>0:assert evaluate(q,vertex)>=0


def parsed_layer(data):
    out=dict(data);out['candidates']=[dict(c,**{key:tuple(map(F,c[key])) for key in ['domain','bias_interval','affine','cost']}) for c in data['candidates']]
    out['cells']=[dict(c,lo=algebra.decode_root(c['lo']),hi=algebra.decode_root(c['hi']),representative=F(c['representative'])) for c in data['cells']]
    out['pieces']=[dict(c,lo=algebra.decode_root(c['lo']),hi=algebra.decode_root(c['hi']),affine=tuple(map(F,c['affine']))) for c in data['pieces']]
    return out


def audit_layer(data,previous,target,direction,old,checks):
    layer=parsed_layer(data);n=len(previous);gamma=F(float(.01));bound=F(float(.12));knots=[F(0),F(1,2),F(1)]
    breaks=sorted({-bound,bound}|{k-p for p in previous for k in knots if -bound<k-p<bound});expected_intervals=set(zip(breaks[:-1],breaks[1:]));assert {c['bias_interval'] for c in layer['candidates']}==expected_intervals
    for interval in expected_intervals:
        cc=sorted((c for c in layer['candidates'] if c['bias_interval']==interval),key=lambda c:c['domain']);assert cc[0]['domain'][0]==0 and cc[-1]['domain'][1]==1
        for a,b in zip(cc[:-1],cc[1:]):assert a['domain'][1]==b['domain'][0]
    for c in layer['candidates']:
        lo,hi=c['domain'];left,right=c['bias_interval'];a,b=c['affine'];q=c['cost'];assert lo<hi
        slopes=[(0,2,-2,0)[region(p+(left+right)/2)] for p in previous]
        def derivative(t):
            bias=a+b*t
            return sum(s*(fold(p+bias)-h-t*u) for s,p,h,u in zip(slopes,previous,target,direction))/n+gamma*(bias-old)
        for t in [lo,(lo+hi)/2,hi]:
            bias=a+b*t;assert left<=bias<=right
            actual=sum((fold(p+bias)-h-t*u)**2 for p,h,u in zip(previous,target,direction))/n+gamma*(bias-old)**2
            assert actual==evaluate(q,t);checks['direct_objective_points']+=1
        if not b and a==left:assert derivative(lo)>=0 and derivative(hi)>=0
        elif not b and a==right:assert derivative(lo)<=0 and derivative(hi)<=0
        else:assert derivative(lo)==derivative(hi)==0
        checks['exact_interval_argmins']+=1
    previous_right=algebra.rational(0)
    for c in layer['cells']:
        lo,hi,t=c['lo'],c['hi'],c['representative'];assert algebra.compare(previous_right,lo)==0 and algebra.compare(lo,hi)<0;assert lo.compare_rational(t)<0 and hi.compare_rational(t)>0
        active=[z for z in layer['candidates'] if z['domain'][0]<t<z['domain'][1]];assert [z['index'] for z in active]==c['active']
        assert {z['bias_interval'] for z in active}==expected_intervals
        winner=layer['candidates'][c['winner']];assert winner in active
        for z in active:
            assert lo.compare_rational(z['domain'][0])>=0 and hi.compare_rational(z['domain'][1])<=0
            minimum_nonnegative(tuple(a-b for a,b in zip(z['cost'],winner['cost'])),lo,hi);checks['whole_cell_minimum_inequalities']+=1
        chosen=min(active,key=lambda z:(evaluate(z['cost'],t),z['index']));assert chosen['index']==c['winner'];previous_right=hi;checks['complete_envelope_cells']+=1
    assert algebra.compare(previous_right,algebra.rational(1))==0
    # Coalesced pieces must cover every certified cell with exactly its bias.
    for cell in layer['cells']:
        owners=[p for p in layer['pieces'] if algebra.compare(p['lo'],cell['lo'])<=0 and algebra.compare(p['hi'],cell['hi'])>=0]
        assert len(owners)==1 and owners[0]['affine']==layer['candidates'][cell['winner']]['affine']
    return layer


def audit_path(segments,x,layers,chord,checks):
    previous_right=algebra.rational(0);found=set()
    for record in segments:
        lo,hi=map(algebra.decode_root,[record['lo'],record['hi']]);t=F(record['representative']);bias=[tuple(map(F,b)) for b in record['bias']]
        assert algebra.compare(previous_right,lo)==0 and algebra.compare(lo,hi)<0 and lo.compare_rational(t)<0 and hi.compare_rational(t)>0
        if chord is not None:assert bias==chord
        else:
            for j,layer in enumerate(layers):
                owners=[p for p in layer['pieces'] if algebra.compare(p['lo'],lo)<=0 and algebra.compare(p['hi'],hi)>=0 and p['affine']==bias[j]];assert owners
        outputs=[(xx,F(0)) for xx in x];codes=[]
        for a,b in bias:
            after=[]
            for u,v in outputs:
                p,q=u+a,v+b;k=region(p+q*t);codes.append(k)
                if k>0:minimum_nonnegative((p-[F(0),F(1,2),F(1)][k-1],q,F(0)),lo,hi)
                if k<3:minimum_nonnegative(([F(0),F(1,2),F(1)][k]-p,-q,F(0)),lo,hi)
                slope=(0,2,-2,0)[k];offset=(0,0,2,0)[k];after.append((slope*p+offset,slope*q));checks['whole_forward_branch_intervals']+=1
            outputs=after
        key=bytes(codes).hex();assert key==record['mode'];found.add(key);previous_right=hi;checks['complete_forward_segments']+=1
    assert algebra.compare(previous_right,algebra.rational(1))==0
    return found


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/actual_credit_response';inp=base/'primitive';out=base/'primitive_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    ss=json.loads((inp/'summary.json').read_text());assert ss['passed']
    for n,h in ss['outputs_sha256'].items():assert sha(inp/n)==h
    p=json.loads((inp/'protocol.json').read_text());hashes=dict(p['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    dump(out/'protocol.json',dict(source_sha256=hashes,primitive_summary_sha256=sha(inp/'summary.json'),phase_accesses_query_targets=False,
        scope='Direct physical objective at3 rational points plus exact KKT, complete-cell global-minimum inequalities, full-path interval branch inequalities; not dense-grid proof'))
    rows=json.loads((inp/'rows.json').read_text());files=json.loads((inp/'files.json').read_text());checks=Counter();groups=Counter();begin=time.perf_counter()
    reference=root/'results/confirmation_conditional_risk/reference';refs={r['seed']:r for r in json.loads((reference/'coverage.json').read_text())};positive={}
    for seed in p['seeds']:
        rr=refs[seed];assert sha(reference/rr['file'])==rr['sha256'];positive[seed]={r['pattern'] for r in json.loads((reference/rr['file']).read_text())['reference']['positive_regions']}
    for row in rows:
        assert row['status']=='complete';assert sha(inp/row['file'])==row['sha256']==files[row['file']];data=json.loads((inp/row['file']).read_text())
        x=list(map(F,map(str,data['x'])));old=list(map(F,map(str,data['old_bias'])))
        # JSON decimals came from Python floats: recover the same binary
        # values used by the mathematical generator, not decimal rationals.
        x=list(map(lambda v:F(float(v)),data['x']));old=list(map(lambda v:F(float(v)),data['old_bias']))
        h=[[F(float(v)) for v in layer] for layer in data['activity']];u=[[F(float(v)) for v in layer] for layer in data['dual']]
        layers=[audit_layer(data['layers'][j],x if j==0 else h[j-1],h[j],u[j],old[j],checks) for j in range(4)]
        ends=[list(map(F,b)) for b in data['mathematical_endpoints']]
        for t,b in zip([F(0),F(1)],ends):
            for j,layer in enumerate(layers):
                active=[c for c in layer['candidates'] if c['domain'][0]<=t<=c['domain'][1]];best=min(active,key=lambda c:(evaluate(c['cost'],t),c['index']));assert b[j]==best['affine'][0]+t*best['affine'][1];checks['exact_endpoints']+=1
        chord=[(a,b-a) for a,b in zip(*ends)];rp=audit_path(data['response_segments'],x,layers,None,checks);cp=audit_path(data['chord_segments'],x,layers,chord,checks)
        endpoint_modes=set(data['endpoint_modes']);assert rp|endpoint_modes==set(data['response_modes']) and cp|endpoint_modes==set(data['chord_modes'])
        rp=(rp|endpoint_modes)&positive[row['seed']];cp=(cp|endpoint_modes)&positive[row['seed']];ep=endpoint_modes&positive[row['seed']]
        for field,value in [('endpoint_positive',ep),('response_positive',rp),('chord_positive',cp),('response_only_positive',rp-cp),('chord_only_positive',cp-rp)]:assert sorted(value)==row[field]
        for field,value in [('response_only_states',rp-cp),('chord_only_states',cp-rp),('response_adds_to_endpoints',rp-ep),('chord_adds_to_endpoints',cp-ep)]:groups[field]+=bool(value)
        checks['states']+=1
        if checks['states']%33==0:print(json.dumps(dict(states=checks['states'],total=132,seconds=time.perf_counter()-begin)),flush=True)
    assert checks['states']==132
    ans=dict(passed=True,checks=checks,groups=groups,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),phase_accesses_query_targets=False,
        scope='Open-segment completeness only, isolated breakpoints explicitly excluded; counts across states not distinct task-level modes; no online resource or query claim')
    dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)


if __name__=='__main__':main()
