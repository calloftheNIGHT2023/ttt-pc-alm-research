"""Independent first-crossing audit: nested three-point interpolation.

Never imports the candidate threshold, coefficients, or algebraic comparator.
Root ordering uses rational interval refinement and rational polynomial ties.
"""
import argparse
from collections import Counter
from fractions import Fraction as F
from functools import cmp_to_key
import json
from math import isqrt
from pathlib import Path
import time
import numpy as np
from audit_local_dual_jump import minima,energy,branch,g,unpack
from run_local_dual_jump_v2 import dump
from run_multiplier_fixed_point_screen import sha

def val(p,t):return p[0]*t*t+p[1]*t+p[2]
def interpolate(lo,hi,ys):
    f0,fm,f1=ys;span=hi-lo;aa=2*(f1+f0-2*fm);bb=f1-f0-aa
    return [aa/span**2,bb/span-2*aa*lo/span**2,f0-bb*lo/span+aa*lo*lo/span**2]
def peak(p,lo,hi):
    points=[lo,hi]
    if p[0]<0 and lo<-p[1]/(2*p[0])<hi:points.append(-p[1]/(2*p[0]))
    return max([(val(p,t),t) for t in points],key=lambda z:(z[0],-z[1]))
def bisect(p,lo,hi,n):
    assert val(p,lo)<=0<val(p,hi)
    for _ in range(n):
        mid=(lo+hi)/2
        if val(p,mid)>0:hi=mid
        else:lo=mid
    return lo,hi
def rational_root(p):
    a,b,c=p
    if not a:return -c/b
    d=b*b-4*a*c;n,m=isqrt(d.numerator),isqrt(d.denominator)
    if n*n==d.numerator and m*m==d.denominator:return (-b+F(n,m))/(2*a)
    return None
def compare(a,b):
    pa,pb=a['poly'],b['poly'];ra,rb=rational_root(pa),rational_root(pb)
    if ra is not None and rb is not None:return (ra>rb)-(ra<rb)
    if pa[0] and pb[0] and pa[0]*pb[0]>0 and [z/pa[0] for z in pa]==[z/pb[0] for z in pb]:return 0
    la,ha=(ra,ra) if ra is not None else (a['lower'],a['upper']);lb,hb=(rb,rb) if rb is not None else (b['lower'],b['upper'])
    for _ in range(1025):
        if ha<=lb:return -1
        if hb<=la:return 1
        if ra is None:la,ha=bisect(pa,la,ha,1)
        if rb is None:lb,hb=bisect(pb,lb,hb,1)
    raise AssertionError('Independent root ordering remains unresolved.')

def independent_cuts(s,d,cap):
    prev,bias,nb,z0,y=s;ans={F(0),cap};knots=[F(0),F(1,2),F(1)];slopes=[0,2,-2,0];offsets=[0,0,2,0]
    for k in range(4):
        lo=F(0) if k==0 else max(F(0),knots[k-1]-nb);hi=F(1) if k==3 else min(F(1),knots[k]-nb)
        if lo>hi:continue
        # Extend this branch's affine relation only to interpolate its vertex.
        def objective(z,t):return (z-g(prev+bias)+t*d[0])**2+(y-slopes[k]*(z+nb)-offsets[k]+t*d[1])**2+F(.01)*(z-z0)**2
        vertices=[]
        zl,zh=(lo,hi) if lo<hi else (F(0),F(1))
        for t in [F(0),cap]:
            p=interpolate(zl,zh,[objective(z,t) for z in [zl,(zl+zh)/2,zh]]);assert p[0]>0;vertices.append(-p[1]/(2*p[0]))
        velocity=(vertices[1]-vertices[0])/cap
        if velocity:
            for endpoint in [lo,hi]:
                t=(endpoint-vertices[0])/velocity
                if 0<t<cap:ans.add(t)
    return sorted(ans)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    parent=root/'results/minimum_sufficient_dual/threshold';out=root/'results/minimum_sufficient_dual/threshold_audit';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    p=json.loads((parent/'protocol.json').read_text());summary=json.loads((parent/'summary.json').read_text());hashes=dict(p['source_sha256'])
    for n,h in hashes.items():assert sha(src/n)==h,n
    for n in ['protocol','files','rows']:assert sha(parent/f'{n}.json')==summary[f'{n}_sha256']
    hashes[Path(__file__).name]=sha(Path(__file__));dump(out/'protocol.json',dict(source_sha256=hashes,parent_summary_sha256=sha(parent/'summary.json'),oracle='exact nested three-point interpolation; independent rational root refinement',query_targets_accessed=False))
    counts=Counter();rows=[];begin=time.perf_counter()
    for item in json.loads((parent/'files.json').read_text()):
        assert sha(parent/item['file'])==item['sha256'];source=root/'results/cold_stagnation_switch/development'/item['source_file'];assert sha(source)==item['source_sha256']
        with np.load(source) as a:x=a['x'];bank=a['b'][16];activities=a['h'][16]
        for record in json.loads((parent/item['file']).read_text()):
            counts['states']+=1;old=record['old_event'];event=record['selected']
            if old is None:assert event is None and record['certificate'] is None;continue
            r=record['restart'];j,i=old['j'],old['i'];b=bank[r];h=activities[:,r]
            s=list(map(lambda z:F(float(z)),[x[i] if j==0 else h[j-1,i],b[j],b[j+1],h[j,i],h[j+1,i]]));cap=F(old['tau']);d=[F(float(z))/cap for z in old['u']];cert=record['certificate'];k0=branch(s[3]+s[2]);cuts=independent_cuts(s,d,cap)
            assert d==list(map(unpack,cert['direction'])) and cuts==list(map(unpack,cert['cuts'])) and k0==cert['current_branch'];assert len(cert['pieces'])==len(cuts)-1
            assert [s[3]-g(s[0]+s[1]),s[4]-g(s[3]+s[2])]==list(map(unpack,cert['ideal_residual']))
            zero={z['k']:z for z in minima(s,[F(0),F(0)])};assert all(zero[k0]['energy']<v['energy'] for k,v in zero.items() if k!=k0)
            pieces=[]
            for lo,hi,saved in zip(cuts[:-1],cuts[1:],cert['pieces']):
                assert lo==unpack(saved['lo']) and hi==unpack(saved['hi']);samples=[{z['k']:z['energy'] for z in minima(s,[t*z for z in d])} for t in [lo,(lo+hi)/2,hi]]
                polys={k:interpolate(lo,hi,[z[k] for z in samples]) for k in zero};assert set(map(int,saved['polynomials']))==set(polys)
                for k,poly in polys.items():assert poly==list(map(unpack,saved['polynomials'][str(k)]));counts['interpolated_polynomials']+=1
                pieces.append((lo,hi,polys));counts['pieces']+=1
            roots=[]
            for comp in cert['competitors']:
                k=comp['k'];found=None;checked=[]
                for lo,hi,polys in pieces:
                    poly=[a-bb for a,bb in zip(polys[k0],polys[k])];v,t=peak(poly,lo,hi);checked.append((lo,hi,poly,v,t));counts['segment_maxima']+=1
                    if v>0:
                        lower,upper=bisect(poly,lo,t,32);found=dict(k=k,poly=poly,lower=lower,upper=upper);break
                assert len(checked)==len(comp['checked'])
                for (lo,hi,poly,v,t),c in zip(checked,comp['checked']):
                    assert [lo,hi,v,t]==[unpack(c[z]) for z in ['lo','hi','maximum','maximum_at']] and poly==list(map(unpack,c['poly']))
                saved=comp['threshold']
                if found is None:assert saved is None
                else:
                    assert found['lower']==unpack(saved['lower']) and found['upper']==unpack(saved['upper']);assert found['poly']==list(map(unpack,saved['poly']));roots.append(found);counts['root_brackets']+=1
            ordered=sorted(roots,key=cmp_to_key(lambda a,b:compare(a,b) or (a['k']>b['k'])-(a['k']<b['k'])));assert cert['root_order']==[z['k'] for z in ordered];assert cert['chosen']['k']==ordered[0]['k']
            for saved in cert['comparisons']:assert saved['sign']==compare(ordered[0],next(z for z in roots if z['k']==saved['k']));counts['independent_root_order_checks']+=1
            chosen=ordered[0];assert event['j']==j and event['i']==i and F(event['tau'])<=cap
            u=list(map(lambda z:F(float(z)),event['u']));z=F(event['z']);physical={v['k']:v for v in minima(s,u)};gap=physical[k0]['energy']-energy(s,u,z)
            assert F(0)<=z<=F(1) and branch(z+s[2])!=k0 and gap>0 and gap==unpack(event['rounded_margin'])
            newnorm=sum(v*v for v in u);oldnorm=sum(F(float(v))**2 for v in old['u']);assert newnorm<=oldnorm and newnorm==unpack(cert['new_norm2']) and oldnorm==unpack(cert['old_norm2']);assert all(abs(a)<=abs(F(float(b))) for a,b in zip(u,old['u']))
            tau=F(event['tau']);assert u==[F(float(tau*dd)) for dd in d];assert cert['attempts'][-1]['accepted'] and tau==unpack(cert['implementation_tau']);assert record['fallbacks']==len(cert['attempts'])-1
            for ix,attempt in enumerate(cert['attempts']):
                actual_tau=unpack(attempt['tau']);ideal={v['k']:v for v in minima(s,[actual_tau*z for z in d])};positive=ideal[k0]['energy']>min(v['energy'] for k,v in ideal.items() if k!=k0)
                assert positive==attempt['ideal_positive']
                if ix==0:assert actual_tau==F(min(float(cap),float(chosen['upper'])))
                else:assert actual_tau==F(float(cap) if ix==16 else float((unpack(cert['attempts'][ix-1]['tau'])+cap)/2))
                if ix<len(cert['attempts'])-1:assert not attempt['accepted']
                counts['implementation_attempts']+=1
            counts['physical_certificates']+=1;rows.append(dict(seed=item['seed'],restart=r,tau_ratio=float(tau/cap),norm2_ratio=float(newnorm/oldnorm),fallbacks=record['fallbacks'],gap_float=float(gap)))
        print(json.dumps(dict(seed=item['seed'],counts=counts)),flush=True)
    assert counts['states']==272 and counts['physical_certificates']==260
    dump(out/'rows.json',rows);ans=dict(passed=True,counts=counts,seconds=time.perf_counter()-begin,protocol_sha256=sha(out/'protocol.json'),rows_sha256=sha(out/'rows.json'),query_targets_accessed=False);dump(out/'summary.json',ans);print(json.dumps(ans),flush=True)

if __name__=='__main__':main()
