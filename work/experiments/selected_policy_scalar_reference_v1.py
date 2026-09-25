"""315 scalar Fraction evaluator with fixed choices, no affine coefficient algebra.

Intended for off-policy algebra checks too: choices stay fixed at every probe.
It does not assert that those choices minimize energy at the probe point.
"""
from fractions import Fraction as F

S=[0,2,-2,0];C=[0,0,2,0];K=[F(0),F(1,2),F(1)]


def decode(policies,schema):
    result={}
    for group,fields in schema.items():
        row=list(policies[group]);offset=0;data={}
        for field in fields:
            stop=offset+field['width'];data[field['name']]=[int(t) for t in row[offset:stop]];offset=stop
        assert offset==len(row);result[group]=data
    return result


def scalar(policies,schema,point,x,v,method,*,bound=.12,trust=.01,eps=.001):
    p=decode(policies,schema);d=sum(g.startswith('bias_') for g in p);n=len(x)
    x,v=list(map(F,x)),list(map(F,v));B,T,E=F(bound),F(trust),F(eps)
    point=list(map(F,point));b=point[:d]
    oldh=[point[d+j*n:d+(j+1)*n] for j in range(d)]
    u=[point[d+d*n+j*n:d+d*n+(j+1)*n] for j in range(d)]
    h=[row[:] for row in oldh]
    for j in range(d-1,-1,-1):
        q=p[f'activity_{j}']
        for i in range(n):
            prev=x[i] if j==0 else oldh[j-1][i];reg=q['read_regions'][i]
            a=S[reg]*(prev+b[j])+C[reg]-u[j][i]
            if j==d-1:
                lo=max(F(0),v[i]-E);hi=min(F(1),v[i]+E)
                raw=(a+T*oldh[j][i])/(1+T);low=q['candidate_vs_low'][i];high=q['candidate_vs_high'][i]
            else:
                k=q['winner'][i];ss=S[k]
                if k==0 or q['low_source'][k]<=0:lo=F(0)
                else:lo=K[k-1]-b[j+1]
                if k==3 or q['high_source'][k]>=0:hi=F(1)
                else:hi=K[k]-b[j+1]
                raw=(a+ss*(h[j+1][i]+u[j+1][i]-ss*b[j+1]-C[k])+T*oldh[j][i])/(1+ss*ss+T)
                low=q['candidate_vs_low'][k*n+i];high=q['candidate_vs_high'][k*n+i]
            if high>=0:h[j][i]=hi
            elif low<=0:h[j][i]=lo
            else:h[j][i]=raw
    newb=[]
    for j in range(d):
        q=p[f'bias_{j}'];prev=x if j==0 else h[j-1];target=[h[j][i]+u[j][i] for i in range(n)]
        aa=F(0);bb=F(0)
        for i,reg in enumerate(q['initial_regions']):
            aa+=F(S[reg]**2,n);bb+=S[reg]*(target[i]-S[reg]*prev[i]-C[reg])/n
        winner=q['winner'][0]
        for event in q['stable_order'][:winner]:
            if q['event_valid'][event]:
                k,i=divmod(event,n)
                aa+=F(S[k+1]**2-S[k]**2,n)
                bb+=(S[k+1]*(target[i]-S[k+1]*prev[i]-C[k+1])-S[k]*(target[i]-S[k]*prev[i]-C[k]))/n
        raw=(bb+T*b[j])/(max(aa,F(0))+T)
        ordered=[]
        for event in q['stable_order']:
            k,i=divmod(event,n)
            if q['event_vs_box_low'][event]<=0:value=-B
            elif q['event_vs_box_high'][event]>=0:value=B
            else:value=K[k]-prev[i]
            ordered.append(value)
        lo=([ -B ]+ordered)[winner];hi=(ordered+[B])[winner]
        if q['candidate_vs_high'][winner]>=0:newb.append(hi)
        elif q['candidate_vs_low'][winner]<=0:newb.append(lo)
        else:newb.append(raw)
    newu=[]
    for j in range(d):
        prev=x if j==0 else h[j-1];out=[]
        for i,reg in enumerate(p[f'residual_{j}']['regions']):
            residual=h[j][i]-S[reg]*(prev[i]+newb[j])-C[reg]
            out.append(u[j][i]+(F(1,2)*residual if method=='alm' else F(0)))
        newu.append(out)
    return newb+[t for row in h for t in row]+[t for row in newu for t in row]
