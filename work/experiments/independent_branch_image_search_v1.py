"""321 independent exact checker: suffix minima plus best-first enumeration.

Does not import the candidate selector. Uses both clamp breakpoints, reverse
cost-to-go recursion, and best-first search rather than forward K truncation.
"""
from fractions import Fraction as F
from functools import lru_cache
from itertools import product
import heapq

BOUND=F(.12)
EPS=F(.001)


def image_mask(row):return sum(2**i for i,label in enumerate(row) if label==1 or label==2)


@lru_cache(maxsize=128)
def layer_table(low,high,previous,credit):
    n=len(low);intervals=[(-BOUND,F(0)),(F(0),F(1,2)),(F(1,2),F(1)),(F(1),1+BOUND)]
    slopes=[0,2,-2,0];offsets=[0,0,2,0];result=[]
    for row in product(range(4),repeat=n):
        zl=[intervals[k][0] for k in row];zh=[intervals[k][1] for k in row]
        left=max([-BOUND]+[z-h for z,h in zip(zl,high)])
        right=min([BOUND]+[z-h for z,h in zip(zh,low)])
        if left>right:continue
        points={left,right}
        for z0,z1,l,h in zip(zl,zh,low,high):
            for point in [z0-l,z1-h]:
                if left<=point<=right:points.add(point)
        sc=[slopes[r]*a for r,a in zip(row,credit)]
        linear=[p-s for p,s in zip(previous,sc)]
        constant=-sum((a*offsets[r] for r,a in zip(row,credit)),F(0));values=[]
        for bias in points:
            hlow=[max(l,z-bias) for l,z in zip(low,zl)]
            hhigh=[min(h,z-bias) for h,z in zip(high,zh)]
            values.append(constant+sum((min(c*l,c*h)-s*bias for c,l,h,s in zip(linear,hlow,hhigh,sc)),F(0)))
        result.append((row,image_mask(row),min(values)))
    return tuple(result)


class IndependentSearch:
    def __init__(self,x,v,credit,original):
        self.n=len(x);self.d=len(original);self.original=tuple(tuple(map(int,row)) for row in original)
        aa=[tuple(F(float(a)) for a in row) for row in credit];self.transitions={};self.grouped={};self.expansions=0
        terminal={}
        for bits in range(2**self.n):
            lo=[max(F(0),F(float(y))-EPS) for y in v]
            hi=[min(F(int(bool(bits&(1<<i)))),F(float(y))+EPS) for i,y in enumerate(v)]
            if all(l<=h for l,h in zip(lo,hi)):
                terminal[bits]=sum((min(a*l,a*h) for a,l,h in zip(aa[-1],lo,hi)),F(0))
        for j in range(self.d):
            for incoming in [-1] if j==0 else range(2**self.n):
                low=tuple(map(F,x)) if j==0 else (F(0),)*self.n
                high=low if j==0 else tuple(F(int(bool(incoming&(1<<i)))) for i in range(self.n))
                previous=(F(0),)*self.n if j==0 else aa[j-1];edges=[];groups={}
                for row,outgoing,value in layer_table(low,high,previous,aa[j]):
                    if j==self.d-1:
                        if outgoing not in terminal:continue
                        value+=terminal[outgoing]
                    delta=sum(a!=b for a,b in zip(row,self.original[j]));edges.append((row,outgoing,delta,value))
                    key=(outgoing,delta);pair=(value,row)
                    if key not in groups or pair<groups[key]:groups[key]=pair
                self.transitions[j,incoming]=edges;self.grouped[j,incoming]=groups
        self.best=lru_cache(maxsize=None)(self._best)

    def _best(self,j,incoming,remaining):
        if remaining<0 or remaining>(self.d-j)*self.n:return None
        if j==self.d:return (F(0),()) if remaining==0 else None
        answer=None
        for (outgoing,delta),(value,row) in self.grouped[j,incoming].items():
            suffix=self.best(j+1,outgoing,remaining-delta)
            if suffix is None:continue
            pair=(value+suffix[0],row+suffix[1])
            if answer is None or pair<answer:answer=pair
        return answer

    def kbest(self,changes,k=8):
        best=self.best(0,-1,changes)
        if best is None:return []
        queue=[(best[0],best[1],0,-1,changes,F(0),())];output=[]
        while queue and len(output)<k:
            _,_,j,incoming,left,cost,prefix=heapq.heappop(queue);self.expansions+=1
            if j==self.d:
                assert left==0;output.append((cost,prefix));continue
            for row,outgoing,delta,value in self.transitions[j,incoming]:
                suffix=self.best(j+1,outgoing,left-delta)
                if suffix is None:continue
                path=prefix+row;total=cost+value
                heapq.heappush(queue,(total+suffix[0],path+suffix[1],j+1,outgoing,left-delta,total,path))
        assert output==sorted(output)
        return output

    def fixed(self,pattern):
        incoming=-1;total=F(0)
        for j,row in enumerate(pattern):
            target=tuple(map(int,row));matched=next((edge for edge in self.transitions[j,incoming] if edge[0]==target),None)
            if matched is None:return None
            _,incoming,_,value=matched;total+=value
        return total

    def clear(self):self.best.cache_clear()
