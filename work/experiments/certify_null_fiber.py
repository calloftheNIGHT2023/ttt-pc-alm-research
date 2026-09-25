"""Exact rational null-fiber witnesses for observed toy networks.

Treat stored floating input/weight/parameter values as rational constants in the
mathematical network. Query separation remains a numerical evaluator statistic.
"""
import argparse,hashlib,json,time
from fractions import Fraction as F
from pathlib import Path
import numpy as np
import vector_interval_memory as m


def rational_array(a):return [[F(float(v)) for v in row] for row in a]


def activation(z):return max(F(-1),F(1)-2*abs(z))


def exact_network(point,x,weights):
    h=x
    for layer,weight in enumerate(weights):h=[[activation(sum((weight[o][k]*row[k] for k in range(len(row))),F(0))+point[layer][o]) for o in range(len(row))] for row in h]
    return h


def affine_rows(point,x,weights):
    depth=len(weights);width=len(point[0]);p=depth*width;n=len(x);h=x;jac=[[[F(0)]*p for _ in range(width)] for _ in range(n)];constraints=[]
    for layer,weight in enumerate(weights):
        newh=[];newjac=[]
        for i in range(n):
            hh=[];jj=[]
            for out in range(width):
                z=sum((weight[out][k]*h[i][k] for k in range(width)),F(0))+point[layer][out]
                coeff=[sum((weight[out][k]*jac[i][k][param] for k in range(width)),F(0)) for param in range(p)]
                coeff[layer*width+out]+=1
                if z<F(-1):s,k,lo,hi=0,-1,None,F(-1)
                elif z<F(0):s,k,lo,hi=2,1,F(-1),F(0)
                elif z<F(1):s,k,lo,hi=-2,1,F(0),F(1)
                else:s,k,lo,hi=0,-1,F(1),None
                if lo is not None:constraints.append(([-v for v in coeff],z-lo))
                if hi is not None:constraints.append((coeff,hi-z))
                hh.append(s*z+k);jj.append([s*v for v in coeff])
            newh.append(hh);newjac.append(jj)
        h,jac=newh,newjac
    return [row for sample in jac for row in sample],constraints


def null_vector(matrix):
    rows=[r.copy() for r in matrix];ncols=len(rows[0]);pivots=[];at=0
    for col in range(ncols):
        pivot=next((i for i in range(at,len(rows)) if rows[i][col]),None)
        if pivot is None:continue
        rows[at],rows[pivot]=rows[pivot],rows[at];value=rows[at][col]
        for i in range(at+1,len(rows)):
            if not rows[i][col]:continue
            factor=rows[i][col]/value
            for k in range(col,ncols):rows[i][k]-=factor*rows[at][k]
        pivots.append(col);at+=1
        if at==len(rows):break
    if len(pivots)==ncols:return len(pivots),None
    free=next(i for i in range(ncols) if i not in pivots);direction=[F(0)]*ncols;direction[free]=F(1)
    for i,col in reversed(list(enumerate(pivots))):direction[col]=-sum((rows[i][k]*direction[k] for k in range(col+1,ncols)),F(0))/rows[i][col]
    assert all(sum((a*b for a,b in zip(row,direction)),F(0))==0 for row in matrix)
    return len(pivots),direction


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args();assert not args.out.exists();args.out.parent.mkdir(parents=True,exist_ok=True)
    weights_float=m.family.make_weights(3,8);weights=[rational_array(w) for w in weights_float];records=[];start=time.perf_counter()
    for seed in [5400000,5400009]:
        rng=np.random.default_rng(seed);truth=rng.uniform(-m.PRIOR,m.PRIOR,(3,8));x=rng.uniform(-1,1,(24,8));q=rng.uniform(-1,1,(512,8))
        point=rational_array(truth);inputs=rational_array(x[:8]);jac,inequalities=affine_rows(point,inputs,weights);rank,direction=null_vector(jac)
        assert rank==23 and direction is not None,rank
        flat=[v for row in point for v in row];bound=F(float(m.PRIOR))
        for j,value in enumerate(flat):
            row=[F(0)]*24;row[j]=1;inequalities.append((row,bound-value));row2=[-v for v in row];inequalities.append((row2,bound+value))
        low=None;high=None
        for coeff,margin in inequalities:
            slope=sum((a*b for a,b in zip(coeff,direction)),F(0))
            if slope>0:high=min(high,margin/slope) if high is not None else margin/slope
            elif slope<0:low=max(low,margin/slope) if low is not None else margin/slope
            else:assert margin>=0
        assert low<0<high
        endpoints=[[flat[j]+F(19,20)*step*direction[j] for j in range(24)] for step in [low,high]]
        tasks=[[values[8*j:8*(j+1)] for j in range(3)] for values in endpoints]
        outputs=[exact_network(task,inputs,weights) for task in tasks];assert outputs[0]==outputs[1]
        assert all(abs(value)<=bound for task in endpoints for value in task)
        pred=m.forward(np.array([[[float(v) for v in row] for row in task] for task in tasks]),q,weights_float)
        separation=float(np.mean((pred[0]-pred[1])**2))
        records.append({'seed':seed,'exact_rank':rank,'exact_null_residual_zero':True,'exact_context_outputs_identical':True,
            'exact_prior_membership':True,'direction':[str(v) for v in direction],'interval':[str(low),str(high)],
            'lower_task':[[str(v) for v in row] for row in tasks[0]],'upper_task':[[str(v) for v in row] for row in tasks[1]],
            'query_separation_numerical':separation,'two_point_risk_bound_numerical':separation/4})
        print(json.dumps({'seed':seed,'exact_rank':rank,'exact_context_equality':True,'query_separation':separation}),flush=True)
    result={'passed':True,'records':records,'seconds':time.perf_counter()-start,
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'scope':'exact equality for rationalized stored constants; two-point minimax theorem, not original-prior average risk or PC-specific advantage'}
    args.out.write_text(json.dumps(result,indent=2),encoding='utf-8')


if __name__=='__main__':main()
