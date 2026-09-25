"""414 exact segment enumeration and sampled multidimensional containment."""
from fractions import Fraction as F
from pathlib import Path
import time
import traceback
import numpy as np
import frontier_range_certificate_v1 as core
import deadline_risk_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'results/frontier_range_certificate/preflight_v1'
DESIGN='outputs/ttt-pc-alm-research/414_frontier_range_certificate_protocol_v1.md'


def line_extrema(q,inv,low,high,constants):
    segments=[(low,high,F(0),q)]
    for row in inv:
        b_slope=row[0];b_const=sum((a*b for a,b in zip(row[1:],constants)),F(0));next_segments=[]
        for left,right,slope,offset in segments:
            slope+=b_slope;offset+=b_const;knots={left,right}
            if slope:
                knots|={t for t in ((k-offset)/slope for k in [F(0),F(1,2),F(1)]) if left<t<right}
            knots=sorted(knots)
            for a,b in zip(knots,knots[1:]):
                middle=slope*(a+b)/2+offset
                if middle<=0 or middle>=1:s,c=F(0),F(0)
                elif middle<=F(1,2):s,c=2*slope,2*offset
                else:s,c=-2*slope,2-2*offset
                next_segments.append((a,b,s,c))
        segments=next_segments
    values=[s*t+c for a,b,s,c in segments for t in [a,b]]
    return min(values),max(values)


def main():
    begin=time.perf_counter();rng=np.random.default_rng(414731)
    files=[Path(__file__),Path(core.__file__),Path(core.core.__file__),ROOT/DESIGN]
    hashes={p.relative_to(ROOT).as_posix():io.sha(p) for p in files}
    counts=dict(inverse_identities=0,line_exact_ranges=0,box_point_containments=0,zero_width_cases=0)
    matrices=[]
    while len(matrices)<32:
        matrix=rng.integers(-3,4,(4,4)).tolist()
        if core.core.exact_rank(matrix)==4:matrices.append(matrix)
    for matrix in matrices:
        inv=core.inverse(matrix);counts['inverse_identities']+=1
        low,high=F(-1,20),F(1,20);constant=[F(int(i),50) for i in rng.integers(-2,3,3)]
        q=F(int(rng.integers(0,101)),100)
        intervals=[(low,high)]+[(a,a) for a in constant]
        enclosure,_=core.enclose(q,inv,intervals)
        exact=line_extrema(q,inv,low,high,constant)
        assert enclosure[0]<=exact[0]<=exact[1]<=enclosure[1];counts['line_exact_ranges']+=1
        for _ in range(16):
            z=[F(int(t),1000) for t in rng.integers(-50,51,4)]
            b=[sum((a*v for a,v in zip(row,z)),F(0)) for row in inv]
            bounds,_=core.enclose(q,inv,[(low,high)]*4)
            assert bounds[0]<=core.core.mathcore.forward(q,b)<=bounds[1];counts['box_point_containments']+=1
        z=[F(0)]*4
        point_range,_=core.enclose(q,inv,[(t,t) for t in z])
        assert point_range[0]==point_range[1]==core.core.mathcore.forward(q,z);counts['zero_width_cases']+=1
    identity=tuple(tuple(F(int(i==j)) for j in range(4)) for i in range(4))
    peak,_=core.enclose(F(0),identity,[(F(0),F(1))]+[(F(0),F(0))]*3)
    assert peak==(0,1), 'Vertex-only range would incorrectly miss internal peaks'
    io.verify_hashes(ROOT,hashes)
    ss=dict(passed=True,counts=counts,source_sha256=hashes,seconds=time.perf_counter()-begin,
        real_support_range_diagnostic_run=False,task_risk_compared=False)
    io.save(OUT/'summary.json',ss);print(ss,flush=True)


if __name__=='__main__':
    assert io.read(ROOT/'results/partial_support_certificate/audit_v1/summary.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
