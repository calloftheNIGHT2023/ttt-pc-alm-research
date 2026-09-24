"""Exact coverage of [0,1] from pointwise signed-mean bounds and a Lipschitz constant.

If |D(q_i)| <= u_i and |D(q)-D(q_i)| <= K|q-q_i|, while the
separate geometry displacement is uniformly <= e, each point certifies radius
(B-e-u_i)/K. Unequal radii may cover the interval even when a common grid-gap
test does not. Input pointwise bounds must have their own integral proofs.
"""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def cover(points,budget,position,lipschitz):
    budget,position,lipschitz=map(F,(budget,position,lipschitz))
    assert budget>=0 and position>=0 and lipschitz>=0
    intervals=[]
    for index,(query,bound) in enumerate(points):
        query,bound=F(query),F(bound);assert 0<=query<=1 and bound>=0
        allowance=budget-position-bound
        if allowance<0:continue
        if lipschitz==0:left,right=F(0),F(1)
        else:
            radius=allowance/lipschitz;left=max(F(0),query-radius);right=min(F(1),query+radius)
        intervals.append(dict(point=index,left=str(left),right=str(right)))
    ordered=sorted(intervals,key=lambda row:(F(row['left']),F(row['right']),row['point']))
    frontier=F(0);gaps=[]
    for interval in ordered:
        left,right=F(interval['left']),F(interval['right'])
        if left>frontier:gaps.append((frontier,left))
        frontier=max(frontier,right)
    if frontier<1:gaps.append((frontier,F(1)))
    return dict(complete=not gaps,intervals=intervals,gaps=[[str(a),str(b)] for a,b in gaps],
        exact_uncovered_length=str(sum((b-a for a,b in gaps),F(0))),
        exact_budget=str(budget),exact_position_bound=str(position),exact_lipschitz=str(lipschitz))


def verify_complete(record,points,budget,position,lipschitz):
    # Recompute each certified radius, then check every segment induced by the
    # endpoints; this does not call the producer's interval-union scan.
    budget,position,lipschitz=map(F,(budget,position,lipschitz))
    assert record['complete'] and record['gaps']==[] and F(record['exact_uncovered_length'])==0
    assert F(record['exact_budget'])==budget and F(record['exact_position_bound'])==position
    assert F(record['exact_lipschitz'])==lipschitz and budget>=0 and position>=0 and lipschitz>=0
    seen=set();intervals=[]
    for row in record['intervals']:
        index=row['point'];assert type(index) is int and 0<=index<len(points) and index not in seen;seen.add(index)
        query,bound=map(F,points[index]);assert 0<=query<=1 and bound>=0
        remaining=budget-position-bound;assert remaining>=0
        if lipschitz==0:low,high=F(0),F(1)
        else:low=max(F(0),query-remaining/lipschitz);high=min(F(1),query+remaining/lipschitz)
        assert F(row['left'])==low and F(row['right'])==high
        intervals.append((low,high))
    assert seen=={i for i,(_,u) in enumerate(points) if budget-position-F(u)>=0}
    endpoints=sorted({F(0),F(1)}|{p for interval in intervals for p in interval})
    probes=endpoints+[(a+b)/2 for a,b in zip(endpoints,endpoints[1:])]
    assert all(any(a<=p<=b for a,b in intervals) for p in probes), 'Uncovered segment or endpoint'
    return dict(passed=True,point_bounds=len(points),certified_intervals=len(intervals),segments_checked=len(endpoints)-1)


def dyadic_inside(left,right,max_power=30):
    left,right=F(left),F(right);assert 0<=left<right<=1
    midpoint=(left+right)/2
    for power in range(max_power+1):
        denominator=2**power;numerator=(midpoint.numerator*denominator)//midpoint.denominator
        for value in [F(numerator,denominator),F(numerator+1,denominator)]:
            if left<value<right:return value
    return None


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();out=root/'results/probe_credit_confirmation/query_cover_selftests_v1'
    assert not out.exists();tests={}
    examples=[
        ('touching_intervals',[(F(0),F(0)),(F(1),F(0))],F(1,2),F(0),F(1),True),
        ('overlapping_intervals',[(F(1,4),F(0)),(F(3,4),F(0))],F(1,2),F(0),F(1),True),
        ('middle_gap',[(F(0),F(0)),(F(1),F(0))],F(1,3),F(0),F(1),False),
        ('endpoint_gaps',[(F(1,2),F(0))],F(1,4),F(0),F(1),False),
        ('zero_radius_not_coverage',[(F(1,2),F(1))],F(1),F(0),F(1),False),
        ('zero_lipschitz',[(F(1,3),F(1,4))],F(1,2),F(1,4),F(0),True),
        ('over_budget_point',[(F(1,3),F(1))],F(1,2),F(0),F(0),False),
        ('unequal_radii',[(F(0),F(0)),(F(3,4),F(1,4))],F(1,2),F(0),F(1),True),
    ]
    for name,points,budget,position,constant,complete in examples:
        result=cover(points,budget,position,constant);assert result['complete']==complete
        if complete:checked=verify_complete(result,points,budget,position,constant)
        else:checked=None
        tests[name]=dict(passed=True,certificate=result,independent_check=checked)
    gap=cover(examples[2][1],F(1,3),0,1);assert gap['gaps']==[['1/3','2/3']]
    point=dyadic_inside(F(1,3),F(2,3));assert point==F(1,2)
    assert dyadic_inside(F(1,3),F(2,3),max_power=0) is None
    tests['dyadic_gap_choice']=dict(passed=True,value=str(point),exhaustion_not_success=True)
    points=examples[0][1];valid=cover(points,F(1,2),0,1)
    corruptions=[]
    bad=deepcopy(valid);bad['intervals'][0]['right']='3/4';corruptions.append(('inflated_radius',bad,points))
    bad=deepcopy(gap);bad.update(complete=True,gaps=[],exact_uncovered_length='0');corruptions.append(('hidden_gap',bad,examples[2][1]))
    bad=deepcopy(valid);bad['intervals'].pop();corruptions.append(('missing_interval',bad,points))
    bad=deepcopy(valid);bad['intervals'].append(deepcopy(bad['intervals'][0]));corruptions.append(('duplicate_interval',bad,points))
    for name,bad,pp in corruptions:
        budget=F(1,3) if name=='hidden_gap' else F(1,2)
        try:verify_complete(bad,pp,budget,0,1)
        except (AssertionError,ValueError,IndexError):tests[name]=dict(passed=True,corruption_rejected=True)
        else:raise AssertionError('Bad coverage accepted: '+name)
    out.mkdir();exclusive_json(out/'tests.json',tests)
    summary=dict(passed=True,tests=len(tests),query_targets_accessed=False,audit_gate_passed=False,
        scope='Abstract interval-coverage arithmetic only; actual readout bounds need separate certificates',
        source_sha256=sha(Path(__file__)),tests_sha256=sha(out/'tests.json'))
    exclusive_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
