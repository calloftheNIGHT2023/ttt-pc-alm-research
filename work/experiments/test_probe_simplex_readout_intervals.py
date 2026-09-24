"""Analytic integral fixtures before any new geometry-integral diagnosis."""
import argparse
from fractions import Fraction as F
from pathlib import Path
import json
from probe_simplex_readout_intervals import integrate, enclosure, range_image
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/probe_credit_confirmation/simplex_interval_selftests_v1'
    assert not out.exists()
    vertices=tuple([tuple(F(0) for _ in range(4))]+
        [tuple(F(i==j) for j in range(4)) for i in range(4)])
    small=tuple(tuple(t/1024 for t in v) for v in vertices)
    tests={};records={}
    expected=F(16,128)+F(30,5*1024)
    r=integrate(F(1,128),[small],[F(1)],max_width=F(0))
    assert F(r['exact_lower'])==F(r['exact_upper'])==expected and r['splits']==0
    tests['four_layer_affine_mean']=dict(passed=True,exact_integral=str(expected));records['affine']=r
    for name,q,value in [('constant_zero',F(-2),F(0)),('clipped_peak',F(1,2),F(1))]:
        point=tuple([tuple(F(0) for _ in range(4))]*5)
        r=integrate(q,[point],[F(1)],layers=1,max_width=F(0))
        assert F(r['exact_lower'])==F(r['exact_upper'])==value
        tests[name]=dict(passed=True,exact_integral=str(value));records[name]=r
    assert range_image(F(1,4),F(3,4))==(F(1,2),F(1))
    assert range_image(F(-1),F(2))==(F(0),F(1))
    tests['interval_peak_not_missed']=dict(passed=True)
    # b0 ~ Beta(1,4) on the standard 4-simplex:
    # E[(b0-a)+]=(1-a)^5/5 for a in [0,1].
    for name,a in [('unshifted_tent',F(0)),('crossing_both_knots',F(1,4)),('tail_hinge',F(3,4))]:
        expected=2*(1-a)**5/5
        if a+F(1,2)<1:expected-=4*(1-a-F(1,2))**5/5
        coarse=integrate(-a,[vertices],[F(1)],layers=1,max_width=F(1,1000),max_splits=0)
        refined=integrate(-a,[vertices],[F(1)],layers=1,max_width=F(1,1000),max_splits=4096)
        assert F(refined['exact_lower'])<=expected<=F(refined['exact_upper'])
        assert F(refined['exact_lower'])>=F(coarse['exact_lower'])
        assert F(refined['exact_upper'])<=F(coarse['exact_upper'])
        assert refined['status']=='width_met',name
        tests[name]=dict(passed=True,exact_integral=str(expected),splits=refined['splits'],
                         width=str(F(refined['exact_upper'])-F(refined['exact_lower'])))
        records[name]=refined
    r=integrate(F(1,128),[small,small],[F(3,7),-F(3,7)],target=F(0),max_splits=0)
    assert r['status']=='upper_bound_met' and F(r['exact_lower'])==F(r['exact_upper'])==0
    tests['signed_affine_cancellation']=dict(passed=True);records['signed']=r
    r=integrate(F(1,128),[small],[F(1)],target=F(1,1000),max_splits=0)
    assert r['status']=='lower_bound_exceeds_target'
    tests['actual_exceedance_reported']=dict(passed=True);records['exceedance']=r
    r=integrate(F(0),[vertices],[F(1)],layers=1,max_width=F(1,1000),max_splits=0)
    assert r['status']=='work_limit_unresolved'
    tests['work_limit_not_success']=dict(passed=True);records['work_limit']=r
    out.mkdir();exclusive_json(out/'tests.json',tests);exclusive_json(out/'records.json',records)
    summary=dict(passed=True,tests=len(tests),query_targets_accessed=False,audit_gate_passed=False,
        scope='Analytic scalar integrals and exact-status behavior; independent proof replay still required',
        source_sha256={n:sha(src/n) for n in [Path(__file__).name,'probe_simplex_readout_intervals.py']},
        outputs_sha256={n:sha(out/n) for n in ['tests.json','records.json']})
    exclusive_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
