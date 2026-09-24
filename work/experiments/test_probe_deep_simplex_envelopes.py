"""Analytic four-fold tent integrals and correlated-bias cancellation fixtures."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
from probe_simplex_readout_intervals_v3 import integrate
from audit_probe_simplex_readout_intervals_v3 import verify
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def analytic_tent4_beta(query,scale):
    # The first standard-simplex barycentric coordinate t has density
    # 4(1-t)^3. T^4(query+scale*t) is affine on each inverse dyadic segment.
    assert scale>0 and 0<=query and query+scale<=1
    answer=F(0)
    for segment in range(16):
        left=max(F(0),(F(segment,16)-query)/scale)
        right=min(F(1),(F(segment+1,16)-query)/scale)
        if left>=right:continue
        slope_z=F(16 if segment%2==0 else -16)
        intercept_z=F(-segment if segment%2==0 else segment+1)
        slope=slope_z*scale;intercept=slope_z*query+intercept_z
        polynomial=[4*intercept,4*slope-12*intercept,12*intercept-12*slope,12*slope-4*intercept,-4*slope]
        answer+=sum((c*(right**(power+1)-left**(power+1))/F(power+1)
                    for power,c in enumerate(polynomial)),F(0))
    return answer


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/probe_credit_confirmation/deep_simplex_envelope_selftests_v1';assert not out.exists()
    prior=root/'results/probe_credit_confirmation/simplex_interval_selftests_v3'
    previous=json.loads((prior/'summary.json').read_text());assert previous['passed']
    for name,digest in previous['source_sha256'].items():assert sha(src/name)==digest
    out.mkdir();tests={};files={}
    for name,query,scale in [('fourfold_full_beta',F(0),F(1)),('fourfold_shifted_beta',F(1,10),F(1,4))]:
        vertices=[(F(0),F(0),F(0),F(0))]*5
        vertices[1]=(scale,F(0),F(0),F(0))
        expected=analytic_tent4_beta(query,scale)
        record=integrate(query,[vertices],[F(1)],max_width=F(1,1000),max_splits=32768)
        checked=verify([vertices],[F(1)],query,4,record)
        assert F(record['exact_lower'])<=expected<=F(record['exact_upper'])
        assert record['status']=='width_met',name
        tests[name]=dict(passed=True,exact_integral=str(expected),check=checked)
        filename=name+'.json';exclusive_json(out/filename,record);files[filename]=sha(out/filename)
        print(json.dumps(dict(test=name,passed=True,splits=record['splits'],exact_integral=str(expected))),flush=True)
    # For every t in [0,1], T(t)<=2t, so T(T(t)-2t)=0 and
    # the remaining zero-bias tent layers stay zero. This is nonlinear at
    # the first layer but globally cancels via the correlated second bias.
    vertices=[(F(0),F(0),F(0),F(0))]*5;vertices[1]=(F(1),F(-2),F(0),F(0))
    record=integrate(F(0),[vertices],[F(1)],target=F(0),max_splits=32768)
    checked=verify([vertices],[F(1)],F(0),4,record)
    assert record['status']=='upper_bound_met' and F(record['exact_lower'])==F(record['exact_upper'])==0
    tests['nonlinear_correlated_zero']=dict(passed=True,exact_integral='0',check=checked)
    exclusive_json(out/'nonlinear_correlated_zero.json',record);files['nonlinear_correlated_zero.json']=sha(out/'nonlinear_correlated_zero.json')
    exclusive_json(out/'tests.json',tests);files['tests.json']=sha(out/'tests.json')
    result=dict(passed=True,analytic_four_layer_tests=len(tests),query_targets_accessed=False,audit_gate_passed=False,
        source_sha256={**previous['source_sha256'],Path(__file__).name:sha(Path(__file__))},
        previous_selftest_sha256=sha(prior/'summary.json'),outputs_sha256=files)
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
