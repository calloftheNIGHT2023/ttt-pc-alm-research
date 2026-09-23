"""Freeze round245/246 evidence, exact risk cross-check, report and next plan."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import re
import sys
import numpy as np
import multiplier_fixed_point_exact as core
import exact_escape_query_risk as integral
from audit_multiplier_fixed_point import exact_forward
from run_multiplier_fixed_point_screen import sha,dump

def read(p):return json.loads(p.read_text())
def main():
    sys.set_int_max_str_digits(0);ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/multiplier_fixed_point';docs=root/'outputs/ttt-pc-alm-research';target=root/'results/round_246_audit.json';assert not target.exists()
    screen=read(base/'screen/summary.json');exact=read(base/'exact/summary.json');cont=read(base/'continuation/summary.json');audit=read(base/'audit/summary.json')
    witness=read(base/'witness/summary.json');risk=read(base/'witness/query_risk.json');basin=read(base/'witness/stationary_basin.json');analysis=read(base/'analysis/summary.json');figure=read(base/'analysis/figure_audit.json')
    assert all(t['passed'] for t in [screen,exact,cont,audit,witness,risk,basin,analysis,figure])
    assert screen['proposals']==exact['proposals']==audit['independent_points']==73
    assert exact['accepted']==audit['accepted']==cont['certified_points']==69
    assert cont['method_points']==analysis['checks']['query_replays']==analysis['checks']['support_points']==414
    assert audit['all_block_checks']==1460 and audit['interval_checks']==3772
    for phase in ['screen','exact','continuation']:
        p=read(base/phase/'protocol.json')
        for name,h in p['source_sha256'].items():assert sha(src/name)==h,name
    p=read(base/'continuation/protocol.json');sources=dict(p['source_sha256'])
    assert p['exact_summary_sha256']==sha(base/'exact/summary.json') and p['accepted_sha256']==sha(base/'exact/accepted.json') and p['certificates_sha256']==sha(base/'exact/certificates.json')
    assert exact['certificates_sha256']==p['certificates_sha256'] and exact['accepted_sha256']==p['accepted_sha256']
    for task in read(base/'screen/rows.json'):assert sha(base/'screen'/task['file'])==task['sha256']
    manifest=read(base/'continuation/before_query_manifest.json')
    for name,h in manifest['arrays'].items():assert sha(base/'continuation'/name)==h
    for name,h in manifest['geometry'].items():assert sha(base/'continuation'/name)==h
    assert manifest['support_rows_sha256']==sha(base/'continuation/support_rows.json') and cont['before_query_manifest_sha256']==sha(base/'continuation/before_query_manifest.json')
    for name,h in audit['output_sha256'].items():assert sha(base/'audit'/name)==h
    assert audit['source_sha256']==sha(src/'audit_multiplier_fixed_point.py')
    assert witness['trajectory_sha256']==sha(base/'witness/trajectory.json') and witness['first_primal_move']==witness['first_forward_branch_change']==7 and witness['first_strict_support_feasible']==62
    assert basin['source_sha256']==sha(src/'prove_escape_stationary_basin.py') and risk['source_sha256']==sha(src/'exact_escape_query_risk.py')
    assert risk['witness_summary_sha256']==sha(base/'witness/summary.json') and basin['certificates_sha256']==sha(base/'exact/certificates.json')
    assert analysis['source_sha256']==sha(src/'analyze_multiplier_fixed_point.py') and analysis['task_rows_sha256']==sha(base/'analysis/task_balanced_rows.json')
    assert figure['source_sha256']==sha(src/'plot_multiplier_fixed_point.py') and figure['figure_sha256']==sha(base/'analysis/multiplier_fixed_point.png')
    assert figure['analysis_sha256']==sha(base/'analysis/summary.json') and figure['risk_sha256']==sha(base/'witness/query_risk.json')
    # Independent exact Simpson integration of the frozen FLOAT predictors.
    # A difference is affine on each union interval, so Simpson is exact.
    seed=risk['seed'];rid=risk['restart'];teacher=[F(float(t)) for t in np.random.default_rng(seed).uniform(-.12,.12,4)];simpson={}
    for method,result in risk['saved_float_methods'].items():
        a=np.load(base/'continuation'/f'{seed}_{method}.npz');i=list(a['restarts']).index(rid);b=[F(float(t)) for t in a['best'][-1,i]]
        knots=sorted({p for lo,hi,s,c in integral.segments(b)+integral.segments(teacher) for p in [lo,hi]});total=F(0)
        def squared(t):return (exact_forward([t],b)[0][0]-exact_forward([t],teacher)[0][0])**2
        for lo,hi in zip(knots[:-1],knots[1:]):total+=(hi-lo)*(squared(lo)+4*squared((lo+hi)/2)+squared(hi))/6
        assert total==core.unpack(result['exact']);simpson[method]=float(total)
    assert simpson['alm64']<min(simpson[m] for m in simpson if m!='alm64')
    # Exact local quadratic identity: gradient is zero and Hessian is J^T J/n.
    jac=[[core.unpack(t) for t in row] for row in basin['jacobian']];rr=[core.unpack(t) for t in basin['residual']];radius=core.unpack(basin['radius_infinity'])
    assert all(sum(row[j]*r for row,r in zip(jac,rr))==0 for j in range(4)) and radius>0
    for c in basin['preactivation_constraints']:
        norm=sum(abs(core.unpack(t)) for t in c['coefficient']);assert radius*norm<core.unpack(c['margin'])
    for j in range(4):
        for k in range(4):assert core.unpack(basin['hessian_gram'][j][k])==sum(row[j]*row[k] for row in jac)/4
    extra=['audit_multiplier_fixed_point.py','prove_multiplier_escape_witness.py','analyze_multiplier_fixed_point.py','exact_escape_query_risk.py',
        'prove_escape_stationary_basin.py','plot_multiplier_fixed_point.py',Path(__file__).name]
    sources.update({n:sha(src/n) for n in extra});reports={};links=0
    for name in ['245_multiplier_fixed_point_escape_protocol.md','246_multiplier_fixed_point_escape_results.md','247_cold_start_stagnation_switch_protocol.md']:
        path=docs/name;reports[name]=sha(path)
        for ref in re.findall(r'\]\(([^)]+)\)',path.read_text(encoding='utf-8')):
            if '://' in ref or ref.startswith('#'):continue
            assert (path.parent/ref).resolve().exists(),ref;links+=1
    assert reports['245_multiplier_fixed_point_escape_protocol.md']==read(base/'screen/protocol.json')['design_sha256']
    result=dict(passed=True,source_sha256=sources,frozen_sources=len(sources),report_sha256=reports,links=links,
        summaries_sha256={name:sha(base/name) for name in ['screen/summary.json','exact/summary.json','continuation/summary.json','audit/summary.json','witness/summary.json',
            'witness/query_risk.json','witness/stationary_basin.json','analysis/summary.json','analysis/figure_audit.json']},
        exact_simpson_risks=simpson,primal_fixed_point_checks=1460,interval_checks=3772,continuation_points=414,
        scope='Exact conditional local mechanism and old-development continuation; not cold-start online superiority or project completion')
    dump(target,result);print(json.dumps(dict(passed=True,frozen_sources=len(sources),reports=3,links=links,exact_simpson_risks=simpson)),flush=True)

if __name__=='__main__':main()
