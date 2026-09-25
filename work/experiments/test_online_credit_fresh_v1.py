"""328 unit gates before full old-task serialization preflight."""
from pathlib import Path
from fractions import Fraction as F
from itertools import product
import math
import traceback
import numpy as np
import online_credit_fresh_suite_v1 as suite
import run_online_credit_fresh_v1 as pipeline
import evaluate_online_credit_fresh_v1 as evaluation
import probe_confirmation_statistics as statistics
from evaluate_complete_credit_mode_geometry_v1 import read,sha


def exact_readout_identity():
    # Exact finite posterior, independent true parameter and two readout draws.
    prior=[F(1,5),F(3,10),F(1,10),F(2,5)];q=[F(1,3),F(2,3)]
    functions=[[F(0),F(1,4)],[F(1),F(3,4)],[F(1,2),F(1)],[F(3,4),F(0)]]
    def moments(pool):
        mass=sum(prior[i] for i in pool);weights={i:prior[i]/mass for i in pool}
        mean=[sum(weights[i]*functions[i][j] for i in pool) for j in range(2)]
        var=sum(q[j]*sum(weights[i]*(functions[i][j]-mean[j])**2 for i in pool) for j in range(2))
        return mass,weights,mean,var
    _,_,full,bayes=moments(range(4))
    def actual(pool):
        _,weights,_,_=moments(pool)
        return sum(prior[t]*weights[i]*weights[k]*q[j]*((functions[i][j]+functions[k][j])/2-functions[t][j])**2
                   for t in range(4) for i,k in product(pool,repeat=2) for j in range(2))
    old=[0,1];mass,_,mu,var=moments(old);oldrisk=actual(old)
    assert oldrisk==bayes+sum(q[j]*(mu[j]-full[j])**2 for j in range(2))+var/2
    signs=[]
    for added in [2,3]:
        new=old+[added];_,_,nm,nv=moments(new);rrisk=actual(new)
        assert rrisk==bayes+sum(q[j]*(nm[j]-full[j])**2 for j in range(2))+nv/2
        a=prior[added]/(mass+prior[added]);d=[functions[added][j]-mu[j] for j in range(2)]
        dd=sum(q[j]*d[j]**2 for j in range(2));inner=sum(q[j]*(mu[j]-full[j])*d[j] for j in range(2))
        assert nv==(1-a)*var+a*(1-a)*dd
        delta=2*a*inner+a*a*dd+(-a*var+a*(1-a)*dd)/2
        assert delta==rrisk-oldrisk;signs.append((delta>0)-(delta<0))
    assert signs==[1,-1]
    return dict(passed=True,exact_posterior_cases=3,added_region_identities=2,contains_improvement_and_counterexample=True)


def run(root,out):
    hashes=suite.gate(root);configs=suite.resources.catalogue(root);names=[c['name'] for c in configs]
    protocol=dict(source_sha256=hashes,design_sha256=sha(root/suite.DESIGN),query_targets_accessed=False)
    pipeline.exclusive(out/'protocol.json',protocol);checks={}
    assert len(configs)==51 and len(set(names))==51 and all(c in names for c in suite.CANDIDATES)
    assert not set(suite.PILOT_SEEDS)&(set(range(5910000,5910064))|set(range(294000000,294000128))|set(range(307000000,307000256)))
    q=np.linspace(0,1,257);x=np.array([.113,.227,.379,.683]);v=np.array([.1,.2,.3,.4])
    a=dict(x_observed=x,v_observed=v,q_observed=q,prediction=np.zeros(257),point_prediction=np.ones(257))
    pipeline.validate(a);signature=pipeline.signature(x,v,q);xx=x.copy();xx[0]+=.001;assert signature!=pipeline.signature(xx,v,q)
    errors=0
    for field,value in [('prediction',np.full(257,np.nan)),('prediction',np.full(257,1.01)),('point_prediction',np.zeros(129)),('x_observed',np.array([0.,.227,.379,.683]))]:
        try:pipeline.validate({**a,field:value})
        except AssertionError:errors+=1
    assert errors==4;checks['invalid_prediction_or_overlap_rejected']=errors
    meta=dict(a_bytes=8,nested=dict(b_bytes=16,excluded=True),notbytes='description')
    assert pipeline.byte_fields(meta)=={'a_bytes':8,'nested.b_bytes':16};checks['byte_field_extraction']=1
    risk=np.arange(3*51*4,dtype=float).reshape(3,51,4)/1000;pairs,deltas=evaluation.paired_comparisons(risk,names)
    for i,(candidate,control) in enumerate(pairs):assert np.array_equal(deltas[:,i],risk[:,names.index(candidate)]-risk[:,names.index(control)])
    checks['paired_comparison_columns']=100
    c=evaluation.concentration(np.array([-.375,0.,0.]),[1,2,3]);assert c['largest_absolute_share']==1 and c['leave_largest_absolute_out_mean']==c['worst_leave_one_out_mean']==0.
    c=evaluation.concentration(np.zeros(3),[1,2,3]);assert c['equal']==3 and c['largest_absolute_share']==0.
    checks['concentration_cases']=2;checks['bootstrap_primitives']=statistics.selftest()
    checks['finite_particle_risk_identity']=exact_readout_identity()
    rows=[dict(method=name,seconds=(j+1)/100,execution_failed=False,named_byte_fields={'fake_bytes':8+j}) for j,name in enumerate(names) for _ in range(3)]
    costs=pipeline.cost_table(rows,names);assert len(costs['methods'])==51
    for j,row in enumerate(costs['methods']):assert math.isclose(row['mean_seconds'],(j+1)/100) and row['maximum_named_byte_fields']['fake_bytes']==8+j
    checks['cost_methods']=51
    # The complete old-task preflight separately exercises I/O, live calls and seal verification.
    assert suite.gate(root)==hashes
    result=dict(passed=True,checks=checks,query_targets_accessed=False,outputs_sha256={'protocol.json':sha(out/'protocol.json')})
    pipeline.exclusive(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];out=root/suite.BASE/'tests_v1';out.mkdir(parents=True,exist_ok=False)
    try:run(root,out)
    except Exception:pipeline.exclusive(out/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
