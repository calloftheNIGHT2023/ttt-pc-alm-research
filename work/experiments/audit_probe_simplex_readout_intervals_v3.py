"""Independent scalar-line and tree verification for affine-envelope proofs."""
import argparse
from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
from audit_probe_simplex_readout_intervals import leaf_enclosure,clip_tent
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def bound_leaf(vertices,query,layers,leaf):
    from audit_probe_simplex_readout_intervals_v2 import bound_leaf as exact_envelope_check
    lower,upper=exact_envelope_check(vertices,query,layers,leaf)
    if not leaf['affine']:
        unit=2**64
        lower=Q((lower.numerator*unit)//lower.denominator,unit)
        upper=Q(-((-upper.numerator*unit)//upper.denominator),unit)
    assert lower<=upper
    return lower,upper


def verify(roots,coefficients,query,layers,record):
    coefficients=[Q(c) for c in coefficients];query=Q(query)
    assert record['enclosure_version']==3 and record['rounding_bits']==64 and 1<=layers<=4 and len(roots)==len(coefficients)
    assert Q(record['query_coordinate'])==query and record['layers']==layers
    assert [Q(c) for c in record['coefficients']]==coefficients
    nodes={(i,''):(tuple(tuple(Q(t) for t in p) for p in root),c)
           for i,(root,c) in enumerate(zip(roots,coefficients)) if c}
    assert all(len(v)==5 and all(len(p)==4 for p in v) for v,_ in nodes.values())
    for operation in record['split_operations']:
        key=(operation['root'],operation['path']);vertices,mass=nodes.pop(key)
        i,j=operation['edge'];assert type(i) is int and type(j) is int and 0<=i<j<5
        midpoint=tuple((a+b)/2 for a,b in zip(vertices[i],vertices[j]))
        first=list(vertices);first[i]=midpoint;second=list(vertices);second[j]=midpoint
        nodes[(key[0],key[1]+'0')]=(tuple(first),mass/2)
        nodes[(key[0],key[1]+'1')]=(tuple(second),mass/2)
    seen=set();lower=Q(0);upper=Q(0);affine_count=0;line_checks=0
    for leaf in record['leaves']:
        key=(leaf['root'],leaf['path']);assert key not in seen;seen.add(key)
        vertices,mass=nodes[key];assert Q(leaf['mass'])==mass
        lo,hi=bound_leaf(vertices,query,layers,leaf)
        assert Q(leaf['low'])==lo and Q(leaf['high'])==hi and 0<=lo<=hi<=1
        a,b=mass*lo,mass*hi;lower+=min(a,b);upper+=max(a,b)
        affine_count+=int(leaf['affine']);line_checks+=2*len(leaf['envelope'])
    assert seen==set(nodes)
    assert sum((mass for _,mass in nodes.values()),Q(0))==sum(coefficients,Q(0))
    assert record['splits']==len(record['split_operations'])<=record['max_splits']
    assert record['leaf_count']==len(nodes) and record['affine_leaves']==affine_count
    assert Q(record['exact_lower'])==lower and Q(record['exact_upper'])==upper
    absolute=max(abs(lower),abs(upper));assert Q(record['exact_absolute_upper_bound'])==absolute
    assert record['lower']==float(lower) and record['upper']==float(upper)
    assert record['absolute_upper_bound']==float(absolute)
    target=None if record['target'] is None else Q(record['target'])
    width=None if record['max_width'] is None else Q(record['max_width'])
    assert target is None or target>=0
    assert width is None or width>=0
    status=record['status']
    if status=='upper_bound_met':assert target is not None and absolute<=target
    elif status=='lower_bound_exceeds_target':assert target is not None and (lower>target or upper<-target)
    elif status=='width_met':assert width is not None and upper-lower<=width
    elif status=='exact_integral':assert lower==upper
    elif status=='work_limit_unresolved':assert record['splits']==record['max_splits']
    else:raise AssertionError('Unknown status')
    return dict(passed=True,status=status,leaves=len(nodes),splits=record['splits'],
        exact_lower=str(lower),exact_upper=str(upper),exact_absolute_upper_bound=str(absolute),
        independently_checked_scalar_lines=line_checks)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    out=root/'results/probe_credit_confirmation/simplex_interval_selftests_v3';assert not out.exists()
    from probe_simplex_readout_intervals_v3 import integrate
    vertices=tuple([tuple(Q(0) for _ in range(4))]+[tuple(Q(i==j) for j in range(4)) for i in range(4)])
    small=tuple(tuple(t/1024 for t in p) for p in vertices);point=tuple([tuple(Q(0) for _ in range(4))]*5)
    cases={
        'affine':([small],[Q(1)],Q(1,128),4,dict(max_width=Q(0)),Q(16,128)+Q(30,5120)),
        'constant_zero':([point],[Q(1)],Q(-2),1,dict(max_width=Q(0)),Q(0)),
        'clipped_peak':([point],[Q(1)],Q(1,2),1,dict(max_width=Q(0)),Q(1)),
        'unshifted_tent':([vertices],[Q(1)],Q(0),1,dict(max_width=Q(1,1000)),Q(3,8)),
        'crossing_both_knots':([vertices],[Q(1)],Q(-1,4),1,dict(max_width=Q(1,1000)),Q(241,2560)),
        'tail_hinge':([vertices],[Q(1)],Q(-3,4),1,dict(max_width=Q(1,1000)),Q(1,2560)),
        'signed':([small,small],[Q(3,7),Q(-3,7)],Q(1,128),4,dict(target=Q(0)),Q(0)),
        'exceedance':([small],[Q(1)],Q(1,128),4,dict(target=Q(1,1000)),Q(16,128)+Q(30,5120)),
        'work_limit':([vertices],[Q(1)],Q(-1,4),1,dict(max_width=Q(1,1000),max_splits=0),Q(241,2560)),
    }
    checked={};records={}
    for name,(rr,cc,query,layers,options,truth) in cases.items():
        record=integrate(query,rr,cc,layers=layers,**options)
        check=verify(rr,cc,query,layers,record)
        assert Q(record['exact_lower'])<=truth<=Q(record['exact_upper'])
        if name=='work_limit':assert record['status']=='work_limit_unresolved'
        elif name=='exceedance':assert record['status']=='lower_bound_exceeds_target'
        else:assert record['status'] in ['width_met','upper_bound_met']
        checked[name]=dict(**check,exact_known_integral=str(truth));records[name]=record
    rejected={}
    def reject(name,case,mutate):
        bad=deepcopy(records[case]);mutate(bad);rr,cc,query,layers,_,_=cases[case]
        try:verify(rr,cc,query,layers,bad)
        except (AssertionError,KeyError,ValueError,TypeError,IndexError):
            rejected[name]=dict(passed=True,rejected=True);return
        raise AssertionError('Invalid proof accepted: '+name)
    case='crossing_both_knots'
    reject('missing_leaf',case,lambda r:r['leaves'].pop())
    reject('duplicate_leaf',case,lambda r:r['leaves'].append(deepcopy(r['leaves'][0])))
    reject('wrong_mass',case,lambda r:r['leaves'][0].update(mass='-1'))
    reject('invalid_edge',case,lambda r:r['split_operations'][0].update(edge=[0,0]))
    reject('wrong_parent',case,lambda r:r['split_operations'][0].update(path='111'))
    reject('wrong_upper',case,lambda r:r.update(exact_upper='0'))
    reject('wrong_query',case,lambda r:r.update(query_coordinate='0'))
    reject('false_success','work_limit',lambda r:r.update(status='upper_bound_met',target='0'))
    reject('wrong_sign','signed',lambda r:r.update(coefficients=['3/7','3/7']))
    reject('false_scalar_upper','work_limit',lambda r:r['leaves'][0]['envelope'][0].update(upper_line=['0','0']))
    reject('false_scalar_lower','work_limit',lambda r:r['leaves'][0]['envelope'][0].update(lower_line=['0','1']))
    reject('false_scalar_domain','work_limit',lambda r:r['leaves'][0]['envelope'][0].update(exact_z_upper='0'))
    reject('missing_envelope_stage','work_limit',lambda r:r['leaves'][0].update(envelope=[]))
    reject('false_affine_flag','work_limit',lambda r:r['leaves'][0].update(affine=True))
    reject('wrong_rounding_bits','work_limit',lambda r:r.update(rounding_bits=63))
    reject('inward_endpoint','work_limit',lambda r:r['leaves'][0].update(high=str(Q(r['leaves'][0]['high'])-Q(1,2**64))))
    out.mkdir();exclusive_json(out/'checked.json',checked);exclusive_json(out/'records.json',records);exclusive_json(out/'corruptions.json',rejected)
    names=[Path(__file__).name,'probe_simplex_readout_intervals_v3.py','audit_probe_simplex_readout_intervals_v2.py','audit_probe_simplex_readout_intervals.py','probe_simplex_readout_intervals.py']
    result=dict(passed=True,analytic_independent_replays=len(checked),corruption_rejections=len(rejected),
        query_targets_accessed=False,audit_gate_passed=False,
        source_sha256={n:sha(src/n) for n in names},
        outputs_sha256={n:sha(out/n) for n in ['checked.json','records.json','corruptions.json']})
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
