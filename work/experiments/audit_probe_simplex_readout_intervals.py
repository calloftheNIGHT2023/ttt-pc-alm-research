"""Independent exact replay of midpoint-simplex integral certificates.

The verifier does not import the generating enclosure or subdivision routines.
It represents affine stages explicitly as coefficients/offsets, and evaluates
nonlinear interval images by extrema at endpoints and included tent knots.
"""
import argparse
from copy import deepcopy
from fractions import Fraction as Q
import json
from pathlib import Path
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def clip_tent(value):
    return 2*max(Q(0),value)-4*max(Q(0),value-Q(1,2))+2*max(Q(0),value-1)


def interval_image(lower,upper):
    points=[lower,upper]+[k for k in [Q(0),Q(1,2),Q(1)] if lower<=k<=upper]
    values=[clip_tent(t) for t in points]
    return min(values),max(values)


def leaf_enclosure(vertices,query,layers):
    coefficients=[Q(0)]*4;offset=query
    for j in range(layers):
        coefficients[j]+=1
        values=[sum((a*b for a,b in zip(coefficients,p)),Q(0))+offset for p in vertices]
        lower,upper=min(values),max(values)
        if upper<=0 or lower>=1:slope,shift=Q(0),Q(0)
        elif 0<=lower and upper<=Q(1,2):slope,shift=Q(2),Q(0)
        elif Q(1,2)<=lower and upper<=1:slope,shift=Q(-2),Q(2)
        else:
            lower,upper=interval_image(lower,upper)
            for k in range(j+1,layers):
                lower,upper=interval_image(lower+min(p[k] for p in vertices),upper+max(p[k] for p in vertices))
            return lower,upper,False
        coefficients=[slope*a for a in coefficients];offset=slope*offset+shift
    centroid=[sum((p[j] for p in vertices),Q(0))/5 for j in range(4)]
    answer=sum((a*b for a,b in zip(coefficients,centroid)),Q(0))+offset
    return answer,answer,True


def verify(roots,coefficients,query,layers,record):
    coefficients=[Q(c) for c in coefficients];query=Q(query)
    assert len(roots)==len(coefficients) and 1<=layers<=4
    assert record['query_coordinate']==str(query) and record['layers']==layers
    assert [Q(c) for c in record['coefficients']]==coefficients
    nodes={(i,''):(tuple(tuple(Q(t) for t in p) for p in root),c)
           for i,(root,c) in enumerate(zip(roots,coefficients)) if c}
    assert all(len(v)==5 and all(len(p)==4 for p in v) for v,_ in nodes.values())
    for operation in record['split_operations']:
        key=(operation['root'],operation['path']);vertices,mass=nodes.pop(key)
        i,j=operation['edge'];assert type(i) is int and type(j) is int and 0<=i<j<5
        midpoint=tuple((a+b)/2 for a,b in zip(vertices[i],vertices[j]))
        first=list(vertices);first[i]=midpoint
        second=list(vertices);second[j]=midpoint
        nodes[(key[0],key[1]+'0')]=(tuple(first),mass/2)
        nodes[(key[0],key[1]+'1')]=(tuple(second),mass/2)
    seen=set();lower=Q(0);upper=Q(0);affine_count=0
    for leaf in record['leaves']:
        key=(leaf['root'],leaf['path']);assert key not in seen;seen.add(key)
        vertices,mass=nodes[key];assert Q(leaf['mass'])==mass
        lo,hi,affine=leaf_enclosure(vertices,query,layers)
        assert Q(leaf['low'])==lo and Q(leaf['high'])==hi and leaf['affine']==affine
        assert 0<=lo<=hi<=1
        first,last=mass*lo,mass*hi;lower+=min(first,last);upper+=max(first,last)
        affine_count+=int(affine)
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
                exact_lower=str(lower),exact_upper=str(upper),exact_absolute_upper_bound=str(absolute))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/'simplex_interval_selftests_v1'
    out=base/'simplex_interval_independent_selftests_v1';assert not out.exists()
    prior=json.loads((inp/'summary.json').read_text());assert prior['passed'] and prior['tests']==10
    for name,digest in prior['outputs_sha256'].items():assert sha(inp/name)==digest
    for name,digest in prior['source_sha256'].items():assert sha(src/name)==digest
    records=json.loads((inp/'records.json').read_text())
    vertices=tuple([tuple(Q(0) for _ in range(4))]+[tuple(Q(i==j) for j in range(4)) for i in range(4)])
    small=tuple(tuple(t/1024 for t in p) for p in vertices)
    point=tuple([tuple(Q(0) for _ in range(4))]*5)
    cases={
        'affine':([small],[Q(1)],Q(1,128),4),
        'constant_zero':([point],[Q(1)],Q(-2),1),
        'clipped_peak':([point],[Q(1)],Q(1,2),1),
        'unshifted_tent':([vertices],[Q(1)],Q(0),1),
        'crossing_both_knots':([vertices],[Q(1)],Q(-1,4),1),
        'tail_hinge':([vertices],[Q(1)],Q(-3,4),1),
        'signed':([small,small],[Q(3,7),Q(-3,7)],Q(1,128),4),
        'exceedance':([small],[Q(1)],Q(1,128),4),
        'work_limit':([vertices],[Q(1)],Q(0),1),
    }
    checked={name:verify(*args,records[name]) for name,args in cases.items()}
    corruptions={}
    def reject(name,record_name,mutate):
        bad=deepcopy(records[record_name]);mutate(bad)
        try:verify(*cases[record_name],bad)
        except (AssertionError,KeyError,ValueError,TypeError,IndexError):
            corruptions[name]=dict(passed=True,rejected=True);return
        raise AssertionError('Invalid certificate accepted: '+name)
    name='crossing_both_knots'
    reject('missing_leaf',name,lambda r:r['leaves'].pop())
    reject('duplicate_leaf',name,lambda r:r['leaves'].append(deepcopy(r['leaves'][0])))
    reject('incorrect_mass',name,lambda r:r['leaves'][0].update(mass='-1'))
    reject('invalid_split_edge',name,lambda r:r['split_operations'][0].update(edge=[0,0]))
    reject('nonexistent_split_parent',name,lambda r:r['split_operations'][0].update(path='111'))
    reject('understated_leaf_upper',name,lambda r:r['leaves'][0].update(high='-1'))
    reject('wrong_total_upper',name,lambda r:r.update(exact_upper='0'))
    reject('wrong_query',name,lambda r:r.update(query_coordinate='0'))
    reject('false_success','work_limit',lambda r:r.update(status='upper_bound_met',target='0'))
    reject('wrong_coefficient','signed',lambda r:r.update(coefficients=['3/7','3/7']))
    out.mkdir();exclusive_json(out/'checked.json',checked);exclusive_json(out/'corruptions.json',corruptions)
    result=dict(passed=True,independent_fixture_replays=len(checked),corruption_rejections=len(corruptions),
        query_targets_accessed=False,audit_gate_passed=False,
        source_sha256={n:sha(src/n) for n in [Path(__file__).name,'probe_simplex_readout_intervals.py']},
        producer_selftests_sha256=sha(inp/'summary.json'),
        outputs_sha256={n:sha(out/n) for n in ['checked.json','corruptions.json']})
    exclusive_json(out/'summary.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
