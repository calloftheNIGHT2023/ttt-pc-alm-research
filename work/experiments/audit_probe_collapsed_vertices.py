"""Independent collapsed-mesh checks and an analytic triangle-times-box volume.

Uses the previous independent Gauss-Jordan implementation, not the producer's
determinant, vertex enumerator, orientation solver or face-lattice volume.
The analytic volume proof is specific to this diagnosed 12-vertex polytope.
"""
import argparse
from collections import defaultdict
from copy import deepcopy
from fractions import Fraction as Q
from itertools import combinations, product
import json
from pathlib import Path
import numpy as np
from audit_probe_exact_polytope_certificate import gauss, independent_constraints
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def verify_mesh(poly, rows, rhs, result, witness, vertices, volume):
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    recorded=[tuple(Q(t) for t in v) for v in witness['vertices']]
    assert set(recorded)==set(vertices) and len(recorded)==len(vertices)
    assert witness['rows']==[list(t) for t in rows] and [Q(t) for t in witness['rhs']]==rhs
    assert Q(witness['reference_volume'])==Q(result['exact_reference_volume'])==volume>0
    assert result['reference_volume']==float(volume)
    raw,indices=np.unique(poly['facets'].reshape(-1,4),axis=0,return_inverse=True)
    simplices=indices.reshape(-1,4).tolist();matching=witness['matching']
    assert len(matching)==len(raw) and all(type(t) is int for t in matching)
    assert set(matching)==set(range(len(recorded)))
    assert result['saved_vertices']==len(raw) and result['exact_vertices']==len(recorded)
    assert result['collapsed_saved_vertices']==len(raw)-len(recorded)
    assert result['vertex_mapping_bijective']==(len(raw)==len(recorded))
    if len(raw)>len(recorded):
        assert witness['mapping_type']=='surjective_vertex_collapse'
        assert witness['raw_vertex_fibers']==[[i for i,t in enumerate(matching) if t==j] for j in range(len(recorded))]
    origin=tuple(Q(t) for t in witness['origin'])
    expected=tuple(Q(float(c))+Q(float(s))*Q(float(y)) for c,s,y in zip(poly['center'],poly['scale'],poly['interior']))
    assert origin==expected and all(dot(a,origin)<b for a,b in zip(rows,rhs))
    assert len({tuple(sorted(s)) for s in simplices})==len(simplices)
    raw_chain=defaultdict(int);mapped_chain=defaultdict(int);adjacency=defaultdict(list);volumes=[]
    assert len(witness['signs'])==len(witness['cone_volumes'])==len(witness['supporting_constraints'])==len(simplices)
    for index,simplex in enumerate(simplices):
        sign=witness['signs'][index];assert sign in [-1,1] and len(set(simplex))==4
        points=[recorded[matching[i]] for i in simplex]
        cone=sign*gauss([[x-y for x,y in zip(v,origin)] for v in points])[0]/24
        assert cone>=0 and cone==Q(witness['cone_volumes'][index]);volumes.append(cone)
        if cone:
            constraint=witness['supporting_constraints'][index]
            assert type(constraint) is int and 0<=constraint<len(rows)
            assert all(dot(rows[constraint],v)==rhs[constraint] for v in points)
        else:
            assert witness['supporting_constraints'][index] is None
            edges=[[x-y for x,y in zip(v,points[0])] for v in points[1:]]
            assert all(gauss([[edge[j] for j in cols] for edge in edges])[0]==0 for cols in combinations(range(4),3))
        for omitted in range(4):
            ridge=[simplex[i] for i in range(4) if i!=omitted]
            parity=(-1)**sum(ridge[i]>ridge[j] for i in range(3) for j in range(i+1,3))
            key=tuple(sorted(ridge));raw_chain[key]+=sign*(-1)**omitted*parity;adjacency[key].append(index)
            mapped=[matching[i] for i in ridge]
            if len(set(mapped))==3:
                parity=(-1)**sum(mapped[i]>mapped[j] for i in range(3) for j in range(i+1,3))
                mapped_chain[tuple(sorted(mapped))]+=sign*(-1)**omitted*parity
    assert all(v==0 for v in raw_chain.values()) and all(v==0 for v in mapped_chain.values())
    assert all(len(t)==2 for t in adjacency.values())
    neighbors=defaultdict(set)
    for a,b in adjacency.values():neighbors[a].add(b);neighbors[b].add(a)
    reached={0};front=[0]
    while front:
        for child in neighbors[front.pop()]:
            if child not in reached:reached.add(child);front.append(child)
    assert len(reached)==len(simplices) and sum(volumes)==volume
    physical=[tuple(Q(float(c))+Q(float(s))*Q(float(t)) for c,s,t in zip(poly['center'],poly['scale'],row)) for row in raw]
    shifts=[max(abs(point[j]-recorded[matching[i]][j]) for i,point in enumerate(physical)) for j in range(4)]
    assert [Q(t) for t in result['exact_maximum_coordinate_displacements']]==shifts
    weights=[Q(float(t)) for t in poly['simplex_probs']];assert len(weights)==len(volumes) and min(weights)>=0
    total=sum(weights);assert total>0;weights=[t/total for t in weights]
    tv=sum(abs(p-v/volume) for p,v in zip(weights,volumes))/2
    bound=tv+sum(Q(s)*d for s,d in zip([16,8,4,2],shifts))
    assert Q(result['exact_weight_total_variation'])==tv
    assert Q(result['exact_nominal_readout_expectation_error_bound'])==bound
    assert result['nominal_readout_expectation_error_bound']==float(bound)
    assert result['mapped_zero_cones']==sum(v==0 for v in volumes)
    return dict(passed=True,saved_vertices=len(raw),exact_vertices=len(recorded),collapsed=len(raw)-len(recorded),
        cones=len(volumes),exact_zero_cones=sum(v==0 for v in volumes),exact_volume=str(volume),nominal_mean_bound=float(bound))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve();src=Path(__file__).parent;base=root/'results/probe_credit_confirmation'
    inp=base/'collapsed_vertex_diagnosis_v1';out=base/'collapsed_vertex_independent_audit_v1';assert not out.exists()
    prior=json.loads((inp/'summary.json').read_text());assert prior['diagnosis_complete'] and not prior['query_targets_accessed']
    for name,digest in prior['outputs_sha256'].items():assert sha(inp/name)==digest
    for name,digest in prior['source_sha256'].items():assert sha(src/name)==digest
    assert sha(base/prior['geometry_file'])==prior['geometry_sha256']
    with np.load(base/prior['geometry_file']) as saved:poly={k:saved[k].copy() for k in saved.files}
    result=json.loads((inp/'certificate.json').read_text());witness=json.loads((inp/'witness.json').read_text())
    pred=base/'predictions';commit=json.loads((pred/'tasks/5500005/commit.json').read_text())
    first=json.loads((pred/commit['rows_file']).read_text())[0];assert sha(pred/first['file'])==first['sha256']
    with np.load(pred/first['file']) as saved:x=saved['x_observed'].copy();v=saved['v_observed'].copy()
    pattern=np.frombuffer(bytes.fromhex(prior['pattern']),dtype=np.uint8).reshape(4,4)
    constraints=independent_constraints(x,v,pattern);rows=[a for a,b in constraints];rhs=[b for a,b in constraints]
    dot=lambda a,b:sum(x*y for x,y in zip(a,b))
    vertices=set();tested=0
    for indices in combinations(range(len(rows)),4):
        tested+=1;det,point=gauss([rows[i] for i in indices],[rhs[i] for i in indices])
        if det and all(dot(a,point)<=b for a,b in constraints):vertices.add(point)
    assert len(vertices)==12
    # In coordinates M*b, the seven active constraints form two intervals
    # and a triangle. Check its exact twelve vertices against EVERY original
    # constraint: active constraints give P subset Q, these checks give Q subset P.
    assert rows[22]==tuple(-t for t in rows[2]) and rows[24]==tuple(-t for t in rows[0])
    assert rows[23]==tuple(-2*a-b for a,b in zip(rows[4],rows[12]))
    matrix=[rows[i] for i in [0,2,4,12]];jac=abs(gauss(matrix)[0]);assert jac>0
    width1=rhs[0]+rhs[24];width2=rhs[2]+rhs[22];height=rhs[23]+2*rhs[4]+rhs[12]
    assert width1>0 and width2>0 and height>0
    analytic_vertices=set()
    for a,b in product([-rhs[24],rhs[0]],[-rhs[22],rhs[2]]):
        for u,w in [(Q(0),Q(0)),(height/2,Q(0)),(Q(0),height)]:
            _,point=gauss(matrix,[a,b,rhs[4]-u,rhs[12]-w])
            assert all(dot(row,point)<=value for row,value in constraints)
            analytic_vertices.add(point)
    assert analytic_vertices==vertices
    volume=width1*width2*height**2/(4*jac)
    checked=verify_mesh(poly,rows,rhs,result,witness,vertices,volume)
    rejected={}
    mutations=[
        ('wrong_vertex_fiber',lambda r,w:w['raw_vertex_fibers'][0].append(15)),
        ('missing_exact_vertex',lambda r,w:w['matching'].__setitem__(15,0)),
        ('wrong_matching',lambda r,w:w['matching'].__setitem__(0,11)),
        ('wrong_orientation',lambda r,w:w['signs'].__setitem__(0,-w['signs'][0])),
        ('wrong_volume',lambda r,w:w.update(reference_volume='1')),
        ('wrong_cone_volume',lambda r,w:w['cone_volumes'].__setitem__(0,'1')),
        ('forged_error',lambda r,w:r.update(exact_nominal_readout_expectation_error_bound='0')),
        ('hidden_collapse',lambda r,w:r.update(collapsed_saved_vertices=0)),
        ('forged_bijection',lambda r,w:r.update(vertex_mapping_bijective=True)),
    ]
    for name,mutate in mutations:
        bad_result,bad_witness=deepcopy(result),deepcopy(witness);mutate(bad_result,bad_witness)
        try:verify_mesh(poly,rows,rhs,bad_result,bad_witness,vertices,volume)
        except (AssertionError,IndexError,KeyError):rejected[name]=dict(rejected=True)
        else:raise AssertionError('Corruption accepted: '+name)
    out.mkdir();exclusive_json(out/'corruptions.json',rejected)
    answer=dict(passed=True,query_targets_accessed=False,audit_gate_passed=False,independent_check=checked,
        equation_subsets_checked=tested,analytic_volume=str(volume),corruption_rejections=len(rejected),
        diagnosis_summary_sha256=sha(inp/'summary.json'),observed_input_prediction_sha256=first['sha256'],
        source_sha256={n:sha(src/n) for n in [Path(__file__).name,'audit_probe_exact_polytope_certificate.py']},
        scope='One diagnosed geometry: independent input constraints, rational vertex enumeration, analytic product volume, boundary chains and mean-error bound',
        outputs_sha256={'corruptions.json':sha(out/'corruptions.json')})
    exclusive_json(out/'summary.json',answer);print(json.dumps(answer),flush=True)


if __name__=='__main__':main()
