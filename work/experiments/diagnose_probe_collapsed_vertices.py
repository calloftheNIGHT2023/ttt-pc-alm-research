"""Reproduce the v4 sixth-task vertex-matching failure with no query targets."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path
import numpy as np
from probe_exact_polytope_certificate_v2 import certify
from probe_exact_polytope_certificate import dot, rank
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,required=True)
    root=parser.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';inp=base/'task_5500005_geometry_census_v1'
    out=base/'collapsed_vertex_diagnosis_v1';assert not out.exists()
    old=json.loads((inp/'summary.json').read_text());assert old['diagnosis_complete'] and not old['query_targets_accessed']
    for name,digest in old['outputs_sha256'].items():assert sha(inp/name)==digest
    for name,digest in json.loads((inp/'files.json').read_text()).items():assert sha(inp/name)==digest
    key='01020102010202010203020202010102'
    prior=json.loads((inp/'geometries'/str(key+'.json')).read_text());assert not prior['certificate_valid']
    diagnosis=prior['matching_diagnosis'];rows=diagnosis['exact_rows'];rhs=[F(t) for t in diagnosis['exact_rhs']]
    with np.load(inp/prior['geometry_file']) as saved:poly={k:saved[k].copy() for k in saved.files}
    result,witness=certify(poly,rows,rhs)
    vertices=[tuple(F(t) for t in row) for row in witness['vertices']]
    facets=[]
    for i,(row,value) in enumerate(zip(rows,rhs)):
        indices=[j for j,v in enumerate(vertices) if dot(row,v)==value]
        if rank([vertices[j] for j in indices])==3:facets.append(dict(constraint=i,row=row,rhs=str(value),vertices=indices))
    assert result['saved_vertices']==16 and result['exact_vertices']==12 and result['collapsed_saved_vertices']==4
    out.mkdir();exclusive_json(out/'certificate.json',result);exclusive_json(out/'witness.json',witness)
    exclusive_json(out/'active_facets.json',facets)
    summary=dict(diagnosis_complete=True,audit_gate_passed=False,query_targets_accessed=False,
        seed=5500005,pattern=key,prior_summary_sha256=sha(inp/'summary.json'),
        geometry_file=str((inp/prior['geometry_file']).relative_to(base)),geometry_sha256=sha(inp/prior['geometry_file']),
        collapsed_vertices=4,exact_vertices=12,active_facets=len(facets),
        nominal_readout_bound=result['nominal_readout_expectation_error_bound'],
        source_sha256={n:sha(src/n) for n in [Path(__file__).name,'probe_exact_polytope_certificate_v2.py','probe_exact_polytope_certificate.py']},
        outputs_sha256={n:sha(out/n) for n in ['certificate.json','witness.json','active_facets.json']})
    exclusive_json(out/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
