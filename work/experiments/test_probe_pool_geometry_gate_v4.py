"""Verify the refined pool gate on all prior diagnostic tasks and bad inputs."""
import argparse
from copy import deepcopy
from fractions import Fraction as F
import json
from pathlib import Path
import time
import numpy as np
from probe_pool_geometry_gate_v3 import checked_component, pool_coupling
from run_probe_credit_confirmation_v2 import exclusive_json
from run_multiplier_fixed_point_screen import sha


def read(path): return json.loads(path.read_text(encoding='utf-8'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';out=base/'pool_geometry_gate_selftests_v4'
    assert not out.exists();started=time.perf_counter()
    prerequisites={};hashes={};tests={}
    old=base/'pool_geometry_gate_selftests_v3';ss=read(old/'summary.json')
    assert ss['passed'] and ss['tests']==67 and not ss['query_targets_accessed']
    assert sha(old/'tests.json')==ss['tests_sha256']
    hashes.update(ss['source_sha256']);prerequisites[str((old/'summary.json').relative_to(base))]=sha(old/'summary.json')
    diagnostic=base/'barycentric_position_diagnosis_v1';ds=read(diagnostic/'summary.json')
    assert ds['diagnosis_complete'] and ds['analytic_cases']==4 and ds['corruption_rejections']==9
    assert ds['all_original_budget_bounds_pass'] and not ds['query_targets_accessed']
    for name,h in ds['outputs_sha256'].items():assert sha(diagnostic/name)==h
    hashes.update(read(diagnostic/'protocol.json')['source_sha256'])
    prerequisites[str((diagnostic/'summary.json').relative_to(base))]=sha(diagnostic/'summary.json')
    for name in [Path(__file__).name,'probe_pool_geometry_gate_v3.py','probe_barycentric_position_bound.py']:hashes[name]=sha(src/name)
    for name,h in hashes.items():assert sha(src/name)==h
    geometry_counts={};pool_counts={}
    for folder in ['first_task_geometry_census_v2','task_5500005_geometry_census_v1','task_5500037_geometry_census_v1']:
        inp=base/folder;summary=read(inp/'summary.json')
        assert summary['diagnosis_complete'] and not summary['query_targets_accessed']
        for name,h in summary['outputs_sha256'].items():assert sha(inp/name)==h
        for name,h in read(inp/'files.json').items():assert sha(inp/name)==h
        prerequisites[str((inp/'summary.json').relative_to(base))]=sha(inp/'summary.json')
        pools=read(inp/'pools.json');cache={};certificates={}
        for key in sorted({key for row in pools for key in row.get('patterns',[])}):
            row=read(inp/'geometries'/f'{key}.json')
            with np.load(inp/row['geometry_file']) as z:poly={name:z[name].copy() for name in z.files}
            ww=row.get('witness') or row['matching_diagnosis']
            rows=ww.get('rows',ww.get('exact_rows'));rhs=[F(t) for t in ww.get('rhs',ww.get('exact_rhs'))]
            cert,witness=checked_component(poly,rows,rhs)
            if 'witness' in row:assert witness==row['witness']
            cache[key]=poly;certificates[key]=cert
        geometry_counts[folder]=len(cache);pool_counts[folder]=0
        for row in pools:
            if row.get('empty_pool'):continue
            result=pool_coupling(row['patterns'],cache,certificates)
            tests[folder+'/'+row['method']]=dict(passed=True,certificate=result)
            pool_counts[folder]+=1
        assert pool_counts[folder]==16
    # Structural validation stays mandatory, even when a forged refined error
    # would appear numerically small. Start from the final actual fixture.
    key=next(iter(cache));good=cache[key];row=read(inp/'geometries'/f'{key}.json')
    rr=row['witness']['rows'];bb=[F(t) for t in row['witness']['rhs']]
    def reject(name,change):
        bad=deepcopy(good);change(bad)
        try:checked_component(bad,rr,bb)
        except (AssertionError,ValueError):tests[name]=dict(passed=True,deliberate_corruption_rejected=True)
        else:raise AssertionError(name)
    reject('new_gate_missing_facet',lambda p:p.update(facets=p['facets'][:-1],simplex_probs=p['simplex_probs'][:-1]))
    reject('new_gate_false_volume',lambda p:p.update(volume=p['volume']*2))
    reject('new_gate_wrong_origin',lambda p:p.update(interior=np.full(4,100.)))
    reject('new_gate_negative_probability',lambda p:p.update(simplex_probs=-p['simplex_probs']))
    for name,h in hashes.items():assert sha(src/name)==h
    out.mkdir();exclusive_json(out/'tests.json',tests)
    result=dict(passed=True,tests=len(tests),pool_checks=sum(pool_counts.values()),geometry_checks=sum(geometry_counts.values()),
        geometry_counts=geometry_counts,structural_corruption_rejections=4,retained_old_gate_tests=67,
        barycentric_analytic_tests=4,barycentric_corruption_rejections=9,
        query_targets_accessed=False,audit_gate_passed=False,source_sha256=hashes,
        prerequisite_summary_sha256=prerequisites,barycentric_note_sha256=sha(root/'outputs/ttt-pc-alm-research/287_barycentric_position_certificate.md'),
        tests_sha256=sha(out/'tests.json'),seconds=time.perf_counter()-started)
    assert result['tests']==52 and result['geometry_checks']==34
    exclusive_json(out/'summary.json',result);print(json.dumps({k:v for k,v in result.items() if k!='source_sha256'}),flush=True)


if __name__=='__main__':main()
