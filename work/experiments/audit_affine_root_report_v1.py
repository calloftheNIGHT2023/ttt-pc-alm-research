"""Independent 316 table and invariant-coordinate certificate verification."""
import argparse
from collections import Counter
from fractions import Fraction as F
from pathlib import Path
import re
from effective_affine_map_v1 import restore
from multiplier_fixed_point_exact import rref_solve
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    source=root/'results/affine_root_proposal/development_v1';report=root/'results/affine_root_proposal/report_v1'
    aggregate=read(root/'results/affine_root_proposal/audit_v1/aggregate.json')
    diagnosis=read(report/'invariant_coordinate_diagnosis.json');counts=Counter()
    for family,group in diagnosis['families'].items():
        for witness in group['invariant_witnesses']:
            item=read(source/witness['file']);s=item['solution'];rows=restore(item['formula']);ids=s['active']
            a=[[F(i==j)-rows[i].terms.get(j,F(0)) for j in ids] for i in ids]
            rhs=[rows[i].terms.get(-1,F(0)) for i in ids]
            value,meta=rref_solve(a,rhs,[F(item['point'][j]) for j in ids]);assert value is not None
            col=ids.index(witness['coordinate']);pivot=next(p for p in meta['parameterization'] if p['pivot']==col)
            assert all(int(p[0])==0 for p in pivot['free_coefficients'])
            constant=F(int(pivot['constant'][0]),int(pivot['constant'][1]))
            assert constant==F(witness['value'])==value[col]
            assert not F(witness['lower'])<=constant<=F(witness['upper'])
            counts['independent_invariant_coordinate_certificates']+=1
    text=(report/'report.md').read_text(encoding='utf-8')
    rows=[line.split('|')[1:-1] for line in text.splitlines() if line.startswith('| ') and not line.startswith('| 方法')]
    data=[[v.strip() for v in row] for row in rows]
    families=['alm_keep','alm_reset','nodual'];names=['保留乘子','重置乘子','无乘子']
    assert len(data)==9
    for i,f in enumerate(families):
        c=aggregate[f]['counts'];assert data[i][0]==names[i]
        expected=[c.get(k,0) for k in ['no_fixed_point_in_selected_formula','root_outside_domain',
                                     'root_wrong_actual_update','actual_fixed_point','fixed_support_feasible']]
        assert list(map(int,data[i][1:]))==expected;counts['report_table_numbers']+=5
        e=diagnosis['families'][f]['counts']
        assert list(map(int,data[3+i][1:]))==[e.get('all_roots_outside_domain_certified',0),e.get('outside_chosen_root_only_unresolved',0)]
        counts['report_table_numbers']+=2
        expected=[aggregate[f]['seconds']['current_local_sweep']/131,aggregate[f]['mean_total_core_seconds']]
        assert [f'{v:.6f}' for v in expected]==data[6+i][1:3]
        assert int(data[6+i][3])==aggregate[f]['max_rational_bits'];counts['report_table_numbers']+=3
    save(out/'summary.json',dict(passed=True,counts=dict(counts),report_sha256=sha(report/'report.md'),
                                image_sha256=sha(report/'root_or_drift.png'),visual_qa_performed_by_agent=True,
                                source_sha256=sha(Path(__file__))))
    print(dict(counts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    a.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],a.out)
