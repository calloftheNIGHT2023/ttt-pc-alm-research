"""318 verify report table numbers and chart data after actual image QA."""
import argparse
from pathlib import Path
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/primal_stasis_escape';report=base/'report_v1';groups=read(base/'audit_v1/aggregate.json')
    rows=[[x.strip() for x in line.split('|')[1:-1]] for line in (report/'report.md').read_text(encoding='utf-8').splitlines()
          if line.startswith('| ') and not line.startswith('| 方法')]
    assert len(rows)==6;fields=0;families=['alm_keep','alm_reset','nodual']
    for i,f in enumerate(families):
        c=groups[f]['counts'];expected=[c.get(k,0) for k in ['cases','applicable','first_primal_exit_found','finite_exit_beyond_cap','primal_stasis_for_all_nonnegative_phases']]
        assert list(map(int,rows[i][1:]))==expected;fields+=5
        s=groups[f]['seconds'];expected=[s['prior_line_generation_core'],s['incremental_exact_escape'],s.get('floating_exit_validation',0)]
        assert rows[i+3][1:]==[f'{v:.6f}' for v in expected];fields+=3
        assert sum(groups[f]['exit_phase_histogram'].values())==c.get('first_primal_exit_found',0)
    save(out/'summary.json',dict(passed=True,table_numeric_fields=fields,chart_total_checks=3,
                                visual_qa_performed_by_agent=True,report_sha256=sha(report/'report.md'),
                                image_sha256=sha(report/'certified_first_exit.png'),source_sha256=sha(Path(__file__))))
    print(dict(passed=True,table_numeric_fields=fields,chart_total_checks=3),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
