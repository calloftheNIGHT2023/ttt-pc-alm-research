"""Final 312 comparison aggregates and displayed table verification."""
import argparse
from pathlib import Path
import re
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/paired_primal_dual_predictor';geo=base/'geometry_v1';report=base/'report_v1'
    tasks,states=read(geo/'tasks.json'),read(geo/'states.json')
    checks=0;controls=['normal_one','normal_two','bias_only','dual_only','dual_full','paired_b_zero_u',
                      'paired_b_last_du','paired_b_random_du','random_db_paired_u']
    for row in tasks+states:
        data=read(geo/next(t['file'] for t in tasks if t['seed']==row['seed']))
        p={n:set(r['positive_modes']) for n,r in row['methods'].items()}
        old=set(data['original_positive_modes']);prior=set(data['all_308_309_positive_modes']);main=p['paired_bu']
        expected={f'paired_new_vs_original_and_{n}':main-old-p[n] for n in controls}
        expected['paired_new_vs_original_and_single_blocks']=main-old-p['bias_only']-p['dual_only']-p['dual_full']
        expected['paired_new_vs_original_and_all_308_309']=main-prior
        expected['paired_new_vs_original_and_controls']=main-old-set().union(*(p[n] for n in controls))
        expected['paired_new_vs_original_and_prior_and_controls']=main-prior-set().union(*(p[n] for n in controls))
        assert set(expected)==set(row['comparisons'])
        for n,value in expected.items():assert sorted(value)==row['comparisons'][n];checks+=1
    aggregate=read(geo/'comparisons.json')
    for name,a in aggregate.items():
        assert a['tasks']==sum(bool(t['comparisons'][name]) for t in tasks)
        assert a['pairs']==sum(len(t['comparisons'][name]) for t in tasks)
        assert a['state_mode_occurrences']==sum(len(s['comparisons'][name]) for s in states)
        checks+=3
    text=(report/'report.md').read_text(encoding='utf-8')
    values=[tuple(map(int,m)) for m in re.findall(r'^\| [^|]+ \| (\d+) \| (\d+) \| (\d+) \|$',text,re.M)]
    agg=read(geo/'aggregate.json')
    assert values==[(a['tasks_with_new_positive'],a['task_positive_pairs'],a['new_pairs_vs_all_308_309']) for a in agg.values()]
    assert len(values)==12
    diag=read(base/'audit_v1/state_diagnostics.json')
    assert sum(d['db_max']==0 for d in diag)==57 and sum(d['db_max']<=1e-12 for d in diag)==89
    assert '131处有57处' in text and '89处最大参数增量≤1e-12' in text
    final=dict(passed=True,comparison_set_and_aggregate_checks=checks,table_numeric_cells=36,
        displayed_parameter_change_counts_checked=True,query_targets_accessed=False,
        source_sha256=sha(Path(__file__)),report_summary_sha256=sha(report/'summary.json'),
        independent_task_gain_established=False)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
