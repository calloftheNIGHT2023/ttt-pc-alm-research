"""313-315 report table/plot values and success-event prefix verification."""
import argparse
from collections import Counter
from pathlib import Path
import numpy as np
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def run(root,out):
    base=root/'results/solver_policy_recurrence';source=base/'development_v1';report=base/'report_v1'
    rs=read(report/'summary.json')
    for n,digest in rs['outputs_sha256'].items():assert sha(report/n)==digest
    for n,digest in rs['input_summaries_sha256'].items():assert sha(base/n/'summary.json')==digest
    a=read(base/'analysis_v1/aggregate.json');c=read(base/'two_cycle_v1/aggregate.json')
    families=['alm_keep','alm_reset','nodual'];labels=['ALM保留乘子','ALM重置起点乘子','无乘子更新']
    tablelines=[line for line in (report/'report.md').read_text(encoding='utf-8').splitlines()
                if any(line.startswith('| '+name+' |') for name in labels)]
    assert len(tablelines)==6;checks=Counter()
    for i,(f,name) in enumerate(zip(families,labels)):
        cells=[t.strip() for t in tablelines[i].strip('|').split('|')]
        assert cells==[name,str(a[f]['full']['segments']),f"{a[f]['full']['singleton_fraction']:.2%}",
            f"{a[f]['full']['thresholds']['4']['trajectories_with_segment']}/131",
            f"{a[f]['full']['thresholds']['4']['steps_in_segments']}/8384",
            f"{a[f]['success_events']['events_with_completed_segment_at_least_4']}/{a[f]['success_events']['count']}"]
        cells=[t.strip() for t in tablelines[i+3].strip('|').split('|')]
        assert cells==[name,f"{c[f]['full']['trajectories_with_cycle']}/131",str(c[f]['full']['segments']),
            f"{c[f]['full']['covered_steps']}/8384",str(c[f]['full']['maximum_length']),
            f"{c[f]['full']['success_events_strictly_before']['with_cycle']}/{c[f]['full']['success_events_strictly_before']['total']}"]
        checks['table_numeric_cells']+=10
    schema=read(source/'policy_schema.json')
    rows={(r['seed'],r['family']):r for r in read(source/'rows.json')}
    actual_events=read(base/'analysis_v1/events.json');prefixes={f:[] for f in families}
    for e in actual_events:
        row=rows[e['seed'],e['family']]
        with np.load(source/row['file'],allow_pickle=False) as z:
            i=next(i for i,loc in enumerate(z['locations']) if loc.tolist()==e['location'])
            full=np.concatenate([z['policy_'+g][:,i] for g in schema],axis=1)
            k=e['first_step'];start=k-1
            while start>0 and np.array_equal(full[start-1],full[start]):start-=1
            assert e['policy_start']==start+1 and e['policy_age_at_arrival']==k-start
            # Independently enumerate completed runs strictly before current run.
            lens=[];last=None
            for value in full[:start]:
                value=tuple(value)
                if value!=last:lens.append(1)
                else:lens[-1]+=1
                last=value
            maximum=max(lens,default=0)
            assert e['longest_completed_policy_segment_before_arrival']==maximum
            assert e['immediately_preceding_completed_segment_length']==(lens[-1] if lens else 0)
            prefixes[e['family']].append(maximum);checks['event_prefix_fields']+=4
    for f in families:
        assert sum(v>=4 for v in prefixes[f])==a[f]['success_events']['events_with_completed_segment_at_least_4']
        checks['success_plot_counts']+=1
    plot={f:{kind:100*a[f][kind]['thresholds']['4']['steps_in_segments']/8384 for kind in ['full','forward']} for f in families}
    save(out/'plot_values.json',plot)
    final=dict(passed=True,checks=dict(checks),source_sha256=sha(Path(__file__)),report_summary_sha256=sha(report/'summary.json'),
        outputs_sha256={'plot_values.json':sha(out/'plot_values.json')},query_targets_accessed=False,
        independent_task_gain_established=False)
    save(out/'summary.json',final);print(final,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
