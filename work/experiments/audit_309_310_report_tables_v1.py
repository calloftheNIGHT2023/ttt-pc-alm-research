"""Independently parse delivered Markdown tables and the 310 plotted matrices."""
from pathlib import Path
from evaluate_complete_credit_mode_geometry_v1 import read,save,sha


def table(path):
    return [l.strip('|').split('|') for l in path.read_text(encoding='utf-8').splitlines() if l.startswith('|')]


def main():
    root=Path(__file__).resolve().parents[2]
    a=root/'results/layer_credit_interaction'
    agg=read(a/'geometry_v1/aggregate.json');comp=read(a/'geometry_v1/comparisons.json')
    rows=table(a/'report_v1/report.md');checks=0
    for name,r in agg.items():
        rr=[x for x in rows if x[0]==name];assert len(rr)==1
        expected=[str(r['tasks_with_new_positive']),str(r['task_positive_pairs']),str(r['unknown_pairs']),format(r['numeric_new_volume'],'.12g')]
        assert rr[0][1:]==expected;checks+=4
    fields=['new_vs_original','new_vs_original_and_own_continuous','new_vs_original_and_all_308','exclusive_vs_all_308_and_other_masks']
    for name,r in comp.items():
        rr=[x for x in rows if x[0]==name];assert len(rr)==1
        assert rr[0][1:]==[str(r[f]['pairs']) for f in fields];checks+=4
    save(a/'report_v1/table_audit_v1.json',dict(passed=True,rows=108,table_fields=checks,
        report_sha256=sha(a/'report_v1/report.md'),source_sha256=sha(Path(__file__))))
    b=root/'results/successful_transition_census'
    census=read(b/'development_v2/aggregate.json');audit=read(b/'block_audit_v1/aggregate.json')
    rr=table(b/'report_v1/report.md');fields_checked=0
    for name,r in census.items():
        row=[x for x in rr if x[0]==name and len(x)==7];assert len(row)==1
        steps=sorted(map(int,r['first_step_histogram']))
        expected=[str(r['tasks']),str(r['task_mode_pairs']),f'{steps[0]}–{steps[-1]}',
                  str(r['step_zero_pairs']),str(r['step_one_pairs']),str(r['first_parameters_in_support_band_pairs'])]
        assert row[0][1:]==expected;fields_checked+=6
    for name,r in audit.items():
        row=[x for x in rr if x[0]==name and len(x)==4];assert len(row)==1
        assert row[0][1:]==[str(r['events']),str(r['all_current_only_events']),str(r['old_b_any_new_positive_events'])]
        fields_checked+=3
    summary=read(b/'report_v1/summary.json');events=read(b/'block_audit_v1/events.json')
    for name,digest in summary['outputs_sha256'].items():assert sha(b/'report_v1'/name)==digest
    cells=0
    for i in range(8):
        mask=format(i,'03b')
        for j,family in enumerate(['alm_keep','alm_reset','nodual','pc']):
            chosen=[e for e in events if e['event']['family']==family]
            assert summary['event_totals'][j]==len(chosen)
            for field,key in [('counts_target','target_reached'),('counts_any_new','new_positive')]:
                count=sum(next(p for p in e['alternatives'] if p['mask']==mask)[key] for e in chosen)
                assert summary[field][i][j]==count;cells+=1
    result=dict(passed=True,table_rows=11,table_fields=fields_checked,heatmap_cells= cells,
        report_summary_sha256=sha(b/'report_v1/summary.json'),source_sha256=sha(Path(__file__)))
    save(b/'report_v1/table_audit_v1.json',result)
    print(dict(stage309_table_fields=checks,stage310=result))


if __name__=='__main__':main()
