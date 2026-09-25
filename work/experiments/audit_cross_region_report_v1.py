"""Independent 359 table/figure values versus frozen scoring and component audit."""
from pathlib import Path
from report_search_radius_development_v1 import read,sha,save

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];base=root/'results/cross_region_online';out=base/'report_v1'
    tables=read(out/'table_data.json');fig=read(out/'figure_data.json');report=(out/'report.md').read_text(encoding='utf-8')
    assert all(ord(c)>=32 or c in '\n\r\t' for c in report)
    by={m['method']:m for m in read(base/'development_evaluation_v1/methods.json')}
    comps={(r['candidate'],r['control'],r['metric']):r for r in read(base/'development_evaluation_v1/comparisons.json')}
    mechanisms=read(base/'development_audit_v1/mechanisms.json');primary='cross_dual_reuse_g8';cells=0;values=0
    for row in tables['current']:
        name=row[0];m=by[name];c=None if name==primary else comps[primary,name,'mse257'];mm=[r for r in mechanisms if r['method']==name]
        expected=[name,f"{m['metrics']['mse257']:.10f}",f"{m['mean_current_seconds']:.6f}",str(m['failures']),
            str(sum(r['geometry_calls'] for r in mm)) if mm else '—','—' if c is None else f"{c['mean_difference']:+.10f}",
            '—' if c is None else f"{c['improved']}/{c['equal']}/{c['worse']}"]
        assert row==expected
    for row in tables['references']:
        name=row[0];c=comps[primary,name,'mse257']
        assert row==[name,f"{by[name]['metrics']['mse257']:.10f}",f"{c['mean_difference']:+.10f}",
                     f"{c['improved']}/{c['equal']}/{c['worse']}",f"{c['worst_leave_one_out_mean']:+.10f}"]
    component=read(root/'results/cross_region_credit/audit_v1/summary.json')['totals']
    for row in tables['component']:
        name=row[0];a=component[name+'_independent'];b=component[name+'_reuse']
        assert row==[name,a['rejected'],b['rejected'],b['additional_rejections'],b['transfer_rejections'],a['total_response_pairs'],b['total_response_pairs'],
                     f"{a['seconds']:.6f}",f"{b['seconds']:.6f}"]
    for group in tables.values():
        for row in group:assert '| '+' | '.join(map(str,row))+' |' in report;cells+=len(row)
    for row in fig['methods']:
        assert row['mse']==by[row['method']]['metrics']['mse257'] and row['seconds']==by[row['method']]['mean_current_seconds'];values+=2
    for c,pairs in fig['component'].items():
        for suffix,n in pairs.items():assert n==component[c+'_'+suffix]['rejected'];values+=1
    manifest=read(out/'manifest.json')
    for n,h in manifest['outputs_sha256'].items():assert sha(out/n)==h
    result=dict(passed=True,table_cells=cells,figure_data_values=values,manifest_sha256=sha(out/'manifest.json'),
        evaluation_summary_sha256=sha(base/'development_evaluation_v1/summary.json'))
    save(out/'qa_numeric.json',result);print(result,flush=True)
