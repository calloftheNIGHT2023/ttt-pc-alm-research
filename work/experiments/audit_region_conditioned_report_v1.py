"""Independent report table values and plot data against sealed artifacts."""
from pathlib import Path
import numpy as np
from report_search_radius_development_v1 import read,sha,save

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];base=root/'results/region_conditioned_credit'
    out=base/'report_v1';report=(out/'report.md').read_text(encoding='utf-8')
    assert all(ord(c)>=32 or c in '\n\r\t' for c in report), 'Unexpected control character in report'
    a=read(base/'audit_v1/summary.json');tables=read(out/'table_data.json');fig=read(out/'figure_data.json')
    ch=['dual','dual_plus_residual','residual','bp','random_sign','zero'];p=[1,4,16,64,128]
    extra={c:[0]*5 for c in ch};exclusive={c:0 for c in ch}
    for task in read(base/'development_v1/tasks.json'):
        d=base/'development_v1'/str(task['seed']);ss={}
        for c in ch:
            with np.load(d/(c+'.npz'),allow_pickle=False) as z:ss[c]=z['first_step']
        with np.load(d/'controls.npz',allow_pickle=False) as z:mask=z['c20']
        for c in ch:
            for i,n in enumerate(p):extra[c][i]+=sum(0<int(s)<=n and not bool(v) for s,v in zip(ss[c],mask))
            exclusive[c]+=sum(int(ss[c][i])>0 and not bool(mask[i]) and all(int(ss[b][i])==0 for b in ch if b!=c) for i in range(len(mask)))
    expected=[]
    for c in ch:
        t=a['totals'][c]
        expected.append([c,*[t['prefix_'+str(n)] for n in p],t['beyond_c20'],t['response_pairs'],f"{t['seconds']:.6f}"])
        assert fig['coverage'][c]==[t['prefix_'+str(n)] for n in p]
        assert fig['beyond_c20_prefixes'][c]==extra[c]
        assert fig['exclusive_after_c20_and_other_credits'][c]==exclusive[c]
    assert tables['main']==expected
    assert tables['controls']==[[c,a['totals'][c]['rejected'],f"{a['totals'][c]['seconds']:.6f}"] for c in ['c5','c20','pdhg60']]
    assert tables['contrasts']==[[c,*[a['contrasts'][c][b] for b in ch],exclusive[c]] for c in ch]
    cells=0
    for group in tables.values():
        for row in group:
            assert '| '+' | '.join(map(str,row))+' |' in report;cells+=len(row)
    for c,seconds in fig['component_seconds'].items():assert seconds==a['totals'][c]['seconds']
    manifest=read(out/'manifest.json')
    for n,h in manifest['outputs_sha256'].items():assert sha(out/n)==h
    result=dict(passed=True,table_cells=cells,figure_values=75,manifest_sha256=sha(out/'manifest.json'),
        audit_summary_sha256=sha(base/'audit_v1/summary.json'),independent_posthoc_union_recalculation=True)
    save(out/'qa_numeric.json',result);print(result,flush=True)
