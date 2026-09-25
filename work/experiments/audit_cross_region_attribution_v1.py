"""Post-seal descriptive attribution: equal predictions versus no-credit control."""
from pathlib import Path
import numpy as np
from report_search_radius_development_v1 import read,sha,save,complete

if __name__=='__main__':
    root=Path(__file__).resolve().parents[2];base=root/'results/cross_region_online'
    complete(base/'development_audit_v1');pred=base/'development_predictions_v1'
    rows=read(pred/'rows.json');index={(r['seed'],r['method']):r for r in rows}
    names=['cross_'+c+'_reuse_g8' for c in ['dual','dual_plus_residual','residual','bp','random_sign','zero']]
    names+=['cross_dual_independent_g8','cross_zero_independent_g8']
    count=0;poolchecks=0;saved={n:0 for n in names}
    for seed in range(328000000,328000032):
        reference=index[seed,'cross_c20_g8'];rm=read(root/reference['metadata_file'])['metadata']
        with np.load(root/reference['file'],allow_pickle=False) as z:ref={k:z[k] for k in ['points','allocation','prediction']}
        for name in names:
            row=index[seed,name];m=read(root/row['metadata_file'])['metadata']
            assert not m['execution_failed'] and not rm['execution_failed']
            assert m['positive_modes']==rm['positive_modes'];poolchecks+=1
            with np.load(root/row['file'],allow_pickle=False) as z:
                for k in ref:
                    assert ref[k].dtype==z[k].dtype and ref[k].shape==z[k].shape and ref[k].tobytes()==z[k].tobytes();count+=1
            saved[name]+=len(rm['new_mode_classifications_detail'])-len(m['new_mode_classifications_detail'])
    out=base/'attribution_v1';out.mkdir(parents=True,exist_ok=False)
    result=dict(passed=True,posthoc_descriptive_attribution=True,bitwise_arrays=count,identical_positive_pools=poolchecks,
        methods=names,reference='cross_c20_g8',saved_geometry_calls_vs_reference=saved,
        source_sha256=sha(Path(__file__)),prediction_rows_sha256=sha(pred/'rows.json'),
        independent_audit_sha256=sha(base/'development_audit_v1/summary.json'),
        inference='On these 32 tasks, the no-credit same-budget control exactly reproduces all eight credit paths predictions. No universal claim.')
    save(out/'summary.json',result);print(result,flush=True)
