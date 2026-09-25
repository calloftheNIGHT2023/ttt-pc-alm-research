"""328 frozen51 catalogue and independent new-development protocol gate."""
import numpy as np
import online_credit_resource_suite_v1 as resources
from evaluate_complete_credit_mode_geometry_v1 import read,sha

BASE='results/online_credit_fresh_pilot'
DESIGN='outputs/ttt-pc-alm-research/328_fresh_online_credit_protocol_v1.md'
REVISION='outputs/ttt-pc-alm-research/328_serialization_correction_v2.md'
SOURCES=['online_credit_fresh_suite_v2.py','run_online_credit_fresh_v2.py','evaluate_online_credit_fresh_v2.py','test_online_credit_fresh_v2.py','audit_online_credit_fresh_v2.py',
         'probe_confirmation_statistics.py','run_independent_hybrid_memory.py','analyze_recovered_online_comparison.py','counterfactual_fresh_pilot_v1.py','continue_online_credit_fresh_v2.ps1','exact_quadratic_events_v1.py']
FIELDS=['prediction','point_prediction']
PILOT_SEEDS=list(range(328000000,328000512))
OLD_SEEDS=[5910000,5910001,5910063]
CANDIDATES=[resources.PRIMARY,resources.SECONDARY]
_REFERENCES={}


def gate(root):
    assert sha(root/BASE/'preflight_predictions_v1/failure.json')=='210fe6251dbe9bf248d192bf37e609c30d3f0db6cc68c5ed68c40e1478c0b240'
    assert sha(root/BASE/'tests_v1/protocol.json')=='e002d77e0caf5fafb85ce70baced04cb39eef3f9b09f83afa87ef645bdf167ab'
    hashes=resources.gate(root);folder=root/resources.BASE/'calibration_v1'
    summary=resources.old.complete(folder);audit=read(root/resources.BASE/'audit_v1/summary.json')
    assert audit['passed'] and audit['calibration_summary_sha256']==sha(folder/'summary.json')
    protocol=read(folder/'protocol.json');assert protocol['source_sha256']==hashes
    assert summary['methods']==51 and summary['timing_runs']==816 and summary['memory_runs']==102
    selection=read(folder/'selection.json')
    for candidate in CANDIDATES:assert not selection[candidate]['highest_adam_still_within'],'Need a higher Adam budget before new-task execution'
    assert resources.catalogue(root)==protocol['configs']
    for n in SOURCES:hashes[n]=sha(root/'work/experiments'/n)
    return hashes


def reference(root,cfg,seed,arrays,meta):
    folder=root/resources.BASE/'calibration_v1';filename=f'{seed}_{cfg["name"]}.npz'
    if str(root) not in _REFERENCES:
        records={}
        for row in read(folder/'timings.json'):
            records.setdefault((row['seed'],row['method']),{n:row['metadata'][n] for n in ['execution_failed','positive_modes'] if n in row['metadata']})
        _REFERENCES[str(root)]=read(folder/'files.json'),records
    files,records=_REFERENCES[str(root)]
    assert sha(folder/filename)==files[filename]
    count=0
    with np.load(folder/filename,allow_pickle=False) as z:
        assert set(z.files)==set(arrays)-{'x_observed','v_observed','q_observed'}
        for n in z.files:assert z[n].shape==arrays[n].shape and z[n].dtype==arrays[n].dtype and z[n].tobytes()==arrays[n].tobytes(),(seed,cfg['name'],n);count+=1
    expected=records[seed,cfg['name']]
    assert meta['execution_failed']==expected['execution_failed']
    if 'positive_modes' in meta:assert meta['positive_modes']==expected['positive_modes']
    return count
