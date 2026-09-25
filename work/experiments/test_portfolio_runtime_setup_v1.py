"""402 paired cold/warm runtime diagnostic with exact archived-array replay."""
from pathlib import Path
import time
import traceback
import numpy as np
import deadline_risk_io_v1 as io
import prefix_deadline_worker_v1 as old
import prefix_deadline_worker_v2 as new
import prefix_matched_controls_v1 as controls

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/portfolio_runtime_setup/preflight_v1'
DESIGN = 'outputs/ttt-pc-alm-research/402_portfolio_runtime_setup_correction_v1.md'


def main():
    begin=time.perf_counter()
    reference=ROOT/'results/prefix_deadline_pilot/preflight_v1'
    reference_summary=io.read(reference/'summary.json')
    assert reference_summary['passed']
    for name,digest in reference_summary['outputs_sha256'].items():assert io.sha(reference/name)==digest
    reference_rows={r['directory']:r for r in io.read(reference/'rows.json')}
    for name,digest in io.read(reference/'inputs_manifest.json').items():assert io.sha(reference/'inputs'/name)==digest
    original_protocol=io.read(ROOT/'results/prefix_deadline_pilot/development_v1/protocol.json')
    hashes={**original_protocol['source_sha256'],
        **{name:io.sha(ROOT/name) for name in [DESIGN,
            'work/experiments/prefix_deadline_worker_v2.py',
            'work/experiments/test_portfolio_runtime_setup_v1.py']}}
    io.verify_hashes(ROOT,hashes)
    io.save(OUT/'protocol.json',dict(source_sha256=hashes,seed=5920002,stages=list(controls.STAGES),
        portfolios=list(controls.PORTFOLIOS),variants=['original','preloaded'],repeats=2,budget=20.,
        expected_calls=32,query_targets_accessed=False,
        reference_summary_sha256=io.sha(reference/'summary.json'),
        purpose='Runtime-boundary diagnosis, not new task risk or isolated wall-time speedup'))
    inputs=io.load_arrays(reference/'inputs/5920002.npz')
    rows=[];sessions=[];array_checks=0;packet_checks=0
    for ni,n in enumerate(controls.STAGES):
        x,v,q=inputs['x'][:n].copy(),inputs['v'][:n].copy(),inputs['q'].copy()
        for ci,name in enumerate(controls.PORTFOLIOS):
            ref=reference/'calls'/f'5920002_n{n}_{name}_b20000'
            for name_file,digest in reference_rows[ref.relative_to(ROOT).as_posix()]['files'].items():
                assert io.sha(ref/name_file)==digest
            saved=io.load_arrays(ref/'state.npz')
            for label,module in ([('original',old),('preloaded',new)] if (ni+ci)%2==0 else [('preloaded',new),('original',old)]):
                session=module.Session(dict(name=name,kind='portfolio'),ROOT)
                try:
                    for repeat in range(2):
                        result=session.run(x,v,q,seed=5920002,budget=20.)
                        prediction,meta,events,archive=result
                        directory=OUT/'calls'/f'{label}_{name}_n{n}_r{repeat}'
                        record=io.save_call(ROOT,directory,*result)
                        assert meta['received_final'] and meta['selected']=='final' and meta['error'] is None
                        assert meta['generation']==repeat and meta['model_fingerprint_verified']
                        assert not meta['shared_scientific_task_state_reused']
                        aa=archive['arrays']
                        assert set(aa)==set(saved)
                        for key in saved:
                            np.testing.assert_array_equal(aa[key],saved[key]);array_checks+=1
                        np.testing.assert_array_equal(prediction,saved['prediction'])
                        expected=[saved['fallback_prediction']]+[saved[f'stage_{stage}_prediction'] for stage in range(len(controls.PORTFOLIOS[name]))]+[saved['prediction']]
                        assert len(events)==len(expected)
                        for event,wanted in zip(events,expected):
                            np.testing.assert_array_equal(event['prediction'],wanted)
                            assert event['received_seconds']<=20.;packet_checks+=1
                        chosen,kind=old.parent.old.choose(events,20.,q)
                        np.testing.assert_array_equal(chosen,prediction);assert kind=='final'
                        rows.append(dict(variant=label,method=name,n=n,repeat=repeat,**record,
                            first_stage_received_seconds=events[1]['received_seconds'],
                            prior_received_seconds=events[0]['received_seconds'],
                            first_member_compute_seconds=archive['metadata']['stages'][0]['cumulative_seconds'],
                            all_packet_received_seconds=[e['received_seconds'] for e in events]))
                finally:
                    sessions.append(dict(variant=label,method=name,n=n,**session.close()))
            print(dict(n=n,method=name,calls=len(rows),seconds=time.perf_counter()-begin),flush=True)
    for label in ['original','preloaded']:
        for name in controls.PORTFOLIOS:
            for n in controls.STAGES:
                rr=[r for r in rows if (r['variant'],r['method'],r['n'])==(label,name,n)]
                assert len(rr)==2 and rr[0]['worker_pid']==rr[1]['worker_pid']
                assert rr[0]['setup_seconds']>0 and rr[1]['setup_seconds']==0
    io.verify_hashes(ROOT,hashes)
    io.save(OUT/'rows.json',rows);io.save(OUT/'sessions.json',sessions)
    io.save(OUT/'summary.json',dict(passed=True,calls=len(rows),saved_arrays_replayed=array_checks,
        event_predictions_replayed=packet_checks,query_targets_accessed=False,
        fixed_budget_risk_comparison=False,source_sha256=hashes,seconds=time.perf_counter()-begin,
        outputs_sha256={f:io.sha(OUT/f) for f in ['protocol.json','rows.json','sessions.json']}))
    print({k:v for k,v in io.read(OUT/'summary.json').items() if k!='source_sha256'},flush=True)


if __name__=='__main__':
    assert io.read(ROOT/'results/prefix_deadline_pilot/report_v1/qa_numeric.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False)
    try:main()
    except Exception:
        io.save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
