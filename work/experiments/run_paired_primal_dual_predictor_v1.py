"""312: seal causal predictions on all old support-only development states."""
import argparse
from collections import Counter
from pathlib import Path
import time
import traceback

import numpy as np
import scipy.optimize as opt
import cold_stagnation_switch as cold
from posterior_confirmation_pipeline import discovery_box
from run_complete_credit_amplitude_support_v2 import inputs
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha
from paired_primal_dual_predictor_v1 import NAMES, evaluate, local


def run(root, out):
    begin = time.perf_counter()
    tests = root/'results/paired_primal_dual_predictor/tests_v1/summary.json'
    tested = read(tests)
    assert tested['passed'] and not tested['real_task_data_accessed']
    for n, digest in tested['source_sha256'].items():
        assert sha(root/'work/experiments'/n) == digest
    items, task_manifest, input_hashes = inputs(root)
    raw = root/'results/certificate_activity_attribution/development'
    originals = {r['seed']: r for r in read(raw/'rows.json') if r['method']=='credit_control_probe33'}
    # Use only observed inputs and past/current/next solver states, never query truth.
    for seed in sorted({item['seed'] for item in items}):
        source = raw/originals[seed]['file']
        assert sha(source) == originals[seed]['sha256']
        with np.load(source, allow_pickle=False) as z:
            for item in (i for i in items if i['seed']==seed):
                phase, step, origin = item['location']; assert step >= 2
                prefix = 'prefix' if phase==0 else 'anchor'
                previous = {}
                for k in ['b','h','u']:
                    arr = z[prefix+'_'+k]
                    before = arr[step-2,origin] if k=='b' else arr[step-2,:,origin]
                    current = arr[step-1,origin] if k=='b' else arr[step-1,:,origin]
                    assert np.array_equal(current, item[k])
                    previous[k] = before.copy()
                item['previous'] = previous
                item['expected_u'] = z[prefix+'_u'][step,:,origin].copy()
    prior_protocol = read(root/'results/complete_credit_amplitude_events/development_v2/protocol.json')
    for n, digest in prior_protocol['source_sha256'].items():
        assert sha(root/'work/experiments'/n)==digest
    names = set(prior_protocol['source_sha256']) | set(tested['source_sha256']) | {
        Path(__file__).name, 'layer_credit_interaction_v1.py', 'evaluate_complete_credit_mode_geometry_v1.py'}
    protocol = dict(source_sha256={n:sha(root/'work/experiments'/n) for n in sorted(names)},
        input_sha256=input_hashes, tests_summary_sha256=sha(tests),
        design_sha256=sha(root/'outputs/ttt-pc-alm-research/312_paired_primal_dual_predictor_protocol.md'),
        tasks=64, states=131, configurations=NAMES, task_manifest=task_manifest,
        minimum_original_step=min(i['location'][1] for i in items),
        query_targets_accessed=False, geometry_or_posterior_accessed=False,
        new_confirmation=False, resources_matched=False, alpha_grid_searched=False,
        reference_uses_actual_float_inputs=True, predicted_parameter_points_retained=True,
        future_state_use='Original next states ONLY for normal-step implementation checks, never predictions.')
    save(out/'protocol.json', protocol)
    files, rows, counts, guarded = {}, [], Counter(), []
    max_gap = max_du_gap = 0.
    def forbid(*args, **kwargs):
        raise AssertionError('Global BP or global optimizer called during local candidate generation')
    try:
        for obj, name in [(cold.bp,'evaluate'),(cold.bp,'refine'),(opt,'linprog'),(opt,'minimize')]:
            guarded.append((obj,name,getattr(obj,name))); setattr(obj,name,forbid)
        with discovery_box(.12):
            for index, item in enumerate(items):
                current = {k:item[k] for k in ['b','h','u']}
                check = local(item['previous'],item['x'],item['v']); check.step()
                for k, actual in [('b',check.b[0]),('h',check.h[:,0]),('u',check.u[:,0])]:
                    assert np.array_equal(actual,current[k]), (item['seed'],item['location'],k,'past replay')
                    counts['previous_to_current_array_checks'] += 1
                evaluated, metadata = evaluate(current,item['previous'],item['x'],item['v'],
                                               [item['seed'],*item['location']])
                for method in evaluated:
                    if method['method'] in ['normal_one','normal_two']:
                        for k in ['b','h','u']:
                            assert np.array_equal(method['states'][1][k],item['expected_'+k])
                            counts['normal_endpoint_array_checks'] += 1
                    counts['configurations'] += 1
                    counts['local_steps'] += method['costs']['local_steps']
                    for rr in method['references']:
                        counts['value_discrepancies'] += rr['value_discrepancy']
                        counts['mode_discrepancies'] += rr['mode_discrepancy']
                        max_gap = max(max_gap,*rr['gaps'].values())
                counts['alpha_zero'] += metadata['alpha_zero']
                counts['alpha_less_than_one'] += metadata['alpha_less_than_one']
                max_du_gap = max(max_du_gap,metadata['du_vs_half_current_residual_max_gap'])
                filename = f"{item['seed']}_{'_'.join(map(str,item['location']))}.json"
                payload = dict(seed=item['seed'],location=item['location'],
                    current={k:a.tolist() for k,a in current.items()},
                    previous={k:a.tolist() for k,a in item['previous'].items()},
                    x_observed=item['x'].tolist(),v_observed=item['v'].tolist(),
                    metadata=metadata,methods=evaluated)
                save(out/filename,payload); files[filename]=sha(out/filename)
                rows.append(dict(seed=item['seed'],location=item['location'],file=filename,sha256=files[filename],
                    alpha=metadata['alpha'],alpha_zero=metadata['alpha_zero'],
                    methods={m['method']:m['proposal_modes'] for m in evaluated}))
                if (index+1)%16==0 or index+1==len(items):
                    print(dict(states=index+1,counts=dict(counts),max_gap=max_gap),flush=True)
    finally:
        for obj,name,value in guarded: setattr(obj,name,value)
    assert len(rows)==131 and counts['configurations']==1572 and counts['local_steps']==1703
    assert counts['value_discrepancies']==counts['mode_discrepancies']==0
    for n,digest in protocol['source_sha256'].items(): assert sha(root/'work/experiments'/n)==digest
    save(out/'rows.json',rows)
    for n in ['rows.json','protocol.json']: files[n]=sha(out/n)
    save(out/'before_geometry_manifest.json',dict(files_sha256=files,all_states_processed=True,
        query_targets_accessed=False,geometry_or_posterior_accessed=False,states=131))
    final=dict(passed=True,states=131,tasks=64,counts=dict(counts),max_exact_reference_gap=max_gap,
        max_du_vs_half_residual_gap=max_du_gap,runtime_no_global_bp_guard_passed=True,
        query_targets_accessed=False,geometry_evaluated=False,resources_matched=False,
        independent_task_gain_established=False,seconds=time.perf_counter()-begin,
        manifest_sha256=sha(out/'before_geometry_manifest.json'),outputs_sha256=files)
    save(out/'summary.json',final); print({k:v for k,v in final.items() if k!='outputs_sha256'},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    try:run(Path(__file__).resolve().parents[2],args.out)
    except Exception:
        save(args.out/'failure.json',dict(traceback=traceback.format_exc()));raise
