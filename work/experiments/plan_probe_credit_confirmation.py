"""Pre-data sample-size/cost planning from audited development differences."""
import argparse,json,math,statistics
from pathlib import Path
import numpy as np
from run_multiplier_fixed_point_screen import sha,dump


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);root=ap.parse_args().project.resolve();src=Path(__file__).parent
    base=root/'results/probe_credit_confirmation';out=base/'planning';out.mkdir(parents=True,exist_ok=True);assert not (out/'protocol.json').exists()
    old=root/'results/probe_credit_budget/evaluation';audit=old.parent/'evaluation_audit';aa=json.loads((audit/'summary.json').read_text());assert aa['passed'];ap0=json.loads((audit/'protocol.json').read_text());assert sha(old/'summary.json')==ap0['evaluation_summary_sha256']
    hashes=dict(ap0['source_sha256']);hashes[Path(__file__).name]=sha(Path(__file__))
    for n,h in hashes.items():assert sha(src/n)==h,n
    resources=root/'results/probe_credit_resources/calibration';selection=json.loads((resources/'selected_configs.json').read_text());res=json.loads((resources/'summary.json').read_text())
    for n,h in res['outputs_sha256'].items():assert sha(resources/n)==h
    cfgs=json.loads((resources/'protocol.json').read_text())['configs'];core=[c for c in cfgs if c['name'] in selection['sensitivity_110_percent'] or c['name'] in selection['over_budget_background']];assert len(core)==19
    additional=['cold__linear_ls','cold__residual_linear_ls','cold__residual_linear_ridge','cold__residual_linear_rls','cold__rbf_loocv','cold__meta_ridge64','cold__meta_ridge128','cold__meta_shallow64_5']
    # 27 total, excluding primary and the explicitly over-budget full probe:
    family=25;alpha=.05;effect=.001;power=.8;z=statistics.NormalDist().inv_cdf(1-alpha/family)+statistics.NormalDist().inv_cdf(power)
    p=dict(source_sha256=hashes,development_evaluation_audit_sha256=sha(audit/'summary.json'),resource_summary_sha256=sha(resources/'summary.json'),primary='credit_control_probe33',
        core_configurations=core,additional_regression_head_names=additional,planned_total_methods=27,conservative_test_family=family,alpha=alpha,planning_effect_absolute_mse=effect,planning_power=power,
        planning_comparisons='Within1.10 archive/local/Adam controls, excluding heads and over-budget full probe; equal hypothetical effect used only for planning, never claimed achieved',
        sample_count_rule='Power-of-two ceiling of maximum normal-approximation required N; minimum1024, maximum8192; no new-context or answer access',
        query_metric='Fixed257-grid actual independent query MSE; no posterior-risk substitution',phase_accesses_new_contexts_or_targets=False)
    dump(out/'protocol.json',p);rows={(r['seed'],r['method'],r['grid']):r for r in json.loads((old/'rows.json').read_text())};seeds=list(range(5910000,5910064));planning=[]
    for cfg in core:
        name=cfg['name']
        if name==p['primary'] or cfg['group'] in ['ridge','shallow','full_probe']:continue
        d=np.array([rows[s,p['primary'],257]['mse']-rows[s,name,257]['mse'] for s in seeds]);sd=float(np.std(d,ddof=1));needed=math.ceil((z*sd/effect)**2)
        planning.append(dict(control=name,old_query_difference=float(d.mean()),paired_sd=sd,normal_approx_required_n=needed,
            warning='Hypothetical .001 effect; pilot mean and variance can be noisy after development. This is not evidence of a true .001 improvement.'))
    maximum=max(r['normal_approx_required_n'] for r in planning);rounded=2**math.ceil(math.log2(max(1024,maximum)));count=min(8192,rounded)
    fullseconds=sum(selection['calibration_means'][c['name']] for c in core);means=np.array([r['paired_sd'] for r in planning]);power_at_n=statistics.NormalDist().cdf(math.sqrt(count)*effect/max(means)-statistics.NormalDist().inv_cdf(1-alpha/family))
    dump(out/'comparisons.json',planning);answer=dict(passed=True,planned_tasks=count,unclipped_power_two_count=rounded,maximum_approx_required_n=maximum,
        worst_pilot_sd=float(max(means)),approx_power_at_hypothetical_effect=float(power_at_n),core19_seconds_per_task=fullseconds,
        core19_estimated_compute_hours=count*fullseconds/3600,estimate_excludes_extra8_heads_io_audits_and_task_variability=True,
        planning_inputs='Only previously opened64 development tasks; no fresh context generation or new targets',phase_accesses_new_contexts_or_targets=False,
        protocol_sha256=sha(out/'protocol.json'),comparisons_sha256=sha(out/'comparisons.json'),next='Freeze full protocol and check eight extra heads/resources before creating untouched confirmation contexts')
    dump(out/'summary.json',answer);print(json.dumps(answer),flush=True)


if __name__=='__main__':main()
