"""391 pre-query secondary decomposition of regional deadline risk.

freeze uses stdlib only, while formal timings run. run is evaluator-only after
387 seal/audit/evaluation. No new predictions are submitted to the experiment.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import traceback

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/new_task_deadline_risk'
PLAN = BASE / 'mechanism_plan_v1'
OUT = BASE / 'mechanism_decomposition_v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def freeze():
    before = BASE / 'evaluation_v1/before_query.json'
    assert not before.exists(), 'Secondary plan must be frozen before any query-target creation'
    original = read(BASE / 'development_v1/protocol.json')
    sources = [Path(__file__), ROOT / 'outputs/ttt-pc-alm-research/391_deadline_mechanism_decomposition_v1.md']
    plan = dict(primary=original['primary'], primary_budget=original['primary_budget'],
                methods=[c['name'] for c in original['configs'] if c['kind'] == 'regional'],
                budgets=original['budgets'], seeds=original['seeds'],
                task_protocol_sha256=sha(BASE / 'development_v1/protocol.json'),
                source_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sources},
                frozen_before_query=True, primary_protocol_unchanged=True,
                secondary_descriptive_not_confirmatory=True,
                reference_prediction='lexicographically first (method,budget) timely complete regional output for each task',
                offline_fallback='recompute common prior4096 from observed x/v/q only; never supplied to timed predictors',
                effect_orientation='control minus primary MSE; positive favors primary',
                contributions=['fallback_receipt', 'complete_receipt', 'complete_prediction_difference'],
                exact_identity='L(C)-L(A)=(JrA-JrC)*(Lc-Lr)+(JpA-JpC)*(Lr-Lpstar)+JpC*(LpC-Lpstar)-JpA*(LpA-Lpstar)',
                missing_complete_reference='If no timely full prediction exists, both Jp are zero and these terms are exactly zero; no full solver is run',
                uncertainty='No new p-values or winner selection; all 9 regional comparisons at both original budgets',
                query_targets_accessed=False)
    PLAN.mkdir(parents=True, exist_ok=False)
    save(PLAN / 'protocol.json', plan)
    assert not before.exists(), 'Query generation raced the secondary-plan freeze'
    print(dict(passed=True, frozen_before_query=True, methods=len(plan['methods']), plan=str(PLAN / 'protocol.json')), flush=True)


def run():
    import numpy as np
    import deadline_risk_io_v1 as archive
    import independent_hybrid_memory as regression

    plan = read(PLAN / 'protocol.json')
    for relative, digest in plan['source_sha256'].items():
        assert sha(ROOT / relative) == digest, relative
    assert plan['task_protocol_sha256'] == sha(BASE / 'development_v1/protocol.json')
    evaluation = BASE / 'evaluation_v1'
    evaluated = read(evaluation / 'summary.json')
    audited = read(BASE / 'audit_v1/summary.json')
    assert evaluated['passed'] and audited['passed']
    for relative, digest in evaluated['outputs_sha256'].items():
        assert sha(evaluation / relative) == digest, relative
    original = read(BASE / 'development_v1/protocol.json')
    archive.verify_hashes(ROOT, original['source_sha256'])
    archive.verify_hashes(ROOT, original['pretrained_sha256'])
    risk = read(evaluation / 'risk_rows.json')
    rows = [r for r in risk if r['method'] in plan['methods']]
    assert len(rows) == len(plan['methods']) * len(plan['budgets']) * len(plan['seeds'])
    truth = archive.load_arrays(evaluation / 'query_truth.npz')
    targets = {int(seed):target for seed, target in zip(truth['seeds'], truth['targets'])}
    predictions = {}; roles = {}; reference = {}; reference_origin = {}
    for row in sorted(rows, key=lambda r:(r['seed'], r['method'], r['budget'])):
        directory = ROOT / row['directory']
        assert sha(directory / 'outputs.npz') == row['files']['outputs.npz']
        pred = archive.load_arrays(directory / 'outputs.npz')['prediction']
        key = row['seed'], row['method'], row['budget']
        predictions[key] = pred
        role = row['selected']
        if role == 'final':
            assert sha(directory / 'state.json') == row['files']['state.json']
            state = read(directory / 'state.json')
            role = 'complete' if state['full_candidate_set_resolved'] and state['readout_available'] else 'fallback'
        assert role in ('constant', 'fallback', 'complete')
        roles[key] = role
        if role == 'complete' and row['seed'] not in reference:
            reference[row['seed']] = pred
            reference_origin[row['seed']] = dict(method=row['method'], budget=row['budget'])

    fallback = {}; risk_constant = {}; risk_fallback = {}; risk_reference = {}
    complete_comparisons = []; maximum_complete_difference = 0.
    for seed in plan['seeds']:
        inputs = archive.load_arrays(BASE / f'development_v1/inputs/{seed}.npz')
        function, _, _ = regression.regression(inputs['x'], inputs['v'], 'prior4096_ridge')
        fallback[seed] = np.clip(function(inputs['q']), 0., 1.)
        risk_constant[seed] = float(np.mean((.5 - targets[seed]) ** 2))
        risk_fallback[seed] = float(np.mean((fallback[seed] - targets[seed]) ** 2))
        risk_reference[seed] = float(np.mean((reference[seed] - targets[seed]) ** 2)) if seed in reference else None
    own_risk = {}
    for row in rows:
        key = row['seed'], row['method'], row['budget']; role = roles[key]; pred = predictions[key]; seed = row['seed']
        if role == 'fallback':
            np.testing.assert_array_equal(pred, fallback[seed])
        elif role == 'constant':
            np.testing.assert_array_equal(pred, np.full_like(pred, .5))
        else:
            gap = float(np.max(abs(pred - reference[seed])))
            maximum_complete_difference = max(maximum_complete_difference, gap)
            complete_comparisons.append(dict(seed=seed, method=row['method'], budget=row['budget'],
                                            max_prediction_difference=gap, exactly_equal=gap == 0.))
        own_risk[key] = float(np.mean((pred - targets[seed]) ** 2))
        assert own_risk[key] == row['query_mse']

    terms = []; grouped = []
    for budget in plan['budgets']:
        for control in plan['methods']:
            if control == plan['primary']:
                continue
            pair = []
            for seed in plan['seeds']:
                ka = seed, plan['primary'], budget; kc = seed, control, budget
                ra, rc = roles[ka], roles[kc]
                jra, jrc = int(ra != 'constant'), int(rc != 'constant')
                jpa, jpc = int(ra == 'complete'), int(rc == 'complete')
                fallback_gain = risk_constant[seed] - risk_fallback[seed]
                complete_gain = risk_fallback[seed] - risk_reference[seed] if seed in reference else None
                part_fallback = (jra - jrc) * fallback_gain
                part_complete = (jpa - jpc) * complete_gain if complete_gain is not None else 0.
                part_prediction = 0.
                if jpc:
                    part_prediction += own_risk[kc] - risk_reference[seed]
                if jpa:
                    part_prediction -= own_risk[ka] - risk_reference[seed]
                difference = own_risk[kc] - own_risk[ka]
                reconstructed = math.fsum((part_fallback, part_complete, part_prediction))
                error = abs(difference - reconstructed)
                assert error < 2e-14, (seed, control, budget, error)
                item = dict(seed=seed, budget=budget, control=control, primary_role=ra, control_role=rc,
                            primary_complete=jpa, control_complete=jpc,
                            direct_control_minus_primary=difference, fallback_receipt=part_fallback,
                            complete_receipt=part_complete, complete_prediction_difference=part_prediction,
                            realized_fallback_to_complete_gain=complete_gain,
                            reference_origin=reference_origin.get(seed), reconstruction_error=error)
                terms.append(item); pair.append(item)
            grouped.append(dict(budget=budget, control=control, tasks=len(pair),
                primary_only_complete=sum(r['primary_complete'] and not r['control_complete'] for r in pair),
                control_only_complete=sum(r['control_complete'] and not r['primary_complete'] for r in pair),
                both_complete=sum(r['primary_complete'] and r['control_complete'] for r in pair),
                neither_complete=sum(not (r['primary_complete'] or r['control_complete']) for r in pair),
                **{field:math.fsum(r[field] for r in pair)/len(pair) for field in
                    ('direct_control_minus_primary', 'fallback_receipt', 'complete_receipt', 'complete_prediction_difference')},
                maximum_reconstruction_error=max(r['reconstruction_error'] for r in pair)))
    save(OUT / 'per_task_terms.json', terms)
    save(OUT / 'groups.json', grouped)
    save(OUT / 'complete_prediction_consistency.json', complete_comparisons)
    table = ['| Budget | Control | MSE(C)-MSE(A) | Fallback receipt | Complete receipt | Prediction difference | A only / C only / both / neither |',
             '| --- | --- | --- | --- | --- | --- | --- |']
    for group in grouped:
        table.append('| ' + ' | '.join([str(group['budget']), group['control'],
            *(f'{group[k]:.10g}' for k in ('direct_control_minus_primary', 'fallback_receipt', 'complete_receipt', 'complete_prediction_difference')),
            f"{group['primary_only_complete']} / {group['control_only_complete']} / {group['both_complete']} / {group['neither_complete']}"]) + ' |')
    body = '\n'.join(['# 391｜截止时间查询风险的机制分解', '',
        '在查询答案生成前冻结的次要描述性分析；不修改387主指标、主方法或显著性检验。正数表示主方法误差更低。', '',
        *table, '',
        '每项使用全部32任务。误差差异严格拆成备用到达、完整读出到达及完整预测差异；如后者不为零，不把它混入计算速度贡献。完成子组仅作描述，不用其条件均值选择方法或宣称新的显著性。', '',
        '完整参考预测只来自已封存且及时交付的结果，按方法名和预算固定排序选取；没有为未完成作业补跑完整求解。备用仅为离线核验从已观察支持重算，不交回任何预测方法、不从在线预算扣除。', '',
        f'已封存完整预测的最大跨配置差：{maximum_complete_difference:.17g}。没有及时完整参考的任务数：{len(plan["seeds"])-len(reference)}。', '',
        '此处增益是实际查询标签上的误差差，不是未知Bayes条件均值的估计保证，也不是对完成子组的独立随机化因果实验。总体风险与统计结论仍以388为准。', '',
        '[预查询分析协议](../mechanism_plan_v1/protocol.json) · [逐任务分解](per_task_terms.json) · [完整预测一致性](complete_prediction_consistency.json)', ''])
    (OUT / 'report.md').write_text(body, encoding='utf-8')
    summary = dict(passed=True, secondary_descriptive_not_confirmatory=True,
                   original_predictions_unchanged=True, per_task_identities=len(terms), comparisons=len(grouped),
                   maximum_reconstruction_error=max(r['reconstruction_error'] for r in terms),
                   complete_prediction_comparisons=len(complete_comparisons),
                   maximum_complete_prediction_difference=maximum_complete_difference,
                   tasks_without_timely_complete_reference=len(plan['seeds']) - len(reference),
                   no_missing_complete_solver_rerun=True,
                   protocol_sha256=sha(PLAN / 'protocol.json'),
                   evaluated_summary_sha256=sha(evaluation / 'summary.json'),
                   outputs_sha256={name:sha(OUT / name) for name in ('per_task_terms.json', 'groups.json', 'complete_prediction_consistency.json', 'report.md')},
                   core_research_goal_complete=False)
    save(OUT / 'summary.json', summary)
    print(summary, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('action', choices=('freeze', 'run')); args = parser.parse_args()
    if args.action == 'freeze':
        freeze()
    else:
        assert (BASE / 'evaluation_v1/summary.json').exists()
        OUT.mkdir(parents=True, exist_ok=False)
        try:
            run()
        except Exception:
            save(OUT / 'failure.json', dict(traceback=traceback.format_exc(), automatic_retry=False))
            raise
