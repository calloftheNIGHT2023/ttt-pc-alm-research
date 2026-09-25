"""288 full supplement report; requires completed independent audit first."""
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from known_range_projection_pipeline_v1 import complete, now, paths, read, save, sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--stage', choices=['preflight', 'confirmation'], required=True)
    args = parser.parse_args()
    root = args.project.resolve()
    loc = paths(root, args.stage)
    ev, audit = loc['evaluation'], loc['audit']
    es, aus = complete(ev), complete(audit)
    assert aus['evaluation_summary_sha256'] == sha(ev/'summary.json')
    p = read(ev/'protocol.json')
    methods, contrasts = read(ev/'methods.json'), read(ev/'comparisons.json')
    names, controls, primary = p['methods'], p['controls'], p['primary']
    assert [r['method'] for r in methods] == names and len(methods) == 27
    assert len(contrasts) == 104 and aus['method_tables'] == 27
    mm = {r['method']: r for r in methods}
    cc = {(r['control'], r['metric']): r for r in contrasts}
    source = sha(Path(__file__))
    if args.stage == 'confirmation':
        prev = loc['extra']/'figures_preflight_v1'
        complete(prev)
        assert read(prev/'protocol.json')['report_source_sha256'] == source
    out = loc['extra']/f'figures_{args.stage}_v1'
    out.mkdir(parents=True, exist_ok=False)
    save(out/'protocol.json', dict(created_utc=now(), stage=args.stage,
         report_source_sha256=source, evaluation_summary_sha256=sha(ev/'summary.json'),
         audit_summary_sha256=sha(audit/'summary.json'), all_methods_retained=True,
         supplementary_analysis=True, core_research_goal_complete=False))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'axes.spines.top': False, 'axes.spines.right': False})
    labels = [n+(' [primary]' if n == primary else ' [background]' if n == 'probe_all_alm64' else '') for n in names]
    y = np.arange(27)
    fig, ax = plt.subplots(figsize=(14, 12), layout='constrained')
    ax.barh(y-.18, [r['metrics']['mse257']['raw_mean'] for r in methods],
            height=.32, color='#9eabb5', label='Original frozen readout')
    ax.barh(y+.18, [r['metrics']['mse257']['projected_mean'] for r in methods],
            height=.32, color='#27817c', label='Same readout clipped to [0,1]')
    ax.set(yticks=y, yticklabels=labels, xlabel='Query MSE on 257-point grid; lower is better')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.legend(loc='lower right')
    scope = 'OLD2 FUNCTIONAL PREFLIGHT' if args.stage == 'preflight' else '8192 ALREADY-EVALUATED TASKS'
    ax.set_title('288 range-projection supplement | '+scope+'\nAll 27 fixed methods; no new adaptation or independent confirmation', loc='left')
    fig.savefig(out/'288_all_methods.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(15, 12), layout='constrained')
    for i, name in enumerate(controls):
        r = cc[name, 'mse257']
        lo, hi = r['descriptive95']
        color = '#27817c' if r['upper_below_zero'] else '#777c83'
        ax.hlines(i, lo, hi, color=color, lw=2)
        ax.plot(r['mean_difference'], i, 'o', color=color, ms=5)
        if r['adjusted_upper'] is not None:
            ax.plot(r['adjusted_upper'], i, '>', color=color, ms=7)
    ax.axvline(0, color='#a94c41', ls='--', lw=1)
    ax.set_xscale('symlog', linthresh=.001)
    ax.set(yticks=np.arange(26), yticklabels=[n+(' [background]' if n == 'probe_all_alm64' else '') for n in controls],
           xlabel='Projected primary minus projected control MSE; negative favors primary')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=.2)
    ax.set_title('288 | '+scope+'\nAll 26 contrasts; approximate bootstrap, not a replacement of the original 287 criterion', loc='left')
    ax.legend(handles=[Line2D([0],[0], color='#777c83', marker='o', label='Mean + descriptive 95% interval'),
               Line2D([0],[0], color='#777c83', marker='>', ls='', label='25-family upper quantile (.998)')],
               loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)
    fig.supxlabel('Symmetric log axis, linear within +/-0.001. Task-level resampling; no finite-sample familywise guarantee.', fontsize=9)
    fig.savefig(out/'288_all_contrasts.png', dpi=160)
    plt.close(fig)
    missing = [r['control'] for r in contrasts if r['supplementary_main_comparison'] and not r['upper_below_zero']]
    lines = ['# 288｜统一输出范围后的完整补充比较', '',
             f'身份：{scope}。全部方法、参数和观察数据不变；只把两种读出统一投影到已知范围 [0,1]。', '',
             '**这是同一批已评估数据的补充分析，不是新独立确认，也不替换 287 的原统计判据。**', '',
             '## 1．结果', '',
             f'候选投影后主 MSE 为 {mm[primary]["metrics"]["mse257"]["projected_mean"]:.12g}；',
             f'补充 25 项主比较中 {es["supplementary_negative_upper_bounds"]} 项上界低于零，原比较为 {es["original_main_negative_upper_bounds"]} 项。',
             '未通过补充主阈值：'+', '.join(missing)+ '。不显著不等于等价。', '',
             '![全部方法原读出与范围投影](288_all_methods.png)', '',
             '![全部补充比较](288_all_contrasts.png)', '',
             '## 2．全部方法与变化', '',
             '|方法|原主MSE|投影主MSE|平均变化|主读出改变任务|主读出改变坐标|点读出改变任务|失败数|',
             '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in methods:
        m, a, b = r['metrics']['mse257'], r['projection']['prediction'], r['projection']['point_prediction']
        lines.append(f'|{r["method"]}|{m["raw_mean"]:.12g}|{m["projected_mean"]:.12g}|{m["mean_change"]:.12g}|{a["changed_tasks"]}|{a["changed_values"]}|{b["changed_tasks"]}|{r["failures"]}|')
    lines += ['', '## 3．全部 104 项比较', '',
              '横线是描述性 95% 区间，三角是主比较族的 .998 单侧上分位；两者不是同一区间。',
              '主指标仅 mse257，三个次要指标不能替代主判据。全试探 ALM64 是超预算背景。', '']
    for metric in p['metrics']:
        lines += [f'### {metric}', '', '|控制|主减控制|95%下限|95%上限|校正上界|好/同/差任务|',
                  '|---|---:|---:|---:|---:|---:|']
        for name in controls:
            r = cc[name, metric]
            lo, hi = r['descriptive95']
            upper = '—' if r['adjusted_upper'] is None else f'{r["adjusted_upper"]:.12g}'
            lines.append(f'|{name}|{r["mean_difference"]:.12g}|{lo:.12g}|{hi:.12g}|{upper}|{r["improved_tasks"]}/{r["equal_tasks"]}/{r["worse_tasks"]}|')
        lines.append('')
    lines += ['## 4．数学保证与资源边界', '',
              '对任意 y∈[0,1]，P(a)=clip(a,0,1) 满足 (a−y)²−(P(a)−y)²≥(a−P(a))²≥0。',
              '这保证投影不增加任一坐标的平方误差，不能归因于 PC-ALM。主指标风险有界于 [0,1]；原始未投影 OLS 的无限任务方差警告不应直接搬给有界投影后的损失。有限样本 bootstrap 的覆盖率仍不是精确保证。', '',
              '原完整调用耗时不包含这次新投影。新增两数组为 4112 bytes 小计，不是进程峰值内存。投影微调用在文件循环中测量，不能把它直接加到旧耗时后声称公平端到端加速；投影运行时还有阶段备份并发。', '',
              '|方法|原完整均秒（不含投影）|投影两数组孤立均秒|新增输出bytes小计|',
              '|---|---:|---:|---:|']
    for r in methods:
        lines.append(f'|{r["method"]}|{r["original_complete_mean_seconds"]:.9g}|{r["isolated_projection_pair_mean_seconds"]:.9g}|{r["projected_output_numeric_bytes"]}|')
    lines += ['', '## 5．复核和结论范围', '',
              f'独立复核 {aus["tasks"]} 任务、{aus["predictors"]} 个预测器、{aus["arrays"]} 数组、{aus["comparisons"]} 比较与 {aus["bootstrap_scalar_means"]} 个重抽均值。',
              f'新风险最大重算差 {aus["maximum_independent_risk_gap"]:.12g}；原风险 {aus["maximum_independent_original_risk_gap"]:.12g}；重抽 {aus["maximum_independent_bootstrap_gap"]:.12g}；逐点风险最大增加 {aus["maximum_pointwise_risk_increase"]:.12g}。', '',
              '支持预测先全部封存再作补充评价，但原 287 查询结果在本协议前已经看过。没有新训练、没有额外标签、没有修改原预测。',
              '候选与强同起点 Adam 的独立优势尚未建立；不能以通过 21 项或保留部分回归优势宣称整个研究完成。仍缺官方 TTT 同任务桥接、一般 MLP 和最终下游验证。', '',
              f'[评价摘要](../{ev.name}/summary.json)；[独立核验](../{audit.name}/summary.json)；[补充协议](../../../outputs/ttt-pc-alm-research/288_full_projection_protocol.md)。', '',
              '图像生成不代替人工视检。']
    if args.stage == 'preflight':
        lines.insert(2, '**仅旧两任务功能前检，所有数值不作为科学证据。**')
    with (out/'report.md').open('x', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    result = dict(passed=True, stage=args.stage, functional_preflight_only=args.stage=='preflight',
                  supplementary_analysis=True, tasks=aus['tasks'], methods=27, comparisons=104,
                  report_source_sha256=source, visual_qa_required=True, core_research_goal_complete=False,
                  outputs_sha256={n: sha(out/n) for n in ['protocol.json','report.md','288_all_methods.png','288_all_contrasts.png']})
    save(out/'summary.json', result)
    print(result, flush=True)


if __name__ == '__main__':
    main()
