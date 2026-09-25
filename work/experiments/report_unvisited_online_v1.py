"""354 assemble audited results only; no candidate tuning or resampling."""
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save, write, complete, table, f

BASE = 'results/unvisited_online'
ENTRY = 'outputs/ttt-pc-alm-research/354_unvisited_online_results_v1.md'
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
PRIMARY = 'unvisited_dual'


def figures(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    xx = np.arange(6); labels = ['Dual', 'Dual + residual', 'Residual', 'BP', 'Random sign', 'Zero']
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    differences = [r['delta_old'] for r in data['credits']]
    axes[0].bar(xx, differences, color=['#157b75' if x < 0 else '#b66552' for x in differences])
    axes[0].axhline(0, color='#52616a', linewidth=1)
    axes[0].set_title('Exclude visited modes before K-best truncation', loc='left', pad=15)
    axes[0].set_ylabel('Mean MSE change vs same-credit old method')
    if all(x == 0 for x in differences):
        axes[0].set_ylim(-1, 1); axes[0].set_yticks([0]); axes[0].text(.5, .8, 'All six changes are zero', transform=axes[0].transAxes, ha='center')
    axes[1].bar(xx, [r['seconds'] for r in data['credits']], color='#527cad')
    axes[1].axhline(data['native_seconds'], color='#b54556', linestyle='--', label='Native ALM64')
    axes[1].axhline(data['adam_seconds'], color='#6858a2', linestyle=':', label='Same-start Adam240')
    axes[1].set_title('Complete live fits, measured in this run', loc='left', pad=15)
    axes[1].set_ylabel('Mean seconds per task; one repetition'); axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(xx, labels, rotation=24, ha='right'); axis.spines[['top', 'right']].set_visible(False); axis.grid(axis='y', alpha=.18)
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.87, wspace=.32)
    fig.text(.08, .065, 'Negative error change is improvement. All six credits receive the same exclusion and search budget.', fontsize=10)
    fig.text(.08, .025, '32 exposed development tasks. No new-task significance, fixed-time match or peak-memory match claimed.', fontsize=10)
    fig.savefig(out/'unvisited_risk_and_cost.png', dpi=160); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]; folders = {n: root/BASE/f'development_{n}_v1' for n in ['predictions', 'evaluation', 'audit']}
    summaries = {n: complete(p) for n, p in folders.items()}; audit = summaries['audit']
    assert audit['prediction_summary_sha256'] == sha(folders['predictions']/'summary.json')
    assert audit['evaluation_summary_sha256'] == sha(folders['evaluation']/'summary.json')
    protocol = read(folders['predictions']/'protocol.json'); assert protocol['primary'] == PRIMARY
    for p, digest in protocol['source_sha256'].items(): assert sha(root/p) == digest
    methods = read(folders['evaluation']/'methods.json'); by = {m['method']: m for m in methods}
    comps = {(r['candidate'], r['control'], r['metric']): r for r in read(folders['evaluation']/'comparisons.json')}
    mechanisms = read(folders['audit']/'mechanisms.json')
    native = 'unvisited_native_alm64'; adam = 'unvisited_native_adam240'
    data = dict(native_seconds=by[native]['mean_current_seconds'], adam_seconds=by[adam]['mean_current_seconds'], credits=[])
    for ch in CHANNELS:
        name = 'unvisited_'+ch; old = 'strong_language_'+ch; c = comps[name, old, 'mse257']; mm = [r for r in mechanisms if r['method'] == name]
        data['credits'].append(dict(method=name, old_method=old, mse=by[name]['metrics']['mse257'], old_mse=by[old]['metrics']['mse257'],
            delta_old=c['mean_difference'], improved=c['improved'], equal=c['equal'], worse=c['worse'],
            extra_positive=sum(len(r['additional_positive_vs_old']) for r in mm), seconds=by[name]['mean_current_seconds']))
    main_rows = [[r['method'], f(r['old_mse']), f(r['mse']), f(r['delta_old']), f"{r['improved']}/{r['equal']}/{r['worse']}", str(r['extra_positive']), f(r['seconds'])] for r in data['credits']]
    controls = [c['name'] for c in protocol['configs'] if c['name'] != PRIMARY]+['strong_language_dual', 'strong_native_alm128', 'strong_native_alm256', 'strong_native_adam3840', 'cold__prior16384_ridge', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
    comp_rows = []
    for name in controls:
        c = comps[PRIMARY, name, 'mse257']; comp_rows.append([name, f(by[name]['metrics']['mse257']), f(c['mean_difference']), f"{c['improved']}/{c['equal']}/{c['worse']}", f(c['worst_leave_one_out_mean'])])
    full_rows = [[m['method']]+[f(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+[f(m['mean_current_seconds']), str(m['failures'])] for m in methods]
    out = root/BASE/'report_v1'; out.mkdir(parents=True, exist_ok=False); figures(out, data)
    save(out/'figure_data.json', data); save(out/'table_data.json', dict(main=main_rows, comparisons=comp_rows, full=full_rows))
    delta = comps[PRIMARY, 'strong_language_dual', 'mse257']; nz = comps[PRIMARY, 'unvisited_zero', 'mse257']; na = comps[PRIMARY, adam, 'mse257']
    intro = f"主候选MSE257 **{by[PRIMARY]['metrics']['mse257']:.10f}**；相对旧同信用均差 **{delta['mean_difference']:+.10f}**，改善/相同/变差 **{delta['improved']}/{delta['equal']}/{delta['worse']}**。相对同增强零信用均差 **{nz['mean_difference']:+.10f}**，相对同期Adam240均差 **{na['mean_difference']:+.10f}**。负值表示主方法更低。"
    text = ['# 354｜截断前排除已访问模式：在线查询与完整控制', '', intro, '',
        '32个已暴露开发任务，256次新完整fit，137方法4384预测器。主unvisited_dual在评分前固定；所有预测封存后才评分。不是新任务确认，也不是论文完成。', '',
        '## 机制与可证明范围', '',
        '旧过程在每个Hamming壳先取K=8，再去掉母算法已知区域。新过程令F只等于本次自己的真实轨迹已访问模式，用F的前缀树补集状态加入支持语言DP，在截断前搜索未访问集合。六信用全部同增强；F不是不可行集合，F的原几何仍由母算法完整处理。', '',
        '同一DP状态的合法后缀与增量成本相同，因此按精确整数成本和字节序截断前K不丢失全局前K。原前K中的未访问路径在补集中排名只会提前；已访问有效区域由母流程保留。该证明保证搜索排名和池保留，不保证风险降低，也不证明PC独有能力。', '',
        '350小例360排名检查，351进一步独立枚举全部真实单观察语言笛卡尔积，共1665624模式/192份排名，逐壳最小值和全部提议精确一致。离线共38536几何证书通过。未把这些存档状态或共用几何免费提供给在线候选。', '',
        '## 相对旧同信用的实际变化', '',
        table(['方法', '旧MSE257', '新MSE257', '新−旧', '改善/同/变差', '额外正模式对', '本次完整均秒'], main_rows), '',
        '![误差变化及完整在线成本](unvisited_risk_and_cost.png)', '',
        '真实费用包括重新准备、629条状态更新、支持触发、当前F捕获、前缀树、搜索、几何、2048粒子、读出与轨迹指纹。native和Adam240在本轮重跑；旧129方法时间置空。每任务仅一次交错运行，不当成重复统计、等时间或等峰值内存证据。', '',
        '## 主候选与强控制', '',
        table(['对照', '对照MSE257', '主−对照', '改善/同/变差', '最差留一均差'], comp_rows), '',
        '全部8新方法对其余136方法、四指标的4352配对比较原样保留。留一检查是开发集中度诊断，不是显著性结论；不能根据表中较优变体事后换主。', '',
        '## 核验与交付边界', '',
        f"执行失败{summaries['predictions']['checks']['failures']}。独立审计{audit['counts']['risk_fields']}风险字段、{audit['counts']['comparison_rows']}比较、{audit['counts']['independent_certificates']}几何证书；最大标量风险差{audit['maximum_scalar_risk_gap']:.3g}。实际在线提议和支持侧普查逐项一致，原轨迹、触发和母池保持。", '',
        '原语、覆盖和任务效果分开报告。即使相对旧版改善，仍需同增强非乘子、强BP、资源匹配和新任务支持独立机制；固定回归与浅层头的实验比较不等于数学排除任意闭式方法或任意网络。尚无官方TTT-MLP/LLM/VLM实验。核心目标不因本轮流程完成而标完成。', '',
        '## 全部137方法', '',
        table(['方法', 'MSE257', 'MSE129', '单点MSE257', '单点MSE129', '本次均秒', '失败'], full_rows), '',
        '依据：[评分](../development_evaluation_v1/methods.json)、[全比较](../development_evaluation_v1/comparisons.json)、[独立审计](../development_audit_v1/summary.json)、[在线新增区域](../development_audit_v1/mechanisms.json)。紧凑发布不含完整粒子和逐调用几何日志，不是clean-clone完整科学复现。', '']
    write(out/'report.md', '\n'.join(text)); write(root/ENTRY, '# 354｜未访问补集在线比较已完成\n\n'+intro+'\n\n[完整图文和137方法](../../results/unvisited_online/report_v1/report.md)\n\n开发结果，核心独立收益目标仍未完成。\n')
    manifest = dict(passed=True, source_sha256=sha(Path(__file__)), entry_file=ENTRY, entry_sha256=sha(root/ENTRY),
        input_summary_sha256={p.relative_to(root).as_posix()+'/summary.json': sha(p/'summary.json') for p in folders.values()},
        outputs_sha256={n: sha(out/n) for n in ['report.md', 'figure_data.json', 'table_data.json', 'unvisited_risk_and_cost.png']})
    save(out/'manifest.json', manifest); print(dict(passed=True, report=str(out/'report.md')), flush=True)


if __name__ == '__main__': main()
