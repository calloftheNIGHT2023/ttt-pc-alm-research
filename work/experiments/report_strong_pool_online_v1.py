"""349 audited strong-parent marginal-value report, figures and every method."""
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save, write, complete, table, f

BASE = 'results/strong_pool_online'
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
PRIMARY = 'strong_language_dual'
NATIVE = 'strong_native_alm64'
ENTRY = 'outputs/ttt-pc-alm-research/349_strong_pool_results_v1.md'


def figures(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    import numpy as np
    fig, ax = plt.subplots(figsize=(14, 5.5)); ax.set_xlim(0, 14); ax.set_ylim(0, 5.5); ax.axis('off')
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color, edgecolor='none'))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=15, color='#506471', linewidth=1.5))
    ax.text(.2, 5.1, 'Test credit-search value on top of the actual strong discovery pipeline', fontsize=15, weight='bold')
    box(.2, 2.6, 2.25, 1.2, 'Same observed support\n17 prior starts\n16 preparation steps', '#e7edf3')
    box(3.0, 2.6, 2.25, 1.2, '37 trials per start\n629 live states\n64 ALM steps each', '#d6e9e7')
    box(5.9, 3.25, 2.65, 1.0, 'Original visited-mode pool T\nUnchanged baseline path', '#dae8f7')
    box(5.9, 1.35, 2.65, 1.15, 'First exact support-fit state\nSame six credit controls\nLanguage-constrained K8', '#f1e2c9')
    box(9.2, 2.4, 2.15, 1.2, 'Charged geometry\nT union accepted R\nSame prior + particles', '#e8def0')
    box(11.95, 2.4, 1.8, 1.2, 'Query prediction\nTargets used\nonly for scoring', '#e9edf0')
    arrow((2.5, 3.2), (2.93, 3.2)); arrow((5.3, 3.45), (5.83, 3.75)); arrow((5.3, 2.9), (5.83, 2.0))
    arrow((8.62, 3.75), (9.13, 3.2)); arrow((8.62, 1.95), (9.13, 2.75)); arrow((11.42, 3.0), (11.87, 3.0))
    ax.text(.2, .63, 'Required attribution: added value beyond native, watch-only, equally enhanced non-dual credits and stronger optimizers.', fontsize=11)
    ax.text(.2, .15, 'No archived strong pool is given to the candidate. All preparation, search, geometry, sampling and readout are charged.', fontsize=10)
    fig.tight_layout(); fig.savefig(out/'strong_pool_mechanism.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 6)); xx = np.arange(6)
    labels = ['Dual', 'Dual + residual', 'Residual', 'BP', 'Random sign', 'Zero']
    differences = [r['delta_to_native'] for r in data['credits']]
    axes[0].bar(xx, differences, color=['#187c77' if y < 0 else '#bb6d55' for y in differences])
    axes[0].axhline(0, color='#53616c', linewidth=1)
    axes[0].set_title('Incremental query error: candidate minus native', loc='left', pad=14)
    axes[0].set_ylabel('Mean MSE difference; negative is improvement')
    if all(y == 0 for y in differences):
        axes[0].set_ylim(-1, 1); axes[0].set_yticks([0])
        axes[0].text(.5, .8, 'All six stored mean differences are zero', transform=axes[0].transAxes, ha='center')
    axes[1].bar(xx, [r['mean_seconds'] for r in data['credits']], color='#527cad')
    axes[1].axhline(data['native_seconds'], color='#b74659', linestyle='--', label='Native ALM64')
    axes[1].axhline(data['watch_seconds'], color='#8d6aa6', linestyle=':', label='Watch only')
    axes[1].set_title('Complete calls measured in this same run', loc='left', pad=14)
    axes[1].set_ylabel('Mean seconds; one repetition per task')
    axes[1].legend(frameon=False)
    for axis in axes:
        axis.set_xticks(xx, labels, rotation=24, ha='right'); axis.spines[['top', 'right']].set_visible(False); axis.grid(axis='y', alpha=.18)
    fig.subplots_adjust(left=.08, right=.98, bottom=.25, top=.87, wspace=.32)
    fig.text(.08, .025, '32 exposed development tasks. No new-task significance, fixed-wall-time match or peak-memory match claimed.', fontsize=10)
    fig.savefig(out/'incremental_risk_and_cost.png', dpi=160); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    folders = {n: root/BASE/f'development_{n}_v1' for n in ['predictions', 'evaluation', 'audit']}
    summaries = {n: complete(p) for n, p in folders.items()}
    audit = summaries['audit']; assert audit['prediction_summary_sha256'] == sha(folders['predictions']/'summary.json')
    assert audit['evaluation_summary_sha256'] == sha(folders['evaluation']/'summary.json')
    protocol = read(folders['predictions']/'protocol.json')
    for path, digest in protocol['source_sha256'].items(): assert sha(root/path) == digest
    assert protocol['primary'] == PRIMARY and len(protocol['seeds']) == 32
    methods = read(folders['evaluation']/'methods.json'); by = {m['method']: m for m in methods}
    comps = {(r['candidate'], r['control'], r['metric']): r for r in read(folders['evaluation']/'comparisons.json')}
    mechanisms = read(folders['audit']/'mechanisms.json')
    data = dict(native_mse257=by[NATIVE]['metrics']['mse257'], native_seconds=by[NATIVE]['mean_current_seconds'],
                watch_seconds=by['strong_watch_alm64']['mean_current_seconds'], credits=[])
    for ch in CHANNELS:
        name = 'strong_language_'+ch; c = comps[name, NATIVE, 'mse257']; mm = [r for r in mechanisms if r['method'] == name]
        data['credits'].append(dict(channel=ch, method=name, mse257=by[name]['metrics']['mse257'],
            delta_to_native=c['mean_difference'], improved=c['improved'], equal=c['equal'], worse=c['worse'],
            new_positive_pairs=sum(len(r['new_positive_modes']) for r in mm),
            tasks_with_new_positive=sum(bool(r['new_positive_modes']) for r in mm), mean_seconds=by[name]['mean_current_seconds']))
    out = root/BASE/'report_v1'; out.mkdir(parents=True, exist_ok=False); figures(out, data); save(out/'figure_data.json', data)
    main_rows = [[r['method'], f(r['mse257']), f(r['delta_to_native']), f"{r['improved']}/{r['equal']}/{r['worse']}",
                  str(r['new_positive_pairs']), str(r['tasks_with_new_positive']), f(r['mean_seconds'])] for r in data['credits']]
    controls = [c['name'] for c in protocol['configs'] if c['name'] != PRIMARY]+['language_first_fit_dual', 'cold__prior16384_ridge', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
    comp_rows = []
    for name in controls:
        c = comps[PRIMARY, name, 'mse257']
        comp_rows.append([name, f(by[name]['metrics']['mse257']), f(c['mean_difference']),
                          f"{c['improved']}/{c['equal']}/{c['worse']}", f(c['worst_leave_one_out_mean'])])
    full_rows = [[m['method']]+[f(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+
                [f(m['mean_current_seconds']), str(m['failures'])] for m in methods]
    save(out/'table_data.json', dict(main=main_rows, comparisons=comp_rows, full=full_rows))
    main = by[PRIMARY]; marginal = comps[PRIMARY, NATIVE, 'mse257']; zero = comps[PRIMARY, 'strong_language_zero', 'mse257']
    text = ['# 349｜强母池上的信用搜索：实际增益、成本和完整对照', '',
        '2026-09-25中午交付补充。32个已暴露开发任务，19新方法加110封存对照，全部预测先封存、后评分与独立审计。主候选事前固定，未根据评分改名或删任务。', '',
        '## 先看是否有新增价值', '',
        f"主 `strong_language_dual` 的MSE257为 **{main['metrics']['mse257']:.10f}**，原强母算法本次为 **{by[NATIVE]['metrics']['mse257']:.10f}**。主−母均差 **{marginal['mean_difference']:+.10f}**，32任务改善/相同/变差为 **{marginal['improved']}/{marginal['equal']}/{marginal['worse']}**。主−同增强零信用均差 **{zero['mean_difference']:+.10f}**。负数才表示主方法开发均值更低。", '',
        '这些是完整实测，不预先将桥接成功等同于任务成功。即便候选相对旧弱候选更好，也必须先扣除强母算法本来就有的效果。新盲任务与严格资源匹配尚未完成，核心独立收益目标不因此标完成。', '',
        '![强流程中的信用追加位置](strong_pool_mechanism.png)', '',
        '## 原因：母算法不同，不是简单多跑几步', '',
        '旧候选33起点各展开分支试探但只保留一个优胜状态，再续接32步和一个锚点32步。强probe-all ALM64从17起点各保留37个试探，共629状态，每个续接64步。后者搜索路径更丰富，批量更新仍较快。新实现忠实保留强方法的全部原状态、实际参数模式和几何池，不载入历史结果作免费母池。', '',
        '只在该强轨迹上选择第一个精确满足支持带的状态；若全部不满足，选观测带外误差最小者。六种信用从同一b/h/u读取，再执行同一支持条件语言全半径K8。BP只在明确控制中计算；无BP候选不先执行BP更新或利用BP初始化乘子。公共几何LP仍存在且收费。', '',
        '## 数学可证明的部分', '',
        r'若母算法正池为T，当前信用提议经几何验收的新集合为R，则输出使用T∪R。保持母轨迹与几何确定性时，T不丢失；R为空时，相同规范顺序、相同粒子随机数和读出函数保证预测逐位不变。此结论是实现不变量，不是R非空时风险必降。', '',
        r'对于理想独立粒子，记e=μ_T−μ_S、d=μ_R−μ_T、a为新增区域的合并权重，则条件风险差仍为 $2a\langle e,d\rangle+a^2\|d\|^2+\{a(V_R-V_T)+a(1-a)\|d\|^2\}/M$。只有该量为负才保证改善。增加池或使用非零乘子本身不构成充分条件。', '',
        'watch控制支付只读触发器费用但不追加候选，必须完整复现母算法输出；native不支付观察费用，是更低开销的真实基线。更长ALM、普通PC、无乘子、Adam都获得同样629起点，不把它们限制在旧33起点上比较。', '',
        '## 六信用相对同一个强母算法', '',
        table(['信用方法', 'MSE257', '该方法−native', '改善/同/变差', '新增正模式对', '新增任务', '本次完整均秒'], main_rows), '',
        '![新增风险与同期成本](incremental_risk_and_cost.png)', '',
        '时间是同一轮、随机调用顺序下的每任务一次完整fit，包含实际准备、局部更新、精确触发、语言搜索、几何、采样、读出及轨迹指纹。不是重复计时统计，也不是等墙钟或等峰值内存约束；旧110方法历史时间不混入。', '',
        '## 主候选对全部新强控制及代表性旧控制', '',
        table(['对照', '对照MSE257', '主−对照', '改善/同/变差', '最差留一均差'], comp_rows), '',
        '所有19新方法与其余128方法的四指标配对差均在原始JSON中保留，总9728条。表内留一是集中度诊断，不是显著性结论。不能在多个开发变体中挑获胜者后当作预指定主候选。', '',
        '## 执行与核验结果', '',
        f"完整608次新fit，加110旧方法封存预测，32×129=4128预测器。执行失败{summaries['predictions']['checks']['failures']}。独立标量风险{audit['counts']['risk_fields']}字段，最大差{audit['maximum_scalar_risk_gap']:.3g}；比较{audit['counts']['comparison_rows']}行；轨迹{audit['counts']['trajectory_states']}状态；新几何证书{audit['counts']['independent_certificates']}。", '',
        '347额外检查8任务64调用，逐位核对母算法6个数组及只读原实现逐步轨迹。348第一任务19配置预检还逐数组复核全部12个native强控制的原legacy实现；正式首任务重放19配置、前8任务重放347，输入与支持粒子全部检查。', '',
        '本轮仍为4层标量tent模型，不是官方TTT-MLP、LLM或VLM验证。局部信用搜索可因母池已覆盖建议而不改变输出；这必须据实际完整结果报告。闭式回归与附加网络层的普遍不可能性没有证明。', '',
        '## 全部129方法', '',
        table(['方法', 'MSE257', 'MSE129', '单点MSE257', '单点MSE129', '本次均秒', '失败'], full_rows), '',
        '依据：[评分](../development_evaluation_v1/methods.json)、[全部比较](../development_evaluation_v1/comparisons.json)、[独立审计](../development_audit_v1/summary.json)、[实际新增模式](../development_audit_v1/mechanisms.json)。图文数字和视觉QA另有收据。阶段发布省略全部粒子数组及完整几何日志，不是全量远程数据备份。', '']
    write(out/'report.md', '\n'.join(text))
    write(root/ENTRY, '# 349｜强母池信用追加实验已完成\n\n[完整数学、两幅图、六信用和129方法](../../results/strong_pool_online/report_v1/report.md)\n\n'+
        f"32开发任务，608次新fit。主MSE257={main['metrics']['mse257']:.10f}；主−原强母方法={marginal['mean_difference']:+.10f}；改善/同/变差={marginal['improved']}/{marginal['equal']}/{marginal['worse']}。主−同增强zero={zero['mean_difference']:+.10f}。\n\n"+
        '所有预测封存后评分与独立审计通过，仍不等于新盲任务和匹配资源下的独立优势；核心目标继续保持未完成。\n')
    manifest = dict(passed=True, source_sha256=sha(Path(__file__)), entry_file=ENTRY, entry_sha256=sha(root/ENTRY),
        input_summary_sha256={p.relative_to(root).as_posix()+'/summary.json': sha(p/'summary.json') for p in folders.values()},
        outputs_sha256={n: sha(out/n) for n in ['report.md', 'figure_data.json', 'table_data.json', 'strong_pool_mechanism.png', 'incremental_risk_and_cost.png']})
    save(out/'manifest.json', manifest); print(dict(passed=True, report=str(out/'report.md')), flush=True)


if __name__ == '__main__': main()
