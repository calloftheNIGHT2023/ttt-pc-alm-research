"""341 display only, gated on all 340 predictions, scores and independent audit."""
import hashlib
import json
from pathlib import Path

BASE = 'results/search_radius_development'
PRIMARY = 'radius_first_fit_dual__k8__sall'
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
REFERENCES = ['online_first_fit_dual', 'online_first_fit_dual__k32', 'probe_all_alm64']
CONTROLS = [f'radius_first_fit_{c}__k8__sall' for c in CHANNELS if c != 'dual'] + [
    'online_first_fit_dual', 'online_first_fit_dual__k32', 'online_uniform_state_dual_plus_residual__k64',
    'probe_all_alm64', 'probe_then_adam15360_33', 'probe_pc1024_33', 'probe_nodual1024_33',
    'plain_alm1024_33', 'probe_alm512_33', 'cold__prior16384_ridge', 'cold__meta_ridge128', 'cold__meta_shallow64_20']
RESOURCE_PANEL = ['reference_fraction_first_fit_dual', 'online_first_fit_dual__k32',
    'online_uniform_state_dual_plus_residual__k64', 'probe_all_alm64', 'probe_then_adam3840_33',
    'probe_pc256_33', 'probe_nodual256_33', 'plain_alm256_33']


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
def write(p, value):
    with p.open('x', encoding='utf-8') as f: f.write(value)
def complete(p):
    if (p/'failure.json').exists(): raise RuntimeError('Preserved failure needs resolution before reporting')
    s = read(p/'summary.json'); assert s['passed']
    for n, digest in s.get('outputs_sha256', {}).items(): assert sha(p/n) == digest
    return s
def table(headers, rows):
    return '\n'.join(['| '+' | '.join(headers)+' |', '| '+' | '.join(['---']*len(headers))+' |']+
                     ['| '+' | '.join(map(str, r))+' |' for r in rows])+'\n'
def f(value): return '—' if value is None else f'{value:.10f}'


def gate(root):
    paths = {n: root/BASE/f'development_{n}_v1' for n in ['predictions', 'evaluation', 'audit']}
    # Refuse even to open partial quality outputs before every phase is terminal.
    for p in paths.values():
        assert (p/'summary.json').is_file() and not (p/'failure.json').exists(), p
    summaries = {n: complete(p) for n, p in paths.items()}
    a = summaries['audit']
    assert a['prediction_summary_sha256'] == sha(paths['predictions']/'summary.json')
    assert a['evaluation_summary_sha256'] == sha(paths['evaluation']/'summary.json')
    p = read(paths['predictions']/'protocol.json')
    assert p['seeds'] == list(range(328000000, 328000032)) and p['primary'] == PRIMARY
    for path, digest in p['source_sha256'].items(): assert sha(root/path) == digest
    assert not summaries['evaluation']['statistical_superiority_claim']
    assert not summaries['evaluation']['matched_resource_superiority_claim']
    cal = root/'results/budget_reinvestment/calibration_v1'; complete(cal)
    ca = complete(root/'results/budget_reinvestment/audit_v1')
    assert ca['calibration_summary_sha256'] == sha(cal/'summary.json')
    reach = root/'results/search_radius_reachability/audit_v1'; complete(reach)
    primitive = root/'results/search_radius_primitive/preflight_v1'; complete(primitive)
    return paths, summaries, p, cal, reach, primitive


def figures(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(14, 5.8)); ax.set_xlim(0, 14); ax.set_ylim(0, 5.8); ax.axis('off')
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.08', facecolor=color, edgecolor='none'))
        ax.text(x+w/2, y+h/2, label, ha='center', va='center', fontsize=10)
    def arrow(a, b): ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=14, linewidth=1.3, color='#586573'))
    box(.15, 3.7, 2.3, 1.0, 'Observed support\nSame state + credit', '#e5edf4')
    box(3.0, 3.7, 2.5, 1.0, 'Exact dynamic program\nAll Hamming shells', '#d6e9ed')
    box(6.1, 4.2, 2.5, .9, 'Original extraction\nShells m ... m + 2', '#f8e1c2')
    box(6.1, 2.6, 2.5, .9, 'Wider extraction\nShells m ... 16', '#cbe5db')
    box(9.3, 3.3, 2.0, 1.0, 'Exact feasibility\nVolume + sampling', '#e7e0f1')
    box(11.85, 3.3, 1.95, 1.0, 'Query readout\nMeasured risk', '#e4e8ec')
    arrow((2.5, 4.2), (2.95, 4.2)); arrow((5.55, 4.3), (6.03, 4.63)); arrow((5.55, 4.0), (6.03, 3.08))
    arrow((8.65, 4.62), (9.23, 4.0)); arrow((8.65, 3.05), (9.23, 3.58)); arrow((11.35, 3.8), (11.78, 3.8))
    r = data['reachability']
    ax.text(.15, 5.5, 'Change the extraction radius, not the information or trainable parameters', fontsize=16, weight='bold')
    ax.text(.15, 1.9, f"Exact audit: {r['outside_pairs']:,} missing positive-volume mode pairs are outside the old window.", fontsize=13)
    ax.text(.15, 1.35, f"They occur in {r['outside_tasks']}/512 development tasks ({r['fit_outside_tasks']} already fit their support).", fontsize=12)
    ax.text(.15, .65, 'At fixed K: no additional DP recurrence. Extra geometry, sampling and readout costs must still be charged.', fontsize=11)
    ax.text(.15, .18, 'Mode counts are not posterior mass or prediction gains. Every credit control receives the same radius extension.', fontsize=10, color='#485560')
    fig.tight_layout(); fig.savefig(out/'radius_mechanism.png', dpi=160); plt.close(fig)
    fig, ax = plt.subplots(figsize=(12, 7.5))
    colors = ['#b52845', '#8349b2', '#1c8c81', '#2b6eb4', '#da9037', '#707980']
    labels = ['Dual', 'Dual + residual', 'Residual', 'BP', 'Random sign', 'Zero']
    for channel, color, label in zip(CHANNELS, colors, labels):
        yy = [r['mse257'] for r in data['curves'] if r['channel'] == channel]
        assert len(yy) == 4
        ax.plot(range(4), yy, color=color, marker='o', linewidth=2, label=label)
    styles = [('Original dual / K8 / 3 shells', '--'), ('Dual / K32 / 3 shells', ':'), ('Strong probe-all ALM64', '-.')]
    for row, (label, style) in zip(data['references'], styles):
        ax.axhline(row['mse257'], color='#333b45', linestyle=style, linewidth=1.3, label=label)
    ax.set_xticks(range(4), [1, 2, 4, 8]); ax.set_xlabel('K retained paths per shell; all-shell extraction')
    ax.set_ylabel('Task-equal independent-query MSE (zoomed y-axis; lower is better)')
    ax.set_title('32 previously exposed development tasks — descriptive means only', loc='left', pad=18)
    ax.grid(alpha=.18); ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=.10, right=.97, top=.89, bottom=.30)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.17), ncol=3, frameon=False, fontsize=10)
    fig.text(.10, .035, 'All six credits share the same support, trajectory, radius and particle count. No new-task significance or matched-resource claim.', fontsize=9)
    fig.savefig(out/'radius_query_curves.png', dpi=160); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]; paths, summaries, p, cal, reach, primitive = gate(root)
    out = root/BASE/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    methods = read(paths['evaluation']/'methods.json'); by = {r['method']: r for r in methods}
    comparisons = {(r['candidate'], r['control'], r['metric']): r for r in read(paths['evaluation']/'comparisons.json')}
    union = next(r for r in read(reach/'methods.json') if r['method'] == 'strong_union')
    resource = {r['method']: r for r in read(cal/'methods.json')}; selection = read(cal/'selection.json')
    inputs = {(folder/'summary.json').relative_to(root).as_posix(): sha(folder/'summary.json')
              for folder in [*paths.values(), cal, root/'results/budget_reinvestment/audit_v1', reach, primitive]}
    data = dict(input_summary_sha256=inputs,
        reachability=dict(missing_pairs=union['certified_positive_mode_pairs'],
            outside_pairs=union['certified_categories']['above_window'], inside_pairs=union['certified_categories']['within_window'],
            outside_tasks=union['tasks_with_certified_above_window'], fit_outside_tasks=union['support_fit_tasks_with_certified_above_window']),
        curves=[dict(channel=c, k=k, method=f'radius_first_fit_{c}__k{k}__sall',
                     mse257=by[f'radius_first_fit_{c}__k{k}__sall']['metrics']['mse257']) for c in CHANNELS for k in [1, 2, 4, 8]],
        references=[dict(method=n, mse257=by[n]['metrics']['mse257']) for n in REFERENCES])
    figures(out, data); save(out/'figure_data.json', data)
    key_rows = []
    for name in CONTROLS:
        c = comparisons[PRIMARY, name, 'mse257']
        key_rows.append([name, f(by[name]['metrics']['mse257']), f(c['mean_difference']),
            f"{c['improved']}/{c['equal']}/{c['worse']}", f(c['worst_leave_one_out_mean'])])
    full_rows = [[r['method']]+[f(r['metrics'][metric]) for metric in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+
                 [f(r['mean_current_seconds']), str(r['failures'])] for r in methods]
    resource_rows = [[name, f(resource[name]['mean_seconds']), str(resource[name]['maximum_traced_peak_bytes']),
                      str(resource[name]['maximum_absolute_lifetime_peak_wset'])] for name in RESOURCE_PANEL]
    tables = dict(key_comparisons=key_rows, full_methods=full_rows, resources=resource_rows); save(out/'table_data.json', tables)
    primary = by[PRIMARY]
    text = ['# 341｜从搜索范围瓶颈到完整在线查询检验', '',
        '本报告为2026-09-25中午阶段交付。32个任务全部是已暴露开发数据，不是新盲集，也不是论文级确认。所有新预测先封存，再单独评分和独立审计；主候选在评分前已指定。', '',
        '## 本轮实际完成了什么', '',
        f"先完成88配置的1408次计时和116次独立内存测量；再在512旧开发任务中精确确认搜索范围限制；随后验证半径原语，并运行32任务×53新配置={summaries['predictions']['new_calls']}次真实fit。另引用同任务原51方法的封存预测，共比较{len(methods)}种方法。", '',
        f"预指定主候选为`{PRIMARY}`，本轮MSE257为**{primary['metrics']['mse257']:.10f}**，执行失败{primary['failures']}次。下面给出全部关键对照和全部方法，不能仅凭相对回归改善就归因于PC-ALM。", '',
        '## 数学机制：为何这次扩范围有明确对象', '',
        r'固定支持数据、触发状态和信用。精确DP为每个汉明距离h求按下界与字典序排列的K个候选。令m为至少一个下界非正的最小正距离。旧提议集只包含m至m+2三个壳；不论K多大，距离大于m+2的模式都不能由这条追加搜索输出。', '',
        f"四个强发现器的正池并集中，有{union['certified_positive_mode_pairs']}个主候选漏掉的任务—模式对，全部重新通过严格正体积证书。其中{data['reachability']['outside_pairs']}个在旧窗口外，涉及{data['reachability']['outside_tasks']}/512任务，含{data['reachability']['fit_outside_tasks']}个已拟合支持任务。它们不是随机无效分支，但数量也不是后验质量。", '',
        '![半径改变的位置与成本边界](radius_mechanism.png)', '',
        '新实现只扩大从同一DP表提取候选的距离范围。固定K时，DP递推及保留状态计数不变；扩大span得到嵌套提议。额外精确几何检查、采样、读出仍然计费。all表示覆盖所有距离壳，不表示枚举了所有模式或得到完整后验，因为每壳仍只保留有限K。', '',
        '## 乘子为什么可能仍有用，以及必须如何归因', '',
        '当支持损失已为零时，仅按该损失做普通梯度下降不再移动参数。但在有限步、尚未同时满足层间约束与驻点条件的ALM状态中，乘子仍可能非零，保留内部约束和历史信息。可检验假设是：这些信号能否在同预算搜索中找出更有预测价值的可行解释。', '',
        '这不是说非零乘子在此等于正确BP信用；那种对应需要约束与驻点条件。零信用、残差、BP信用和随机符号都获得同一全壳搜索、同一轨迹与观测，因而只有与这些控制比较，才能判断乘子是否提供独立收益。扩半径本身不是PC-ALM专属贡献。', '',
        '## 为什么“模式更多”不能替代查询检验', '',
        r'令S为支持数据，T为当前保留的可行参数集合，μ_S为完整后验预测均值，μ_T为截断集合预测均值，V_T为该集合的预测方差，M为理想独立粒子数。以查询网格上的平均平方范数计，条件风险分解为：', '',
        r'$$R_M(T\mid S)=R_{\mathrm{Bayes}}(S)+\|\mu_T-\mu_S\|_q^2+V_T/M.$$','',
        r'增加不相交区域R，令a为其在合并集合中的后验权重，d=μ_R−μ_T，e=μ_T−μ_S，则风险变化为：', '',
        r'$$\Delta R=2a\langle e,d\rangle_q+a^2\|d\|_q^2+\{a(V_R-V_T)+a(1-a)\|d\|_q^2\}/M.$$','',
        '只有右侧为负才保证在上述理想条件下改善。实际数值体积和采样器还要额外验证；本轮对固定teacher的开发MSE不是这个条件期望恒等式的无条件证明。因此直接运行独立查询，并保留全部成功与失败样本。', '',
        '## 同半径、不同信用与K的查询结果', '',
        '![全壳搜索的查询风险曲线](radius_query_curves.png)', '',
        '六条曲线来自同一32任务。纵轴为放大刻度；没有置信区间或显著性标记。虚线为固定对照，不因本轮查询表现挑选。', '',
        '## 主候选的预列关键对照', '',
        '差值=主候选MSE−对照MSE，负数表示主候选开发均值较低。改善/相同/变差以任务为单位；最差留一均差用于观察是否由单个任务主导，不是显著性检验。', '',
        table(['对照', '对照MSE257', '主−对照', '改善/同/变差', '最差留一均差'], key_rows), '',
        '## 资源证据及不能混用的时间', '',
        f"337的同期8任务重复校准预算B={selection['budget_seconds']:.10f}秒，12信用共同K={selection['common_k']}；uniform控制自身K64也保留。Adam3840仍在预算内，故340新增7680/15360步及更长PC、无乘子和ALM。", '',
        table(['337方法', '完整均时/秒', 'tracemalloc峰值/B', '绝对进程生命周期峰值/B'], resource_rows), '',
        '上表来自同一337资源校准，内存为首尾两个任务独立进程的最大记录。绝对进程峰值包含导入和预加载模型；tracemalloc不一定涵盖原生库分配，不能相减当纯算法峰值。340全表中的时间仅为新配置本次单次开发运行，旧方法填空，不与旧时间拼成同预算胜出。新候选要进入新任务确认，仍需同期重复计时、峰值测量和足够强的预算前沿。', '',
        '## 全部方法，不按效果删行', '',
        table(['方法', 'MSE257', 'MSE129', 'point MSE257', 'point MSE129', '本轮新fit均时/秒', '失败'], full_rows), '',
        '## 验证、交付边界与下一步', '',
        f"独立审计检查{summaries['audit']['independent_risk_fields']}个风险字段、{summaries['audit']['comparison_rows']}条比较，最大风险差{summaries['audit']['maximum_risk_gap']:.3g}；预检重放、上游状态、支持粒子和候选嵌套均验证。图像仍须单独实际查看后才标记视觉验收。", '',
        '当前是4层标量tent模型、4个支持观测、4个可调偏置及2048粒子的可控理论实验。不是官方TTT-MLP、LLM/VLM验证，不声称普遍胜过所有闭式解或任意新增网络层。后续只在开发效果具有竞争力且资源边界核验后进入新任务确认；研究总目标仍进行中。', '',
        '[完整评分表](../development_evaluation_v1/methods.json) · [全部描述性比较](../development_evaluation_v1/comparisons.json) · [独立审计](../development_audit_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(text))
    entry = root/'outputs/ttt-pc-alm-research/341_radius_development_results_v1.md'
    write(entry, '# 341｜搜索半径机制与32任务在线开发结果\n\n[完整图文报告](../../results/search_radius_development/report_v1/report.md)\n\n'
        '此版本仅描述已暴露开发任务；不宣称新任务显著性或匹配资源优势。请结合报告中的全部104方法及独立审计读取，核心研究目标仍进行中。\n')
    save(out/'manifest.json', dict(input_summary_sha256=inputs, generator_source_sha256=sha(Path(__file__)),
        primary=PRIMARY, tasks=32, methods=len(methods), visual_qa_passed=False, scientific_source_changed=False,
        outputs_sha256={n: sha(out/n) for n in ['report.md', 'figure_data.json', 'table_data.json', 'radius_mechanism.png', 'radius_query_curves.png']},
        entry_file=entry.relative_to(root).as_posix(), entry_sha256=sha(entry)))
    print(dict(report=str(out/'report.md'), methods=len(methods), visual_qa_passed=False), flush=True)


if __name__ == '__main__': main()
