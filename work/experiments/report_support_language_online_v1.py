"""346 result-driven figures and complete tables, only after 345 independent audit."""
from collections import Counter
from pathlib import Path
from report_search_radius_development_v1 import read, sha, save, write, complete, table, f

BASE = 'results/support_language_online'
CHANNELS = ['dual', 'dual_plus_residual', 'residual', 'bp', 'random_sign', 'zero']
PRIMARY = 'language_first_fit_dual'
ENTRY = 'outputs/ttt-pc-alm-research/346_support_language_results_v1.md'
STRONG = ['probe_all_alm64', 'probe_alm256_33', 'probe_alm512_33', 'probe_then_adam15360_33',
          'plain_alm1024_33', 'probe_pc1024_33', 'cold__prior16384_ridge', 'cold__meta_ridge128', 'cold__meta_shallow64_20']


def figures(out, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    labels = ['Dual', 'Dual + residual', 'Residual', 'BP', 'Random sign', 'Zero']
    x = np.arange(6)
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(x, [r['old_mse257'] for r in data['credits']], marker='o', color='#7c8794', label='Original all-shell K8')
    ax.plot(x, [r['new_mse257'] for r in data['credits']], marker='s', color='#137f79', label='Support-language K8')
    ax.axhline(data['strong_alm'], color='#bc3e57', linestyle='--', label='Strong probe-all ALM64')
    ax.axhline(data['strong_adam'], color='#9165a9', linestyle=':', label='Strong Adam15360')
    ax.set_xticks(x, labels); ax.set_ylabel('Mean independent-query MSE (zoomed scale; lower is better)')
    ax.set_title('32 exposed development tasks: same support, state and K; six credit controls', loc='left', pad=18)
    ax.grid(alpha=.2); ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=.1, right=.97, top=.90, bottom=.22)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False)
    fig.text(.10, .025, 'Descriptive development means only. No new-task significance or matched-resource superiority claim.', fontsize=10)
    fig.savefig(out/'query_comparison.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    axes[0].bar(x, [r['added_positive_pairs'] for r in data['credits']], color='#137f79', label='New positive mode pairs')
    axes[0].bar(x, [-r['lost_positive_pairs'] for r in data['credits']], color='#d39753', label='Lost old mode pairs')
    axes[0].axhline(0, color='#54616c', linewidth=1)
    axes[0].set_title('Actual change in the feasible memory pool', loc='left', pad=14)
    axes[0].set_ylabel('Task-mode pairs vs each original credit pool')
    axes[0].legend(frameon=False, fontsize=9)
    axes[1].bar(x, [r['mean_current_seconds'] for r in data['credits']], color='#527aac')
    axes[1].set_title('Complete new fit + readout cost', loc='left', pad=14)
    axes[1].set_ylabel('Mean seconds; no historical baseline time mixed in')
    for ax in axes:
        ax.set_xticks(x, labels, rotation=24, ha='right'); ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='y', alpha=.2)
    fig.subplots_adjust(left=.07, right=.97, bottom=.24, top=.88, wspace=.3)
    fig.text(.07, .025, 'More feasible modes do not guarantee lower risk. Times include the actual online geometry and sampling work.', fontsize=10)
    fig.savefig(out/'memory_and_cost.png', dpi=160); plt.close(fig)


def main():
    root = Path(__file__).resolve().parents[2]
    folders = {n: root/BASE/f'development_{n}_v1' for n in ['predictions', 'evaluation', 'audit']}
    summaries = {n: complete(p) for n, p in folders.items()}
    assert summaries['audit']['prediction_summary_sha256'] == sha(folders['predictions']/'summary.json')
    assert summaries['audit']['evaluation_summary_sha256'] == sha(folders['evaluation']/'summary.json')
    census = complete(root/'results/radius_candidate_census/audit_v1')
    primitive = complete(root/'results/support_language_primitive/preflight_v1')
    protocol = read(folders['predictions']/'protocol.json')
    for path, digest in protocol['source_sha256'].items(): assert sha(root/path) == digest
    assert protocol['primary'] == PRIMARY and len(protocol['seeds']) == 32
    methods = read(folders['evaluation']/'methods.json'); by = {m['method']: m for m in methods}
    comparisons = {(r['candidate'], r['control'], r['metric']): r for r in read(folders['evaluation']/'comparisons.json')}
    mechanisms = read(folders['audit']/'mechanisms.json')
    data = dict(credits=[], strong_alm=by['probe_all_alm64']['metrics']['mse257'],
                strong_adam=by['probe_then_adam15360_33']['metrics']['mse257'])
    for ch in CHANNELS:
        name = 'language_first_fit_'+ch; rr = [r for r in mechanisms if r['method'] == name]
        data['credits'].append(dict(channel=ch, method=name, new_mse257=by[name]['metrics']['mse257'],
            old_mse257=by[f'radius_first_fit_{ch}__k8__sall']['metrics']['mse257'],
            added_positive_pairs=sum(len(r['new_vs_old_credit_pool']) for r in rr),
            lost_positive_pairs=sum(len(r['old_credit_pool_lost']) for r in rr),
            tasks_with_new_positive=sum(bool(r['new_vs_old_credit_pool']) for r in rr),
            mean_current_seconds=by[name]['mean_current_seconds'],
            max_dp_entries=max(r['dp_max_entries'] for r in rr)))
    out = root/BASE/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    figures(out, data); save(out/'figure_data.json', data)
    main_rows = [[r['method'], f(r['new_mse257']), f(r['old_mse257']),
                  str(r['added_positive_pairs']), str(r['lost_positive_pairs']), f(r['mean_current_seconds'])] for r in data['credits']]
    controls = ['online_first_fit_dual', 'online_first_fit_dual__k32']+['language_first_fit_'+c for c in CHANNELS[1:]]+STRONG
    comp_rows = []
    for name in controls:
        r = comparisons[PRIMARY, name, 'mse257']
        comp_rows.append([name, f(by[name]['metrics']['mse257']), f(r['mean_difference']),
                          f"{r['improved']}/{r['equal']}/{r['worse']}", f(r['worst_leave_one_out_mean'])])
    all_rows = [[m['method']]+[f(m['metrics'][k]) for k in ['mse257', 'mse129', 'point_mse257', 'point_mse129']]+
                [f(m['mean_current_seconds']), str(m['failures'])] for m in methods]
    save(out/'table_data.json', dict(main=main_rows, comparisons=comp_rows, full=all_rows))
    main = by[PRIMARY]; aa = comparisons[PRIMARY, 'probe_all_alm64', 'mse257']; ad = comparisons[PRIMARY, 'probe_then_adam15360_33', 'mse257']
    previous = comparisons[PRIMARY, 'online_first_fit_dual', 'mse257']
    text = ['# 346｜支持条件路径语言：完整在线结果与数学边界', '',
        '2026-09-25中午阶段交付补充。全部32个任务为已暴露开发集；先封存预测，后评分，独立审计通过。不是未见任务确认。', '',
        '## 结果先行', '',
        f"预指定主候选 `language_first_fit_dual` 的查询MSE257为 **{main['metrics']['mse257']:.10f}**。相对原主候选均差 {previous['mean_difference']:+.10f}，相对强probe-all ALM64 {aa['mean_difference']:+.10f}，相对Adam15360 {ad['mean_difference']:+.10f}。差值为新方法减对照，负数才是开发均值改善。", '',
        '不能只选其中有利比较宣布成功。是否有独立PC-ALM收益，还要看同样增强的非乘子信用、强优化器以及后续同期资源和未见任务实验。当前核心研究目标保持未完成。', '',
        '![六信用查询比较](query_comparison.png)', '',
        '## 从实际瓶颈到这次改变', '',
        '343对32任务发现：旧主候选扩大距离后新增2696个提议，2694个严格不可行，2个已在旧正池；所有返回数组逐字节不变。四个强对照正池中，主候选漏掉93个真正可行模式，全部在其距离壳的前8名以外（92个非平局）。此前323已经指出过排序瓶颈；这里补的是新32任务的完整证据，不把重复诊断写成新贡献。', '',
        '原排序把更负的信用下界排在前面。但“下界≤0”只意味着尚未证明不可行，不意味着真的可行。许多单个支持样本就无法满足的路径，仍可能占满有限名额。', '',
        '新原语为每个已观察支持对预计算允许的层间分支序列，编译成有限无环自动机，再把自动机状态纳入精确K-best动态规划。先约束候选空间，再截断前K条；不是先挑K条再过滤。未访问查询答案、teacher参数或参考正池。', '',
        '## 可检验的数学结论', '',
        r'对单个支持样本，固定分支路径，令 $I_0=\{x_i\}$，$J_l=(I_{l-1}+[-B,B])\cap Z_{p_{li}}$，$I_l=s_{p_{li}}J_l+c_{p_{li}}$。每层新增偏置的连续区间像及固定分支仿射像都是精确区间；末层与观测带相交当且仅当该单样本路径存在偏置见证。整数公共尺度保持binary64输入的精确有理解释。', '',
        '任何全网络可行模式的每个样本列必然通过。因此删去不通过的模式是安全必要条件；不同样本各自通过却未必共享同一组偏置，所以仍需后续全网络几何检查。这个必要条件本身不属于PC-ALM专有能力。', '',
        'DP状态为“距离、上一层image mask、各样本的自动机状态”。相同状态具有相同未来选项和成本，故每状态保留K个最小前缀仍是精确K-best，而不是束搜索近似。约束后壳最小下界不会变小，也不会错误排除真实可行模式；但有限K选出的旧池与新池未必嵌套，更不能推出风险单调改善。', '',
        '一个可直接手算的严格筛选例子：两层、单样本x=0.01、v=0.1、B=0.12、eps=0.001，原模式(0,0)，信用(-1,-1)，每壳K=1。旧松弛在距离2选(1,2)，新语言选择(1,1)。前者在第一层至多得到h1=0.26，第二层预激活至多0.38，不能进入从0.5开始的分支2；后者可取b1=0、b2=0.03满足观测。实际精确程序得到旧下界约−1.101、新下界约−0.221：更负的旧值反而对应不可能路径。这个事后构造的数学示例说明严格筛选能力，不作为预注册任务收益证据，也不证明只有乘子能做到。', '',
        '边界：单列路径枚举含4^d的深度指数项，产品自动机也可能增长；本实验深度仅4。完整枚举器过去已有单样本筛查，本轮的新工程机制是把该约束放进信用DP的截断之前，不宣称首次提出区间可达性。', '',
        '## 六信用共同增强后的实际记忆和成本', '',
        table(['新方法', '新MSE257', '旧全壳MSE257', '新增正模式对', '丢失旧正模式对', '本次完整均秒'], main_rows), '',
        '![记忆与完整成本](memory_and_cost.png)', '',
        '新增/丢失以各自原信用正池为参照；同一模式在不同任务分开计数。正模式数量不是后验质量或风险收益。时间仅比较本次六个真实完整调用，旧104方法的时间留空；不能用本表宣称相对历史强对照同预算更优。', '',
        '## 预指定主候选的完整关键比较', '',
        table(['对照', '对照MSE257', '主−对照', '改善/同/变差', '最差留一均差'], comp_rows), '',
        '留一只是集中度诊断，不是显著性证明。所有32任务均保留，不删除困难任务。支持拟合/回退分组及所有比较在原始JSON中公开。', '',
        '## 执行与核验', '',
        f"六配置预检后完成192次新fit，加同任务104旧对照，共110方法、3520预测器。原轨迹/触发/输入、首任务重放及支持粒子核验通过，失败{summaries['predictions']['checks']['failures']}。独立标量风险{summaries['audit']['counts']['risk_fields']}项，最大差{summaries['audit']['maximum_scalar_risk_gap']:.3g}；比较行{summaries['audit']['counts']['comparison_rows']}，新几何有理证书{summaries['audit']['counts']['independent_certificates']}。", '',
        '本轮仍是4层标量tent少样本记忆原型，不是官方TTT-MLP或LLM/VLM结果。几何读出明确使用公共全局LP；局部提议器禁止全局BP/LP，不能因此把整个系统写成纯局部PC。外部闭式头的普遍不可能性并未证明。', '',
        '下一步需区分两个可改进环节：一是单样本语言仍未约束跨样本共享偏置，二是当前追加搜索的基础发现池弱于强ALM。后续必须明确检验更紧约束或对强母池的新增价值，并为所有信用/有无追加搜索提供相同条件，不能只增强候选而保留弱对照。', '',
        '## 全部110方法', '',
        table(['方法', 'MSE257', 'MSE129', '单点MSE257', '单点MSE129', '本次均秒', '失败'], all_rows), '',
        '原始依据：[评分](../development_evaluation_v1/methods.json)、[全部配对差](../development_evaluation_v1/comparisons.json)、[独立审计](../development_audit_v1/summary.json)、[记忆变化](../development_audit_v1/mechanisms.json)。', '',
        '图文的数字检查与实际视觉检查另有收据。Git阶段包为紧凑归档，省略完整粒子预测数组和每次几何日志，不等于全量远程备份。', '']
    write(out/'report.md', '\n'.join(text))
    write(root/ENTRY, '# 346｜支持条件路径语言实验已完成\n\n[完整数学推导、两幅图、六信用和全部110方法](../../results/support_language_online/report_v1/report.md)\n\n'+
        f"32开发任务，192新fit，主MSE257={main['metrics']['mse257']:.10f}。主−原={previous['mean_difference']:+.10f}；主−强ALM={aa['mean_difference']:+.10f}；主−Adam15360={ad['mean_difference']:+.10f}。负数表示本次开发均值较低。\n\n"+
        '全部预测封存后评分、独立审计通过；不是新盲测、不是同期资源匹配证明，核心独立收益目标仍未标完成。\n')
    manifest = dict(passed=True, source_sha256=sha(Path(__file__)), entry_file=ENTRY, entry_sha256=sha(root/ENTRY),
        input_summary_sha256={p.relative_to(root).as_posix()+'/summary.json': sha(p/'summary.json') for p in folders.values()},
        outputs_sha256={n: sha(out/n) for n in ['report.md', 'figure_data.json', 'table_data.json', 'query_comparison.png', 'memory_and_cost.png']})
    save(out/'manifest.json', manifest); print(dict(passed=True, report=str(out/'report.md')), flush=True)


if __name__ == '__main__': main()
