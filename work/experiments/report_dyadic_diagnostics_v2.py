"""336 report only audited arithmetic-equivalence and measured runtime evidence."""
import hashlib
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

BASE = 'results/online_credit_fresh_pilot'
DOCUMENT = 'outputs/ttt-pc-alm-research/336_dyadic_compute_results_v2.md'


def read(p): return json.loads(p.read_text(encoding='utf-8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def write(p, value):
    with p.open('x', encoding='utf-8') as f: json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)


def complete(p):
    result = read(p/'summary.json'); assert result['passed']
    for name, digest in result.get('outputs_sha256', {}).items(): assert sha(p/name) == digest
    return result


def label(name):
    return name.replace('online_first_fit_', 'First-fit / ').replace('online_uniform_state_', 'Uniform / ').replace('dual_plus_residual', 'dual+residual').replace('random_sign', 'random-sign')


def main():
    root = Path(__file__).resolve().parents[2]; base = root/BASE
    audit = complete(base/'dyadic_diagnostics_audit_v1')
    for folder, digest in audit['input_summary_sha256'].items(): assert sha(base/folder/'summary.json') == digest
    ds = complete(base/'cost_readout_diagnostic_v1'); ks = complete(base/'dyadic_branch_kernel_v1'); os = complete(base/'dyadic_online_v1')
    old = {r['method']: r for r in read(base/'cost_readout_diagnostic_v1/methods.json')}
    records = read(base/'dyadic_online_v1/calls.json'); results = read(base/'dyadic_online_v1/methods.json')
    names = [r['method'] for r in results]; by = {r['method']: r for r in results}
    primary = 'online_first_fit_dual'; secondary = 'online_uniform_state_dual_plus_residual'
    p, s = by[primary], by[secondary]; original = old[primary]
    out = base/'dyadic_report_v2'; out.mkdir(parents=True, exist_ok=False)
    implementations = ['fraction', 'integer_dp', 'integer_dp_trigger']
    styles = ['#9babc0', '#437bc0', '#138b81']
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.2), gridspec_kw={'width_ratios': [1.55, 1]})
    y = np.arange(len(names)); offset = .23
    for i, impl in enumerate(implementations):
        axes[0].barh(y+(i-1)*offset, [by[n]['mean_seconds'][impl] for n in names], height=.21,
                     color=styles[i], label=['Original Fraction', 'Integer search only', 'Integer search + trigger'][i])
    axes[0].set_yticks(y, [label(n) for n in names], fontsize=9); axes[0].invert_yaxis()
    axes[0].set_xlabel('Complete online fit + query readout (seconds)')
    axes[0].set_title('All 12 credit controls receive the same optimization', loc='left', fontsize=12)
    axes[0].set_ylim(len(names)-.05, -2.0)
    axes[0].legend(loc='upper right', fontsize=8, frameon=False)
    buckets = []
    for impl in implementations:
        rr = [r for r in records if r['method'] == primary and r['implementation'] == impl]
        totals = {k: statistics.fmean(r[k] for r in rr) for k in ['seconds', 'continuation_seconds', 'search_seconds', 'geometry_seconds']}
        totals['other'] = totals['seconds']-sum(totals[k] for k in ['continuation_seconds', 'search_seconds', 'geometry_seconds'])
        assert totals['other'] >= 0; buckets.append(dict(implementation=impl, **totals))
    bottom = np.zeros(3)
    for field, name, color in [('search_seconds', 'Exact branch search', '#d58b42'),
                              ('continuation_seconds', 'Trajectory + trigger', '#715bae'),
                              ('geometry_seconds', 'Geometry', '#437bc0'), ('other', 'Other charged work', '#a8b9c9')]:
        height = np.array([b[field] for b in buckets])
        axes[1].bar(np.arange(3), height, bottom=bottom, width=.6, label=name, color=color); bottom += height
    axes[1].set_xticks(np.arange(3), ['Original', 'Integer\nsearch', 'Integer\nsearch + trigger'], fontsize=9)
    axes[1].set_ylabel('Seconds per task'); axes[1].set_title('Primary: unchanged predictions, lower cost', loc='left', fontsize=12)
    axes[1].set_ylim(0, max(bottom)*1.30)
    for i, value in enumerate(bottom): axes[1].text(i, value+.014, f'{value:.3f}s', ha='center', fontsize=10)
    axes[1].legend(loc='upper right', frameon=False, fontsize=8)
    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x' if ax is axes[0] else 'y', alpha=.15); ax.set_axisbelow(True)
    fig.suptitle('Exact arithmetic implementation — not a new accuracy result', fontsize=15, x=.05, ha='left')
    fig.text(.05, .02, '8 development tasks × 12 methods × 3 implementations × 2 timing repeats. All returned arrays match the sealed reference bitwise.', fontsize=9)
    fig.tight_layout(rect=(0, .05, 1, .95)); fig.savefig(out/'online_equivalent_runtime.png', dpi=170); plt.close(fig)
    write(out/'figure_data.json', dict(methods=results, primary_components=buckets, source_audit_sha256=sha(base/'dyadic_diagnostics_audit_v1/summary.json')))
    lines = ['# 336｜保留原预测的精确算术提速：完整在线路径验证', '',
        '2026-09-25，美国东部时间上午。此轮是已发布512任务之后的计算实现改进，不是第二次独立准确率确认，研究总目标仍进行中。', '',
        '## 结果', '',
        f'主方法在8个预先固定开发任务上的完整均时由 **{p["mean_seconds"]["fraction"]:.6f}秒**降至 **{p["mean_seconds"]["integer_dp_trigger"]:.6f}秒**，减少 **{100*p["fractional_reduction"]:.2f}%**。全部576次真实在线调用的{os["bitwise_array_checks"]}个返回数组与原封存结果逐位一致；不是拿缓存预测代替计算。', '',
        f'次候选由{s["mean_seconds"]["fraction"]:.6f}秒降至{s["mean_seconds"]["integer_dp_trigger"]:.6f}秒。全部12种信用控制都获得同一整数实现，不能将通用算术提速归成PC独有准确率收益。', '',
        '![完整在线计时与主方法成本分项](../../results/online_credit_fresh_pilot/dyadic_report_v2/online_equivalent_runtime.png)', '',
        '## 为什么选择这项改动', '',
        f'先检查完整512任务、16方法、8192个保存预测器。主方法原均时{original["charged_seconds"]:.9f}秒，其中精确分支搜索{original["components"]["search_seconds"]:.9f}秒；支持触发器单独计时{original["overlapping_trigger_seconds"]:.9f}秒，已经包含在轨迹收集/搜索中，不能再次累加。', '',
        '有限读出方差的估计约为2.6×10⁻⁵；扣除这一项后，主方法相对强ALM、随机信用和Adam的主要数值差距基本保留。这个估计不是某次固定粒子种子的误差界，也不能证明采样噪声完全没有作用。当前证据支持先降低搜索开销，而不是把差距归咎于粒子数。', '',
        '| 方法 | 原封存MSE | 估计读出方差项 | 扣除后的开发估计 |', '| --- | ---: | ---: | ---: |']
    for n in [primary, 'online_first_fit_random_sign', 'probe_all_alm64', 'probe_then_adam1920_33']:
        r = old[n]; lines.append(f'| {n} | {r["actual_mse"]:.10f} | {r["estimated_readout_variance"]:.10f} | {r["corrected_fixed_teacher_risk_estimate"]:.10f} |')
    lines += ['', '扣除列仅为理想独立粒子模型下固定teacher风险的无偏估计形式，未执行任何新预测器，不是完整后验偏差或新显著性检验。', '',
        '## 数学上为什么不会改变答案', '',
        'binary64输入都是分母为2的幂的有理数。把全部坐标乘共同尺度X、全部信用乘共同尺度A，321中的局部目标就恰好变成原目标的AX倍。盒交、分支断点和min/max全用任意精度整数计算，目标次序、符号、并列破同分及空盒判定均保持。最后只在接口把整数/AX约分回原Fraction字符串。', '',
        '支持损失同理：整数tent递推为H←max(0,X−|2(H+B)−X|)，Rᵢ=max(0,|Hᵢ−Vᵢ|−E)，原损失恰好等于ΣRᵢ²/(2nX²)。这既不近似梯度，也不改变最早零损失状态的选择。', '',
        '实现验收包含48个小问题的3504模式穷举、144次精确K-best对照、非正规数及1032个触发状态测试。全部512×12=6144个原搜索状态的有序提议、证书和壳最小值逐项相同。另在4任务48状态作192次随机交错内核计时，搜索内核提速区间为' +
        f'{min(r["fraction_over_integer"] for r in ks["timing_methods"]):.2f}–{max(r["fraction_over_integer"] for r in ks["timing_methods"]):.2f}倍；这不是完整在线提速倍数。', '',
        '## 全部完整在线计时', '',
        '| 方法 | 原Fraction / s | 仅整数搜索 / s | 整数搜索＋触发 / s | 总时间减少 | 更快任务 |',
        '| --- | ---: | ---: | ---: | ---: | ---: |']
    for r in results:
        t = r['mean_seconds']; lines.append(f'| {r["method"]} | {t["fraction"]:.6f} | {t["integer_dp"]:.6f} | {t["integer_dp_trigger"]:.6f} | {100*r["fractional_reduction"]:.2f}% | {r["faster_tasks"]}/8 |')
    lines += ['', '每任务先平均两次重复，再对8任务等权平均。576次调用不是576个独立任务；没有新增显著性检验，没有把8任务的时间当作512任务完整计时。持久粒子和模型状态未变，但尚未测量新实现的进程峰值内存。', '',
        '## 对核心研究问题的推进', '',
        '这项结果把一个可去除的计算瓶颈落到了真实在线路径。它保留了之前的独有区域和预测能力，并为同一总预算下更充分的分支探索提供了可检验的空间。下一步必须另行冻结预算和全部强控制，检验节省的计算是否能换来独立查询收益；不能仅凭内核更快就宣称已经超过强Adam或证明PC不可替代。', '',
        '原512任务的准确率和机制结论仍以[332报告](332_fresh_pilot_results_v1.md)为准；没有官方TTT、LLM/VLM或完整论文新增验证。', '',
        '证据：[333诊断协议](333_fresh_cost_readout_diagnostic_v1.md)、[334整数证明与内核协议](334_dyadic_branch_search_protocol_v1.md)、[335完整在线协议](335_dyadic_online_equivalence_protocol_v1.md)、[独立汇总审计](../../results/online_credit_fresh_pilot/dyadic_diagnostics_audit_v1/summary.json)。', '']
    with (root/DOCUMENT).open('x', encoding='utf-8') as f: f.write('\n'.join(lines))
    summary = dict(passed=True, generation_only=True, visual_qa_pending=True, document=DOCUMENT,
        report_source_sha256=sha(Path(__file__)), audit_summary_sha256=sha(base/'dyadic_diagnostics_audit_v1/summary.json'),
        outputs_sha256={n: sha(out/n) for n in ['online_equivalent_runtime.png', 'figure_data.json']}, document_sha256=sha(root/DOCUMENT))
    write(out/'summary.json', summary); print(summary, flush=True)


if __name__ == '__main__': main()
