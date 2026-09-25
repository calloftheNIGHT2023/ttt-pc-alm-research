"""308 deterministic full-table report; no task fitting or query scoring."""
import argparse
from fractions import Fraction as F
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from evaluate_complete_credit_mode_geometry_v1 import read, save, sha, METHODS


def run(root, out):
    base = root / 'results/complete_credit_amplitude_events'
    folders = {name: base / name for name in ['development_v2', 'geometry_v1', 'range_audit_v1', 'float_diagnosis_v1']}
    summaries = {name: read(folder / 'summary.json') for name, folder in folders.items()}
    for name, summary in summaries.items():
        for filename, digest in summary.get('outputs_sha256', {}).items():
            assert sha(folders[name] / filename) == digest
    support, geom, audit, fp = [summaries[n] for n in folders]
    assert support['all_selected_states_processed'] and geom['execution_passed'] and audit['passed'] and fp['passed']
    manifest = read(folders['development_v2'] / 'before_geometry_manifest.json')
    for name, digest in manifest['files_sha256'].items():
        assert sha(folders['development_v2'] / name) == digest
    agg = read(folders['geometry_v1'] / 'aggregate.json')
    tasks = read(folders['geometry_v1'] / 'tasks.json')
    rows = read(folders['development_v2'] / 'rows.json')
    assert set(agg) == set(METHODS) and len(tasks) == 64 and len(rows) == 393
    path_counts = {}
    for family in ['dual', 'residual', 'random_sign']:
        rr = [r for r in rows if r['family'] == family]
        path_counts[family] = dict(directions=len(rr), cells=sum(r['cells'] for r in rr),
            extra_open_mode_directions=sum(bool(r['additional_open_modes_vs_fixed']) for r in rr),
            search_seconds=sum(r['search_seconds'] for r in rr),
            median_search_seconds=float(np.median([r['search_seconds'] for r in rr])))
    labels = METHODS
    values = [agg[n]['task_positive_pairs'] for n in labels]
    colors = ['#265eaa' if n in ['dual_event_all', 'residual_event_all', 'random_sign_event_all'] else
              '#278877' if n.startswith('control_') else '#aab4c3' for n in labels]
    fig, ax = plt.subplots(figsize=(12.8, 12.8))
    yy = np.arange(len(labels))
    ax.barh(yy, values, color=colors, height=.73)
    ax.set_yticks(yy, labels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) + 3)
    ax.set_xlabel('New positive-volume (task, mode) pairs beyond the original pool', fontsize=11)
    ax.set_title('308: complete amplitude paths and all same-state controls', fontsize=15, pad=17)
    for y, value in zip(yy, values):
        ax.text(value + .15, y, str(value), va='center', fontsize=9)
    ax.xaxis.grid(True, alpha=.15)
    ax.set_axisbelow(True)
    ax.spines[['top', 'right']].set_visible(False)
    fig.subplots_adjust(left=.34, right=.97, bottom=.085, top=.945)
    fig.text(.34, .035, '64 old development tasks; 131 identical starting states.\nCounts are not query-risk gains; runtime is NOT matched.', fontsize=10)
    fig.savefig(out / 'all_mode_controls.png', dpi=160)
    plt.close(fig)
    lines = ['# 308阶段结果：完整幅度搜索已完成，独立收益门槛未满足', '',
        '本报告包含整个固定实验，不是新任务确认；本轮未读取教师查询答案或后验均值。整体研究目标仍未完成。', '',
        '## 结果定位', '',
        '原乘子连续路径在原方法池外增加1个正体积模式，分布在1个任务；同方向七点方案也发现了同一个模式。连续枚举没有在原池与七点之外新增有效区域。残差路径较自身七点多1个新区域，随机符号路径较自身七点多3个，其中1个只出现在路径边界。不能将这些结果归因于原乘子信用的独立优势。', '',
        '这使下一项问题更明确：仅加密原乘子的标量幅度不能补齐本实验缺失解释。后续应分析内部状态/信用方向的可达性与阻断约束，先检查已有工作避免重复；不是扩大样本来追求当前方案显著性。', '',
        '## 覆盖与独立审计', '',
        f"- 全部131位置×3方向完成，{support['counts']['cells']}开区间、{support['counts']['boundaries']}边界；无安全上限未完成项。全流程诊断耗时{support['seconds']:.3f}秒。",
        f"- {support['counts']['directions_with_extra_open_modes']}个方向存在七点遗漏的开区间模式，但模式数不等于有效解释数。",
        f"- 独立端点/顶点审计通过{audit['counts']['uniform_guard_root_exclusions']}个无内部根检查、{audit['counts']['independent_scalar_witness_checks']}个标量内部点和{audit['counts']['independent_scalar_boundary_checks']}个有理边界。",
        f"- {audit['counts']['algebraic_boundaries_not_scalar_checked']}个无理边界检查了精确符号，但没有用独立有理标量求解器重放；{audit['nonrepresentable_cells']}个开区间不存在内部binary64幅度。不得把这些区间默认当成可直接浮点部署的搜索点。", '',
        '|方向|完整路径数|分段数|存在七点遗漏的路径数|纯搜索总秒数|每路径搜索中位秒数|',
        '|---|---:|---:|---:|---:|---:|']
    for family, row in path_counts.items():
        lines.append(f"|{family}|{row['directions']}|{row['cells']}|{row['extra_open_mode_directions']}|{row['search_seconds']:.6f}|{row['median_search_seconds']:.6f}|")
    lines += ['', '这些费用是精确原型的开发诊断计时，不是完整在线fit，也未匹配303子集控制的墙钟。', '',
        '## 全部35方法', '',
        '所有方法共享相同先验盒和观察带。额外模式加入不变原池；没有触发的任务仍计入全部64任务。boundary_only是边界诊断，不是主候选。', '',
        '|方法|有新正体积模式的任务数|新(task,mode)对数|新模式数值体积合计|未解决(task,mode)对数|',
        '|---|---:|---:|---:|---:|']
    for name in METHODS:
        row = agg[name]
        lines.append(f"|{name}|{row['tasks_with_new_positive']}|{row['task_positive_pairs']}|{row['known_new_positive_numeric_volume_sum']:.12g}|{row['task_unknown_mode_pairs']}|")
    lines += ['', '![全部方法新增正体积模式对数](all_mode_controls.png)', '',
        f"共同几何实际分类{geom['counts']['mode_classifications']}个任务-模式：{geom['counts']['positive_volume']}个有严格内部证书、{geom['counts']['infeasible']}个有精确不可行证书，零未解决。包含原池模式，不能把总正体积数量当作新增贡献。已复核{geom['counts']['certificate_rechecks']}项证书；发生{geom['counts']['geometry_repairs']}次既定条件化数值修复。体积仍为数值结果而非精确积分。", '',
        '## 两处浮点差异', '',
        f"固定1e-10检查线发现{fp['cases']}例，最大偏置差{fp['max_production_gap']:.12g}，活动差没有超线。独立直接成本标量求解与精确模型最大差{fp['max_scalar_gap']:.12g}。逐位重放原浮点输出后，在两例共享偏置块都验证了累计浮点成本相对有理直接成本的候选排序倒置；不是将它们归零或放宽阈值。两例输出模式相同。", '',
        '## 保留与后续门槛', '',
        'v1错误地要求带噪观察值属于[0,1]，在第6位置初始化处停止；原15方向和失败记录保留。v2只修正观察带相交条件，并逐字段复核原15方向结果不变。', '',
        '本轮没有形成原乘子特有的新正体积区域，因此不进入308协议的旧后验风险诊断或新任务确认。下一机制仍须给出数学瓶颈、支持侧选择规则、同参数BP/PC/ALM及强回归/浅头对照、完整资源计费；不能以覆盖实现替代实际任务收益。', '']
    (out / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    save(out / 'path_counts.json', path_counts)
    save(out / 'summary.json', dict(passed=True, all_methods=35, tasks=64,
         input_summaries_sha256={name: sha(folder / 'summary.json') for name, folder in folders.items()},
         outputs_sha256={p.name: sha(p) for p in out.iterdir() if p.is_file()},
         visual_review_required=True, independent_task_gain_established=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    run(Path(__file__).resolve().parents[2], args.out)
