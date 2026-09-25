"""309 full fixed-matrix report; scope conclusions are computed from sealed sets."""
import argparse
from pathlib import Path
from evaluate_complete_credit_mode_geometry_v1 import read, save, sha


def run(root,out):
    base=root/'results/layer_credit_interaction'
    folders=[base/n for n in ['development_v1','geometry_v1','audit_v2']]
    support,geo,audit=[read(p/'summary.json') for p in folders]
    assert all(s['passed'] for s in [support,geo,audit])
    for folder,summary in zip(folders,[support,geo,audit]):
        for name,digest in summary.get('outputs_sha256',{}).items():assert sha(folder/name)==digest
    agg=read(folders[1]/'aggregate.json');comp=read(folders[1]/'comparisons.json')
    h1=comp['dual_float']['new_vs_original_and_own_continuous']['pairs']>0
    h2=comp['dual_float']['exclusive_vs_all_308_and_other_masks']['pairs']>0
    lines=['# 309：完整层开关因果诊断', '',
        '四层全部16种保留/清零组合，三种信用方向，131个相同内部状态；全部64旧任务保留。不是新任务确认或已经可部署的选择器。', '',
        '## 执行与数学范围', '',
        f"- {support['counts']['proposals']}个更新全部完成，逐一通过独立有理完整步与符号完整步交叉核验。",
        f"- {support['counts']['old_endpoint_array_checks']}项旧端点数组核验通过；全关和全开未改变原定义。",
        f"- 超过1e-10的数值差异：{support['counts']['float_value_discrepancies']}；模式差异：{support['counts']['float_mode_discrepancies']}；最大b/h差：{support['max_float_gap']:.12g}。",
        f"- 独立归档审计：{audit['counts']['exact_forward_replays']}精确前向、{audit['counts']['floating_forward_replays']}浮点前向、{audit['counts']['method_set_rechecks']}方法集合、{audit['counts']['aggregate_numbers_rechecked']}聚合数值检查。",
        '- 固定局部策略的响应满足H_l=(-u_l e_l+s H_(l+1)+s u_(l+1)e_(l+1))/(1+s²+τ)。共同幅度只保留行和；行非零而行和为零可产生条件性抵消，但不是全部真实状态的前提，更不保证有效新模式。',
        '- 七组合成状态46个更新、4096项有理导数检查通过，同时保留全夹紧反例。首次测试因旧模式函数固定四层而失败，失败记录保留，通用深度编码修正后重测通过。', '',
        '## 六个完整并集（不使用事后最佳掩码）', '',
        '|方向与实现|原池外新正体积对|原池及本方向完整连续路径外|全部308方法外|再排除其他方向层开关后独有|',
        '|---|---:|---:|---:|---:|']
    fields=['new_vs_original','new_vs_original_and_own_continuous','new_vs_original_and_all_308','exclusive_vs_all_308_and_other_masks']
    for name,values in comp.items():
        lines.append('|'+name+'|'+'|'.join(str(values[f]['pairs']) for f in fields)+'|')
    lines+=['', '单位是(task,mode)对，不是独立任务数，也不是查询误差收益；逐任务列表完整保留。', '',
        '## 假设门槛', '',
        f'- H1（原乘子真实浮点层开关找回自身完整连续路径遗漏的正体积模式）：{h1}。',
        f'- H2的本轮必要线索（原乘子存在全部308强控制及另外两族层开关之外的正体积模式）：{h2}。',
        '- 这两个布尔值只对应本轮集合判据。即使成立，也还需支持侧在线选择规则、完整费用和新任务查询比较，不能据此宣布独立PC-ALM贡献。', '',
        '## 全部102配置', '',
        '|方法|有新增模式的任务|新增(task,mode)对|未确定对|新数值体积|',
        '|---|---:|---:|---:|---:|']
    for n,r in agg.items():
        lines.append(f"|{n}|{r['tasks_with_new_positive']}|{r['task_positive_pairs']}|{r['unknown_pairs']}|{r['numeric_new_volume']:.12g}|")
    rows=read(folders[0]/'rows.json')
    lines+=['', '## 完整资源边界', '',
        f"支持诊断总时间{support['seconds']:.6f}秒；包含两个精确求解器的核验和归档。浮点批初始化、一步更新与模式编码总计{sum(r['costs']['float_batch_initialization_step_and_mode_seconds'] for r in rows):.6f}秒，批量16状态；没有把诊断总时间或这个小计当端到端在线费用。",
        '16掩码枚举的中间状态、选择/几何、历史池、读出及完整对照成本还需统一计入；不把掩码并集当免费方法。查询答案及后验均值未读取，未做风险评估或扩大任务数。整体研究目标仍未完成。', '']
    report='\n'.join(lines)
    (out/'report.md').write_text(report,encoding='utf-8')
    # Separate from scientific arithmetic audit: ensure every table row was emitted once.
    table=[l for l in report.splitlines() if l.startswith('|')]
    for n in agg:assert sum(l.startswith('|'+n+'|') for l in table)==1
    result=dict(passed=True,all_methods=102,union_comparisons=6,H1_float=h1,H2_float_necessary_signal=h2,
        input_summaries_sha256={p.name:sha(p/'summary.json') for p in folders},
        report_sha256=sha(out/'report.md'),source_sha256=sha(Path(__file__)),
        independent_task_gain_established=False)
    save(out/'summary.json',result);print(result,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    args.out.mkdir(parents=True,exist_ok=False);run(Path(__file__).resolve().parents[2],args.out)
