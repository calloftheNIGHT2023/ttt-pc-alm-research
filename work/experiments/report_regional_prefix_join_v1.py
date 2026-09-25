"""379 full search results; no query-risk or publication-readiness overclaim."""
from pathlib import Path
import math
import numpy as np
import run_regional_prefix_join_v1 as run
from report_search_radius_development_v1 import table, write


def main():
    root = Path(__file__).resolve().parents[2]; base = root/run.BASE
    development = run.complete(base/'development_v1'); audit = run.complete(base/'audit_v1')
    assert audit['development_summary_sha256'] == run.sha(base/'development_v1/summary.json')
    assert run.read(base/'development_v1/protocol.json')['source_sha256'] == run.hashes(root)
    groups = run.read(base/'audit_v1/groups.json'); rows = run.read(base/'development_v1/rows.json')
    idx = {(g['method'], g['schedule'], g['ordering']): g for g in groups}
    assert len(idx) == 36 and all(g['calls'] == 16 for g in groups)
    names = [c[0] for c in run.model.CONFIGS]; table_rows = []; resource_rows = []; summaries = []
    for schedule in ['bfs', 'dfs']:
        for order in ['observed', 'farthest_x']:
            main = idx[run.model.PRIMARY, schedule, order]; base_none = idx['none', schedule, order]
            summaries.append(f"{schedule.upper()}/{order}：主完成{main['completed']}/16、均{main['mean_seconds']:.3f}秒；无额外筛选完成{base_none['completed']}/16、均{base_none['mean_seconds']:.3f}秒。")
            for name in names:
                g = idx[name, schedule, order]
                # Independently recompute all displayed aggregate numbers from raw per-call records.
                rr = [r for r in rows if (r['method'], r['schedule'], r['ordering']) == (name, schedule, order)]
                assert g['completed'] == sum(r['completed'] for r in rr)
                assert g['mean_seconds'] == math.fsum(r['seconds'] for r in rr)/16
                assert g['expanded'] == sum(r['expanded'] for r in rr)
                assert g['local_rejected'] == sum(r['local_rejected'] for r in rr)
                assert g['solver_coordinate_steps'] == sum(r['solver_coordinate_steps'] for r in rr)
                table_rows.append([schedule, order, name, g['completed'], f"{g['mean_seconds']:.3f}",
                                   g['expanded'], g['local_rejected'], g['final_candidates'], g['solver_coordinate_steps']])
                metas = [run.read(root/r['directory']/'metadata.json') for r in rr]
                resource_rows.append([schedule, order, name,
                    max(m['peak_search_states'] for m in metas),
                    f"{max(m['live_named_array_bytes_lower_bound'] for m in metas)/2**20:.3f}",
                    f"{g['mean_returned_bytes']/2**20:.3f}",
                    f"{max(m['total_seconds'] for m in metas):.3f}"])
    out = base/'report_v1'; out.mkdir(parents=True, exist_ok=False)
    run.save(out/'table_data.json', table_rows); run.save(out/'resource_data.json', resource_rows); run.save(out/'figure_data.json', groups)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 10))
    for ax, (schedule, order) in zip(axes.ravel(), [(s, o) for s in ['bfs', 'dfs'] for o in ['observed', 'farthest_x']]):
        values = [idx[name, schedule, order] for name in names]; yy = np.arange(len(names))
        colors = ['#158579' if name == run.model.PRIMARY else '#567698' if name == 'none' else '#b4bec9' for name in names]
        ax.barh(yy, [g['completed'] for g in values], color=colors, height=.66)
        ax.set_yticks(yy, [name.replace('regional_', '').replace('pdhg_', 'PDHG ').replace('_', ' ') for name in names], fontsize=9)
        ax.invert_yaxis(); ax.set_xlim(0, 20); ax.set_xticks([0, 4, 8, 12, 16])
        ax.set_title(f'{schedule.upper()} | {order.replace("_", " ")}', loc='left', pad=12, fontsize=12)
        for y, g in enumerate(values):
            ax.text(g['completed']+.15, y, str(g['completed']), va='center', fontsize=9)
            ax.text(18.7, y, f"{g['mean_seconds']:.2f}s", ha='center', va='center', fontsize=8.5, color='#435267')
        ax.text(18.7, -.85, 'mean time', ha='center', fontsize=8.5, color='#435267')
        ax.set_xlabel('Completed full searches / 16 tasks', fontsize=9)
        ax.spines[['top', 'right']].set_visible(False); ax.grid(axis='x', alpha=.14); ax.set_axisbelow(True)
    fig.subplots_adjust(left=.14, right=.97, top=.91, bottom=.14, wspace=.48, hspace=.43)
    fig.suptitle('Active regional feedback in full candidate search', fontsize=16, y=.975)
    fig.text(.14, .070, 'Fixed primary: active128. All nine controls, both schedules and both observation orders are shown.', fontsize=10)
    fig.text(.14, .039, 'Common expansion/time thresholds; per-schedule state accounting. Old tasks; final geometry and query reading excluded.', fontsize=9)
    fig.savefig(out/'search_budget.png', dpi=160); plt.close(fig)
    body = ['# 379｜从局部证书到完整候选搜索', '',
            '本轮把376的主动区域反馈真正接入逐观察建树，比较完整搜索而非固定旧前沿。所有576次调用及独立审计完成后才生成本报告。仍不是最终查询预测实验。', '',
            '## 完整结果', '', *summaries, '',
            '![完整搜索完成率及实际成本](search_budget.png)', '',
            '主方法固定为regional_active128。无额外筛选不是不做约束，而是保留共同C5/C20。PDHG256/512是真实显式工作步数，不是名字更改。表中候选数含不可行假阳性，不是正体积解释数，也不是预测准确率。', '',
            table(['调度', '排序', '方法', '完成/16', '均搜索秒', '展开总数', '局部排除', '完整候选总数', '局部坐标步'], table_rows), '',
            '## 结论与机制归因', '',
            '固定主active128在四组设置中均16/16完成；none在observed为9/16、farthest_x为13/16，passive128均13/16。主动反馈的组件增量因此真实进入了完整搜索，不只是固定旧候选批上的证书数量变化。', '',
            '但cold/box PDHG512同样四组全完成，平均时间均低于主方法，且展开更少。较长局部求证可以通过更早剪枝减少后续工作；不能把短128步局部比较直接推广为完整算法优势。本轮因此支持继续优化真实搜索预算分配，不支持直接启动以现有active128为胜出方法的新查询确认。', '',
            '## 为什么剪枝不会删除可行延伸', '',
            '设当前前缀盒约束为D_k，层间残差为r_p、r_a。对任何信用p、a，严格下界L=inf_{D_k}(p·r_p+a·r_a)>0意味着当前前缀没有同时满足两组等式的状态。任何合法完整延伸限制到该前缀都应使残差为零，因此不可能延伸被严格证明不可行的前缀。实现只接受向外舍入的严格正下界，独立审计再以Fraction复核。', '',
            '该论证保证剪枝安全，不保证固定步数找到所有证书、不保证ALM全局收敛，也不保证求证费低于节约的搜索费。累计局部坐标步、完整时间与最终候选必须同时看。', '',
            '## 有限预算与反例防护', '',
            'BFS逐层展开；批式DFS把已产生子批送往深层，允许中断前已有完整候选。未完成的搜索保存待访问父批、观察深度与偏移；空完整候选不是空真实后验。审计要求每个已知可行模式位于已输出候选或者尚未搜索的子树中。两种调度和两种排序均对九方法开放，不用一种基线较弱的调度包装主方法优势。', '',
            '共同展开上限65536、时间阈值8秒协作批边界；最后一批可越过阈值。状态20000检查存在调度差异：BFS检查产生的下一层，DFS检查待访问父批加已输出行数，因此不是两调度严格相同峰值内存上限；同调度内九配置规则相同，实际父层/归档和命名状态另计。见[执行中实现核对](../../../outputs/ttt-pc-alm-research/378_state_accounting_note_v1.md)。', '',
            'C20后批内至少64存活才运行额外筛选。实际费用不同，同上限并非相同FLOPs。完整搜索时间计入外层守卫、语言、排序、建树、筛选和证明归档拷贝，不含写盘压缩、独立审计、最终几何和后验读出。数组状态下界不是进程峰值；本轮是单次交错计时，没有重复计时置信区间。', '',
            table(['调度', '排序', '方法', '最大搜索状态', '最大命名数组下界MiB', '均归档MiB', '最大实际秒'], resource_rows), '',
            '## 独立审计与剩余关口', '',
            f"审计通过：{audit['checks']['fraction_single_languages']}个精确单观察语言、{audit['checks']['enumerated_combinations']}次组合重放、{audit['checks']['contractor_mask_replays']}个收缩mask、{audit['checks']['exact_wide_domain_cuts']}个精确宽域剪枝证书、{audit['checks'].get('positive_output_or_pending_coverage', 0)}次可行模式输出/待搜索覆盖，以及全部576次搜索资源核对。", '',
            '预检还完成117原376求解器数组逐位重放、4显式步数前缀、4嵌套LP/BP守卫拦截、584搜索重复数组及8紧预算真模式检查。实验只见支持x/v；旧几何正模式只在全部搜索结果封存后进入离线审计；没有查询答案。', '',
            '下一步检验更充分的局部求证与已证分支提前退役能否减少全树总费用；所有控制必须获得相同执行优化，并保留当前128主的结果，不能事后改名成成功。成本竞争力成立后接入共同几何与anytime后验读出，再测试总费用和未见查询误差。若查询效果能被更简单控制复现，就不能归因于PC-ALM。本轮不替代强回归、浅头、同参数优化器或官方TTT的任务比较，不完成研究目标。', '',
            '[预列协议](../../../outputs/ttt-pc-alm-research/378_regional_join_protocol_v1.md) · [全部调用](../development_v1/rows.json) · [独立审计](../audit_v1/summary.json)', '']
    write(out/'report.md', '\n'.join(body))
    entry = 'outputs/ttt-pc-alm-research/379_regional_join_results_v1.md'
    write(root/entry, '# 379｜完整候选搜索阶段结果\n\n'+'\n\n'.join(summaries)+'\n\n[完整图文与九组控制](../../results/regional_prefix_join/report_v1/report.md)\n\n不是新查询收益，目标保持ACTIVE。\n')
    run.save(out/'manifest.json', dict(source_sha256=run.sha(Path(__file__)), entry_file=entry, entry_sha256=run.sha(root/entry),
        input_summary_sha256={p: run.sha(base/p) for p in ['development_v1/summary.json', 'audit_v1/summary.json']},
        outputs_sha256={f: run.sha(out/f) for f in ['table_data.json', 'resource_data.json', 'figure_data.json', 'report.md', 'search_budget.png']}))
    content = (out/'report.md').read_text(encoding='utf-8')
    for schedule in ['bfs', 'dfs']:
        for order in ['observed', 'farthest_x']:
            primary = idx[run.model.PRIMARY, schedule, order]
            assert primary['completed'] == 16 and idx['regional_passive128', schedule, order]['completed'] == 13
            assert idx['none', schedule, order]['completed'] == (9 if order == 'observed' else 13)
            for name in ['pdhg_cold512', 'pdhg_box512']:
                assert idx[name, schedule, order]['completed'] == 16
                assert idx[name, schedule, order]['mean_seconds'] < primary['mean_seconds']
                assert idx[name, schedule, order]['expanded'] < primary['expanded']
    for row in table_rows+resource_rows: assert '| '+' | '.join(map(str, row))+' |' in content
    run.save(out/'qa_numeric.json', dict(passed=True, table_cells=len(table_rows)*9+len(resource_rows)*7, groups=len(groups),
        raw_rows_recomputed=len(rows), manifest_sha256=run.sha(out/'manifest.json')))
    print(dict(passed=True, report=str(out/'report.md')), flush=True)


if __name__ == '__main__': main()
