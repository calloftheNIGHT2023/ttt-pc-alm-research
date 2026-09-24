"""297 complete development table and diagnostic plot, no selection hidden."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from diagnose_counterfactual_conditional_risk_v1 import read,save,sha


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project',type=Path,required=True)
    root=ap.parse_args().project.resolve()
    inp=root/'results/novel_branch_continuation/development_v1'
    out=root/'results/novel_branch_continuation/report_v1'
    out.mkdir(parents=True,exist_ok=False)
    summary=read(inp/'summary.json')
    assert summary['passed']
    for name,digest in summary['outputs_sha256'].items():
        assert sha(inp/name)==digest
    rows=read(inp/'aggregates.json')
    families=['alm_reset','alm_keep','nodual','pc','adam']
    labels=['ALM: reset dual','ALM: keep dual','No dual','Ordinary PC','Adam control']
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained')
    for family,label in zip(families,labels):
        chosen=[r for r in rows if r['grid']==257 and r['method'].startswith(family+'_')]
        ax.plot([int(r['method'].rsplit('_',1)[1]) for r in chosen],
                [r['delta']*1e6 for r in chosen],'.-',label=label)
    ax.axhline(0,color='black',linewidth=.7)
    ax.set_xticks([1,2,4,8])
    ax.set_xlabel('Shadow continuation steps per triggered state (not matched wall time)')
    ax.set_ylabel('Ideal pool-risk change vs original ALM (x 0.000001)')
    ax.set_title('OLD64 same-state continuation | all 20 variants retained')
    ax.legend()
    fig.savefig(out/'continuation.png',dpi=170)
    plt.close(fig)
    text=['# 297｜短续接分支搜索：完整20配置开发结果','',
      '64个旧开发任务，先封存所有支持侧提议轨迹，再打开完整后验参照；不读取真实查询答案。256个逐位自检通过，2427个相同触发状态、20配置、两种网格全部保留。',
      '', '![短续接全对照](continuation.png)', '',
      '两步重置ALM的理想池条件超额风险为 .00122211235，略好于单步 .00122690971；延长到八步为 .00124321511，并非越长越好。四步/八步Adam为 .00107828275/.00104292517，优于这些局部短续接。这里步数不是等时间，全部结果只是旧数据上的离线池诊断，不是部署任务收益。',
      '', '特别地，重置ALM与无乘子算法的第1步参数和活动严格相同。因此，单步额外分支不能归因于新乘子累积；原ALM轨迹提供了状态，但这与影子更新本身的因果作用必须分开。保留乘子的第1步严格重现原轨迹下一步，新增池为零，这是预定的实现检查。',
      '', '结论：不能靠延长当前局部续接来建立相对强BP的优势。下一步直接检验局部乘子方向的信息价值，同时把事件触发Adam短搜索纳入更强对照。PC-ALM主目标仍未完成；没有删掉优势更强的简单方法。',
      '', '| 方法 | 网格 | 理想条件超额风险 | 相对原池差 | 两批对差值 | 质量覆盖 | 状态步/任务 | 新正区域/任务数 |',
      '|---|---:|---:|---:|---|---:|---:|---|']
    for r in rows:
        pairs=', '.join(f'{v:.10g}' for v in r['pair_delta_means'])
        text.append(f'| {r["method"]} | {r["grid"]} | {r["policy_excess"]:.12g} | {r["delta"]:.12g} | {pairs} | {r["mass"]:.10g} | {r["extra_state_steps"]:.5f} | {r["new_positive_modes"]} / {r["tasks_with_new_positive_modes"]} |')
    text+=['',f'结果摘要SHA256：`{sha(inp/"summary.json")}`。脚本、全量提议、封存清单、逐任务风险位于development_v1。计时仅为该诊断分块计算，缺少实际新方法几何与读出，不能与294完整调用直接比速度。']
    with (out/'report.md').open('x',encoding='utf-8') as stream:
        stream.write('\n'.join(text)+'\n')
    save(out/'summary.json',dict(passed=True,source_sha256=sha(Path(__file__)),
        result_summary_sha256=sha(inp/'summary.json'),visual_review_pending=True,
        outputs_sha256={n:sha(out/n) for n in ['report.md','continuation.png']}))
    print(str(out/'report.md'),flush=True)


if __name__=='__main__':
    main()
