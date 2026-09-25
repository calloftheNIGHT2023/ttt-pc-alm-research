"""406 all 42 corrected-runtime configurations; no winner/significance label."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import runtime_matched_prefix_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/io.BASE
OUT=BASE/'report_v1'
ENTRY='outputs/ttt-pc-alm-research/406_runtime_matched_prefix_results_v1.md'


def main():
    evaluation=BASE/'evaluation_v1';checked=io.read(evaluation/'summary.json')
    assert checked['passed'] and checked['calls']==1344 and checked['descriptive_only']
    for f,digest in checked['outputs_sha256'].items():assert io.sha(evaluation/f)==digest
    development=BASE/'development_v1';protocol=io.read(development/'protocol.json')
    assert protocol['primary']==io.PRIMARY and protocol['portfolio_task_free_module_preload']
    groups=io.read(evaluation/'groups.json');averages=io.read(evaluation/'prefix_averages.json');rows=io.read(evaluation/'risk_rows.json')
    with np.load(evaluation/'query_truth.npz',allow_pickle=False) as z:
        truth=dict(zip(map(int,z['seeds']),z['targets']))
    recomputed={};groups_checked=0
    for row in rows:
        directory=ROOT/row['directory'];assert io.sha(directory/'outputs.npz')==row['files']['outputs.npz']
        prediction=io.load_arrays(directory/'outputs.npz')['prediction']
        risk=math.fsum(float(e)**2 for e in prediction-truth[row['seed']])/len(prediction)
        assert abs(risk-row['query_mse'])<1e-13
        recomputed[row['method'],row['n'],row['budget'],row['seed']]=risk
    for group in groups:
        mean=math.fsum(recomputed[group['method'],group['n'],group['budget'],s] for s in io.SEEDS)/4
        assert abs(mean-group['mean_query_mse'])<1e-13;groups_checked+=1
    for group in averages:
        values=[math.fsum(recomputed[group['method'],n,group['budget'],s] for n in io.STAGES)/4 for s in io.SEEDS]
        assert np.max(abs(np.array(values)-group['task_prefix_average_mse']))<1e-13
        assert abs(math.fsum(values)/4-group['mean_query_mse'])<1e-13;groups_checked+=1
    names=[c['name'] for c in protocol['configs']]
    lookup={(r['method'],r['n'],r['budget']):r for r in groups}
    avg={(r['method'],r['budget']):r for r in averages}
    figures=[];matrices={}
    for budget in io.BUDGETS:
        matrix=np.array([[lookup[name,n,budget]['mean_query_mse'] for n in io.STAGES]+[avg[name,budget]['mean_query_mse']] for name in names])
        matrices[str(budget)]=matrix.tolist();fig,ax=plt.subplots(figsize=(13.5,15.5),layout='constrained')
        image=ax.imshow(np.maximum(matrix,1e-12),norm=LogNorm(vmin=max(1e-12,float(matrix.min())),vmax=float(matrix.max())),cmap='viridis',aspect='auto')
        ax.set_yticks(range(len(names)),labels=names,fontsize=8)
        ax.set_xticks(range(5),labels=['n=4','n=8','n=16','n=24','Mean of 4 prefixes'])
        ax.set_title(f'Corrected portfolio runtime setup; {budget:g}s per independent call\nAll 42 configurations; four old tasks, not independent confirmation',fontsize=12)
        for i in range(len(names)):
            for j in range(5):
                color='black' if image.norm(max(matrix[i,j],1e-12))>.7 else 'white'
                ax.text(j,i,f'{matrix[i,j]:.2e}',ha='center',va='center',fontsize=7.5,color=color)
        primary=names.index(io.PRIMARY);ax.get_yticklabels()[primary].set_color('#b34f00');ax.get_yticklabels()[primary].set_fontweight('bold')
        fig.colorbar(image,ax=ax,label='Mean query MSE (log scale)',fraction=.035,pad=.02)
        filename=f'prefix_risk_{int(budget*1000)}ms.png';fig.savefig(OUT/filename,dpi=170);plt.close(fig);figures.append(filename)
    lines=['# 406｜统一预加载后的四前缀限时开发结果','',
        '四个旧任务、42配置、四前缀、.25/.5秒，共1344次预测。全部预测封存并独立审计之后才由评估器生成本轮查询答案；旧任务历史上已开发使用，仍不是独立确认。','',
        '本轮新主active512 DFS/farthest_x、主预算.5秒，四前缀等权平均；选择来自旧397开发结果，不追溯改变旧主。组合已在无支持setup阶段加载优化器／读出模块，相关setup费用单独保留；没有预跑支持更新。','',
        '没有提供四旧任务的显著性或论文胜出标签。全部方法、预算、前缀及逐任务差异均保留。原397结果不覆盖，本表也不能仅凭前后墙钟波动声称算法加速。','']
    for budget,figure in zip(io.BUDGETS,figures):
        lines += [f'## {budget:g}秒','',f'![全部42方法]({figure})','',
            '| Method | n4 MSE | n8 MSE | n16 MSE | n24 MSE | Prefix average | Final / 16 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in names:
            values=[lookup[name,n,budget]['mean_query_mse'] for n in io.STAGES]+[avg[name,budget]['mean_query_mse']]
            final=sum(lookup[name,n,budget]['final'] for n in io.STAGES)
            lines.append('| '+name+' | '+' | '.join(f'{v:.12g}' for v in values)+f' | {final} |')
    lines += ['','## 新主的全部配对差','',
        '差值为active512减控制的每任务四前缀平均MSE，再取四任务均值。负数不自动意味着独立或统计确认。','',
        '| Control | .25s difference | .5s difference |','| --- | ---: | ---: |']
    comparisons=io.read(evaluation/'comparisons.json');cl={(c['control'],c['budget']):c for c in comparisons}
    for name in names:
        if name!=io.PRIMARY:
            lines.append(f"| {name} | {cl[name,.25]['mean_paired_difference']:.12g} | {cl[name,.5]['mean_paired_difference']:.12g} |")
    lines += ['','## 费用与归因边界','',
        '逐调用保存setup、在线接收、归档传输、任务清理、强制停止、控制器超时、RSS及会话关闭。冷启动并未消失；控制器磁盘压缩含在全轮总时长而非逐调用独立归因，RSS是采样下界，没有严格峰值RAM上限匹配。','',
        '组合的阶段预测仍使用fallback包类型，final不是完整后验保证；几何只在同一次调用内复用。所有快状态在任务间清空；元模型当前前缀内的支持重放全额收费。','',
        'BP/PC/ALM参数盒相同，有限步代理目标尚未严格同式。局部候选还包含收费全局几何LP。400部分后验证书没有加入本轮。','',
        '[全部组表](../evaluation_v1/groups.json) · [全部配对](../evaluation_v1/comparisons.json) · [独立审计](../audit_v1/summary.json) · [冻结协议](../development_v1/protocol.json)','',
        '尚无独立新任务确认，也没有NLP/CV/Graph真实模型结果。完整研究目标未达。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    io.save(OUT/'figure_data.json',dict(methods=names,matrices=matrices))
    entry='# 406｜预加载修正后的限时开发结果\n\n四旧任务、42配置、1344次调用，仅描述性开发结果，旧397不改。\n\n[全部图文、控制与费用](../../results/runtime_matched_prefix/report_v1/report.md)\n'
    (ROOT/ENTRY).write_text(entry,encoding='utf-8')
    io.save(OUT/'manifest.json',dict(source_sha256=io.sha(Path(__file__)),evaluation_summary_sha256=io.sha(evaluation/'summary.json'),
        entry_file=ENTRY,entry_sha256=io.sha(ROOT/ENTRY),outputs_sha256={f:io.sha(OUT/f) for f in ['report.md','figure_data.json',*figures]}))
    io.save(OUT/'qa_numeric.json',dict(passed=True,raw_risks_recomputed=len(rows),group_means_recomputed=groups_checked,
        figure_values=420,all_configurations_retained=True,manifest_sha256=io.sha(OUT/'manifest.json')))
    print(io.read(OUT/'qa_numeric.json'),flush=True)


if __name__=='__main__':
    assert io.read(BASE/'evaluation_v1/summary.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False);main()
