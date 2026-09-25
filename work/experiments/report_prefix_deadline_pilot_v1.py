"""398 all-method prefix pilot report, after audited evaluation only."""
from pathlib import Path
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import prefix_deadline_pilot_io_v1 as io

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/io.BASE
OUT=BASE/'report_v1'
ENTRY='outputs/ttt-pc-alm-research/398_prefix_deadline_pilot_results_v1.md'


def main():
    evaluation=BASE/'evaluation_v1';checked=io.read(evaluation/'summary.json')
    assert checked['passed'] and checked['calls']==1344 and checked['descriptive_only']
    for f,digest in checked['outputs_sha256'].items():assert io.sha(evaluation/f)==digest
    development=BASE/'development_v1';protocol=io.read(development/'protocol.json')
    groups=io.read(evaluation/'groups.json');averages=io.read(evaluation/'prefix_averages.json');rows=io.read(evaluation/'risk_rows.json')
    with np.load(evaluation/'query_truth.npz',allow_pickle=False) as z:
        truth=dict(zip(map(int,z['seeds']),z['targets']))
    # Independently recompute individual squared risk and each plotted aggregate.
    recomputed={};groups_checked=0
    for row in rows:
        directory=ROOT/row['directory'];assert io.sha(directory/'outputs.npz')==row['files']['outputs.npz']
        p=io.load_arrays(directory/'outputs.npz')['prediction']
        risk=math.fsum(float(e)**2 for e in p-truth[row['seed']])/len(p)
        assert abs(risk-row['query_mse'])<1e-13
        recomputed[row['method'],row['n'],row['budget'],row['seed']]=risk
    for group in groups:
        mse=math.fsum(recomputed[group['method'],group['n'],group['budget'],s] for s in io.SEEDS)/4
        assert abs(mse-group['mean_query_mse'])<1e-13;groups_checked+=1
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
        ax.set_title(f'Four old development tasks; {budget:g}s per independent call\nAll 42 configurations; not independent confirmation',fontsize=12)
        for i in range(len(names)):
            for j in range(5):
                color='black' if image.norm(max(matrix[i,j],1e-12))>.7 else 'white'
                ax.text(j,i,f'{matrix[i,j]:.2e}',ha='center',va='center',fontsize=7.5,color=color)
        primary=names.index(io.PRIMARY);ax.get_yticklabels()[primary].set_color('#b34f00');ax.get_yticklabels()[primary].set_fontweight('bold')
        fig.colorbar(image,ax=ax,label='Mean query MSE (log scale)',fraction=.035,pad=.02)
        filename=f'prefix_risk_{int(budget*1000)}ms.png';fig.savefig(OUT/filename,dpi=170);plt.close(fig);figures.append(filename)
    lines=['# 398｜四前缀统一截止时间开发结果','',
        '四个旧任务、42配置、4/8/16/24支持、.25/.5秒，共1344次预测。所有预测先封存，再独立审计，再由评估器生成本轮查询答案。历史上这些任务已用于开发，因此不是独立确认。','',
        '新协议固定主为 active1024 DFS/farthest_x，主预算.5秒、每任务四前缀MSE等权平均；没有更改387主配置或其结果。所有配置与预算均报告。四任务仅提供描述性配对结果，不提供显著性或论文胜出标志。','']
    for budget,figure in zip(io.BUDGETS,figures):
        lines += [f'## {budget:g}秒','',f'![全部方法与前缀]({figure})','',
            '| Method | n4 MSE | n8 MSE | n16 MSE | n24 MSE | Prefix average | Final / 16 |',
            '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
        for name in names:
            values=[lookup[name,n,budget]['mean_query_mse'] for n in io.STAGES]+[avg[name,budget]['mean_query_mse']]
            final=sum(lookup[name,n,budget]['final'] for n in io.STAGES)
            lines.append('| '+name+' | '+' | '.join(f'{v:.12g}' for v in values)+f' | {final} |')
    lines += ['','## 原主的全部配对比较','',
        '差值为“主方法减控制”的四前缀平均MSE；负值仅说明这四个旧任务上的均值更低，不等于统计确认。','',
        '| Control | .25s difference | .5s difference |','| --- | ---: | ---: |']
    comparisons=io.read(evaluation/'comparisons.json');cl={(c['control'],c['budget']):c for c in comparisons}
    for name in names:
        if name!=io.PRIMARY:
            lines.append(f"| {name} | {cl[name,.25]['mean_paired_difference']:.12g} | {cl[name,.5]['mean_paired_difference']:.12g} |")
    lines += ['','## 费用与证据边界','',
        '完整逐调用结果保留setup、在线接收、归档传输、任务清理、强制终止、控制器超时、RSS采样和会话关闭；见原始rows/sessions。控制器磁盘压缩／逐步落盘包含在全运行总时间，但没有逐调用独立归因。RSS采样是下界，不是严格峰值RAM匹配。','',
        '组合的中间预测使用fallback包类型，final不是完整后验声明。组合每阶段几何只在本次调用内复用。所有方法只见当前前缀；元模型按已训练轨迹在此前缀内重放更短支持，全部重放收费。','',
        'BP／PC／ALM参数盒相同，但既有有限步代理目标尚未严格同式；这份结果不能替代同目标归因。候选的全局几何LP明示收费，不能称所有计算均局部。','',
        '[完整组表](../evaluation_v1/groups.json) · [全部配对结果](../evaluation_v1/comparisons.json) · [独立审计](../audit_v1/summary.json)','',
        '未完成独立新任务确认，也没有NLP/CV/Graph真实模型结果；完整研究目标仍未达成。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    io.save(OUT/'figure_data.json',dict(methods=names,matrices=matrices))
    entry='# 398｜四前缀限时开发结果\n\n四旧任务、42配置、1344次调用，仅描述性开发结果。\n\n[全部图文、42配置、两预算与费用](../../results/prefix_deadline_pilot/report_v1/report.md)\n'
    (ROOT/ENTRY).write_text(entry,encoding='utf-8')
    io.save(OUT/'manifest.json',dict(source_sha256=io.sha(Path(__file__)),evaluation_summary_sha256=io.sha(evaluation/'summary.json'),
        entry_file=ENTRY,entry_sha256=io.sha(ROOT/ENTRY),outputs_sha256={f:io.sha(OUT/f) for f in ['report.md','figure_data.json',*figures]}))
    io.save(OUT/'qa_numeric.json',dict(passed=True,raw_risks_recomputed=len(rows),group_means_recomputed=groups_checked,
        figure_values=42*5*2,all_configurations_retained=True,manifest_sha256=io.sha(OUT/'manifest.json')))
    print(io.read(OUT/'qa_numeric.json'),flush=True)


if __name__=='__main__':
    assert io.read(BASE/'evaluation_v1/summary.json')['passed']
    OUT.mkdir(parents=True,exist_ok=False);main()
