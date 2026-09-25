"""396 training-only report and checked plots; never a held-out task ranking."""
from pathlib import Path
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'results/matched_official_ttt'
OUT=BASE/'report_v1'
ENTRY='outputs/ttt-pc-alm-research/396_matched_training_and_coverage_results_v1.md'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p,v):
    p.write_text(json.dumps(v,indent=2,allow_nan=False),encoding='utf-8')


def main():
    training=BASE/'training_v1';audit=BASE/'training_audit_v1'
    ts=read(training/'summary.json');au=read(audit/'summary.json')
    assert ts['passed'] and au['passed'] and au['training_summary_sha256']==sha(training/'summary.json')
    for folder,summary in [(training,ts),(audit,au)]:
        for f,digest in summary['outputs_sha256'].items():
            assert sha(folder/f)==digest
    protocol=read(training/'protocol.json');logs=read(training/'training_log.json')
    rows=read(training/'training_summary.json');models=read(audit/'models.json')
    assert len(rows)==len(models)==8
    labels=['Native prior256 h16 p1','Native prior256 h16 p4','Native prior256 h32 p1 (primary)',
            'Native prior256 h32 p4','Native scalar h32 p1','Legacy prior256 h32 p1',
            'Matched meta-ridge128','Matched shallow64 / 20 steps']
    groups=[list(range(4)),list(range(4,8))]
    for mode in ['training','prefixes']:
        fig,axes=plt.subplots(1,2,figsize=(14,5.1),layout='constrained')
        for ax,indices,title in zip(axes,groups,['Native TTT with common prior features','Feature/rate ablations and matched controls']):
            for i in indices:
                if mode=='training':
                    selected=[e for e in logs if e['method']==rows[i]['method']]
                    x=[e['step'] for e in selected];y=[e['validation_raw_mse'] for e in selected]
                else:
                    x=protocol['stages'];y=models[i]['stage_validation_raw_mse']
                ax.plot(x,y,marker='o',markersize=3,linewidth=1.7,label=labels[i])
            ax.set_title(title,fontsize=11);ax.set_ylabel('Old-validation raw MSE')
            ax.set_xlabel('Outer training step' if mode=='training' else 'Observed support prefix')
            ax.grid(alpha=.22);ax.legend(fontsize=8.2,loc='best')
            if mode=='prefixes':ax.set_xticks(protocol['stages'])
        fig.suptitle('Same stored training cohort, eight models; validation only',fontsize=14)
        fig.savefig(OUT/(mode+'.png'),dpi=170);plt.close(fig)
    lines=['# 396｜同 cohort 官方 TTT 强对照训练完成','',
        '八个模型训练完成，2000 个共同 batch 从 seed 重新生成，8000 个张量逐数组一致；八模型各自消费 2000 batch 的记录全一致。已保存八个 checkpoint，并按旧验证集选择规则独立审计。', '',
        '总训练流程约 %.3f 秒；与阶段发布／轻量诊断有并行，因此此处时间是实际执行账目，不是隔离吞吐结论。每模型 32000 外层任务、2048000 唯一查询值、8192000 四前缀查询损失暴露。八配置总暴露应相加，但它们共享同一组唯一任务，不能称 256000 个独立训练任务。'%ts['seconds'],'',
        '**以下全部是旧验证集原始 MSE，不是独立新任务结果。** 训练集／验证集的外层标签合法且明示计费；研究测试查询答案未使用。外层BP与官方TTT基线的内部梯度更新均明确保留。','',
        '![八模型训练过程](training.png)','',
        '![所选checkpoint的四前缀旧验证误差](prefixes.png)','',
        '| Method | Selected step | Mean raw MSE | n4 | n8 | n16 | n24 | Training seconds | CUDA peak allocated MiB |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    cells=0
    for i,r in enumerate(rows):
        m=models[i];assert m['method']==r['method']
        assert abs(np.mean(m['stage_validation_raw_mse'])-m['validation_cpu_raw_mse'])<1e-14
        event=min([e for e in logs if e['method']==r['method']],key=lambda e:(e['validation_raw_mse'],e['step']))
        assert event['step']==r['selected_step'] and event['validation_raw_mse']==r['best_validation_raw_mse']
        values=[r['best_validation_raw_mse'],*m['stage_validation_raw_mse']]
        lines.append('| '+labels[i]+f" | {r['selected_step']} | "+' | '.join(f'{v:.10g}' for v in values)+f" | {r['outer_training_seconds']:.6f} | {r['cuda_peak_allocated_bytes']/2**20:.6f} |")
        cells+=8
    lines +=['','CPU重放与训练设备验证均值的最大差 %.12g。CUDA列仅是分配器峰值，不包括全部驱动、主机或求解器状态。'%au['maximum_cpu_validation_gap'],'',
        '本表没有候选ALM方法，所以不能据此宣称它优于官方TTT。接下来使用相同前缀、观察信息、截止时间和共同读出进行实际查询比较。','',
        '## 393：为什么下一比较需要优化器组合','',
        '低支持诊断找到了有预测影响的后验遗漏：部分n8情境下Adam64/256漏约一半质量，均值偏差代理约6.73e-4；另一些情境GN遗漏更多。但它们有互补性，四库联合在八个旧情境几乎完整。因此下一正式控制应含实际计费的组合，而不能只比较单独求解器。', '',
        '[全部低支持覆盖数表与独立审计](../../low_support_coverage/audit_v1/report.md) · [训练审计](../training_audit_v1/summary.json)','',
        '真实 NLP / CV / Graph 阶段依389计划执行，当前仍没有其训练或任务结果。完整目标未完成。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    (ROOT/ENTRY).write_text('# 396｜匹配训练与低支持覆盖阶段结果\n\n八模型同cohort训练与独立审计完成；旧验证结果不是新任务胜出证据。\n\n[图文、全部八模型、资源与393连接](../../results/matched_official_ttt/report_v1/report.md)\n',encoding='utf-8')
    manifest=dict(source_sha256=sha(Path(__file__)),training_summary_sha256=sha(training/'summary.json'),
        audit_summary_sha256=sha(audit/'summary.json'),coverage_audit_sha256=sha(ROOT/'results/low_support_coverage/audit_v1/summary.json'),
        entry_file=ENTRY,entry_sha256=sha(ROOT/ENTRY),
        outputs_sha256={f:sha(OUT/f) for f in ['report.md','training.png','prefixes.png']})
    save(OUT/'manifest.json',manifest)
    save(OUT/'qa_numeric.json',dict(passed=True,manifest_sha256=sha(OUT/'manifest.json'),
        training_points=len(logs),prefix_values=4*len(models),table_numeric_cells=cells,
        all_models_in_original_order=True,selection_rules_recomputed=True,
        values_are_old_validation_not_held_out_research_results=True))
    print(json.dumps(read(OUT/'qa_numeric.json')),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    main()
