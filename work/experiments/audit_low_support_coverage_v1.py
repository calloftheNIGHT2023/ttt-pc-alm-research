"""393 independent saved-particle reconstruction and descriptive portfolio check."""
from pathlib import Path
import hashlib
import json
import traceback
import numpy as np
import streaming_branch_projection as teacher

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT/'results/low_support_coverage/development_v1'
OUT = ROOT/'results/low_support_coverage/audit_v1'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def save(p,value):
    p.write_text(json.dumps(value,indent=2,allow_nan=False),encoding='utf-8')


def main():
    summary = read(SOURCE/'summary.json'); assert summary['passed']
    for f,digest in summary['outputs_sha256'].items():
        assert sha(SOURCE/f)==digest
    protocol=read(SOURCE/'protocol.json')
    for f,digest in protocol['source_sha256'].items():
        assert sha(ROOT/f)==digest
    banks=read(SOURCE/'sealed_banks.json'); rows=read(SOURCE/'coverage_rows.json')
    refs=read(SOURCE/'references.json'); report_rows=[]; portfolios=[]
    max_forward_gap=0.; checked_banks=0; checked_means=0
    for ref in refs:
        folder=ROOT/ref['directory']
        for f,digest in ref['files'].items():
            assert sha(folder/f)==digest
        with np.load(folder/'arrays.npz',allow_pickle=False) as z:
            x=z['original_x'];v=z['original_v']
        these=[b for b in banks if (b['seed'],b['n'])==(ref['seed'],ref['n'])]
        for b in these:
            bank_folder=ROOT/b['directory']
            for f,digest in b['files'].items():
                assert sha(bank_folder/f)==digest
            with np.load(bank_folder/'arrays.npz',allow_pickle=False) as z:
                keys=sorted({teacher.pattern(x,point).astype(np.uint8).tobytes().hex() for point in z['best_bank']})
            assert keys==b['mode_keys'];checked_banks+=1
        if not ref['fully_resolved']:
            continue
        assert sha(folder/'integration.npz')==ref['integration_sha256']
        with np.load(folder/'integration.npz',allow_pickle=False) as z:
            keys=ref['positive_modes'];means=z['region_means'];variances=z['region_sample_variances'];weights=z['weights'];q=z['q']
            expected_volumes=np.array([ref['geometry_notes'][k]['volume'] for k in keys])
            assert np.array_equal(expected_volumes,z['volumes'])
            assert np.array_equal(expected_volumes/expected_volumes.sum(),weights)
            for i,k in enumerate(keys):
                points=z[f'points_{i}']
                predictions=np.stack([teacher.forward(q,point) for point in points])
                recomputed=predictions.mean(0)
                gap=float(np.max(abs(recomputed-means[i])));max_forward_gap=max(max_forward_gap,gap)
                assert gap<1e-13
                assert np.max(abs(predictions.var(0,ddof=1)-variances[i]))<1e-13
                assert max(np.max(abs(teacher.forward(x,p)-v)) for p in points)<=.001+1e-8
                checked_means+=1
            full=weights@means
            assert np.array_equal(full,z['full_prediction'])
            for b in these:
                row=next(r for r in rows if (r['seed'],r['n'],r['method'])==(b['seed'],b['n'],b['method']))
                mask=np.array([k in b['mode_keys'] for k in keys]);mass=weights[mask].sum()
                assert row['covered_positive_modes']==int(mask.sum())
                assert row['covered_posterior_mass']==float(mass)
                assert row['omitted_posterior_mass']==float(weights[~mask].sum())
                if mask.any():
                    coefficients=weights*mask/mass-weights
                    bias=float(np.mean((coefficients@means)**2))
                    # Shared per-region sampling makes this the variance of the
                    # estimated mean-difference; it is not an error bar for volume.
                    integration_variance=float(np.mean((coefficients**2)@variances/4096))
                    assert abs(bias-row['conditional_bias_proxy'])<1e-14
                    report_rows.append(dict(**row,estimated_mc_variance_of_difference=integration_variance,
                        bias_minus_estimated_mc_variance=bias-integration_variance))
            # Explicitly post-hoc: union all four independently sealed banks.
            # No queries or reference modes are supplied to these optimizers.
            union=set().union(*(set(b['mode_keys']) for b in these))
            mask=np.array([k in union for k in keys]);mass=weights[mask].sum()
            gap=float(np.mean(((weights*mask/mass-weights)@means)**2)) if mask.any() else None
            portfolios.append(dict(seed=ref['seed'],n=ref['n'],covered_modes=int(mask.sum()),
                positive_modes=len(keys),omitted_mass=float(weights[~mask].sum()),bias_proxy=gap,
                posthoc_descriptive=True,independently_timed_algorithm=False))
    assert checked_banks==32 and len(report_rows)==32 and len(portfolios)==8
    save(OUT/'rows_with_mc_variance.json',report_rows);save(OUT/'posthoc_portfolio.json',portfolios)
    lines=['# 393｜低支持覆盖诊断结果','',
        '两个固定旧任务、四个支持前缀，共八个情境。32 个优化器库先封存，参考搜索与几何随后计算。未生成或读取已实现教师的查询答案。', '',
        '八个参考均完整解析，32 个数值偏差恒等式通过。下面的 bias 是有限数值积分下“条件联合均值与完整均值”的平方差，不是实测查询 MSE。MC 列只估计区域粒子积分噪声，不包含浮点体积误差。','',
        '| Seed | n | Method | Covered / all modes | Omitted mass | Bias proxy | MC variance |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: |']
    for r in report_rows:
        lines.append(f"| {r['seed']} | {r['n']} | {r['method']} | {r['covered_positive_modes']} / {r['positive_reference_modes']} | {r['omitted_posterior_mass']:.10g} | {r['conditional_bias_proxy']:.10g} | {r['estimated_mc_variance_of_difference']:.10g} |")
    lines+=['','## 对机制的直接含义','',
        '可量化的遗漏确实存在：5920001 的 n=8 情境，Adam64 与 Adam256 都遗漏约一半后验质量，对应偏差代理约 6.73e-4；但 GN256 在该情境覆盖全部正体积区域。5920000 的 n=4，GN64 的偏差代理约 0.01016，而 Adam256 完整覆盖。不能用单一优化器的遗漏证明本方法不可替代。','',
        '模式数量也不是充分证据：5920001 的 n=24，64 初始化控制遗漏约 16.1% 质量，但偏差代理只有约 1.85e-9。需要同时看预测分歧。','',
        '下面额外审查四库联合，明确属于事后描述，不是预注册主比较，也没有独立执行／匹配时限。它提醒下一轮必须纳入有实际计费的优化器组合，而非让候选独占多解释读出。','',
        '| Seed | n | Union covered / all | Omitted mass | Bias proxy |',
        '| --- | ---: | ---: | ---: | ---: |']
    for r in portfolios:
        lines.append(f"| {r['seed']} | {r['n']} | {r['covered_modes']} / {r['positive_modes']} | {r['omitted_mass']:.10g} | {r['bias_proxy']:.10g} |")
    lines+=['','下一步的正向问题：在共同截止时间与状态限制下，主动局部求证是否能比带实际费用的强优化器组合更便宜地取得足够预测覆盖。当前诊断支持测这个问题，尚不证明它，也不满足进入三域真实模型的全部门槛。','',
        '实际耗时含与本机 GPU 元训练的并行环境，仅作执行记录，不能用于加速比结论。源计划和原始结果不改写。']
    (OUT/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    save(OUT/'summary.json',dict(passed=True,checked_banks=checked_banks,
        checked_region_means=checked_means,checked_bias_rows=len(report_rows),
        posthoc_portfolios=len(portfolios),maximum_forward_mean_gap=max_forward_gap,
        query_targets_accessed=False,source_summary_sha256=sha(SOURCE/'summary.json'),
        source_sha256=sha(Path(__file__)),
        outputs_sha256={f:sha(OUT/f) for f in ['rows_with_mc_variance.json','posthoc_portfolio.json','report.md']}))
    print(json.dumps(read(OUT/'summary.json')),flush=True)


if __name__=='__main__':
    OUT.mkdir(parents=True,exist_ok=False)
    try:
        main()
    except Exception:
        save(OUT/'failure.json',dict(traceback=traceback.format_exc(),automatic_retry=False));raise
