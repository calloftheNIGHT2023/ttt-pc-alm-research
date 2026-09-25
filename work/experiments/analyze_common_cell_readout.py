import argparse, hashlib, json
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(); p.add_argument('--results', type=Path, required=True); args = p.parse_args()
    protocol = json.loads((args.results / 'protocol.json').read_text())
    rows = json.loads((args.results / 'episodes.json').read_text())
    names = [c['name'] for c in protocol['configs']]; modes = ['raw', 'point', 'midpoint', 'mean']
    seeds = list(range(protocol['seed0'], protocol['seed0'] + protocol['count']))
    assert len(rows) == len(seeds) * len(names) * len(modes) * len(protocol['stages'])
    assert all(hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest() == digest
               for name, digest in protocol['source_sha256'].items())
    summary, arrays, comparisons, interventions = [], {}, [], []
    for name in names:
        for mode in modes:
            group = [r for r in rows if r['method'] == name and r['mode'] == mode]
            mse = np.array([np.mean([r['query_mse'] for r in group if r['seed'] == seed]) for seed in seeds])
            duration = np.array([sum(r['adaptation_seconds'] + r['read_queries_seconds'] for r in group if r['seed'] == seed) for seed in seeds])
            arrays[name, mode] = mse, duration
            summary.append(dict(method=name, mode=mode, mean_trajectory_mse=float(mse.mean()),
                                median_full_seconds=float(np.median(duration)), per_stream_mse=mse.tolist(),
                                stage_mse=[float(np.mean([r['query_mse'] for r in group if r['n_context'] == n])) for n in protocol['stages']],
                                stage_feasible=[sum(r['support_feasible'] for r in group if r['n_context'] == n) for n in protocol['stages']],
                                projection_statuses=dict(Counter(r.get('projection_status', 'not_used') for r in group)),
                                fiber_statuses=dict(Counter(r.get('fiber_status', 'not_used') for r in group)),
                                fiber_enabled=[dict(seed=r['seed'], n_context=r['n_context'], width=r['fiber_width']) for r in group if r.get('fiber_status') == 'valid']))
    sample = np.random.default_rng(722619).integers(0, len(seeds), (20000, len(seeds)))
    for mode in modes:
        cm, ct = arrays[names[0], mode]
        for name in names[1:]:
            mm, tt = arrays[name, mode]; diff = cm - mm
            comparisons.append(dict(mode=mode, baseline=name, mean_mse_difference=float(diff.mean()),
                                    paired_descriptive_95_interval=np.quantile(diff[sample].mean(axis=1), [.025, .975]).tolist(),
                                    candidate_better_streams=int(np.sum(diff < 0)), median_time_difference=float(np.median(ct - tt))))
    for name in names:
        for a, b in [('point', 'raw'), ('mean', 'point'), ('mean', 'midpoint')]:
            diff = arrays[name, a][0] - arrays[name, b][0]
            interventions.append(dict(method=name, comparison=f'{a}_minus_{b}', mean_mse_difference=float(diff.mean()),
                                      descriptive_95_interval=np.quantile(diff[sample].mean(axis=1), [.025, .975]).tolist()))
    points = {(r['method'], r['seed'], r['n_context']): np.array(r['anchor_output']) for r in rows if r['mode'] == 'point'}
    agreement = []
    for name in names[1:]:
        deltas = [float(np.max(np.abs(points[names[0], seed, n] - points[name, seed, n]))) for seed in seeds for n in protocol['stages']]
        agreement.append(dict(baseline=name, max_parameter_delta=max(deltas), stage_pairs_within_1e_6=sum(v <= 1e-6 for v in deltas), total=len(deltas)))
    result = dict(summary=summary, comparisons=comparisons, interventions=interventions, projected_parameter_agreement=agreement,
                  source_hashes_match=True, scope='reused development streams; descriptive intervals only')
    out = args.results.parent / 'analysis'; out.mkdir(exist_ok=True)
    (out / 'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    table = ['|Method|Readout|Trajectory MSE|Full seconds|Feasible at 8/16/24|', '|---|---|---:|---:|---|']
    for r in summary:
        table.append(f"|{r['method']}|{r['mode']}|{r['mean_trajectory_mse']:.9g}|{r['median_full_seconds']:.4f}|{r['stage_feasible']}|")
    (out / 'table.md').write_text('\n'.join(table) + '\n', encoding='utf-8')
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), layout='constrained')
    colors = plt.get_cmap('tab10').colors
    for i, name in enumerate(names):
        values = [next(s for s in summary if s['method'] == name and s['mode'] == mode)['mean_trajectory_mse'] for mode in modes]
        axes[0].plot(range(4), values, marker='o', color=colors[i], label=name, alpha=.85)
        for mode, marker in [('raw', 'o'), ('mean', '*')]:
            r = next(s for s in summary if s['method'] == name and s['mode'] == mode)
            axes[1].scatter(r['median_full_seconds'], r['mean_trajectory_mse'], color=colors[i], marker=marker, s=130 if marker == '*' else 45)
    axes[0].set(xticks=range(4), xticklabels=['Raw', 'Common point', 'Midpoint', 'Function mean'], yscale='log', ylabel='Mean trajectory query MSE', title='Same discovered cell processing for every solver')
    axes[0].tick_params(axis='x', labelrotation=15); axes[0].legend(fontsize=8)
    axes[1].set(xscale='log', yscale='log', xlabel='Full fit + query read seconds', ylabel='Mean trajectory query MSE', title='Circle: raw; star: common function mean')
    for ax in axes: ax.grid(alpha=.2)
    fig.suptitle('12 reused development streams: common geometry and uncertainty readout')
    fig.savefig(out / 'common_readout_frontier.png', dpi=170); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout='constrained')
    for i, name in enumerate(names):
        for ax, mode in zip(axes, ['raw', 'mean']):
            ax.plot(np.arange(len(seeds)), arrays[name, mode][0], marker='o', ms=4, color=colors[i], label=name, alpha=.8)
            ax.set(yscale='log', xlabel='Development stream index', ylabel='Trajectory query MSE', title='Raw point' if mode == 'raw' else 'Common projection + function mean')
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.savefig(out / 'common_readout_streams.png', dpi=170); plt.close(fig)
    print(json.dumps(dict(summary=[{k: v for k, v in s.items() if k not in ['per_stream_mse', 'fiber_enabled']} for s in summary], comparisons=comparisons,
                          projected_parameter_agreement=agreement)), flush=True)


if __name__ == '__main__': main()
