"""Figures for report 441 from results/lifted_credit/*/rows.json (evaluated, sealed outputs)."""
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2] / 'results' / 'lifted_credit'
FIG = ROOT / 'figures'; FIG.mkdir(exist_ok=True)
# reference categorical palette, fixed order (dataviz skill)
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.8, 'figure.dpi': 150,
                     'legend.frameon': False})

SERIES = [  # (label, candidate method keys in priority order)
    ('PC-ALM, exact local inversion (1 start)', ['pc_alm_1x2']),
    ('BP Adam, 64-128 starts x 3 lr', ['bp_adam_128x3', 'bp_adam_64x3', 'bp_adam_8x3']),
    ('BP Levenberg-Marquardt', ['bp_lm_64', 'bp_lm_8']),
    ('BP GD, 64-128 starts x 3 lr', ['bp_gd_128x3', 'bp_gd_64x3', 'bp_gd_8x3']),
    ('PC-ALM, official gradient inference', ['pc_alm_gradact_8x2', 'pc_alm_gradact_1x2']),
    ('PC without multipliers (penalty)', ['pc_penalty_cont']),
    ('Best closed-form / fixed features', ['__best_closed__']),
    ('Bayes transform + GN (constructed)', ['hybrid_transform_lm', 'hybrid_transform_lm_8']),
]
CLOSED = ['linear_ridge', 'rf4096_relu_ridge', 'rf4096_link_ridge', 'rbf_krr', 'poly3_krr', 'mlp_extra_layer']


def load(run):
    return [r for r in json.loads((ROOT / run / 'rows.json').read_text()) if not r['method'].startswith('__')]


def value(rows, d, keys, field):
    for k in keys:
        if k == '__best_closed__':
            c = [r for r in rows if r['d'] == d and r['method'] in CLOSED]
            if c:
                return max(r[field] for r in c) if field == 'success' else min(r.get('nmse', r['mse']) for r in c)
            continue
        c = [r for r in rows if r['d'] == d and r['method'] == k]
        if c:
            return c[0][field]
    return None


def panel(ax, rows, title, field='success', extra8=False):
    ds = sorted(set(r['d'] for r in rows))
    if extra8:
        pts = [(d, value(rows, d, ['pc_alm_8x2'], field)) for d in ds]
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PAL[0], lw=2.0, ls='--', marker='s', ms=4,
                label='PC-ALM, exact local inversion (8 starts)', zorder=3)
    for i, (lab, keys) in enumerate(SERIES):
        ys = [value(rows, d, keys, field) for d in ds]
        if all(v is None for v in ys):
            continue
        pts = [(d, v) for d, v in zip(ds, ys) if v is not None]
        lw = 2.6 if i == 0 else 1.6
        ax.plot([p[0] for p in pts], [p[1] for p in pts], color=PAL[i], lw=lw, marker='o', ms=4.5,
                label=lab, zorder=3 if i == 0 else 2)
    ax.set_xscale('log', base=2); ax.set_xticks(ds); ax.set_xticklabels([str(d) for d in ds])
    ax.set_ylim(-0.03, 1.03); ax.set_title(title, color=INK, fontsize=10, loc='left')
    ax.set_xlabel('input dimension d  (support n = 4d per hidden unit)')


def fig_main():
    runs = [('a_formal_he3', 'A  single-index He3, n = 4d (64 tasks / d)', False),
            ('b2_committee_r2_matched', 'B  2-layer committee r = 2, n = 8d (32 tasks / d)', True)]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9), sharey=True)
    for ax, (run, title, e8) in zip(axes, runs):
        panel(ax, load(run), title, extra8=e8)
    axes[0].set_ylabel('tasks solved (query MSE < 0.05)')
    h, l = axes[0].get_legend_handles_labels()
    h2, l2 = axes[1].get_legend_handles_labels()
    for hh, ll in zip(h2, l2):
        if ll not in l:
            h.append(hh); l.append(ll)
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / 'fig1_dimension_scaling.png', bbox_inches='tight'); plt.close(fig)


def fig_real():
    runs = [(r, r) for r in ['c1_nlp', 'c1_cv', 'c1_graph_autogrid'] if (ROOT / r / 'rows.json').exists()]
    if not runs:
        return
    names = {'c1_nlp': 'NLP  MiniLM on AG News (kurtosis 3.3)', 'c1_cv': 'CV  ResNet-18 on CIFAR-10 (kurtosis 3.2)',
             'c1_graph_autogrid': 'Graph  GCN on PubMed (kurtosis 16-20)'}
    fig, axes = plt.subplots(1, len(runs), figsize=(4.4 * len(runs), 3.9), sharey=True, squeeze=False)
    for ax, (run, _) in zip(axes[0], runs):
        panel(ax, load(run), names.get(run, run), extra8=True)
        ax.set_xlabel('whitened feature dimension d  (n = 4d)')
    axes[0][0].set_ylabel('tasks solved (normalised MSE < 0.05)')
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=3, fontsize=8.2, bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(FIG / 'fig3_real_representations.png', bbox_inches='tight'); plt.close(fig)


def fig_stream():
    p = ROOT / 'd_stream_d64' / 'rows.json'
    if not p.exists():
        return
    rows = [r for r in json.loads(p.read_text()) if r['method'] != '__time__']
    lab = {'pcalm_1x2_k10': ('PC-ALM, persistent multipliers', 0), 'replay_adam_8x3_k10': ('BP Adam on all seen data', 1),
           'replay_lm_8_k10': ('BP Levenberg-Marquardt on all seen data', 2), 'ttt_gd_k1': ('official TTT rule: 1 GD step / chunk', 3)}
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m, (l, i) in lab.items():
        rr = sorted([r for r in rows if r['method'] == m], key=lambda r: r['n'])
        ax.plot([r['n'] / 64 for r in rr], [r['success'] for r in rr], color=PAL[i], lw=2.6 if i == 0 else 1.6, label=l)
    ax.set_xlabel('context examples seen, in units of d  (d = 64, chunks of 16)')
    ax.set_ylabel('tasks solved (query MSE < 0.05)'); ax.set_ylim(-0.03, 1.03)
    ax.set_title('D  streaming inner loop, 10 inner iterations per chunk', loc='left', fontsize=10, color=INK)
    ax.legend(fontsize=8.5, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    fig.tight_layout(); fig.savefig(FIG / 'fig2_streaming.png', bbox_inches='tight'); plt.close(fig)


def fig_switch():
    p = ROOT / 'e2_multiplier_switch_v2' / 'rows.json'
    if not p.exists():
        return
    rows = json.loads(p.read_text())
    arms = [('switch_on_multiplier_rho_sel', 'switch multipliers ON (PC-ALM)', 0),
            ('continue_no_mult', 'keep multipliers OFF (PC)', 5),
            ('bp_adam_x3lr', 'BP Adam from same state', 1), ('bp_lm', 'BP Levenberg-Marquardt from same state', 2)]
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    wbar = 0.19
    for j, (k, lab, ci) in enumerate(arms):
        xs = [i + (j - 1.5) * (wbar + 0.02) for i in range(len(rows))]
        vals = [r[k + '_escape_rate_among_stuck'] for r in rows]
        ax.bar(xs, vals, width=wbar, color=PAL[ci], label=lab)
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.02, f'{v:.2f}', ha='center', fontsize=7.5, color=INK2)
    ax.set_xticks(range(len(rows))); ax.set_xticklabels([f"d = {r['d']}" for r in rows])
    ax.set_ylim(0, 1.12); ax.set_ylabel('escaped (query MSE < 0.05)')
    ax.set_title('E  same stuck state (128/128 tasks), 400 more iterations', loc='left', fontsize=10, color=INK)
    ax.grid(axis='x', visible=False)
    ax.legend(fontsize=8, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout(); fig.savefig(FIG / 'fig4_multiplier_switch.png', bbox_inches='tight'); plt.close(fig)


def fig_phase():
    ps = [ROOT / r / 'rows.json' for r in ['f_phase', 'f_phase_d256']]
    if not ps[0].exists():
        return
    rows = sum([json.loads(p.read_text()) for p in ps if p.exists()], [])
    ds = sorted(set(r['d'] for r in rows))
    bp = [m for m in set(r['method'] for r in rows) if m.startswith('bp_adam_') and m != 'bp_adam_1x3'][0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    shades = ['#b9d6f8', '#6fa7ea', '#2a78d6', '#0d3f80']; oshades = ['#f9cdb9', '#f29a73', '#eb6834', '#8f3510']
    for i, d in enumerate(ds):
        for m, sh, ls, lab in [('pc_alm_1x2', shades, '-', 'PC-ALM 1 start'), (bp, oshades, '--', 'BP Adam 64 starts x 3 lr')]:
            rr = sorted([r for r in rows if r['d'] == d and r['method'] == m], key=lambda r: r['alpha'])
            axes[0].plot([r['alpha'] for r in rr], [r['success'] for r in rr], color=sh[i], ls=ls, marker='o', ms=4,
                         lw=1.8, label=f'{lab}, d = {d}')
    axes[0].set_xscale('log', base=2); axes[0].set_xlabel('samples per dimension  alpha = n / d')
    axes[0].set_ylabel('tasks solved'); axes[0].set_ylim(-0.03, 1.03)
    axes[0].set_title('F  success vs sample size', loc='left', fontsize=10, color=INK)
    axes[0].legend(fontsize=7.2, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.24))
    for m, c, lab in [('pc_alm_1x2', PAL[0], 'PC-ALM, 1 start'), (bp, PAL[1], 'BP Adam, 64 starts x 3 lr'),
                      ('bp_adam_1x3', PAL[2], 'BP Adam, 1 start x 3 lr')]:
        th = []
        for d in ds:
            rr = sorted([r for r in rows if r['d'] == d and r['method'] == m], key=lambda r: r['alpha'])
            hit = [r['n'] for r in rr if r['success'] >= 0.5]
            th.append(hit[0] if hit else None)
        pts = [(d, t) for d, t in zip(ds, th) if t]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=c, marker='o', lw=2.2, label=f'{lab}')
    ref = ds[0]
    axes[1].plot(ds, [4 * ref * (d / ref) for d in ds], color=INK2, lw=1, ls=':', label='n = 4 d')
    axes[1].plot(ds, [20 * ref * (d / ref) for d in ds], color=INK2, lw=1, ls='-.', label='n = 20 d')
    axes[1].set_xscale('log', base=2); axes[1].set_yscale('log', base=2)
    axes[1].set_xticks(ds); axes[1].set_xticklabels([str(d) for d in ds]); axes[1].set_xlabel('dimension d')
    axes[1].set_ylabel('n* = first n with >= 50% solved'); axes[1].legend(fontsize=7.5, loc='upper left')
    axes[1].set_title('F  sample threshold scaling', loc='left', fontsize=10, color=INK)
    fig.tight_layout(); fig.savefig(FIG / 'fig5_phase.png', bbox_inches='tight'); plt.close(fig)


if __name__ == '__main__':
    fig_main(); fig_stream(); fig_real(); fig_switch(); fig_phase()
    print('figures ->', FIG)
