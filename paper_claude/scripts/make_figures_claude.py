"""Figures for paper_claude, regenerated from raw result JSONs following the scientific-figure-making skill
(house palette, Arial/Helvetica stack, top/right spines off, frameless legends, dedicated legend panels,
tight_layout(pad=2), 300 dpi, PDF + PNG, editable SVG text)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

RES = Path(r'C:\Users\callofthenight\Documents\Codex\2026-09-18\ttt-pc-alm-ttt-pc-alm\results\lifted_credit')
FIG = Path(__file__).resolve().parents[1] / 'figures'

PALETTE = {"blue_main": "#0F4D92", "blue_secondary": "#3775BA", "green_1": "#DDF3DE", "green_2": "#AADCA9",
           "green_3": "#8BCF8B", "red_1": "#F6CFCB", "red_2": "#E9A6A1", "red_strong": "#B64342",
           "neutral": "#CFCECE", "highlight": "#FFD700", "teal": "#42949E", "violet": "#9A4D8E",
           "gray_dark": "#4D4D4D", "gray_mid": "#767676"}
plt.rcParams.update({"font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"], "font.size": 16,
                     "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 2,
                     "legend.frameon": False, "svg.fonttype": "none", "lines.linewidth": 2.5, "lines.markersize": 7,
                     "xtick.major.width": 2, "ytick.major.width": 2, "mathtext.fontset": "dejavusans"})
# semantic roles: blue = self-iteration (proposed), reds/violet/teal = baselines, grays = references
ROLE = {"self": PALETTE["blue_main"], "self2": PALETTE["blue_secondary"], "gd_full": PALETTE["red_strong"],
        "gd_off": PALETTE["red_2"], "quad": PALETTE["teal"], "tf": PALETTE["violet"], "ref": PALETTE["gray_mid"]}


def load(run):
    return json.loads((RES / run / 'results.json').read_text())


def finalize(fig, name, pad=2.0):
    fig.tight_layout(pad=pad)
    for ext in ('pdf', 'png', 'svg'):
        fig.savefig(FIG / f'{name}.{ext}', dpi=300)
    plt.close(fig)


def legend_panel(ax, handles, labels, ncol=1, fontsize=15):
    ax.set_axis_off(); ax.legend(handles, labels, loc='center', ncol=ncol, fontsize=fontsize, handlelength=2.2)


def attr(run, model, key='4.0'):
    return load(run)['results'][model]['eval_test'][key]['pooled_nmse_mean']


def attr_curve(run, model):
    ev = load(run)['results'][model]['eval_test']
    return [ev[k]['pooled_nmse_mean'] for k in ('2.0', '4.0', '8.0')]


# ============================================================ Figure 1: controlled bases, conditions B2-B5
Q = load('q1_requirements')
fig, axes = plt.subplots(1, 5, figsize=(32, 6.6))

ax = axes[0]
for sig, ls in ((0.25, '--'), (0.5, '-')):
    rows = [r for r in Q['r1'] if r['sigma'] == sig]; n = [r['n_over_d'] for r in rows]
    ax.plot(n, [r['hard_alm_v0']['param_err'] for r in rows], ls, marker='o', color=ROLE['gd_full'])
    ax.plot(n, [r['gd_erm']['param_err'] for r in rows], ls, marker='s', color=ROLE['ref'])
    ax.plot(n, [r['em_posterior_oracle_tau']['param_err'] for r in rows], ls, marker='^', color=ROLE['self'])
ax.set_xscale('log', base=2); ax.set_yscale('log'); ax.set_xlabel('context size $n/d$'); ax.set_ylabel('parameter error')
ax.set_title('(a) R3: consistent inversion', fontsize=16)
ax.legend([Line2D([], [], color=ROLE['gd_full'], marker='o'), Line2D([], [], color=ROLE['ref'], marker='s'),
           Line2D([], [], color=ROLE['self'], marker='^'), Line2D([], [], color='k', ls='-'), Line2D([], [], color='k', ls='--')],
          ['exact inversion', 'ERM (gradient)', 'posterior inversion', r'$\nu$: $\sigma$=0.5', r'$\sigma$=0.25'], fontsize=12.5, loc='lower left')

ax = axes[1]
cols = {0.0: PALETTE['blue_main'], 0.1: PALETTE['blue_secondary'], 0.3: PALETTE['red_2'], 0.5: PALETTE['red_strong']}
for sig, col in cols.items():
    rows = [r for r in Q['r2'] if r['sigma'] == sig and r['n_over_d'] == 4]
    xs = [0.01 if r['v'] == 0 else (30.0 if not np.isfinite(r['v']) else r['v']) for r in rows]
    ax.plot(xs, [r['success'] for r in rows], '-o', color=col, label=rf'$\sigma$={sig}')
ax.set_xscale('log'); ax.set_xticks([0.01, 0.1, 1, 10, 30]); ax.set_xticklabels(['0', '0.1', '1', '10', r'$\infty$'])
ax.set_xlabel('multiplier leak $v$'); ax.set_ylabel('success rate ($n=4d$)'); ax.set_ylim(-0.03, 1.03)
ax.set_title('(b) R5: leak set by base noise', fontsize=16); ax.legend(fontsize=13, loc='lower left')

ax = axes[2]
for sig, ls in ((0.25, '--'), (0.5, '-')):
    rows = [r for r in Q['r3'] if r['sigma'] == sig]; n = [r['n_over_d'] for r in rows]
    ax.plot(n, [r['raw_ratio'] for r in rows], ls, marker='o', color=ROLE['gd_full'])
    ax.plot(n, [r['corrected_ratio'] for r in rows], ls, marker='^', color=ROLE['self'])
rows = [r for r in Q['r3'] if r['sigma'] == 0.5]
ax.plot([r['n_over_d'] for r in rows], [r['theory_raw_ratio'] for r in rows], ':', color='k', lw=2)
ax.axhline(1, color=ROLE['ref'], lw=1.5); ax.set_xscale('log', base=2); ax.set_xlabel('context size $n/d$')
ax.set_ylabel('estimated / true noise variance'); ax.set_title('(c) R4: in-context noise estimate', fontsize=16)
ax.legend([Line2D([], [], color=ROLE['gd_full'], marker='o'), Line2D([], [], color=ROLE['self'], marker='^'), Line2D([], [], color='k', ls=':')],
          ['RSS$/n$', r'RSS$/(n-\mathrm{df})$', r'theory $1-\mathrm{df}/n$'], fontsize=13, loc='lower right')

ax = axes[3]
rows = Q['r4']; n = [r['n_over_d'] for r in rows]
ax.plot(n, [r['single_chain'] for r in rows], '-o', color=ROLE['gd_full'], label='single chain')
ax.plot(n, [r['posterior_mixture'] for r in rows], '-^', color=ROLE['self'], label='posterior mixture')
ax.axvline(2 - 1 / 32, color='k', lw=1.5, ls=':'); ax.text(2.07, 1.5, '$n=2d-1$', fontsize=13)
ax.set_yscale('log'); ax.set_xlabel('context size $n/d$'); ax.set_ylabel('excess query risk')
ax.set_title('(d) R2: mixtures near identifiability', fontsize=16); ax.legend(fontsize=13, loc='upper right')

ax = axes[4]
rows = [r for r in Q['r5'] if r['local'] == 'hard']
lab = [(r'$\rho$' if r['knob'] == 'log_rho' else '$v$') + f"={np.exp(r['at']):.1f}\n" + rf"$\sigma$={r['sigma']}" for r in rows]
x = np.arange(len(rows))
ax.bar(x - 0.2, [r['autograd'] for r in rows], 0.4, color=ROLE['gd_full'], edgecolor='black', linewidth=1.5, label='backprop through solver')
ax.bar(x + 0.2, [r['finite_diff']['0.05'] for r in rows], 0.4, color=ROLE['self'], edgecolor='black', linewidth=1.5, label='true derivative')
ax.axhline(0, color='k', lw=1.5); ax.set_xticks(x); ax.set_xticklabels(lab, fontsize=11)
ax.set_ylabel(r'$dL/d\log\phi$'); ax.set_title('(e) R6: regime knobs need selection', fontsize=16); ax.legend(fontsize=13, loc='upper left')
finalize(fig, 'fig_controlled')

# ============================================================ Figure 2: base diagnostics predict self-iteration
D = load('b2_base_diagnostics')
nlp_test = ['KF Written Frequency', 'KF Number of Categories', 'KF Number of Samples', 'Thorndike-Lorge Frequency', 'Age of Acquisition Rating']
g_test = ['mu', 'gap', 'cv', 'r2', 'B']
s_test = [f'a{j}' for j in range(15, 20)]
nu = lambda base, attrs: float(np.mean([D[base][a]['nu'] for a in attrs]))
cvj = load('r_cv_pr'); nu_cv = cvj['pca_residual_energy_test']

best = lambda vals: min(vals)
bases = [  # name, nu, matched self-iteration NMSE (n=4d / 4m), best baseline NMSE, member
    ('synthetic, $\\sigma$=0', nu('synth_s0', s_test), attr('r3_synth_s0', 'ttt_pcalm'),
     best([attr('r3_synth_s0', m) for m in ('ttt_gd_full', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer')])),
    ('image PCA-64 (CV)', nu_cv, load('r8_cv_ens')['ensemble']['pca_pcalm_prox']['4.0']['nmse_mean'],
     best([load('r10_cv_quad')[m]['eval_test_images']['4.0']['nmse_mean'] for m in ('ttt_quad_ridge', 'transformer')] + [0.245, 0.273])),
    ('synthetic, $\\sigma$=0.5', nu('synth_s0.5', s_test), attr('r11_synth_s0.5', 'ttt_pcalm_prox'),
     best([attr('r3_synth_s0.5', m) for m in ('ttt_gd_full', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer')])),
    ('GINE (Graph)', nu('gine', g_test), attr('r9_graph_s2_a', 'ttt_pcalm_prox'),
     best([attr('r9_graph_s2_b', m) for m in ('ttt_gd_full', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer')])),
    ('GloVe (NLP)', nu('glove', nlp_test), attr('r9_nlp_s2_a', 'ttt_pcalm_prox'),
     best([attr('r9_nlp_s2_b', m) for m in ('ttt_gd_full', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer')])),
    ('MiniLM (NLP)', nu('minilm', nlp_test), attr('b1_nlp_minilm', 'ttt_pcalm_prox'),
     best([attr('b1_nlp_minilm', m) for m in ('ttt_gd_full', 'ttt_quad_ridge', 'transformer')])),
    ('synthetic, $\\sigma$=1', nu('synth_s1', s_test), attr('b3_synth_s1', 'ttt_pcalm_prox'),
     best([attr('r3_synth_s1', m) for m in ('ttt_gd_full', 'ttt_gd_official', 'ttt_quad_ridge', 'transformer')])),
]
json.dump([dict(base=b[0], nu=b[1], self_iteration=b[2], best_baseline=b[3]) for b in bases], open(FIG / 'fig_regime_data.json', 'w'), indent=1)

BB = load('base_bounds') if (RES / 'base_bounds' / 'results.json').exists() else json.loads((RES / 'base_bounds.json').read_text())
probe_nmse = {'GloVe (NLP)': attr('b1_nlp_glove_probe', 'probe_reference'), 'GINE (Graph)': attr('b1_graph_probe', 'probe_reference'),
              'MiniLM (NLP)': attr('b1_nlp_minilm', 'probe_reference'), r'synthetic, $\sigma$=0.5': attr('r3_synth_s0.5', 'probe_reference'),
              r'synthetic, $\sigma$=1': attr('r3_synth_s1', 'probe_reference'), 'image PCA-64 (CV)': cvj['projected_true_image_floor']['nmse_mean'],
              r'synthetic, $\sigma$=0': 0.0}
fig, axes = plt.subplots(1, 3, figsize=(28, 7.6), gridspec_kw=dict(width_ratios=[1.3, 1.0, 0.6]))
ax = axes[0]
gx = np.array([0.0] + [b['nu'] for b in BB if b['nu'] > 0]); gy = np.array([0.0] + [b['floor'] for b in BB if b['nu'] > 0])
gx, gy = gx[gx <= 0.62], gy[:int((gx <= 0.62).sum())]
ax.fill_between(gx, 0, gy, color=PALETTE['red_1'], alpha=0.75, zorder=0, lw=0)
ax.plot(gx, gy, color=PALETTE['red_strong'], lw=2.5, zorder=1)
ax.text(0.33, 0.30, 'unreachable for\nevery learner\n(Theorem R1)', fontsize=14, color=PALETTE['red_strong'])
for i, (name, v, s_, b) in enumerate(bases):
    ax.plot([v, v], [s_, b], color=PALETTE['neutral'], lw=5, zorder=2, solid_capstyle='round')
    ax.scatter([v], [probe_nmse[name]], s=120, color=ROLE['ref'], edgecolor='black', linewidth=1.3, zorder=3, marker='o')
    ax.scatter([v], [b], s=150, color=ROLE['gd_full'], edgecolor='black', linewidth=1.5, zorder=4)
    ax.scatter([v], [s_], s=180, color=ROLE['self'], edgecolor='black', linewidth=1.5, zorder=5, marker='D')
    off = {r'synthetic, $\sigma$=0': (12, -6), 'image PCA-64 (CV)': (14, -8), r'synthetic, $\sigma$=0.5': (-190, -6), 'GINE (Graph)': (14, -10),
           'GloVe (NLP)': (14, -10), 'MiniLM (NLP)': (-150, 6), r'synthetic, $\sigma$=1': (14, -12)}[name]
    ax.annotate(name, (v, s_), textcoords='offset points', xytext=off, fontsize=13)
ax.set_xlabel(r'base noise ratio $\nu = 1-R^2$ (deployment tasks)'); ax.set_ylabel('held-out NMSE ($n=4d$; CV $4m$)')
ax.set_xlim(-0.02, 0.6); ax.set_ylim(0.0, 1.12)
ax.set_title('(a) The base sets a floor for every learner', fontsize=16)

ax = axes[1]   # bases that violate B1: every learner, including the oracle probe, stays above floor(nu)
fl = lambda v: float(np.interp(v, [b['nu'] for b in BB if b['nu'] > 0], [b['floor'] for b in BB if b['nu'] > 0]))
groups = [('MiniLM (NLP)', 'b1_nlp_minilm', 'b1_nlp_minilm', nu('minilm', nlp_test)), (r'synthetic, $\sigma$=1', 'b3_synth_s1', 'r3_synth_s1', nu('synth_s1', s_test))]
meths = [('self-iteration', 'ttt_pcalm_prox', ROLE['self']), ('TTT, full-batch GD', 'ttt_gd_full', ROLE['gd_full']),
         ('closed form, quadratic', 'ttt_quad_ridge', ROLE['quad']), ('Transformer', 'transformer', ROLE['tf']), ('oracle probe', 'probe_reference', ROLE['ref'])]
w_ = 0.16
for gi, (gname, run_self, run_base, vv) in enumerate(groups):
    for mi, (lab_, m, col) in enumerate(meths):
        run = run_self if m == 'ttt_pcalm_prox' else run_base
        val = attr(run, m)
        ax.bar(gi + (mi - 2) * w_, val, w_, color=col, edgecolor='black', linewidth=1.5, label=lab_ if gi == 0 else None)
    ax.plot([gi - 0.45, gi + 0.45], [fl(vv)] * 2, color=PALETTE['red_strong'], lw=3.5, zorder=5)
    ax.text(gi, fl(vv) - 0.045, rf'floor$(\nu{{=}}{vv:.2f})={fl(vv):.2f}$', fontsize=13.5, color=PALETTE['red_strong'], ha='center', zorder=6,
            bbox=dict(facecolor='white', edgecolor=PALETTE['red_strong'], boxstyle='round,pad=0.25'))
ax.set_xticks(range(len(groups))); ax.set_xticklabels([g[0] for g in groups]); ax.set_ylim(0.5, 1.12)
ax.set_ylabel('held-out NMSE ($n=4d$)'); ax.set_title('(b) Bases violating R1: every learner fails', fontsize=16)
hb, lb = ax.get_legend_handles_labels()
legend_panel(axes[2], [Line2D([], [], color=ROLE['self'], marker='D', ls='', markeredgecolor='black', ms=11),
                       Line2D([], [], color=ROLE['gd_full'], marker='o', ls='', markeredgecolor='black', ms=11),
                       Line2D([], [], color=ROLE['ref'], marker='o', ls='', markeredgecolor='black', ms=10),
                       Line2D([], [], color=PALETTE['red_strong'], lw=2.5)] + hb,
             ['(a) self-iteration (matched)', '(a) best baseline (TTT-GD,\nclosed form, Transformer)', '(a) oracle probe (all labels)',
              r'floor$(\nu)$, any learner (Thm. R1)'] + ['(b) ' + l for l in lb], fontsize=13.5)
finalize(fig, 'fig_regime')

# ============================================================ Figure 3: bases that satisfy B1-B6
fig, axes = plt.subplots(1, 4, figsize=(32, 6.8), gridspec_kw=dict(width_ratios=[1, 1, 1, 0.62]))
handles = []
for ax, (title, ra, rb) in zip(axes[:2], (('(a) NLP: GloVe base', 'r9_nlp_s2_a', 'r9_nlp_s2_b'), ('(b) Graph: GINE base', 'r9_graph_s2_a', 'r9_graph_s2_b'))):
    n = [2, 4, 8]
    h1, = ax.plot(n, attr_curve(ra, 'ttt_pcalm_prox'), '-D', color=ROLE['self'], lw=3.2, ms=9)
    h2, = ax.plot(n, attr_curve(rb, 'ttt_gd_full'), '-o', color=ROLE['gd_full'])
    h3, = ax.plot(n, attr_curve(rb, 'ttt_gd_official'), '--o', color=ROLE['gd_off'])
    h4, = ax.plot(n, attr_curve(rb, 'ttt_quad_ridge'), '-s', color=ROLE['quad'])
    h5, = ax.plot(n, attr_curve(rb, 'transformer'), '-^', color=ROLE['tf'])
    handles = [h1, h2, h3, h4, h5]
    ax.set_xscale('log', base=2); ax.set_xticks(n); ax.set_xticklabels(['2d', '4d', '8d'])
    ax.set_xlabel('context size $n$'); ax.set_ylabel('held-out NMSE'); ax.set_title(title, fontsize=16)
ax = axes[2]; n = [2, 3, 4, 6]
ens = load('r8_cv_ens')['ensemble']['pca_pcalm_prox']
ax.plot(n, [ens[k]['nmse_mean'] for k in ('2.0', '3.0', '4.0', '6.0')], '-D', color=ROLE['self'], lw=3.2, ms=9)
gdf = load('r_cv_pr')['ttt_gd_full']['eval_test_images']; gdo = load('r3_cv_pcainit_b')['ttt_gd_official']['eval_test_images']
ax.plot(n, [gdf[k]['nmse_mean'] for k in ('2.0', '3.0', '4.0', '6.0')], '-o', color=ROLE['gd_full'])
ax.plot(n, [gdo[k]['nmse_mean'] for k in ('2.0', '3.0', '4.0', '6.0')], '--o', color=ROLE['gd_off'])
for m, col, mk in (('ttt_quad_ridge', ROLE['quad'], 's'), ('transformer', ROLE['tf'], '^')):
    ev = load('r10_cv_quad')[m]['eval_test_images']; ax.plot(n, [ev[k]['nmse_mean'] for k in ('2.0', '3.0', '4.0', '6.0')], '-' + mk, color=col)
floor = cvj['projected_true_image_floor']['nmse_mean']
hf = ax.axhline(floor, color=ROLE['ref'], lw=2, ls=':')
ax.set_xticks(n); ax.set_xticklabels(['2m', '3m', '4m', '6m']); ax.set_ylim(0, 0.55)
ax.set_xlabel('measurements $n$'); ax.set_ylabel('test-image NMSE'); ax.set_title('(c) CV: PCA-64 image base', fontsize=16)
legend_panel(axes[3], handles + [hf], ['matched self-iteration\n(TTT$\\times$PC-ALM)', 'TTT, full-batch GD', 'TTT, official GD',
                                        'closed form,\nquadratic features', 'Transformer', 'base floor (CV)'])
finalize(fig, 'fig_real')
print('written:', sorted(p.name for p in FIG.glob('fig_*')))
