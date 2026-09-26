"""Paired per-task comparisons for exp_deep_amp.py outputs (report 447).

For each pair (A, B) on the same tasks: success difference with paired bootstrap 95% CI (20,000 resamples),
exact two-sided McNemar test on discordant pairs, and mean-NMSE difference with paired bootstrap CI.
Usage: python analyze_paired.py <run_dir> [<run_dir> ...]   (pooled over the given runs as well)
"""
import json
import math
import sys
from pathlib import Path
import numpy as np

REF = 'pcalm_deepjoint_8x2+adam'
EXTRA_PAIRS = [('pcalm_deepjoint_2x2+adam', 'amp_full_em_4x2+adam'), ('amp_full_em_16x2+adam', 'amp_full_em_4x2+adam')]


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(ma, mb, rng, B=20000):
    sa, sb = (ma < .05).astype(float), (mb < .05).astype(float)
    idx = rng.integers(0, len(ma), (B, len(ma)))
    ds = (sa[idx] - sb[idx]).mean(1); dm = (ma[idx] - mb[idx]).mean(1)
    b = int(((sa == 1) & (sb == 0)).sum()); c = int(((sa == 0) & (sb == 1)).sum())
    return dict(success_A=float(sa.mean()), success_B=float(sb.mean()), diff=float(sa.mean() - sb.mean()),
                ci=[float(np.quantile(ds, .025)), float(np.quantile(ds, .975))], A_only=b, B_only=c,
                mcnemar_p=mcnemar_exact(b, c), nmse_diff=float((ma - mb).mean()),
                nmse_ci=[float(np.quantile(dm, .025)), float(np.quantile(dm, .975))])


def main(dirs):
    rng = np.random.default_rng(447)
    runs = {}
    for dname in dirs:
        rows = json.loads((Path(dname) / 'rows.json').read_text())
        runs[Path(dname).name] = {r['method']: (np.array(r['per_task_nmse']), r) for r in rows}
    out = {}
    for rn, rr in list(runs.items()) + [('pooled', None)]:
        if rr is None:
            common = set.intersection(*[set(v) for v in runs.values()])
            rr = {m: (np.concatenate([runs[k][m][0] for k in runs]), None) for m in common}
        print(f'== {rn}')
        for m, (v, row) in sorted(rr.items(), key=lambda kv: -float((kv[1][0] < .05).mean())):
            extra = f"  {row['seconds']:.0f}s  ov={row['subspace_overlap']:.3f}" if row else ''
            print(f'   {m:30s} success={float((v < .05).mean()):.3f}  mean_nmse={v.mean():.4f}  median={np.median(v):.4f}{extra}')
        if REF in rr:
            res = {}
            for m in rr:
                if m == REF:
                    continue
                res[m] = paired(rr[REF][0], rr[m][0], rng)
                p = res[m]
                print(f'   {REF} vs {m}: diff={p["diff"]:+.3f} CI[{p["ci"][0]:+.3f},{p["ci"][1]:+.3f}] '
                      f'discordant {p["A_only"]}/{p["B_only"]} McNemar p={p["mcnemar_p"]:.3g}  '
                      f'nmse diff={p["nmse_diff"]:+.4f} CI[{p["nmse_ci"][0]:+.4f},{p["nmse_ci"][1]:+.4f}]')
            for A, Bm in EXTRA_PAIRS:
                if A in rr and Bm in rr:
                    p = paired(rr[A][0], rr[Bm][0], rng); res[f'{A} vs {Bm}'] = p
                    print(f'   {A} vs {Bm}: diff={p["diff"]:+.3f} CI[{p["ci"][0]:+.3f},{p["ci"][1]:+.3f}] '
                          f'discordant {p["A_only"]}/{p["B_only"]} McNemar p={p["mcnemar_p"]:.3g}  '
                          f'nmse diff={p["nmse_diff"]:+.4f} CI[{p["nmse_ci"][0]:+.4f},{p["nmse_ci"][1]:+.4f}]')
            out[rn] = res
    Path(dirs[0]).parent.joinpath('k_paired_summary.json').write_text(json.dumps(out, indent=1))


if __name__ == '__main__':
    main(sys.argv[1:])
