"""Controlled bridge for report 450: synthetic multi-attribute data with a KNOWN noise level inside the |.|.
Items e ~ N(0, I_64); 20 attributes A_j = u_j . e + sigma * eps_j with u_j in a hidden 16-dim subspace (unit norm),
standardised. Pair task label |A_j(a) - A_j(b)| = |u_j.(e_a - e_b) + sigma (eps_a - eps_b)|: sigma = 0 is the exact
multi-branch regime of report 448, sigma = 1 matches the decodability of the real NLP/Graph attributes (R^2 = 0.5).
Train attributes 0..14, held-out 15..19.
"""
import argparse
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument('--sigma', type=float, required=True); ap.add_argument('--out', required=True)
a = ap.parse_args()
rng = np.random.default_rng(450)
N, D0, r, P = 30000, 64, 16, 20
E = rng.standard_normal((N, D0))
Q = np.linalg.qr(rng.standard_normal((D0, r)))[0]
U = Q @ rng.standard_normal((r, P)); U /= np.linalg.norm(U, axis=0)
A = E @ U + a.sigma * rng.standard_normal((N, P))
A = (A - A.mean(0)) / A.std(0)
np.savez(a.out, E=E, A=A, names=np.array([f'a{j}' for j in range(P)]), train_idx=np.arange(15), test_idx=np.arange(15, 20))
print('saved', a.out, 'R2 =', 1 / (1 + a.sigma ** 2))
