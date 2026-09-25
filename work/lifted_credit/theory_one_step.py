"""Prop 3 check: dimension-free overlap after ONE lifted step from an uninformed start.

Theory: with anchor a independent of z*=u.x (overlap m~0), exact inversion gives s = S(y, a).
c = E[s z*], v = E[s^2]. LS of s on x with n = alpha*d Gaussian rows gives w1 = c*u + noise,
||noise||^2 ~ (v - c^2) d/(n-d)  ->  overlap1 = c / sqrt(c^2 + (v-c^2)/(alpha-1)), independent of d.
"""
import math, torch, common as C
torch.manual_seed(0)
link = C.He3; sigma = 0.1; rho = 0.1
N = 2_000_000
z = torch.randn(N, dtype=torch.float64); a = torch.randn(N, dtype=torch.float64)
y = link.f(z) + sigma * torch.randn(N, dtype=torch.float64)
grid = torch.linspace(-5, 5, 241, dtype=torch.float64)
s = torch.cat([C._activity_step(link, y[i:i+200000], a[i:i+200000], rho, grid) for i in range(0, N, 200000)])
c = float((s * z).mean()); v = float((s * s).mean())
print(f'c=E[s z*]={c:.4f}  v=E[s^2]={v:.4f}')
for alpha in [2, 4, 8]:
    pred = c / math.sqrt(c * c + (v - c * c) / (alpha - 1))
    emp = []
    for d in [32, 128, 512]:
        n = int(alpha * d); E = 16
        X = torch.randn(E, n, d, dtype=torch.float64); u = torch.randn(E, d, dtype=torch.float64); u /= u.norm(dim=1, keepdim=True)
        zz = torch.einsum('end,ed->en', X, u); yy = link.f(zz) + sigma * torch.randn(E, n, dtype=torch.float64)
        w0 = torch.randn(E, d, dtype=torch.float64) / math.sqrt(d)
        ss = C._activity_step(link, yy, torch.einsum('end,ed->en', X, w0), rho, grid)
        w1 = torch.linalg.solve(torch.einsum('end,enk->edk', X, X), torch.einsum('end,en->ed', X, ss).unsqueeze(-1)).squeeze(-1)
        emp.append(f'd={d}: {float(C.overlap(w1, u).mean()):.3f}')
    print(f'alpha={alpha}: predicted overlap after one lifted step = {pred:.3f} | empirical ' + ', '.join(emp))
# BP comparison: one full-batch gradient step from the same start has overlap ~ O(1/sqrt(d)) + noise
for d in [32, 128, 512]:
    n = 4 * d; E = 16
    X = torch.randn(E, n, d, dtype=torch.float64); u = torch.randn(E, d, dtype=torch.float64); u /= u.norm(dim=1, keepdim=True)
    yy = link.f(torch.einsum('end,ed->en', X, u)) + sigma * torch.randn(E, n, dtype=torch.float64)
    w0 = torch.randn(E, d, dtype=torch.float64) / math.sqrt(d); sw = torch.einsum('end,ed->en', X, w0)
    gr = torch.einsum('en,end->ed', (link.f(sw) - yy) * link.df(sw), X) / n
    print(f'd={d}: overlap of the BP gradient direction with u = {float(C.overlap(-gr, u).abs().mean()):.3f} (1/sqrt(d)={1/math.sqrt(d):.3f})')
