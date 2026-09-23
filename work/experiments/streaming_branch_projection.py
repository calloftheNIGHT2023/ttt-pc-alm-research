"""Observed-context-only, minimum-change writes into piecewise nonlinear memory.

New local block-ALM prototype, not official TTT / official PC-ALM. All global
Jacobians below belong only to baselines and diagnostics, never the candidate.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize, linprog
from local_branch_memory import g, derivative, forward, fit_regression

BOUND = .15
EPS = 1e-3
TOL = 1e-6
KNOTS = np.array([0., .5, 1.])
SLOPES = np.array([0., 2., -2., 0.])
INTERCEPTS = np.array([0., 0., 2., 0.])


def forward_jacobian(x, b):
    """Global derivative: baseline / diagnostic ONLY."""
    h = x.copy()
    jac = np.zeros((len(x), len(b)))
    for j, bias in enumerate(b):
        d = derivative(h + bias)
        jac *= d[:, None]
        jac[:, j] += d
        h = g(h + bias)
    return h, jac


def pattern(x, b):
    h = x.copy()
    regions = []
    for bias in b:
        regions.append(np.searchsorted(KNOTS, h + bias, side="right"))
        h = g(h + bias)
    return np.stack(regions)


def branch_polytope(x, v, anchor):
    """Exact affine map AND region inequalities for frozen weights / fast biases."""
    depth = len(anchor)
    p = np.zeros((len(x), depth))
    c = x.copy()
    arows, rhs = [], []
    regions = pattern(x, anchor)
    lows = np.array([-np.inf, 0., .5, 1.])
    highs = np.array([0., .5, 1., np.inf])
    for j in range(depth):
        zmat = p.copy()
        zmat[:, j] += 1
        reg = regions[j]
        for i in range(len(x)):
            if np.isfinite(highs[reg[i]]):
                arows.append(zmat[i]); rhs.append(highs[reg[i]] - c[i])
            if np.isfinite(lows[reg[i]]):
                arows.append(-zmat[i]); rhs.append(c[i] - lows[reg[i]])
        p = SLOPES[reg, None] * zmat
        c = SLOPES[reg] * c + INTERCEPTS[reg]
    for i in range(len(x)):
        arows.extend([p[i], -p[i]])
        rhs.extend([v[i] + EPS - c[i], -v[i] + EPS + c[i]])
    return p, c, np.array(arows), np.array(rhs)


def branch_feasibility(x, v, anchor):
    _, _, amat, rhs = branch_polytope(x, v, anchor)
    res = linprog(np.zeros(len(anchor)), A_ub=amat, b_ub=rhs,
                  bounds=[(-BOUND, BOUND)] * len(anchor), method="highs")
    if res.success:
        error = float(np.max(np.abs(forward(x, res.x) - v)))
        assert error <= EPS + 2e-6, error
        return {"current_branch_feasible": True, "lp_status": int(res.status),
                "branch_witness_max_error": error}
    return {"current_branch_feasible": False if res.status == 2 else None,
            "lp_status": int(res.status)}


def bias_solve(previous, target, old, anchor, rho, trust):
    """Exact sorted O(R n log n) local block, including true write anchor."""
    restarts, n = previous.shape
    reg = np.searchsorted(KNOTS, previous - BOUND, side="right")
    s0 = SLOPES[reg]
    c0 = s0 * previous + INTERCEPTS[reg]
    initials = [np.sum(s0*s0, axis=1), np.sum(s0*(target-c0), axis=1),
                np.sum((c0-target)**2, axis=1)]
    events = (KNOTS[None, :, None] - previous[:, None, :]).reshape(restarts, -1)
    sb, sa = SLOPES[:-1][None, :, None], SLOPES[1:][None, :, None]
    cb = sb*previous[:, None, :] + INTERCEPTS[:-1][None, :, None]
    ca = sa*previous[:, None, :] + INTERCEPTS[1:][None, :, None]
    deltas = [np.broadcast_to(sa**2-sb**2, (restarts, 3, n)).reshape(restarts, -1),
              (sa*(target[:, None, :]-ca)-sb*(target[:, None, :]-cb)).reshape(restarts, -1),
              ((ca-target[:, None, :])**2-(cb-target[:, None, :])**2).reshape(restarts, -1)]
    valid = (events > -BOUND) & (events < BOUND)
    positions = np.clip(events, -BOUND, BOUND)
    order = np.argsort(positions, axis=1, kind="stable")
    positions = np.take_along_axis(positions, order, axis=1)
    coefs = []
    for ini, delta in zip(initials, deltas):
        ordered = np.take_along_axis(np.where(valid, delta, 0.), order, axis=1)
        coefs.append(np.concatenate([ini[:, None], ini[:, None]+np.cumsum(ordered, axis=1)], axis=1)/n)
    aa, bb, cc = coefs
    eta = 1/(rho*n)
    lo = np.concatenate([np.full((restarts, 1), -BOUND), positions], axis=1)
    hi = np.concatenate([positions, np.full((restarts, 1), BOUND)], axis=1)
    candidates = np.clip((bb + eta*anchor + trust*old[:, None]) /
                         (np.maximum(aa, 0)+eta+trust), lo, hi)
    cost = aa*candidates**2 - 2*bb*candidates + cc
    cost += eta*(candidates-anchor)**2 + trust*(candidates-old[:, None])**2
    return candidates[np.arange(restarts), np.argmin(cost, axis=1)]


def score(b, x, v, anchor):
    current = np.broadcast_to(x, (len(b), len(x)))
    for j in range(b.shape[1]):
        current = g(current + b[:, j, None])
    err = np.max(np.abs(current-v), axis=1)
    move = .5*np.sum((b-anchor)**2, axis=1)
    return err, move


def better(err, move, olderr, oldmove):
    feasible, oldfeasible = err <= EPS+TOL, olderr <= EPS+TOL
    return ((feasible & ~oldfeasible) |
            (feasible & oldfeasible & (move < oldmove)) |
            (~feasible & ~oldfeasible & (err < olderr)))


def fit_local(x, v, anchor, state=None, *, sweeps=600, restarts=16, rho=10.,
              dual_rate=.5, trust=.01, warm=True, gradient_blocks=False):
    """Only adjacent activities, local derivatives (in ablation), and residuals."""
    depth = len(anchor)
    if np.max(np.abs(forward(x, anchor)-v)) <= EPS:
        return anchor.copy(), state, {"skipped": True, "iterations": 0}
    b = np.vstack([anchor, np.random.default_rng(912).uniform(-BOUND, BOUND, (restarts-1, depth))])
    h = np.empty((depth, restarts, len(x)))
    u = np.zeros_like(h)
    previous = np.broadcast_to(x, (restarts, len(x)))
    for j in range(depth):
        h[j] = g(previous+b[:, j, None])
        previous = h[j]
    reused = 0
    if warm and state is not None:
        reused = state["h"].shape[1]
        assert np.array_equal(x[:reused], state["x"])
        assert np.array_equal(v[:reused], state["v"])
        assert np.array_equal(anchor, state["b"])
        h[:, 0, :reused] = state["h"]
        u[:, 0, :reused] = state["u"]
    best_b, best_h, best_u = b.copy(), h.copy(), u.copy()
    besterr, bestmove = score(b, x, v, anchor)
    best_iter = np.zeros(restarts, dtype=int)
    for it in range(sweeps):
        for j in reversed(range(depth)):
            previous = x if j == 0 else h[j-1]
            a = g(previous+b[:, j, None])-u[j]
            old = h[j].copy()
            if j == depth-1:
                h[j] = np.clip((a+trust*old)/(1+trust), np.maximum(0, v-EPS), np.minimum(1, v+EPS))
            elif gradient_blocks:
                z = old+b[:, j+1, None]
                grad = old-a + derivative(z)*(g(z)-h[j+1]-u[j+1])
                h[j] = np.clip(old-grad/(5+trust), 0, 1)
            else:
                nextb = b[:, j+1]
                lo = np.maximum(0, np.array([-np.inf, 0, .5, 1.])[:, None]-nextb)
                hi = np.minimum(1, np.array([0., .5, 1., np.inf])[:, None]-nextb)
                slope = SLOPES[:, None, None]
                offset = (SLOPES[:, None]*nextb + INTERCEPTS[:, None])[:, :, None]
                target = h[j+1]+u[j+1]
                cand = (a[None]+slope*(target[None]-offset)+trust*old[None])/(1+slope**2+trust)
                cand = np.minimum(np.maximum(cand, lo[:, :, None]), hi[:, :, None])
                energy = (cand-a)**2+(g(cand+nextb[None, :, None])-target)**2+trust*(cand-old)**2
                energy = np.where((lo > hi)[:, :, None], np.inf, energy)
                h[j] = np.take_along_axis(cand, np.argmin(energy, axis=0)[None], axis=0)[0]
        for j in range(depth):
            previous = np.broadcast_to(x, (restarts, len(x))) if j == 0 else h[j-1]
            target, old = h[j]+u[j], b[:, j].copy()
            if gradient_blocks:
                eta = 1/(rho*len(x))
                z = previous+old[:, None]
                grad = np.mean((g(z)-target)*derivative(z), axis=1)+eta*(old-anchor[j])
                b[:, j] = np.clip(old-grad/(4+eta+trust), -BOUND, BOUND)
            else:
                b[:, j] = bias_solve(previous, target, old, anchor[j], rho, trust)
        previous = x
        residual = np.empty_like(h)
        for j in range(depth):
            residual[j] = h[j]-g(previous+b[:, j, None])
            previous = h[j]
        u += dual_rate*residual
        err, move = score(b, x, v, anchor)
        update = better(err, move, besterr, bestmove)
        best_b[update], besterr[update], bestmove[update] = b[update], err[update], move[update]
        best_h[:, update], best_u[:, update] = h[:, update], u[:, update]
        best_iter[update] = it+1
    feasible = besterr <= EPS+TOL
    chosen = int(np.argmin(np.where(feasible, bestmove, np.inf))) if np.any(feasible) else int(np.argmin(besterr))
    out = best_b[chosen].copy()
    stateout = {"b": out.copy(), "h": best_h[:, chosen].copy(), "u": best_u[:, chosen].copy(),
                "x": x.copy(), "v": v.copy()}
    previous = x
    residuals = []
    for j in range(depth):
        residuals.append(stateout["h"][j]-g(previous+out[j]))
        previous = stateout["h"][j]
    return out, stateout, {"skipped": False, "iterations": sweeps, "winning_restart": chosen,
                           "best_iteration": int(best_iter[chosen]), "reused_samples": reused,
                           "selected_local_residual_max": float(np.max(np.abs(residuals))),
                           "retained_state_bytes": sum(a.nbytes for a in stateout.values()),
                           "main_search_arrays_bytes": sum(a.nbytes for a in [b,h,u,best_b,best_h,best_u]),
                           "memory_note": "array subtotals only; excludes temporary branch/sort arrays, Python and library workspace"}


def fit_slsqp(x, v, anchor, restarts=16, steps=300):
    if np.max(np.abs(forward(x, anchor)-v)) <= EPS:
        return anchor.copy(), {"skipped": True, "iterations": 0}
    starts = np.vstack([anchor, np.random.default_rng(912).uniform(-BOUND, BOUND, (restarts-1, len(anchor)))])
    best = anchor.copy()
    err0, move0 = score(best[None], x, v, anchor)
    calls, iterations, solver_success = 0, 0, 0
    def cons(b):
        error = forward(x, b)-v
        return np.r_[EPS-error, EPS+error]
    def cjac(b):
        jac = forward_jacobian(x, b)[1]
        return np.concatenate([-jac, jac])
    for initial in starts:
        res = minimize(lambda b: (.5*np.sum((b-anchor)**2), b-anchor), initial, jac=True,
                       method="SLSQP", bounds=[(-BOUND, BOUND)]*len(anchor),
                       constraints={"type":"ineq", "fun":cons, "jac":cjac},
                       options={"maxiter":steps, "ftol":1e-11})
        calls += res.nfev; iterations += res.nit; solver_success += int(res.success)
        err, move = score(res.x[None], x, v, anchor)
        if better(err, move, err0, move0)[0]:
            best, err0, move0 = res.x.copy(), err, move
    return best, {"skipped": False, "iterations": iterations, "function_calls": calls,
                  "solver_successes": solver_success}


def verify():
    rng = np.random.default_rng(1804)
    # Independent dense grid oracle for the anchored exact parameter block.
    grid = np.linspace(-BOUND, BOUND, 100001)
    maxgap = -np.inf
    for _ in range(20):
        prev = rng.uniform(0,1,(1,7)); target = rng.normal(.5,.5,(1,7))
        old = rng.uniform(-BOUND,BOUND,1); anchor = rng.uniform(-BOUND,BOUND)
        out = bias_solve(prev,target,old,anchor,10.,.01)[0]
        def energy(z):
            return np.mean((g(prev[0][None]+np.asarray(z)[:,None])-target[0])**2,axis=1)+(np.asarray(z)-anchor)**2/70+.01*(np.asarray(z)-old[0])**2
        gap = energy([out])[0]-energy(grid).min()
        maxgap = max(maxgap, gap)
        assert gap < 1e-9, gap
    # True-forward affine map in the exact current branch; Jacobian audit.
    x = rng.uniform(0,1,12); b = rng.uniform(-.12,.12,4); v = forward(x,b)
    p,c,amat,rhs = branch_polytope(x,v,b)
    assert np.max(np.abs(p@b+c-v)) < 1e-12
    assert np.max(amat@b-rhs) <= 1e-12
    assert branch_feasibility(x,v,b)["current_branch_feasible"]
    jac = forward_jacobian(x,b)[1]
    numeric = np.column_stack([(forward(x,b+np.eye(4)[j]*1e-7)-forward(x,b-np.eye(4)[j]*1e-7))/2e-7 for j in range(4)])
    graderr = float(np.max(np.abs(jac-numeric)))
    assert graderr < 1e-6
    # Linear exact equality projection: recovery control, not candidate advantage.
    mat = rng.normal(size=(6,10)); a = rng.normal(size=10); target = rng.normal(size=6)
    closed = a+mat.T@np.linalg.solve(mat@mat.T,target-mat@a)
    res = minimize(lambda z:(.5*np.sum((z-a)**2),z-a),a,jac=True,method="SLSQP",
                   constraints={"type":"eq","fun":lambda z:mat@z-target,"jac":lambda z:mat},
                   options={"ftol":1e-12})
    linearerr = float(np.max(np.abs(closed-res.x)))
    assert linearerr < 1e-9
    return {"passed":True,"bias_grid_cases":20,"max_energy_gap_to_grid":float(maxgap),
            "jacobian_fd_max_error":graderr,"linear_projection_slsqp_max_error":linearerr}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--count",type=int,default=8)
    parser.add_argument("--verify-only",action="store_true")
    parser.add_argument("--config-file",type=Path)
    parser.add_argument("--confirm",action="store_true")
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    checks=verify()
    (args.out/"verification.json").write_text(json.dumps(checks,indent=2),encoding="utf-8")
    print(json.dumps(checks),flush=True)
    if args.verify_only: return
    configs=[{"name":"local_warm","kind":"local","warm":True},
             {"name":"local_reset","kind":"local","warm":False},
             {"name":"local_no_dual","kind":"local","warm":False,"dual_rate":0.},
             {"name":"local_derivative","kind":"local","warm":False,"gradient_blocks":True},
             {"name":"slsqp16","kind":"slsqp","restarts":16},
             {"name":"linear","kind":"regression","family":"linear"},
             {"name":"rbf","kind":"regression","family":"rbf"},
             {"name":"prior1024","kind":"regression","family":"prior_bank","features":1024}]
    if args.config_file:
        configs=json.loads(args.config_file.read_text(encoding="utf-8"))["configs"]
    if args.confirm and not args.config_file:
        raise ValueError("Freeze configs before confirmation")
    seed0=5700000 if args.confirm else 5600000
    stages=[4,8,16,24]
    protocol={"phase":"confirmation" if args.confirm else "development","seed0":seed0,
              "count":args.count,"stages":stages,"depth":4,"queries":2048,"epsilon":EPS,
              "feasibility_slack":TOL,"configs":configs,
              "local_defaults":{"sweeps":600,"restarts":16,"rho":10.,"dual_rate":.5,"trust":.01},
              "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "import_sha256":hashlib.sha256(Path(__file__).with_name("local_branch_memory.py").read_bytes()).hexdigest()}
    (args.out/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf-8")
    rows=[]
    for cfg in configs:
        opts={k:v for k,v in cfg.items() if k not in ["name","kind"]}
        for seed in range(seed0,seed0+args.count):
            rng=np.random.default_rng(seed)
            truth=rng.uniform(-.12,.12,4)
            x=rng.uniform(0,1,24); v=forward(x,truth)
            q=rng.uniform(0,1,2048); target=forward(q,truth)
            b=np.zeros(4); state=None; nold=0
            for n in stages:
                anchor=b.copy()
                diagnostic=branch_feasibility(x[:n],v[:n],anchor) if cfg["kind"]!="regression" else {}
                before=time.perf_counter()
                if cfg["kind"]=="local":
                    b,state,metadata=fit_local(x[:n],v[:n],anchor,state,**opts)
                    predict=lambda z:forward(z,b)
                elif cfg["kind"]=="slsqp":
                    b,metadata=fit_slsqp(x[:n],v[:n],anchor,**opts)
                    predict=lambda z:forward(z,b)
                else:
                    predict,metadata=fit_regression(x[:n],v[:n],4,**opts)
                seconds=time.perf_counter()-before
                pred=predict(x[:n]); qp=predict(q)
                err=float(np.max(np.abs(pred-v[:n])))
                row={"method":cfg["name"],"seed":seed,"n_context":n,"seconds":seconds,
                     "query_mse":float(np.mean((qp-target)**2)),"support_max_error":err,
                     "support_feasible":bool(err<=EPS+TOL),
                     "old_support_max_error":float(np.max(np.abs(pred[:nold]-v[:nold]))) if nold else None,
                     **metadata,**diagnostic}
                if cfg["kind"]!="regression":
                    row.update({"b":b.tolist(),"anchor":anchor.tolist(),"move_squared":float(np.sum((b-anchor)**2)),
                                "branch_change_fraction":float(np.mean(pattern(x[:n],b)!=pattern(x[:n],anchor)))})
                rows.append(row); nold=n
        summary=[]
        for n in stages:
            selected=[r for r in rows if r["method"]==cfg["name"] and r["n_context"]==n]
            summary.append({"method":cfg["name"],"n_context":n,
                            "query_mse":float(np.mean([r["query_mse"] for r in selected])),
                            "feasible_count":sum(r["support_feasible"] for r in selected),
                            "count":len(selected),"median_seconds":float(np.median([r["seconds"] for r in selected]))})
        print(json.dumps(summary),flush=True)
        (args.out/"episodes.json").write_text(json.dumps(rows,indent=2),encoding="utf-8")
    summary=[]
    for cfg in configs:
        for n in stages:
            selected=[r for r in rows if r["method"]==cfg["name"] and r["n_context"]==n]
            summary.append({"method":cfg["name"],"n_context":n,
                            "query_mse":float(np.mean([r["query_mse"] for r in selected])),
                            "feasible_count":sum(r["support_feasible"] for r in selected),
                            "count":len(selected),"median_seconds":float(np.median([r["seconds"] for r in selected]))})
    (args.out/"summary.json").write_text(json.dumps(summary,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
