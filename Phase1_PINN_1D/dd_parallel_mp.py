"""
Real multiprocess (true-parallel) overlapping-Schwarz DD-PINN in 1D.

Why this file exists
--------------------
This module instead executes
the subdomains **genuinely in parallel**, one OS process per subdomain, so the
wall-clock speed-up is measured, not inferred.

Two macOS gotchas this file is built around:
  1. Threads don't help (Python GIL) -> we use processes (torch.multiprocessing).
  2. macOS uses the 'spawn' start method, which CANNOT pickle functions defined
     in a Jupyter notebook. So the worker lives here, at module scope, and is
     imported by the child processes. Run this as a script or import run_dd_mp
     from a notebook -- never define the worker in the notebook.

Each worker is PERSISTENT: it builds its model once and keeps it across all
Schwarz rounds, so the only thing crossing the pipe each round is two scalars
(the neighbour interface values). That keeps IPC negligible.

Run:
    python3.13 dd_parallel_mp.py
"""
import argparse
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

# --------------------------------------------------------------------------
# Problem registry. Keyed by string so it survives 'spawn' pickling.
#
# pde_type controls the residual in the worker/seq/vanilla:
#   "poisson"    : -u'' = f                         residual = -u'' - f
#   "react_diff" : -u'' + lam*u = f                 residual = -u'' + lam*u - f
#   "adv_diff"   : -eps*u'' + c*u' = f              residual = -eps*u'' + c*u' - f
#
# All problems have zero or known Dirichlet BCs at x=0,1 and closed-form
# exact solutions, so L2 error is always well-defined.
# --------------------------------------------------------------------------

# ── Poisson  -u'' = f ────────────────────────────────────────────────────
def _poisson(n):
    """u = sin(n pi x),  f = (n pi)^2 sin(n pi x)"""
    return dict(
        pde_type = "poisson",
        f = lambda x, n=n: (n * torch.pi) ** 2 * torch.sin(n * torch.pi * x),
        u = lambda x, n=n: torch.sin(n * torch.pi * x),
        uL=0.0, uR=0.0,
    )

# ── Reaction-diffusion  -u'' + lam*u = f  ────────────────────────────────
# Exact: u = sin(n pi x),  f = (n^2 pi^2 + lam) sin(n pi x)
def _react_diff(n, lam):
    return dict(
        pde_type = "react_diff",
        lam = float(lam),
        f   = lambda x, n=n, lam=lam: ((n * torch.pi) ** 2 + lam) * torch.sin(n * torch.pi * x),
        u   = lambda x, n=n: torch.sin(n * torch.pi * x),
        uL=0.0, uR=0.0,
    )

# ── Advection-diffusion  -eps*u'' + c*u' = 0,  u(0)=0, u(1)=1  ──────────
# Exact: u = (exp(c x / eps) - 1) / (exp(c / eps) - 1)
# Boundary layer of width ~eps/c near x=1 when eps << c.
# f = 0 here (homogeneous), BCs are non-trivial.
def _adv_diff(eps, c):
    ec = float(np.exp(c / eps))
    return dict(
        pde_type = "adv_diff",
        eps = float(eps),
        c   = float(c),
        f   = lambda x: torch.zeros_like(x),
        u   = lambda x, eps=eps, c=c, ec=ec: (torch.exp(c * x / eps) - 1) / (ec - 1),
        uL  = 0.0,
        uR  = 1.0,
    )

PROBLEMS = {
    # --- Poisson ---
    "sin1":       _poisson(1),
    "sin4":       _poisson(4),
    "exp":        dict(pde_type="poisson",
                       f=lambda x: -torch.exp(x),
                       u=lambda x: torch.exp(x),
                       uL=1.0, uR=float(np.exp(1.0))),
    # --- Reaction-diffusion  -u'' + lam*u = f ---
    "rd_lam1":    _react_diff(1, 1),     # mild reaction
    "rd_lam10":   _react_diff(1, 10),    # strong reaction, harder for vanilla
    # --- Advection-diffusion  -eps*u'' + c*u' = 0 ---
    "ad_e1e-1":   _adv_diff(0.1,  1.0), # gentle layer
    "ad_e1e-2":   _adv_diff(0.01, 1.0), # sharp layer — classic PINN failure
}


# --------------------------------------------------------------------------
# Network: hard Dirichlet BCs via a distance function (same as the notebooks).
# --------------------------------------------------------------------------
def mlp(width, depth=3):
    layers = [nn.Linear(1, width), nn.Tanh()]
    for _ in range(depth - 1):
        layers += [nn.Linear(width, width), nn.Tanh()]
    layers += [nn.Linear(width, 1)]
    net = nn.Sequential(*layers)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class HardPINN(nn.Module):
    """u(x) = lift(x) + (x-a)(x-b) * N(x), exact Dirichlet BCs at a, b."""
    def __init__(self, a, b, width, depth=3):
        super().__init__()
        self.a, self.b = a, b
        self.net = mlp(width, depth)

    def forward(self, x, ua, ub):
        lift = ua + (x - self.a) / (self.b - self.a) * (ub - ua)
        return lift + (x - self.a) * (x - self.b) * self.net(x)


def second_deriv(u, x):
    du = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    return torch.autograd.grad(du, x, torch.ones_like(du), create_graph=True)[0]


def first_deriv(u, x):
    return torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]


def pde_residual(u, x, prob_cfg):
    """Returns the PDE residual tensor for any supported pde_type."""
    pde = prob_cfg["pde_type"]
    f   = prob_cfg["f"]
    uxx = second_deriv(u, x)
    if pde == "poisson":
        return -uxx - f(x)
    elif pde == "react_diff":
        return -uxx + prob_cfg["lam"] * u - f(x)
    elif pde == "adv_diff":
        ux = first_deriv(u, x)
        return -prob_cfg["eps"] * uxx + prob_cfg["c"] * ux - f(x)
    else:
        raise ValueError(f"Unknown pde_type: {pde}")


# --------------------------------------------------------------------------
# Geometry: split [0,1] into K overlapping subdomains.
# Subdomain k spans [a_k, b_k]; interior interfaces overlap their neighbours.
# --------------------------------------------------------------------------
def make_geometry(K, overlap):
    edges = np.linspace(0.0, 1.0, K + 1)          # K+1 nominal split points
    a = np.empty(K); b = np.empty(K)
    for k in range(K):
        a[k] = edges[k] - (overlap / 2 if k > 0 else 0.0)
        b[k] = edges[k + 1] + (overlap / 2 if k < K - 1 else 0.0)
    # Point in subdomain k where its RIGHT neighbour's left edge sits (a_{k+1}),
    # and where its LEFT neighbour's right edge sits (b_{k-1}). Both interior to k.
    right_query = [float(a[k + 1]) if k < K - 1 else None for k in range(K)]
    left_query = [float(b[k - 1]) if k > 0 else None for k in range(K)]
    return a, b, left_query, right_query, edges


# --------------------------------------------------------------------------
# Persistent worker: one subdomain, one process, model kept across rounds.
# Protocol over the pipe, per round:
#   parent -> child : ("step", left_bc_value, right_bc_value)
#   child  -> parent: (u_at_left_query, u_at_right_query, loss)
#   parent -> child : ("done",) then child returns its full state_dict + preds
# --------------------------------------------------------------------------
def subdomain_worker(conn, k, cfg):
    torch.set_num_threads(1)                       # avoid core oversubscription
    torch.manual_seed(42 + k)
    prob = PROBLEMS[cfg["prob"]]
    a, b = cfg["a"], cfg["b"]
    model = HardPINN(a, b, cfg["width"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    xc = torch.linspace(a, b, cfg["n_col"]).view(-1, 1)
    lq = torch.tensor([[cfg["left_query"]]]) if cfg["left_query"] is not None else None
    rq = torch.tensor([[cfg["right_query"]]]) if cfg["right_query"] is not None else None

    while True:
        msg = conn.recv()
        if msg[0] == "done":
            xt = torch.linspace(a, b, cfg["n_eval"]).view(-1, 1)
            with torch.no_grad():
                ua = torch.tensor([[msg[1]]]); ub = torch.tensor([[msg[2]]])
                pred = model(xt, ua, ub).numpy().ravel()
            conn.send((xt.numpy().ravel(), pred))
            conn.close()
            return

        _, lbc, rbc = msg
        ua = torch.tensor([[lbc]]); ub = torch.tensor([[rbc]])
        x = xc.clone().detach().requires_grad_(True)
        loss = None
        for _ in range(cfg["steps_per"]):
            u = model(x, ua, ub)
            loss = torch.mean(pde_residual(u, x, prob) ** 2)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        with torch.no_grad():
            ul = model(lq, ua, ub).item() if lq is not None else 0.0
            ur = model(rq, ua, ub).item() if rq is not None else 0.0
        conn.send((ul, ur, float(loss.item())))


# --------------------------------------------------------------------------
# Driver: spawn K persistent workers, run Schwarz rounds in true parallel.
# --------------------------------------------------------------------------
def run_dd_mp(prob="sin1", K=2, overlap=0.2, width=32, n_iter=15,
              steps_per=400, n_col=256, lr=1e-3, alpha=1.0,
              n_eval=400, verbose=True):
    p = PROBLEMS[prob]
    a, b, left_q, right_q, edges = make_geometry(K, overlap)

    # interface values: left_bc[k], right_bc[k] feeding subdomain k each round
    left_bc = np.zeros(K); right_bc = np.zeros(K)
    left_bc[0] = p["uL"]; right_bc[K - 1] = p["uR"]

    ctx = mp.get_context("spawn")
    parents, procs = [], []
    for k in range(K):
        pc, cc = ctx.Pipe()
        cfg = dict(prob=prob, a=float(a[k]), b=float(b[k]), width=width,
                   lr=lr, n_col=n_col, steps_per=steps_per, n_eval=n_eval,
                   left_query=left_q[k], right_query=right_q[k])
        proc = ctx.Process(target=subdomain_worker, args=(cc, k, cfg))
        proc.start()
        parents.append(pc); procs.append(proc)

    # exact solution on a fine grid for L2 scoring
    xt = torch.linspace(0, 1, 2000).view(-1, 1)
    ut = p["u"](xt).numpy().ravel(); xt = xt.numpy().ravel()

    hist = []
    t0 = time.perf_counter()
    for it in range(n_iter):
        # Jacobi: every worker trains against the *previous* round's interface
        for k in range(K):
            parents[k].send(("step", float(left_bc[k]), float(right_bc[k])))
        # they are now all training simultaneously; collect when each finishes
        ul = np.zeros(K); ur = np.zeros(K)
        for k in range(K):
            ul[k], ur[k], _ = parents[k].recv()
        # route interface values to neighbours, with relaxation
        for k in range(1, K):           # left BC of k <- right-query value of k-1
            left_bc[k] = alpha * ur[k - 1] + (1 - alpha) * left_bc[k]
        for k in range(K - 1):          # right BC of k <- left-query value of k+1
            right_bc[k] = alpha * ul[k + 1] + (1 - alpha) * right_bc[k]
        hist.append(time.perf_counter() - t0)
        if verbose:
            print(f"  round {it + 1:2d}/{n_iter}  t={hist[-1]:5.2f}s")
    wall_par = time.perf_counter() - t0

    # gather final predictions, stitch at nominal edges
    pred = np.empty_like(ut)
    for k in range(K):
        parents[k].send(("done", float(left_bc[k]), float(right_bc[k])))
    pieces = [parents[k].recv() for k in range(K)]
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        mask = (xt >= lo) & (xt <= hi) if k == K - 1 else (xt >= lo) & (xt < hi)
        xk, pk = pieces[k]
        pred[mask] = np.interp(xt[mask], xk, pk)
    for proc in procs:
        proc.join()

    l2 = np.linalg.norm(pred - ut) / np.linalg.norm(ut)
    return dict(K=K, l2=float(l2), wall_par=wall_par, hist=hist,
                x=xt, pred=pred, exact=ut)


# --------------------------------------------------------------------------
# Sequential baseline in THIS process (apples-to-apples: same work, no MP),
# so wall_par / wall_seq is the honest measured speed-up.
# --------------------------------------------------------------------------
def run_dd_seq(prob="sin1", K=4, overlap=0.2, width=32, n_iter=15,
               steps_per=400, n_col=256, lr=1e-3, alpha=1.0):
    torch.set_num_threads(1)
    p = PROBLEMS[prob]
    a, b, left_q, right_q, edges = make_geometry(K, overlap)
    models, opts, xcs, lqs, rqs = [], [], [], [], []
    for k in range(K):
        torch.manual_seed(42 + k)
        m = HardPINN(float(a[k]), float(b[k]), width)
        models.append(m); opts.append(torch.optim.Adam(m.parameters(), lr=lr))
        xcs.append(torch.linspace(a[k], b[k], n_col).view(-1, 1))
        lqs.append(torch.tensor([[left_q[k]]]) if left_q[k] is not None else None)
        rqs.append(torch.tensor([[right_q[k]]]) if right_q[k] is not None else None)
    left_bc = np.zeros(K); right_bc = np.zeros(K)
    left_bc[0] = p["uL"]; right_bc[K - 1] = p["uR"]

    t0 = time.perf_counter()
    for _ in range(n_iter):
        ul = np.zeros(K); ur = np.zeros(K)
        for k in range(K):
            ua = torch.tensor([[left_bc[k]]]); ub = torch.tensor([[right_bc[k]]])
            x = xcs[k].clone().detach().requires_grad_(True)
            for _ in range(steps_per):
                u = models[k](x, ua, ub)
                loss = torch.mean(pde_residual(u, x, p) ** 2)
                opts[k].zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(models[k].parameters(), 1.0)
                opts[k].step()
            with torch.no_grad():
                ul[k] = models[k](lqs[k], ua, ub).item() if lqs[k] is not None else 0.0
                ur[k] = models[k](rqs[k], ua, ub).item() if rqs[k] is not None else 0.0
        for k in range(1, K):
            left_bc[k] = alpha * ur[k - 1] + (1 - alpha) * left_bc[k]
        for k in range(K - 1):
            right_bc[k] = alpha * ul[k + 1] + (1 - alpha) * right_bc[k]
    return time.perf_counter() - t0


def run_vanilla(prob="sin4", width=47, steps=6000, n_col=256, lr=1e-3):
    """One global PINN on all of [0,1] — single network, NO multiprocessing.
    Gets the full machine (threads unpinned). This is the baseline DD must beat."""
    torch.manual_seed(42)
    p = PROBLEMS[prob]
    m = HardPINN(0.0, 1.0, width)
    ua = torch.tensor([[p["uL"]]]); ub = torch.tensor([[p["uR"]]])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(steps):
        x = torch.rand(n_col, 1, requires_grad=True)
        u_v = m(x, ua, ub)
        loss = torch.mean(pde_residual(u_v, x, p) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    xt = torch.linspace(0, 1, 2000).view(-1, 1)
    with torch.no_grad():
        l2 = (torch.norm(m(xt, ua, ub) - p["u"](xt)) / torch.norm(p["u"](xt))).item()
    return dict(wall=wall, l2=float(l2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob", default="sin4", choices=list(PROBLEMS))
    ap.add_argument("--Ks", default="2", help="comma list of #subdomains")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--vs-vanilla", action="store_true",
                    help="measured head-to-head: global PINN vs true-parallel DD")
    args = ap.parse_args()

    print(f"problem={args.prob}  cores={mp.cpu_count()}  "
          f"iters={args.iters}  steps/round={args.steps}  width={args.width}\n")

    if args.vs_vanilla:
        v = run_vanilla(args.prob)
        print(f"{'method':>22} {'L2':>10} {'wall (s)':>10} {'vs vanilla':>11}")
        print(f"{'vanilla PINN (global)':>22} {v['l2']:>10.2e} {v['wall']:>10.2f} "
              f"{'1.00x':>11}")
        for K in [int(x) for x in args.Ks.split(",")]:
            r = run_dd_mp(args.prob, K=K, n_iter=args.iters,
                          steps_per=args.steps, width=args.width, verbose=False)
            tag = f"DD true-parallel K={K}"
            print(f"{tag:>22} {r['l2']:>10.2e} {r['wall_par']:>10.2f} "
                  f"{v['wall'] / r['wall_par']:>10.2f}x")
        return

    print(f"{'K':>3} {'L2':>10} {'seq (s)':>9} {'par (s)':>9} {'speedup':>8}")
    for K in [int(x) for x in args.Ks.split(",")]:
        seq = run_dd_seq(args.prob, K=K, n_iter=args.iters,
                         steps_per=args.steps, width=args.width)
        r = run_dd_mp(args.prob, K=K, n_iter=args.iters,
                      steps_per=args.steps, width=args.width, verbose=False)
        print(f"{K:>3} {r['l2']:>10.2e} {seq:>9.2f} {r['wall_par']:>9.2f} "
              f"{seq / r['wall_par']:>7.2f}x")


if __name__ == "__main__":
    main()