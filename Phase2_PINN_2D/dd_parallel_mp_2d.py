"""
Real multiprocess (true-parallel) overlapping-Schwarz DD-PINN in 2D.

Supports two problem families:

  Poisson   -Laplacian(u) = f on [0,1]^2  (elliptic, space-only)
    keys: sin11, sin13, sin31, sin33
    exact: u = sin(kx pi x) sin(ky pi y)

  Helmholtz -(Laplacian(u) + k^2 u) = f on [0,1]^2
    keys: helmholtz_k1, helmholtz_k4, helmholtz_k9  (k = 1, 4, 9)
    exact: u = sin(kx pi x) sin(ky pi y)  (same ansatz, different RHS)

DD always decomposes on x into K vertical strips.  Interface profiles are
functions of y.

Interface coupling is the SOFT transmission penalty:
each strip matches its neighbour's frozen profile on the shared line x=const.

Run:
    python3.13 dd_parallel_mp_2d.py --prob sin13         --Ks 2
    python3.13 dd_parallel_mp_2d.py --prob sin33         --Ks 2,4
    python3.13 dd_parallel_mp_2d.py --prob helmholtz_k4  --Ks 2,4
    python3.13 dd_parallel_mp_2d.py --prob helmholtz_k9  --Ks 4 --vs-vanilla
"""
import argparse
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

PI = torch.pi

# -------------------------------------------------------------------------
# Problem registry
# -------------------------------------------------------------------------
def _make(kx, ky):
    s = lambda x, y: torch.sin(kx * PI * x) * torch.sin(ky * PI * y)
    f = lambda x, y: -((kx * PI) ** 2 + (ky * PI) ** 2) * s(x, y)
    return dict(u=s, f=f, kx=kx, ky=ky, k2=0.0)

def _make_helmholtz(kx, ky, k):
    """-(∇²u + k²u) = f,  u = sin(kx π x) sin(ky π y)."""
    s   = lambda x, y: torch.sin(kx * PI * x) * torch.sin(ky * PI * y)
    lam = (kx * PI) ** 2 + (ky * PI) ** 2
    f   = lambda x, y: (lam - k ** 2) * s(x, y)
    return dict(u=s, f=f, kx=kx, ky=ky, k2=float(k ** 2))

PROBLEMS = {
    # --- Poisson ---
    "sin11":        _make(1, 1),
    "sin13":        _make(1, 3),
    "sin31":        _make(3, 1),
    "sin33":        _make(3, 3),
    # --- Helmholtz  -(∇²u + k²u) = f ---
    "helmholtz_k1": _make_helmholtz(1, 1, 1),
    "helmholtz_k4": _make_helmholtz(1, 3, 4),
    "helmholtz_k9": _make_helmholtz(3, 3, 9),
}

# -------------------------------------------------------------------------
# Shared hyper-parameters (single set, no per-problem overrides)
# -------------------------------------------------------------------------
N_ITER       = 12
STEPS_PER    = 400
N_COL        = 1200
LR           = 1e-3
IFACE_WEIGHT = 300.0
EPOCHS       = 4500     # vanilla baseline only
N_VANILLA    = 2000     # collocation count for vanilla baseline

# -------------------------------------------------------------------------
# Network: hard zero BC on box walls via φ = x(1-x)y(1-y)
# -------------------------------------------------------------------------
def mlp(layers):
    net = []
    for i in range(len(layers) - 1):
        net.append(nn.Linear(layers[i], layers[i + 1]))
        if i < len(layers) - 2:
            net.append(nn.Tanh())
    net = nn.Sequential(*net)
    for m in net:
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
    return net


class HardBCPINN(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.net = mlp(list(layers))

    def forward(self, x, y):
        return x * (1 - x) * y * (1 - y) * self.net(torch.cat([x, y], 1))


def laplacian(model, x, y):
    u   = model(x, y)
    ux  = torch.autograd.grad(u,  x, torch.ones_like(u),  create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uy  = torch.autograd.grad(u,  y, torch.ones_like(u),  create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    return u, uxx + uyy


def make_geometry(K, overlap):
    edges = np.linspace(0.0, 1.0, K + 1)
    a = np.empty(K); b = np.empty(K)
    for k in range(K):
        a[k] = edges[k] - (overlap / 2 if k > 0     else 0.0)
        b[k] = edges[k + 1] + (overlap / 2 if k < K - 1 else 0.0)
    right_query = [float(a[k + 1]) if k < K - 1 else None for k in range(K)]
    left_query  = [float(b[k - 1]) if k > 0     else None for k in range(K)]
    return a, b, left_query, right_query, edges


# -------------------------------------------------------------------------
# Persistent worker process: one strip, model kept across Schwarz rounds.
# -------------------------------------------------------------------------
def strip_worker(conn, k, cfg):
    torch.set_num_threads(1)
    torch.manual_seed(42 + k)

    prob  = PROBLEMS[cfg["prob"]]
    f     = prob["f"]
    a, b  = cfg["a"], cfg["b"]
    k2    = cfg.get("k2", 0.0)

    model = HardBCPINN(cfg["layers"])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    w    = cfg["iface_weight"]
    N    = cfg["n_col"]
    yif  = torch.linspace(0, 1, cfg["n_if"]).view(-1, 1)

    def line(xval):
        return torch.full_like(yif, xval), yif

    has_L = cfg["left_query"]  is not None
    has_R = cfg["right_query"] is not None

    while True:
        msg = conn.recv()

        if msg[0] == "done":
            lin = torch.linspace(a, b, cfg["n_eval"])
            X, Y = torch.meshgrid(lin, torch.linspace(0, 1, cfg["n_eval"]), indexing="ij")
            with torch.no_grad():
                U = model(X.reshape(-1, 1), Y.reshape(-1, 1)).reshape(cfg["n_eval"], -1)
            conn.send((lin.numpy(), U.numpy()))
            conn.close()
            return

        _, ref_left, ref_right = msg
        rL = torch.tensor(ref_left).view(-1, 1)  if has_L else None
        rR = torch.tensor(ref_right).view(-1, 1) if has_R else None
        xL, yL = line(a) if has_L else (None, None)
        xR, yR = line(b) if has_R else (None, None)

        loss = None
        for _ in range(cfg["steps_per"]):
            p = torch.rand(N, 2)
            x = (a + (b - a) * p[:, 0:1]).requires_grad_(True)
            y = p[:, 1:2].requires_grad_(True)
            u, lap = laplacian(model, x, y)
            res  = lap + k2 * u - f(x, y)
            loss = torch.mean(res ** 2)
            if has_L:
                loss = loss + w * torch.mean((model(xL, yL) - rL) ** 2)
            if has_R:
                loss = loss + w * torch.mean((model(xR, yR) - rR) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            out_R = model(*line(cfg["right_query"])).numpy().ravel() if has_R else np.zeros(cfg["n_if"])
            out_L = model(*line(cfg["left_query"])).numpy().ravel()  if has_L else np.zeros(cfg["n_if"])
        conn.send((out_R, out_L, float(loss.item())))


# -------------------------------------------------------------------------
# True-parallel DD runner
# -------------------------------------------------------------------------
def run_dd_mp_2d(prob="sin13", K=2, overlap=0.3, layers=(2, 64, 64, 1),
                 n_iter=N_ITER, steps_per=STEPS_PER, n_col=N_COL, n_if=200,
                 lr=LR, iface_weight=IFACE_WEIGHT, n_eval=120, verbose=True):

    a, b, left_q, right_q, edges = make_geometry(K, overlap)
    ctx = mp.get_context("spawn")
    parents, procs = [], []

    for k in range(K):
        pc, cc = ctx.Pipe()
        cfg = dict(
            prob=prob, a=float(a[k]), b=float(b[k]), layers=tuple(layers),
            lr=lr, n_col=n_col, n_if=n_if, steps_per=steps_per,
            iface_weight=iface_weight, n_eval=n_eval,
            k2=PROBLEMS[prob].get("k2", 0.0),
            left_query=left_q[k], right_query=right_q[k],
        )
        proc = ctx.Process(target=strip_worker, args=(cc, k, cfg))
        proc.start()
        parents.append(pc); procs.append(proc)

    ref_left  = [np.zeros(n_if) for _ in range(K)]
    ref_right = [np.zeros(n_if) for _ in range(K)]

    t0 = time.perf_counter()
    for it in range(n_iter):
        for k in range(K):
            parents[k].send(("step", ref_left[k], ref_right[k]))
        out_R = [None] * K; out_L = [None] * K
        for k in range(K):
            out_R[k], out_L[k], _ = parents[k].recv()
        for k in range(1, K):
            ref_left[k]  = out_R[k - 1]
        for k in range(K - 1):
            ref_right[k] = out_L[k + 1]
        if verbose:
            print(f"  round {it + 1:2d}/{n_iter}  t={time.perf_counter() - t0:5.2f}s")
    wall_par = time.perf_counter() - t0

    for k in range(K):
        parents[k].send(("done",))
    pieces = [parents[k].recv() for k in range(K)]
    for proc in procs:
        proc.join()

    M   = 160
    lin = np.linspace(0, 1, M)
    Xg, Yg = np.meshgrid(lin, lin, indexing="ij")
    pred = np.zeros((M, M))
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        cols = (Xg[:, 0] >= lo) & (Xg[:, 0] <= hi) if k == K - 1 else \
               (Xg[:, 0] >= lo) & (Xg[:, 0] <  hi)
        xk, Uk = pieces[k]
        yk = np.linspace(0, 1, Uk.shape[1])
        for i in np.where(cols)[0]:
            pred[i, :] = np.interp(lin, yk, Uk[np.argmin(np.abs(xk - lin[i]))])

    ut = PROBLEMS[prob]["u"](
        torch.tensor(Xg, dtype=torch.float32),
        torch.tensor(Yg, dtype=torch.float32),
    ).numpy()
    l2 = np.linalg.norm(pred - ut) / np.linalg.norm(ut)
    return dict(K=K, l2=float(l2), wall_par=wall_par, edges=edges)


# -------------------------------------------------------------------------
# Sequential DD baseline (same work, one process)
# -------------------------------------------------------------------------
def run_dd_seq_2d(prob="sin13", K=2, overlap=0.3, layers=(2, 64, 64, 1),
                  n_iter=N_ITER, steps_per=STEPS_PER, n_col=N_COL, n_if=200,
                  lr=LR, iface_weight=IFACE_WEIGHT):

    torch.set_num_threads(1)
    p  = PROBLEMS[prob]; f = p["f"]; k2 = p.get("k2", 0.0)
    a, b, left_q, right_q, edges = make_geometry(K, overlap)
    yif = torch.linspace(0, 1, n_if).view(-1, 1)
    models = [HardBCPINN(layers) for _ in range(K)]
    for k in range(K):
        torch.manual_seed(42 + k); models[k] = HardBCPINN(layers)
    opts      = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    ref_left  = [torch.zeros(n_if, 1) for _ in range(K)]
    ref_right = [torch.zeros(n_if, 1) for _ in range(K)]

    def line(xval):
        return torch.full_like(yif, xval), yif

    t0 = time.perf_counter()
    for _ in range(n_iter):
        outR = [None] * K; outL = [None] * K
        for k in range(K):
            hasL, hasR = left_q[k] is not None, right_q[k] is not None
            for _ in range(steps_per):
                pr = torch.rand(n_col, 2)
                x  = (a[k] + (b[k] - a[k]) * pr[:, 0:1]).requires_grad_(True)
                y  = pr[:, 1:2].requires_grad_(True)
                u_s, lap = laplacian(models[k], x, y)
                loss = torch.mean((lap + k2 * u_s - f(x, y)) ** 2)
                if hasL:
                    xL, yL = line(a[k])
                    loss = loss + iface_weight * torch.mean((models[k](xL, yL) - ref_left[k]) ** 2)
                if hasR:
                    xR, yR = line(b[k])
                    loss = loss + iface_weight * torch.mean((models[k](xR, yR) - ref_right[k]) ** 2)
                opts[k].zero_grad(); loss.backward(); opts[k].step()
            with torch.no_grad():
                outR[k] = models[k](*line(right_q[k])) if hasR else None
                outL[k] = models[k](*line(left_q[k]))  if hasL else None
        for k in range(1, K):
            ref_left[k]  = outR[k - 1].detach()
        for k in range(K - 1):
            ref_right[k] = outL[k + 1].detach()
    return time.perf_counter() - t0


# -------------------------------------------------------------------------
# Vanilla global PINN baseline
# -------------------------------------------------------------------------
def run_vanilla_2d(prob="sin13", layers=(2, 64, 64, 64, 1),
                   epochs=EPOCHS, N=N_VANILLA, lr=LR):
    """One PINN on the whole square — unpinned on threads so it gets the full machine."""
    torch.manual_seed(42)
    p  = PROBLEMS[prob]; f = p["f"]; k2 = p.get("k2", 0.0)
    m  = HardBCPINN(layers)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(epochs):
        pr = torch.rand(N, 2)
        x  = pr[:, 0:1].requires_grad_(True)
        y  = pr[:, 1:2].requires_grad_(True)
        u_v, lap = laplacian(m, x, y)
        loss = torch.mean((lap + k2 * u_v - f(x, y)) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.perf_counter() - t0
    M   = 160
    lin = torch.linspace(0, 1, M)
    X, Y = torch.meshgrid(lin, lin, indexing="ij")
    with torch.no_grad():
        up = m(X.reshape(-1, 1), Y.reshape(-1, 1)).reshape(M, M).numpy()
    ut = p["u"](X, Y).numpy()
    l2 = np.linalg.norm(up - ut) / np.linalg.norm(ut)
    return dict(wall=wall, l2=float(l2))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob",       default="sin13", choices=list(PROBLEMS))
    ap.add_argument("--Ks",         default="2")
    ap.add_argument("--iters",      type=int, default=N_ITER)
    ap.add_argument("--steps",      type=int, default=STEPS_PER)
    ap.add_argument("--vs-vanilla", action="store_true",
                    help="head-to-head: global PINN vs true-parallel DD")
    args = ap.parse_args()
    print(f"problem={args.prob}  cores={mp.cpu_count()}  "
          f"iters={args.iters}  steps/round={args.steps}\n")

    if args.vs_vanilla:
        v = run_vanilla_2d(args.prob)
        print(f"{'method':>22} {'L2':>10} {'wall (s)':>10} {'vs vanilla':>11}")
        print(f"{'vanilla PINN (global)':>22} {v['l2']:>10.2e} {v['wall']:>10.2f} {'1.00x':>11}")
        for K in [int(x) for x in args.Ks.split(",")]:
            r   = run_dd_mp_2d(args.prob, K=K, n_iter=args.iters,
                               steps_per=args.steps, verbose=False)
            tag = f"DD true-parallel K={K}"
            print(f"{tag:>22} {r['l2']:>10.2e} {r['wall_par']:>10.2f} "
                  f"{v['wall'] / r['wall_par']:>10.2f}x")
        return

    print(f"{'K':>3} {'L2':>10} {'seq (s)':>9} {'par (s)':>9} {'speedup':>8}")
    for K in [int(x) for x in args.Ks.split(",")]:
        seq = run_dd_seq_2d(args.prob, K=K, n_iter=args.iters, steps_per=args.steps)
        r   = run_dd_mp_2d(args.prob,  K=K, n_iter=args.iters, steps_per=args.steps, verbose=False)
        print(f"{K:>3} {r['l2']:>10.2e} {seq:>9.2f} {r['wall_par']:>9.2f} "
              f"{seq / r['wall_par']:>7.2f}x")


if __name__ == "__main__":
    main()