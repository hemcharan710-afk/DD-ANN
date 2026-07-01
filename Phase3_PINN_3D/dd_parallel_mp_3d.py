"""
Real multiprocess (true-parallel) overlapping-Schwarz DD-PINN in 3D.

Three problem families, all on the unit cube [0,1]^3 with homogeneous
Dirichlet data, all with a manufactured exact solution so the relative L2
error is always well-defined.  Every operator is written in coercive elliptic
form  -div(eps grad u) + c u = f  so the residual is sign-consistent and the
PINN actually converges (accuracy is a real number here, not a placeholder).

  Poisson    -Laplacian(u) = f                       (uniform-dielectric electrostatics)
    keys: pois_111, pois_112, pois_222
    exact: u = sin(kx pi x) sin(ky pi y) sin(kz pi z)

  LPB        -Laplacian(u) + kappa^2 u = f           (linearized Poisson-Boltzmann,
    keys: lpb_k1, lpb_k3, lpb_k8                      Debye screening, kappa = inverse
    exact: same sinusoidal ansatz                     Debye length)

  COSMO      -div( eps(r) grad u ) = f                (polarizable-continuum / COSMO
    keys: cosmo_lo, cosmo_mid, cosmo_hi               electrostatics with a spatially
    exact: same sinusoidal ansatz                     varying dielectric eps(r))

For COSMO the dielectric is affine,  eps(r) = 1 + ax x + ay y + az z  (> 0 on the
cube), so grad(eps) is constant and the forcing is fully analytic:
    div(eps grad u) = eps * Laplacian(u) + grad(eps) . grad(u)
    f = -( eps * Lap(u) + grad(eps) . grad(u) ).

DD always decomposes on x into K slabs.  Interface conditions live on the shared
plane x = const and are functions of (y, z).  Coupling is the SOFT transmission
penalty (same mechanism as the 2D code): each slab matches its neighbour's frozen
interface profile on the shared plane.

Run:
    python3.13 dd_parallel_mp_3d.py --prob pois_111 --Ks 2
    python3.13 dd_parallel_mp_3d.py --prob lpb_k3   --Ks 2 --vs-vanilla
    python3.13 dd_parallel_mp_3d.py --prob cosmo_mid --vs-vanilla
"""
import argparse
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn

PI = torch.pi


# -------------------------------------------------------------------------
# Problem registry.  Keyed by string so it survives 'spawn' pickling.
#
# pde_type selects the residual form:
#   "poisson" : -Lap(u)               = f      residual = -lap - f
#   "lpb"     : -Lap(u) + kappa^2 u   = f      residual = -lap + kappa2 u - f
#   "cosmo"   : -div(eps grad u)      = f      residual = -(eps*lap + grad_eps.grad_u) - f
#
# Every exact solution is a product of sines that vanishes on the cube walls,
# so the hard-BC ansatz x(1-x)y(1-y)z(1-z) N(x,y,z) enforces u|_bdry = 0 exactly.
# -------------------------------------------------------------------------
def _sines(kx, ky, kz):
    return lambda x, y, z: (torch.sin(kx * PI * x)
                            * torch.sin(ky * PI * y)
                            * torch.sin(kz * PI * z))


def _poisson(kx, ky, kz):
    """-Lap(u) = f,  u = sin(kx pi x) sin(ky pi y) sin(kz pi z)."""
    s   = _sines(kx, ky, kz)
    lam = (kx * PI) ** 2 + (ky * PI) ** 2 + (kz * PI) ** 2
    return dict(pde_type="poisson", kx=kx, ky=ky, kz=kz,
                u=s, f=lambda x, y, z: lam * s(x, y, z))


def _lpb(kx, ky, kz, kappa):
    """-Lap(u) + kappa^2 u = f,  screened Poisson (linearized PB)."""
    s      = _sines(kx, ky, kz)
    lam    = (kx * PI) ** 2 + (ky * PI) ** 2 + (kz * PI) ** 2
    kappa2 = float(kappa) ** 2
    return dict(pde_type="lpb", kx=kx, ky=ky, kz=kz, kappa2=kappa2,
                u=s, f=lambda x, y, z: (lam + kappa2) * s(x, y, z))


def _cosmo(kx, ky, kz, ax, ay, az):
    """-div(eps grad u) = f with affine dielectric eps = 1 + ax x + ay y + az z."""
    s   = _sines(kx, ky, kz)
    lam = (kx * PI) ** 2 + (ky * PI) ** 2 + (kz * PI) ** 2

    def f(x, y, z):
        eps  = 1.0 + ax * x + ay * y + az * z
        sx   = torch.sin(kx * PI * x); cx = torch.cos(kx * PI * x)
        sy   = torch.sin(ky * PI * y); cy = torch.cos(ky * PI * y)
        sz   = torch.sin(kz * PI * z); cz = torch.cos(kz * PI * z)
        lap  = -lam * (sx * sy * sz)
        grad_eps_dot_grad_u = (ax * (kx * PI) * cx * sy * sz
                               + ay * (ky * PI) * sx * cy * sz
                               + az * (kz * PI) * sx * sy * cz)
        return -(eps * lap + grad_eps_dot_grad_u)

    return dict(pde_type="cosmo", kx=kx, ky=ky, kz=kz,
                eps_coeffs=(float(ax), float(ay), float(az)), u=s, f=f)


PROBLEMS = {
    # --- Poisson  -Lap(u) = f ---
    "pois_111": _poisson(1, 1, 1),   # smooth
    "pois_112": _poisson(1, 1, 2),   # mild anisotropy (fast in z)
    "pois_222": _poisson(2, 2, 2),   # higher frequency all axes
    # --- Linearized Poisson-Boltzmann  -Lap(u) + kappa^2 u = f ---
    "lpb_k1":   _lpb(1, 1, 1, 1.0),  # weak screening
    "lpb_k3":   _lpb(1, 1, 2, 3.0),  # moderate Debye screening
    "lpb_k8":   _lpb(2, 2, 2, 8.0),  # strong screening
    # --- COSMO / PCM  -div(eps grad u) = f, eps affine ---
    "cosmo_lo":  _cosmo(1, 1, 1, 0.5, 0.3, 0.2),  # gentle dielectric gradient
    "cosmo_mid": _cosmo(1, 1, 2, 1.0, 0.5, 0.5),  # moderate contrast
    "cosmo_hi":  _cosmo(2, 2, 2, 2.0, 1.0, 1.0),  # strong contrast + high freq
}


# -------------------------------------------------------------------------
# Shared hyper-parameters (single flat set, no per-problem overrides)
# -------------------------------------------------------------------------
N_ITER       = 10      # Schwarz rounds
STEPS_PER    = 300     # Adam steps per slab per round
N_COL        = 2000    # collocation points per slab per step
LR           = 1e-3
IFACE_WEIGHT = 300.0
N_IF         = 16      # interface grid is N_IF x N_IF on the (y,z) plane
EPOCHS       = 3000    # vanilla baseline only
N_VANILLA    = 3000    # collocation count for vanilla baseline
N_EVAL       = 40      # per-axis eval grid inside each slab


# -------------------------------------------------------------------------
# Network: hard zero BC on cube walls via phi = x(1-x)y(1-y)z(1-z)
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

    def forward(self, x, y, z):
        phi = x * (1 - x) * y * (1 - y) * z * (1 - z)
        return phi * self.net(torch.cat([x, y, z], 1))


def _laplacian(model, x, y, z):
    u   = model(x, y, z)
    ux  = torch.autograd.grad(u,  x, torch.ones_like(u),  create_graph=True)[0]
    uxx = torch.autograd.grad(ux, x, torch.ones_like(ux), create_graph=True)[0]
    uy  = torch.autograd.grad(u,  y, torch.ones_like(u),  create_graph=True)[0]
    uyy = torch.autograd.grad(uy, y, torch.ones_like(uy), create_graph=True)[0]
    uz  = torch.autograd.grad(u,  z, torch.ones_like(u),  create_graph=True)[0]
    uzz = torch.autograd.grad(uz, z, torch.ones_like(uz), create_graph=True)[0]
    return u, ux, uy, uz, uxx + uyy + uzz


def pde_residual(model, x, y, z, prob):
    """Residual tensor for any supported 3D pde_type."""
    u, ux, uy, uz, lap = _laplacian(model, x, y, z)
    pde = prob["pde_type"]
    f   = prob["f"]
    if pde == "poisson":
        return -lap - f(x, y, z)
    if pde == "lpb":
        return -lap + prob["kappa2"] * u - f(x, y, z)
    if pde == "cosmo":
        ax, ay, az = prob["eps_coeffs"]
        eps = 1.0 + ax * x + ay * y + az * z
        grad_eps_dot_grad_u = ax * ux + ay * uy + az * uz
        return -(eps * lap + grad_eps_dot_grad_u) - f(x, y, z)
    raise ValueError(f"Unknown pde_type: {pde}")


# -------------------------------------------------------------------------
# Geometry: split [0,1] in x into K overlapping slabs.
# -------------------------------------------------------------------------
def make_geometry(K, overlap):
    edges = np.linspace(0.0, 1.0, K + 1)
    a = np.empty(K); b = np.empty(K)
    for k in range(K):
        a[k] = edges[k] - (overlap / 2 if k > 0     else 0.0)
        b[k] = edges[k + 1] + (overlap / 2 if k < K - 1 else 0.0)
    right_query = [float(a[k + 1]) if k < K - 1 else None for k in range(K)]
    left_query  = [float(b[k - 1]) if k > 0     else None for k in range(K)]
    return a, b, left_query, right_query, edges


def _iface_grid(n_if):
    """Return flattened (y, z) column tensors for an n_if x n_if interface plane."""
    lin = torch.linspace(0, 1, n_if)
    Y, Z = torch.meshgrid(lin, lin, indexing="ij")
    return Y.reshape(-1, 1), Z.reshape(-1, 1)


# -------------------------------------------------------------------------
# Persistent worker: one slab, one process, model kept across Schwarz rounds.
# Protocol per round:
#   parent -> child : ("step", left_profile, right_profile)   # neighbour planes
#   child  -> parent: (out_right_query, out_left_query, loss)
#   parent -> child : ("done",) then child returns its eval cube
# -------------------------------------------------------------------------
def slab_worker(conn, k, cfg):
    torch.set_num_threads(1)
    torch.manual_seed(42 + k)

    prob = PROBLEMS[cfg["prob"]]
    a, b = cfg["a"], cfg["b"]

    model = HardBCPINN(cfg["layers"])
    opt   = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    w   = cfg["iface_weight"]
    N   = cfg["n_col"]
    yif, zif = _iface_grid(cfg["n_if"])

    def plane(xval):
        return torch.full_like(yif, xval), yif, zif

    has_L = cfg["left_query"]  is not None
    has_R = cfg["right_query"] is not None

    while True:
        msg = conn.recv()

        if msg[0] == "done":
            m = cfg["n_eval"]
            lin_x = torch.linspace(a, b, m)
            lin   = torch.linspace(0, 1, m)
            X, Y, Z = torch.meshgrid(lin_x, lin, lin, indexing="ij")
            with torch.no_grad():
                U = model(X.reshape(-1, 1), Y.reshape(-1, 1),
                          Z.reshape(-1, 1)).reshape(m, m, m)
            conn.send((lin_x.numpy(), U.numpy()))
            conn.close()
            return

        _, ref_left, ref_right = msg
        rL = torch.tensor(ref_left).view(-1, 1)  if has_L else None
        rR = torch.tensor(ref_right).view(-1, 1) if has_R else None
        planeL = plane(a) if has_L else None
        planeR = plane(b) if has_R else None

        loss = None
        for _ in range(cfg["steps_per"]):
            p = torch.rand(N, 3)
            x = (a + (b - a) * p[:, 0:1]).requires_grad_(True)
            y = p[:, 1:2].requires_grad_(True)
            z = p[:, 2:3].requires_grad_(True)
            res  = pde_residual(model, x, y, z, prob)
            loss = torch.mean(res ** 2)
            if has_L:
                loss = loss + w * torch.mean((model(*planeL) - rL) ** 2)
            if has_R:
                loss = loss + w * torch.mean((model(*planeR) - rR) ** 2)
            opt.zero_grad(); loss.backward(); opt.step()

        with torch.no_grad():
            out_R = (model(*plane(cfg["right_query"])).numpy().ravel()
                     if has_R else np.zeros(cfg["n_if"] ** 2))
            out_L = (model(*plane(cfg["left_query"])).numpy().ravel()
                     if has_L else np.zeros(cfg["n_if"] ** 2))
        conn.send((out_R, out_L, float(loss.item())))


# -------------------------------------------------------------------------
# True-parallel DD runner
# -------------------------------------------------------------------------
def run_dd_mp_3d(prob="pois_111", K=2, overlap=0.3, layers=(3, 64, 64, 1),
                 n_iter=N_ITER, steps_per=STEPS_PER, n_col=N_COL, n_if=N_IF,
                 lr=LR, iface_weight=IFACE_WEIGHT, n_eval=N_EVAL, verbose=True):

    a, b, left_q, right_q, edges = make_geometry(K, overlap)
    ctx = mp.get_context("spawn")
    parents, procs = [], []

    for k in range(K):
        pc, cc = ctx.Pipe()
        cfg = dict(
            prob=prob, a=float(a[k]), b=float(b[k]), layers=tuple(layers),
            lr=lr, n_col=n_col, n_if=n_if, steps_per=steps_per,
            iface_weight=iface_weight, n_eval=n_eval,
            left_query=left_q[k], right_query=right_q[k],
        )
        proc = ctx.Process(target=slab_worker, args=(cc, k, cfg))
        proc.start()
        parents.append(pc); procs.append(proc)

    npl = n_if ** 2
    ref_left  = [np.zeros(npl) for _ in range(K)]
    ref_right = [np.zeros(npl) for _ in range(K)]

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

    # stitch: assign each global x-slice to the owning slab at nominal edges
    M   = 48
    lin = np.linspace(0, 1, M)
    Xg, Yg, Zg = np.meshgrid(lin, lin, lin, indexing="ij")
    pred = np.zeros((M, M, M))
    for k in range(K):
        lo, hi = edges[k], edges[k + 1]
        cols = ((Xg[:, 0, 0] >= lo) & (Xg[:, 0, 0] <= hi)) if k == K - 1 else \
               ((Xg[:, 0, 0] >= lo) & (Xg[:, 0, 0] <  hi))
        xk, Uk = pieces[k]            # xk: (m,)   Uk: (m, m, m) on [a,b]x[0,1]^2
        m  = Uk.shape[1]
        yk = np.linspace(0, 1, m)
        for i in np.where(cols)[0]:
            ix = int(np.argmin(np.abs(xk - lin[i])))
            plane = Uk[ix]            # (m, m) over (y, z)
            # bilinear interp of the slab plane onto the global (y,z) grid
            iy = np.interp(lin, yk, np.arange(m))
            iz = np.interp(lin, yk, np.arange(m))
            iy0 = np.clip(np.floor(iy).astype(int), 0, m - 1)
            iz0 = np.clip(np.floor(iz).astype(int), 0, m - 1)
            iy1 = np.clip(iy0 + 1, 0, m - 1); iz1 = np.clip(iz0 + 1, 0, m - 1)
            wy  = (iy - iy0)[:, None];        wz  = (iz - iz0)[None, :]
            p00 = plane[np.ix_(iy0, iz0)]; p01 = plane[np.ix_(iy0, iz1)]
            p10 = plane[np.ix_(iy1, iz0)]; p11 = plane[np.ix_(iy1, iz1)]
            pred[i] = ((1 - wy) * (1 - wz) * p00 + (1 - wy) * wz * p01
                       + wy * (1 - wz) * p10 + wy * wz * p11)

    ut = PROBLEMS[prob]["u"](
        torch.tensor(Xg, dtype=torch.float32),
        torch.tensor(Yg, dtype=torch.float32),
        torch.tensor(Zg, dtype=torch.float32),
    ).numpy()
    l2 = np.linalg.norm(pred - ut) / np.linalg.norm(ut)
    return dict(K=K, l2=float(l2), wall_par=wall_par, edges=edges)


# -------------------------------------------------------------------------
# Sequential DD baseline (same work, one process) — honest speed-up denominator
# -------------------------------------------------------------------------
def run_dd_seq_3d(prob="pois_111", K=2, overlap=0.3, layers=(3, 64, 64, 1),
                  n_iter=N_ITER, steps_per=STEPS_PER, n_col=N_COL, n_if=N_IF,
                  lr=LR, iface_weight=IFACE_WEIGHT):

    torch.set_num_threads(1)
    p = PROBLEMS[prob]
    a, b, left_q, right_q, edges = make_geometry(K, overlap)
    yif, zif = _iface_grid(n_if)

    models = []
    for k in range(K):
        torch.manual_seed(42 + k)
        models.append(HardBCPINN(layers))
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]

    npl = n_if ** 2
    ref_left  = [torch.zeros(npl, 1) for _ in range(K)]
    ref_right = [torch.zeros(npl, 1) for _ in range(K)]

    def plane(xval):
        return torch.full_like(yif, xval), yif, zif

    t0 = time.perf_counter()
    for _ in range(n_iter):
        outR = [None] * K; outL = [None] * K
        for k in range(K):
            hasL, hasR = left_q[k] is not None, right_q[k] is not None
            for _ in range(steps_per):
                pr = torch.rand(n_col, 3)
                x = (a[k] + (b[k] - a[k]) * pr[:, 0:1]).requires_grad_(True)
                y = pr[:, 1:2].requires_grad_(True)
                z = pr[:, 2:3].requires_grad_(True)
                loss = torch.mean(pde_residual(models[k], x, y, z, p) ** 2)
                if hasL:
                    loss = loss + iface_weight * torch.mean(
                        (models[k](*plane(a[k])) - ref_left[k]) ** 2)
                if hasR:
                    loss = loss + iface_weight * torch.mean(
                        (models[k](*plane(b[k])) - ref_right[k]) ** 2)
                opts[k].zero_grad(); loss.backward(); opts[k].step()
            with torch.no_grad():
                outR[k] = models[k](*plane(right_q[k])) if hasR else None
                outL[k] = models[k](*plane(left_q[k]))  if hasL else None
        for k in range(1, K):
            ref_left[k]  = outR[k - 1].detach()
        for k in range(K - 1):
            ref_right[k] = outL[k + 1].detach()
    return time.perf_counter() - t0


# -------------------------------------------------------------------------
# Vanilla global PINN baseline (one network on the whole cube, threads unpinned)
# -------------------------------------------------------------------------
def run_vanilla_3d(prob="pois_111", layers=(3, 64, 64, 64, 1),
                   epochs=EPOCHS, N=N_VANILLA, lr=LR):
    torch.manual_seed(42)
    p = PROBLEMS[prob]
    m = HardBCPINN(layers)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    t0 = time.perf_counter()
    for _ in range(epochs):
        pr = torch.rand(N, 3)
        x = pr[:, 0:1].requires_grad_(True)
        y = pr[:, 1:2].requires_grad_(True)
        z = pr[:, 2:3].requires_grad_(True)
        loss = torch.mean(pde_residual(m, x, y, z, p) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    wall = time.perf_counter() - t0

    M   = 48
    lin = torch.linspace(0, 1, M)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    with torch.no_grad():
        up = m(X.reshape(-1, 1), Y.reshape(-1, 1),
               Z.reshape(-1, 1)).reshape(M, M, M).numpy()
    ut = p["u"](X, Y, Z).numpy()
    l2 = np.linalg.norm(up - ut) / np.linalg.norm(ut)
    return dict(wall=wall, l2=float(l2))


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prob",       default="pois_111", choices=list(PROBLEMS))
    ap.add_argument("--Ks",         default="2")
    ap.add_argument("--iters",      type=int, default=N_ITER)
    ap.add_argument("--steps",      type=int, default=STEPS_PER)
    ap.add_argument("--vs-vanilla", action="store_true",
                    help="head-to-head: global PINN vs true-parallel DD")
    args = ap.parse_args()
    print(f"problem={args.prob}  cores={mp.cpu_count()}  "
          f"iters={args.iters}  steps/round={args.steps}\n")

    if args.vs_vanilla:
        v = run_vanilla_3d(args.prob)
        print(f"{'method':>22} {'L2':>10} {'wall (s)':>10} {'vs vanilla':>11}")
        print(f"{'vanilla PINN (global)':>22} {v['l2']:>10.2e} {v['wall']:>10.2f} {'1.00x':>11}")
        for K in [int(x) for x in args.Ks.split(",")]:
            r   = run_dd_mp_3d(args.prob, K=K, n_iter=args.iters,
                               steps_per=args.steps, verbose=False)
            tag = f"DD true-parallel K={K}"
            print(f"{tag:>22} {r['l2']:>10.2e} {r['wall_par']:>10.2f} "
                  f"{v['wall'] / r['wall_par']:>10.2f}x")
        return

    print(f"{'K':>3} {'L2':>10} {'seq (s)':>9} {'par (s)':>9} {'speedup':>8}")
    for K in [int(x) for x in args.Ks.split(",")]:
        seq = run_dd_seq_3d(args.prob, K=K, n_iter=args.iters, steps_per=args.steps)
        r   = run_dd_mp_3d(args.prob,  K=K, n_iter=args.iters, steps_per=args.steps, verbose=False)
        print(f"{K:>3} {r['l2']:>10.2e} {seq:>9.2f} {r['wall_par']:>9.2f} "
              f"{seq / r['wall_par']:>7.2f}x")


if __name__ == "__main__":
    main()
