"""
run_all_problems.py
-------------------
Runs every problem in PROBLEMS (Poisson + Helmholtz) for K=2
through vanilla, sequential DD, and true-parallel DD.

All problems use the same flat hyper-parameters defined in
dd_parallel_mp_2d.py (N_ITER, STEPS_PER, N_COL, LR, IFACE_WEIGHT).
Global --iters/--steps/--lr flags override those defaults if supplied.

Usage:
    python3 run_all_problems.py                         # flat defaults
    python3 run_all_problems.py --Ks 2,4               # multiple K values
    python3 run_all_problems.py --iters 20 --steps 600 # force custom values
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_parallel_mp_2d import (
    PROBLEMS,
    N_ITER, STEPS_PER, N_COL, LR, IFACE_WEIGHT, EPOCHS, N_VANILLA,
    run_dd_seq_2d, run_dd_mp_2d, run_vanilla_2d,
)

POISSON_KEYS   = ["sin11", "sin13", "sin31", "sin33"]
HELMHOLTZ_KEYS = ["helmholtz_k1", "helmholtz_k4", "helmholtz_k9"]


def run_all(Ks, n_iter, steps_per, n_col, lr, iface_weight, epochs, n_vanilla):
    rows = []

    for family, keys in [("Poisson", POISSON_KEYS), ("Helmholtz", HELMHOLTZ_KEYS)]:
        for prob in keys:
            print(f"  [{family}] {prob}  vanilla  "
                  f"(epochs={epochs} lr={lr}) ...", flush=True)
            van = run_vanilla_2d(prob, epochs=epochs, N=n_vanilla, lr=lr)

            for K in Ks:
                print(f"  [{family}] {prob}  K={K}  seq+par  "
                      f"(iters={n_iter} steps={steps_per} "
                      f"iw={iface_weight} lr={lr}) ...", flush=True)
                seq = run_dd_seq_2d(prob, K=K,
                                    n_iter=n_iter, steps_per=steps_per,
                                    n_col=n_col,   lr=lr,
                                    iface_weight=iface_weight)
                res = run_dd_mp_2d(prob, K=K,
                                   n_iter=n_iter, steps_per=steps_per,
                                   n_col=n_col,   lr=lr,
                                   iface_weight=iface_weight,
                                   verbose=False)
                rows.append(dict(
                    family   = family,
                    prob     = prob,
                    K        = K,
                    l2_van   = van["l2"],
                    l2_par   = res["l2"],
                    wall_van = van["wall"],
                    seq      = seq,
                    par      = res["wall_par"],
                    dd_seq   = seq / res["wall_par"],
                    dd_van   = van["wall"] / res["wall_par"],
                ))

    # ---------------------------------------------------------------- print
    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  {'problem':<18} {'K':>2}  {'L2-van':>9}  {'L2-par':>9}  "
          f"{'van (s)':>8}  {'seq (s)':>8}  {'par (s)':>8}  "
          f"{'DD/seq':>7}  {'DD/van':>7}")
    print(sep)

    cur_family = None
    for r in rows:
        if r["family"] != cur_family:
            cur_family = r["family"]
            print(f"\n  ── {cur_family} ──")
        print(f"  {r['prob']:<18} {r['K']:>2}  {r['l2_van']:>9.2e}  {r['l2_par']:>9.2e}  "
              f"{r['wall_van']:>8.2f}  {r['seq']:>8.2f}  {r['par']:>8.2f}  "
              f"{r['dd_seq']:>6.2f}x  {r['dd_van']:>6.2f}x")

    print(f"\n{sep}")
    print("  L2-van  = vanilla PINN relative L2 error")
    print("  L2-par  = parallel DD relative L2 error")
    print("  DD/seq  = parallel DD speedup over sequential DD")
    print("  DD/van  = parallel DD speedup over vanilla  "
          "(vanilla unpinned, DD pins 1 thread/strip)")
    print("  All problems use the same flat hyper-parameters.")
    print(f"{sep}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks",    default="2")
    ap.add_argument("--iters", type=int,   default=N_ITER,       help="Schwarz iterations")
    ap.add_argument("--steps", type=int,   default=STEPS_PER,    help="Adam steps per round")
    ap.add_argument("--ncol",  type=int,   default=N_COL,        help="collocation points per strip")
    ap.add_argument("--lr",    type=float, default=LR,           help="Adam learning rate")
    ap.add_argument("--iw",    type=float, default=IFACE_WEIGHT, help="interface penalty weight")
    ap.add_argument("--epochs",type=int,   default=EPOCHS,       help="vanilla training epochs")
    args = ap.parse_args()

    Ks = [int(k) for k in args.Ks.split(",")]

    import torch.multiprocessing as mp
    print(f"cores={mp.cpu_count()}  Ks={Ks}")
    print(f"  iters={args.iters}  steps={args.steps}  ncol={args.ncol}  "
          f"lr={args.lr}  iw={args.iw}  epochs={args.epochs}\n")

    run_all(
        Ks           = Ks,
        n_iter       = args.iters,
        steps_per    = args.steps,
        n_col        = args.ncol,
        lr           = args.lr,
        iface_weight = args.iw,
        epochs       = args.epochs,
        n_vanilla    = N_VANILLA,
    )


if __name__ == "__main__":
    main()