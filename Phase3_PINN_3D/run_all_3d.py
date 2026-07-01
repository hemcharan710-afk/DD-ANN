"""
run_all_3d.py
-------------
Runs the 3D problem set (Poisson + LPB + COSMO) for K=2 through vanilla PINN,
sequential DD, and true-parallel DD.

All problems use the same flat hyper-parameters defined in dd_parallel_mp_3d.py
(N_ITER, STEPS_PER, N_COL, LR, IFACE_WEIGHT, EPOCHS).  Global --iters/--steps/--lr
flags override those defaults if supplied.

By default one representative problem per family is run (the "3 examples":
3D Poisson, LPB, COSMO).  Pass --all to sweep every registered problem.

Usage:
    python3.13 run_all_3d.py                 # 3 representatives, K=2
    python3.13 run_all_3d.py --all           # every problem
    python3.13 run_all_3d.py --Ks 2,4        # multiple K values
    python3.13 run_all_3d.py --iters 12 --steps 400
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_parallel_mp_3d import (
    PROBLEMS,
    N_ITER, STEPS_PER, N_COL, LR, IFACE_WEIGHT, EPOCHS, N_VANILLA,
    run_dd_seq_3d, run_dd_mp_3d, run_vanilla_3d,
)

# family -> ordered problem keys
FAMILIES = {
    "Poisson": ["pois_111", "pois_112", "pois_222"],
    "LPB":     ["lpb_k1",   "lpb_k3",   "lpb_k8"],
    "COSMO":   ["cosmo_lo", "cosmo_mid", "cosmo_hi"],
}
# one representative per family (the "3 examples")
REPRESENTATIVE = {"Poisson": "pois_111", "LPB": "lpb_k3", "COSMO": "cosmo_mid"}


def run_all(keys_by_family, Ks, n_iter, steps_per, n_col, lr,
            iface_weight, epochs, n_vanilla):
    rows = []
    for family, keys in keys_by_family.items():
        for prob in keys:
            print(f"  [{family}] {prob}  vanilla  "
                  f"(epochs={epochs} lr={lr}) ...", flush=True)
            van = run_vanilla_3d(prob, epochs=epochs, N=n_vanilla, lr=lr)

            for K in Ks:
                print(f"  [{family}] {prob}  K={K}  seq+par  "
                      f"(iters={n_iter} steps={steps_per} "
                      f"iw={iface_weight} lr={lr}) ...", flush=True)
                seq = run_dd_seq_3d(prob, K=K, n_iter=n_iter, steps_per=steps_per,
                                    n_col=n_col, lr=lr, iface_weight=iface_weight)
                res = run_dd_mp_3d(prob, K=K, n_iter=n_iter, steps_per=steps_per,
                                   n_col=n_col, lr=lr, iface_weight=iface_weight,
                                   verbose=False)
                rows.append(dict(
                    family=family, prob=prob, K=K,
                    l2_van=van["l2"], l2_par=res["l2"],
                    wall_van=van["wall"], seq=seq, par=res["wall_par"],
                    dd_seq=seq / res["wall_par"],
                    dd_van=van["wall"] / res["wall_par"],
                ))

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
            print(f"\n  -- {cur_family} --")
        print(f"  {r['prob']:<18} {r['K']:>2}  {r['l2_van']:>9.2e}  {r['l2_par']:>9.2e}  "
              f"{r['wall_van']:>8.2f}  {r['seq']:>8.2f}  {r['par']:>8.2f}  "
              f"{r['dd_seq']:>6.2f}x  {r['dd_van']:>6.2f}x")

    print(f"\n{sep}")
    print("  L2-van  = vanilla PINN relative L2 error")
    print("  L2-par  = parallel DD relative L2 error")
    print("  DD/seq  = parallel DD speedup over sequential DD (genuine concurrency)")
    print("  DD/van  = parallel DD speedup over vanilla global PINN")
    print("  All problems use the same flat hyper-parameters.")
    print(f"{sep}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all",   action="store_true", help="sweep every problem, not just reps")
    ap.add_argument("--Ks",    default="2")
    ap.add_argument("--iters", type=int,   default=N_ITER)
    ap.add_argument("--steps", type=int,   default=STEPS_PER)
    ap.add_argument("--ncol",  type=int,   default=N_COL)
    ap.add_argument("--lr",    type=float, default=LR)
    ap.add_argument("--iw",    type=float, default=IFACE_WEIGHT)
    ap.add_argument("--epochs",type=int,   default=EPOCHS)
    args = ap.parse_args()

    Ks = [int(k) for k in args.Ks.split(",")]
    if args.all:
        keys_by_family = FAMILIES
    else:
        keys_by_family = {fam: [REPRESENTATIVE[fam]] for fam in FAMILIES}

    import torch.multiprocessing as mp
    print(f"cores={mp.cpu_count()}  Ks={Ks}  mode={'all' if args.all else 'representatives'}")
    print(f"  iters={args.iters}  steps={args.steps}  ncol={args.ncol}  "
          f"lr={args.lr}  iw={args.iw}  epochs={args.epochs}\n")

    run_all(keys_by_family, Ks=Ks, n_iter=args.iters, steps_per=args.steps,
            n_col=args.ncol, lr=args.lr, iface_weight=args.iw,
            epochs=args.epochs, n_vanilla=N_VANILLA)


if __name__ == "__main__":
    main()
