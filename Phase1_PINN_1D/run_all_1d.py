"""
run_all_1d.py
-------------
Runs every problem in dd_parallel_mp.py's PROBLEMS registry through:
  - vanilla PINN       (global, threads unpinned)
  - sequential DD      (single process, 1 thread)
  - true-parallel DD   (K processes, 1 thread each)

Prints one consolidated table with L2 error and wall-clock times.

Usage:
    python3 run_all_1d.py
    python3 run_all_1d.py --Ks 2,4 --iters 15 --steps 400 --width 32
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dd_parallel_mp import PROBLEMS, run_vanilla, run_dd_seq, run_dd_mp

# Problem families for grouped display
FAMILIES = {
    "Poisson":            ["sin1", "sin4", "exp"],
    "Reaction-diffusion": ["rd_lam1", "rd_lam10"],
    "Advection-diffusion":["ad_e1e-1", "ad_e1e-2"],
}


def run_all(Ks, iters, steps, width):
    rows = []

    for family, keys in FAMILIES.items():
        for prob in keys:
            print(f"  [{family}] {prob}  vanilla ...", flush=True)
            # Vanilla uses its capacity-matched default width (≈ two DD subdomains
            # of `width`), NOT `width` itself — otherwise the global net is
            # undersized and the DD-vs-vanilla timing is unfair.
            van = run_vanilla(prob)

            for K in Ks:
                print(f"  [{family}] {prob}  K={K}  seq+par ...", flush=True)
                seq = run_dd_seq(prob, K=K, n_iter=iters,
                                 steps_per=steps, width=width)
                par = run_dd_mp(prob, K=K, n_iter=iters,
                                steps_per=steps, width=width, verbose=False)
                rows.append(dict(
                    family   = family,
                    prob     = prob,
                    K        = K,
                    l2_van   = van["l2"],
                    l2_par   = par["l2"],
                    wall_van = van["wall"],
                    seq      = seq,
                    par      = par["wall_par"],
                    dd_seq   = seq / par["wall_par"],
                    dd_van   = van["wall"] / par["wall_par"],
                ))

    # ---------------------------------------------------------------- print
    sep = "-" * 100
    print(f"\n{sep}")
    print(f"  {'problem':<16} {'K':>2}  {'L2-van':>9}  {'L2-par':>9}  "
          f"{'van (s)':>8}  {'seq (s)':>8}  {'par (s)':>8}  "
          f"{'DD/seq':>7}  {'DD/van':>7}")
    print(sep)

    cur_family = None
    for r in rows:
        if r["family"] != cur_family:
            cur_family = r["family"]
            print(f"\n  ── {cur_family} ──")
        print(f"  {r['prob']:<16} {r['K']:>2}  {r['l2_van']:>9.2e}  {r['l2_par']:>9.2e}  "
              f"{r['wall_van']:>8.2f}  {r['seq']:>8.2f}  {r['par']:>8.2f}  "
              f"{r['dd_seq']:>6.2f}x  {r['dd_van']:>6.2f}x")

    print(f"\n{sep}")
    print("  L2-van  = vanilla PINN relative L2 error")
    print("  L2-par  = parallel DD relative L2 error")
    print("  DD/seq  = parallel DD speedup over sequential DD  (measures multiprocessing gain)")
    print("  DD/van  = parallel DD speedup over vanilla        (vanilla runs unpinned)")
    print(f"{sep}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Ks",    default="2",  help="comma-separated K values e.g. 2,4")
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--width", type=int, default=32)
    args = ap.parse_args()

    Ks = [int(k) for k in args.Ks.split(",")]

    import torch.multiprocessing as mp
    print(f"cores={mp.cpu_count()}  Ks={Ks}  iters={args.iters}  "
          f"steps/round={args.steps}  width={args.width}\n")

    run_all(Ks, args.iters, args.steps, args.width)


if __name__ == "__main__":
    main()