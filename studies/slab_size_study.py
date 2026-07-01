"""
slab_size_study.py
------------------
Does the subdomain network *shrink* buy wall-clock without compromising error?

For each dimension (1D, 2D, 3D) this sweeps the per-subdomain network size from
the capacity-matched default DOWN to much smaller nets, keeping the global
vanilla PINN fixed as the baseline, and measures relative L2 error + true-parallel
wall-clock speed-up for every size.

Two modes:
    python3.13 slab_size_study.py            # run the sweeps -> slab_size_results.json
    python3.13 slab_size_study.py --dims 1d  # only one dimension (1d/2d/3d/all)
    python3.13 slab_size_study.py --plot     # read the JSON -> figures (no training)

Figures written next to this script (studies/):
    figure3_slabsize_1d.png
    figure3_slabsize_2d.png
    figure3_slabsize_3d.png
    figure4_slabsize_summary.png

Requires: torch, numpy, matplotlib.  Must be run as a script (DD uses 'spawn').
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                 # repo root (studies/ lives one level down)
for d in ("Phase1_PINN_1D", "Phase2_PINN_2D", "Phase3_PINN_3D"):
    sys.path.insert(0, os.path.join(ROOT, d))

RESULTS = os.path.join(HERE, "slab_size_results.json")


# -------------------------------------------------------------------------
# param counter for a stack of Linear layers with dims [d0, d1, ..., dL]
# -------------------------------------------------------------------------
def nparams(dims):
    return int(sum(dims[i] * dims[i + 1] + dims[i + 1] for i in range(len(dims) - 1)))


# -------------------------------------------------------------------------
# Sweep definitions.  hidden width is the swept knob; "matched" flags the
# capacity-matched default that the README reports.
# -------------------------------------------------------------------------
WIDTHS_1D = [32, 24, 16, 12, 8]                 # global vanilla width = 47
LAYERS_2D = [64, 48, 32, 20]                    # global vanilla 2-64-64-64-1
LAYERS_3D = [64, 48, 32, 20]                    # global vanilla 3-64-64-64-1

PROBS_1D = ["sin1", "sin4", "exp"]
PROBS_2D = ["sin11", "sin13", "sin33"]
PROBS_3D = ["pois_111", "lpb_k3", "cosmo_mid"]

PRETTY = {
    "sin1": "sin(πx)", "sin4": "sin(4πx)", "exp": "e^x",
    "sin11": "sin·sin (1,1)", "sin13": "sin·sin (1,3)", "sin33": "sin·sin (3,3)",
    "pois_111": "Poisson", "lpb_k3": "LPB (κ=3)", "cosmo_mid": "COSMO",
}


def sweep_1d():
    import dd_parallel_mp as M
    out = {"dim": "1D", "matched_width": 32, "vanilla_dims": [1, 47, 47, 47, 1],
           "vanilla_params": nparams([1, 47, 47, 47, 1]), "problems": {}}
    for prob in PROBS_1D:
        van = M.run_vanilla(prob)                       # global width-47
        rows = []
        for w in WIDTHS_1D:
            r = M.run_dd_mp(prob, K=2, width=w, verbose=False)
            dims = [1, w, w, w, 1]
            rows.append(dict(width=w, params=nparams(dims),
                             params_total=2 * nparams(dims),
                             l2=r["l2"], par=r["wall_par"],
                             speedup=van["wall"] / r["wall_par"],
                             matched=(w == 32)))
            print(f"  1D {prob:6s} w={w:3d}  L2={r['l2']:.2e}  "
                  f"par={r['wall_par']:.2f}s  DD/van={van['wall']/r['wall_par']:.2f}x",
                  flush=True)
        out["problems"][prob] = dict(vanilla_l2=van["l2"], vanilla_wall=van["wall"],
                                     rows=rows)
    return out


def sweep_2d():
    import dd_parallel_mp_2d as M
    out = {"dim": "2D", "matched_width": 64, "vanilla_dims": [2, 64, 64, 64, 1],
           "vanilla_params": nparams([2, 64, 64, 64, 1]), "problems": {}}
    for prob in PROBS_2D:
        van = M.run_vanilla_2d(prob)
        rows = []
        for w in LAYERS_2D:
            layers = (2, w, w, 1)
            r = M.run_dd_mp_2d(prob, K=2, layers=layers, verbose=False)
            rows.append(dict(width=w, params=nparams(list(layers)),
                             params_total=2 * nparams(list(layers)),
                             l2=r["l2"], par=r["wall_par"],
                             speedup=van["wall"] / r["wall_par"],
                             matched=(w == 64)))
            print(f"  2D {prob:6s} w={w:3d}  L2={r['l2']:.2e}  "
                  f"par={r['wall_par']:.2f}s  DD/van={van['wall']/r['wall_par']:.2f}x",
                  flush=True)
        out["problems"][prob] = dict(vanilla_l2=van["l2"], vanilla_wall=van["wall"],
                                     rows=rows)
    return out


def sweep_3d():
    import dd_parallel_mp_3d as M
    out = {"dim": "3D", "matched_width": 64, "vanilla_dims": [3, 64, 64, 64, 1],
           "vanilla_params": nparams([3, 64, 64, 64, 1]), "problems": {}}
    for prob in PROBS_3D:
        van = M.run_vanilla_3d(prob)
        rows = []
        for w in LAYERS_3D:
            layers = (3, w, w, 1)
            r = M.run_dd_mp_3d(prob, K=2, layers=layers, verbose=False)
            rows.append(dict(width=w, params=nparams(list(layers)),
                             params_total=2 * nparams(list(layers)),
                             l2=r["l2"], par=r["wall_par"],
                             speedup=van["wall"] / r["wall_par"],
                             matched=(w == 64)))
            print(f"  3D {prob:9s} w={w:3d}  L2={r['l2']:.2e}  "
                  f"par={r['wall_par']:.2f}s  DD/van={van['wall']/r['wall_par']:.2f}x",
                  flush=True)
        out["problems"][prob] = dict(vanilla_l2=van["l2"], vanilla_wall=van["wall"],
                                     rows=rows)
    return out


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------
def make_figures():
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(RESULTS) as fh:
        data = json.load(fh)

    plt.rcParams.update({"font.family": "serif", "font.size": 11})
    NAVY, DD = "#1F3864", "#C0392B"
    LINE_COLORS = ["#1F3864", "#C0392B", "#2E8B57", "#8E44AD"]

    def panel(dim_key, fname):
        d = data[dim_key]
        probs = list(d["problems"])
        fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.4, 4.5), dpi=200)

        for i, prob in enumerate(probs):
            c = LINE_COLORS[i % len(LINE_COLORS)]
            rows = sorted(d["problems"][prob]["rows"], key=lambda r: r["width"])
            ws = [r["width"] for r in rows]
            l2 = [r["l2"] for r in rows]
            sp = [r["speedup"] for r in rows]
            vl = d["problems"][prob]["vanilla_l2"]
            lab = PRETTY.get(prob, prob)
            axL.plot(ws, l2, "-o", color=c, lw=2, ms=6, label=lab, zorder=3)
            axL.axhline(vl, color=c, lw=1.1, ls=":", alpha=0.7, zorder=2)
            axR.plot(ws, sp, "-o", color=c, lw=2, ms=6, label=lab, zorder=3)

        mw = d["matched_width"]
        for ax in (axL, axR):
            ax.axvline(mw, color="#999999", lw=1.1, ls="--", zorder=1)
            ax.set_xlabel("subdomain hidden width")
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
            ax.grid(color="#ececec", lw=0.8); ax.set_axisbelow(True)
        axL.text(mw, axL.get_ylim()[1], "  matched", color="#777777",
                 fontsize=9, va="top", ha="left")

        axL.set_yscale("log")
        axL.set_ylabel("relative $L_2$ error  (log)")
        axL.set_title("Accuracy vs. subdomain size  (dotted = vanilla $L_2$)",
                      color=NAVY, fontsize=11.5, fontweight="bold", pad=8)
        axL.legend(frameon=False, fontsize=9.5)

        axR.axhline(1.0, color="#555555", lw=1.2, ls="--", zorder=1)
        axR.text(axR.get_xlim()[1], 1.02, "parity", ha="right", va="bottom",
                 fontsize=9, color="#555555", style="italic")
        axR.set_ylabel("true-parallel DD speed-up vs. vanilla  (×)")
        axR.set_title("Speed-up vs. subdomain size", color=NAVY,
                      fontsize=11.5, fontweight="bold", pad=8)

        fig.suptitle(f"{d['dim']}: shrinking each subdomain network "
                     f"(vanilla global net fixed, {d['vanilla_params']:,} params)",
                     color=NAVY, fontsize=13, fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.savefig(os.path.join(HERE, fname), bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"saved {fname}")

    if "1D" in data:
        panel("1D", "figure3_slabsize_1d.png")
    if "2D" in data:
        panel("2D", "figure3_slabsize_2d.png")
    if "3D" in data:
        panel("3D", "figure3_slabsize_3d.png")

    # ---- summary: speed-up at the "sweet spot" (smallest width whose L2 stays
    #      within 2x of the matched-capacity DD L2), per dimension ----
    fig, ax = plt.subplots(figsize=(9.0, 4.6), dpi=200)
    dim_order = [k for k in ("1D", "2D", "3D") if k in data]
    colors = {"1D": "#9DB4CE", "2D": "#1F3864", "3D": "#C0392B"}
    xs, labels, bar_colors, notes = [], [], [], []
    xi = 0
    for dk in dim_order:
        d = data[dk]
        for prob in d["problems"]:
            rows = sorted(d["problems"][prob]["rows"], key=lambda r: -r["width"])
            matched = next(r for r in rows if r["matched"])
            thresh = 2.0 * matched["l2"]
            ok = [r for r in rows if r["l2"] <= thresh]
            pick = min(ok, key=lambda r: r["width"]) if ok else matched
            xs.append(xi); labels.append(f"{dk}\n{PRETTY.get(prob, prob)}")
            bar_colors.append(colors[dk])
            notes.append((pick["speedup"], pick["width"], matched["speedup"]))
            xi += 1
        xi += 0.6

    vals = [n[0] for n in notes]
    ax.bar(xs, vals, color=bar_colors, width=0.7, edgecolor="white", zorder=3)
    ax.axhline(1.0, color="#555555", lw=1.2, ls="--", zorder=2)
    for x, (sp, w, msp) in zip(xs, notes):
        ax.text(x, sp + 0.03, f"{sp:.2f}×\nw={w}", ha="center", va="bottom",
                fontsize=8.5, color=NAVY, fontweight="bold")
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("DD speed-up at tuned subdomain size  (×)")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_title("Tuned-subdomain speed-up (smallest net with $L_2$ within 2× of matched)",
                 color=NAVY, fontsize=12, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#ececec", lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "figure4_slabsize_summary.png"),
                bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved figure4_slabsize_summary.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", default="all", choices=["1d", "2d", "3d", "all"])
    ap.add_argument("--plot", action="store_true", help="plot from existing JSON only")
    args = ap.parse_args()

    if args.plot:
        make_figures()
        return

    data = {}
    if os.path.exists(RESULTS):
        with open(RESULTS) as fh:
            data = json.load(fh)

    if args.dims in ("1d", "all"):
        print("== 1D sweep ==", flush=True); data["1D"] = sweep_1d()
        with open(RESULTS, "w") as fh: json.dump(data, fh, indent=2)
    if args.dims in ("2d", "all"):
        print("== 2D sweep ==", flush=True); data["2D"] = sweep_2d()
        with open(RESULTS, "w") as fh: json.dump(data, fh, indent=2)
    if args.dims in ("3d", "all"):
        print("== 3D sweep ==", flush=True); data["3D"] = sweep_3d()
        with open(RESULTS, "w") as fh: json.dump(data, fh, indent=2)

    print(f"\nwrote {RESULTS}")
    make_figures()


if __name__ == "__main__":
    main()
