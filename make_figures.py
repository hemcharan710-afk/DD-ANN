"""
make_figures.py
---------------
Generates the two figures used in the SRIP pre-final report:

  figure1_spectral_bias.png   exact sin(4 pi x) vs vanilla PINN vs DD-PINN
  figure2_speedup.png         measured DD-vs-vanilla wall-clock speed-up bars

Place this file in the SAME folder as dd_parallel_mp.py and run:

    python3.13 make_figures.py

Figure 1 actually trains the networks (uses your dd_parallel_mp.py), so it
takes ~1-2 min on CPU. Figure 2 just plots the measured numbers from your
tables -- edit the SPEEDUP dict below if you re-run and your timings change.

Requires: torch, numpy, matplotlib  (same env you run the solver in).
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# pull the building blocks straight from your solver
from dd_parallel_mp import PROBLEMS, HardPINN, pde_residual, run_dd_mp, run_vanilla

plt.rcParams.update({"font.family": "serif", "font.size": 12})
NAVY, STEEL, EXACT, DD = "#1F3864", "#9DB4CE", "#222222", "#C0392B"


# -------------------------------------------------------------------------
# Figure 1 -- spectral bias on sin(4 pi x)
# -------------------------------------------------------------------------
def vanilla_curve(prob="sin4", width=47, steps=6000, n_col=256, lr=1e-3, n_eval=400):
    """Same training loop as run_vanilla(), but returns the predicted curve."""
    torch.manual_seed(42)
    p = PROBLEMS[prob]
    m = HardPINN(0.0, 1.0, width)
    ua = torch.tensor([[p["uL"]]]); ub = torch.tensor([[p["uR"]]])
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    for _ in range(steps):
        x = torch.rand(n_col, 1, requires_grad=True)
        loss = torch.mean(pde_residual(m(x, ua, ub), x, p) ** 2)
        opt.zero_grad(); loss.backward(); opt.step()
    xt = torch.linspace(0, 1, n_eval).view(-1, 1)
    with torch.no_grad():
        return xt.numpy().ravel(), m(xt, ua, ub).numpy().ravel()


def figure1():
    print("Figure 1: training vanilla PINN on sin(4 pi x) ...", flush=True)
    xv, yv = vanilla_curve("sin4")
    print("Figure 1: running K=2 DD-PINN on sin(4 pi x) ...", flush=True)
    r = run_dd_mp("sin4", K=2, width=32, verbose=False)   # returns x, pred, exact
    xd, yd, ye = r["x"], r["pred"], r["exact"]

    fig, ax = plt.subplots(figsize=(8.2, 4.6), dpi=200)
    ax.plot(xd, ye, color=EXACT, lw=2.2, label="exact  sin(4\u03c0x)", zorder=3)
    ax.plot(xv, yv, color=STEEL, lw=2.0, ls="--",
            label=f"vanilla PINN  (L2={run_vanilla('sin4')['l2']:.2f})", zorder=2)
    ax.plot(xd, yd, color=DD, lw=2.0,
            label=f"DD-PINN K=2  (L2={r['l2']:.2f})", zorder=4)
    ax.axhline(0, color="#cccccc", lw=0.8)
    ax.set_xlabel("x"); ax.set_ylabel("u(x)")
    ax.set_title("Spectral bias on \u2212u\u2033 = f,  exact u = sin(4\u03c0x)",
                 color=NAVY, fontweight="bold", fontsize=12.5, pad=10)
    ax.legend(frameon=False, fontsize=10.5)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(color="#ececec", lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("figure1_spectral_bias.png", bbox_inches="tight", facecolor="white")
    print("saved figure1_spectral_bias.png")


# -------------------------------------------------------------------------
# Figure 2 -- measured DD-vs-vanilla wall-clock speed-up
# (numbers from the report; edit if you re-run and timings change)
# -------------------------------------------------------------------------
SPEEDUP = {
    "1D": {"sin(\u03c0x)": 0.98, "sin(4\u03c0x)": 0.94, "e\u02e3": 0.93},
    "2D": {"sin11": 1.43, "sin13": 1.78, "sin31": 1.58, "sin33": 1.43},
}


def figure2():
    labels = list(SPEEDUP["1D"]) + list(SPEEDUP["2D"])
    vals   = list(SPEEDUP["1D"].values()) + list(SPEEDUP["2D"].values())
    colors = [STEEL] * len(SPEEDUP["1D"]) + [NAVY] * len(SPEEDUP["2D"])
    n1 = len(SPEEDUP["1D"])

    fig, ax = plt.subplots(figsize=(8.2, 4.3), dpi=200)
    x = range(len(vals))
    ax.bar(x, vals, color=colors, width=0.66, edgecolor="white", lw=0.6, zorder=3)
    ax.axhline(1.0, ls="--", lw=1.3, color="#555555", zorder=2)
    ax.text(len(vals) - 0.45, 1.005, "parity (1.00\u00d7)", ha="right", va="bottom",
            fontsize=10, color="#555555", style="italic")
    for xi, v in zip(x, vals):
        ax.text(xi, v + 0.02, f"{v:.2f}\u00d7", ha="center", va="bottom",
                fontsize=10.5, color=NAVY if v >= 1 else "#5a6b80", fontweight="bold")
    ax.axvline(n1 - 0.5, color="#cccccc", lw=1.0, zorder=1)
    ax.text((n1 - 1) / 2, -0.20, "1D  (subdomains too small)", ha="center", va="top",
            fontsize=11, color="#5a6b80", transform=ax.get_xaxis_transform())
    ax.text((n1 + len(vals) - 1) / 2, -0.20, "2D  (above parity, 1.43\u00d7\u20131.78\u00d7)",
            ha="center", va="top", fontsize=11, color=NAVY,
            transform=ax.get_xaxis_transform())
    ax.set_xticks(list(x)); ax.set_xticklabels(labels)
    ax.set_ylabel("DD speed-up vs. global PINN  (\u00d7)")
    ax.set_ylim(0, 2.0)
    ax.set_title("Measured wall-clock speed-up of true-parallel DD over the global PINN",
                 color=NAVY, fontweight="bold", fontsize=12.5, pad=12)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color="#e8e8e8", lw=0.8); ax.set_axisbelow(True)
    plt.tight_layout()
    plt.savefig("figure2_speedup.png", bbox_inches="tight", facecolor="white")
    print("saved figure2_speedup.png")


if __name__ == "__main__":          # required: DD uses multiprocessing 'spawn'
    figure2()                       # fast (no training)
    figure1()                       # ~1-2 min (trains the nets)