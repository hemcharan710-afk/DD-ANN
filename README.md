# Domain Decomposition Accelerated Neural Networks (DD-ANN)

**Scalable, parallel, mesh-free PDE solvers** that combine Physics-Informed
Neural Networks (PINNs) with classical domain decomposition — built toward the
electrostatic models of computational chemistry (Linearized Poisson–Boltzmann,
COSMO).

| | |
|---|---|
| **Program** | Summer Research Internship Program (SRIP) 2026 |
| **Institute** | Indian Institute of Technology Gandhinagar |
| **Students** | Chitiveli Hemcharan Varma (IIT Gandhinagar) · Krishna (VIT Vellore) |
| **Supervisor** | Dr. Abhinav Jha |

---

## Overview

A Physics-Informed Neural Network solves a PDE by training a network so that the
equation's residual — measured through automatic differentiation — vanishes on a
set of collocation points. No mesh is required. This flexibility comes at a cost:
PINNs exhibit **spectral bias** (they learn high-frequency content very slowly)
and become expensive as the domain grows or the solution gets more complex.

This project addresses both limitations with **domain decomposition (DD)**. The
domain is split into overlapping subdomains; a small PINN is trained on each and
the subdomains are coupled by a classical **overlapping Schwarz** iteration that
exchanges interface data between neighbours. Because each subdomain is a smaller,
lower-frequency, *independent* problem, decomposition simultaneously

- **mitigates spectral bias** — each network sees a lower effective frequency, and
- **exposes parallelism** — subdomains train concurrently on separate cores or nodes.

The target application is the **Linearized Poisson–Boltzmann (LPB)** equation and
the **COSMO** solvation model, where domains are large and a single global PINN
does not scale.

---

## Background

### Physics-Informed Neural Networks

A network $N_\theta$ approximates the solution $u$. Boundary conditions are
imposed **exactly** through a distance-function ansatz,

$$u_\theta(x) = \text{lift}(x) + d(x)\,N_\theta(x),$$

where $\text{lift}$ interpolates the boundary data and $d$ vanishes on the
boundary (e.g. $d(x)=(x-a)(x-b)$ in 1D, $d(x,y)=x(1-x)\,y(1-y)$ on the unit
square). Because the boundary conditions hold by construction, the training loss
is the **pure PDE residual** — there is no boundary-penalty term to weight or
balance.

### Overlapping Schwarz domain decomposition

The decomposed solver runs an outer **Schwarz iteration**. In each round every
subdomain trains for a fixed number of steps against its neighbours' current
interface values, then publishes updated interface values for the next round.
The scheme is **Jacobi** (each subdomain uses the *previous* round's data), which
makes the subdomains independent within a round — and therefore parallelisable.

### Why the overlap is essential

With an exact boundary condition, a subdomain evaluated *at its own boundary*
returns the imposed value by construction. If two subdomains merely **touched**,
the transmitted interface value would be self-referential and could never
update — the classical ill-posedness of non-overlapping Dirichlet–Dirichlet
coupling. With **overlap**, each subdomain reads its interface value from the
**neighbour's interior**, a genuine PDE value, and the Schwarz iteration
converges geometrically (faster with larger overlap).

### How the interface value is transmitted

The two phases couple neighbouring subdomains differently — both Jacobi, both
fully parallel:

- **1D — exact (hard) injection.** The neighbour's interior value is fed in as the
  subdomain's Dirichlet datum through the lift term, so the interface condition
  holds by construction each round.
- **2D — soft transmission penalty.** Each strip adds a least-squares term
  $w\,\lVert N_\theta(\text{interface}) - \text{neighbour profile}\rVert^2$
  (weight $w = 300$) that pulls its solution toward the neighbour's frozen profile
  along the shared line $x=\text{const}$.

In both phases the *outer-box* Dirichlet conditions are still imposed exactly
through the distance-function ansatz; only the *internal* interface differs.

---

## Problem set

Both phases solve the **Poisson equation** with homogeneous Dirichlet data; the
forcing $f$ is chosen so that a known analytic solution $u$ can be used to measure
the relative $L_2$ error.

**1D** — $-u''(x) = f(x)$ on $[0,1]$:

| Case | Exact solution $u(x)$ | Character |
|---|---|---|
| `sin1` | $\sin(\pi x)$ | smooth |
| `sin4` | $\sin(4\pi x)$ | high-frequency (spectral-bias stress test) |
| `exp` | $e^{x}$ | non-zero boundary data |

**2D** — $-\Delta u = f(x,y)$ on $[0,1]^2$, with $u = \sin(k_x\pi x)\sin(k_y\pi y)$:

| Case | Exact solution $u(x,y)$ | Character |
|---|---|---|
| `sin11` | $\sin(\pi x)\sin(\pi y)$ | smooth |
| `sin13` | $\sin(\pi x)\sin(3\pi y)$ | anisotropic, higher frequency in $y$ |
| `sin31` | $\sin(3\pi x)\sin(\pi y)$ | anisotropic, higher frequency in $x$ |

In 2D the domain is decomposed into **vertical strips** overlapping in $x$.
Throughout, decomposition uses **$K = 2$ subdomains**.

---

## Repository structure

```
DD-ANN/
├── Phase1_PINN_1D/
│   ├── dd_parallel_mp.py        # 1D: vanilla PINN, DD-PINN, true-parallel DD
│   └── run_all_1d.py            # sweep every 1D problem through all three methods
├── Phase2_PINN_2D/
│   ├── dd_parallel_mp_2d.py     # 2D: vanilla PINN, DD-PINN (strips), true-parallel DD
│   └── run_all_problems.py      # sweep every 2D problem through all three methods
├── References/                  # key reference papers
└── README.md
```

The core scripts also carry extra problem families beyond those reported here —
1D reaction-diffusion and advection-diffusion, 2D anisotropic Poisson and
Helmholtz — used for stress-testing. The headline tables stay focused on the
cases both methods actually solve.

Each script is **self-contained**: it implements the vanilla PINN, the
decomposed PINN, a single-process baseline, and the true-parallel solver, then
prints measured results under matched network capacity and a matched
optimization budget. No numbers are cached or hand-edited.

---

## Methods compared

| | Method | Description |
|---|---|---|
| **A** | **Vanilla PINN** | one global network over the whole domain, using the full machine |
| **B** | **DD-PINN** | $K=2$ overlapping subdomains, a smaller PINN on each, coupled by an overlapping Schwarz iteration; each subdomain trains in its own OS process |

Network capacity is matched between the two methods so the comparison is fair: in
1D a width-47 network (≈4.7k parameters) versus two width-32 networks (≈4.4k
total); in 2D a `2-64-64-64-1` network (≈8.6k) versus two `2-64-64-1` strips
(≈8.8k total).

---

## Results

All results were measured on **Apple M3** (4 performance + 4 efficiency cores),
Python 3.13, PyTorch 2.9, on **CPU** — for networks this small the CPU outperforms
the GPU/MPS backend, where kernel-launch overhead dominates.

### Accuracy (relative $L_2$ error)

**1D Poisson** — 6000 optimization steps (DD: 15 Schwarz rounds × 400 steps):

| Problem | Vanilla PINN | DD-PINN ($K=2$) |
|---|---:|---:|
| $\sin(\pi x)$ | 1.89e-03 | 1.46e-03 |
| **$\sin(4\pi x)$** *(high-freq)* | **2.45e+00** | **5.22e-01** |
| $e^{x}$ | 3.36e-04 | 1.06e-03 |

On smooth problems both methods are accurate. On the high-frequency $\sin(4\pi x)$
the global PINN is crippled by spectral bias, while decomposition is **≈5× more
accurate** (2.45 → 0.52) because each subdomain sees a lower effective frequency.
**In 1D, the value of decomposition is accuracy.**

**2D Poisson** — vanilla 4500 steps, DD 12 Schwarz rounds × 400 steps:

| Problem | Vanilla PINN | DD-PINN ($K=2$) |
|---|---:|---:|
| $\sin(\pi x)\sin(\pi y)$ | 1.24e-04 | 4.96e-03 |
| $\sin(\pi x)\sin(3\pi y)$ | 2.44e-02 | 1.57e-02 |

On the anisotropic $\sin(\pi x)\sin(3\pi y)$ decomposition is more accurate
(2.4e-02 → 1.6e-02); on the very smooth case the single global network retains an
edge. **In 2D, the value of decomposition is speed** — quantified next.

### Parallel speed-up

The two subdomains are executed as **separate OS processes** via
`torch.multiprocessing` and run concurrently; a `time.perf_counter()` wraps the
parallel region. The reported speed-up is

$$\text{speed-up} = \frac{\text{sequential DD (one process)}}{\text{true-parallel DD (two processes)}},$$

an honest, measured ratio of identical work — not an estimate.

| Problem | DD $L_2$ | Sequential (s) | Parallel (s) | **Speed-up** |
|---|---:|---:|---:|---:|
| 1D $\sin(4\pi x)$ | 5.22e-01 | 15.62 | 9.63 | **1.62×** |
| 2D $\sin(\pi x)\sin(3\pi y)$ | 1.57e-02 | 44.66 | 27.49 | **1.62×** |

Running the subdomains in parallel is **≈1.5–1.6× faster** than running them
back-to-back, confirming genuine concurrency across two performance cores.

### Vanilla PINN vs. true-parallel DD

The decisive comparison: a single global PINN with the **whole machine**
available, versus decomposition training its two subdomains in parallel processes.

| Problem | Vanilla $L_2$ | DD $L_2$ | Vanilla (s) | DD (s) | **DD vs. vanilla** |
|---|---:|---:|---:|---:|---:|
| 1D $\sin(\pi x)$ | 1.89e-03 | 1.46e-03 | 9.67 | 9.85 | 0.98× |
| 1D $\sin(4\pi x)$ | 2.45e+00 | 5.22e-01 | 8.52 | 9.09 | 0.94× |
| 1D $e^{x}$ | 3.36e-04 | 1.06e-03 | 8.37 | 9.04 | 0.93× |
| **2D $\sin(\pi x)\sin(\pi y)$** | 1.24e-04 | 4.96e-03 | 42.48 | 29.65 | **1.43×** |
| **2D $\sin(\pi x)\sin(3\pi y)$** | 2.44e-02 | 1.57e-02 | 49.06 | 27.49 | **1.78×** |

- **1D** problems are too small for parallelism to pay off (DD ≈0.9–1.0× vanilla on
  wall-clock); decomposition wins on **accuracy** instead.
- **2D** subdomains are heavy enough that true-parallel DD is **faster** than the
  global PINN (1.43–1.78×) — and on the harder $\sin(\pi x)\sin(3\pi y)$ it is more
  accurate too. This is the project goal: **speed without sacrificing accuracy.**
- The wall-clock speed-up was **consistent across every 2D problem** swept (1.4–1.8×
  on all four Poisson cases plus three Helmholtz cases); it is a property of the
  parallel decomposition, not of any one solution. The accuracy advantage, by
  contrast, is problem-dependent — see *Discussion*.

### Discussion

**Decomposition direction matters.** The 2D strips split the domain in $x$, so the
soft interface coupling fares best when the solution's fast variation runs
*along* the interfaces (in $y$) rather than *across* them. On
$\sin(\pi x)\sin(3\pi y)$ — fast in $y$, parallel to the cuts — decomposition is
more accurate than the global PINN; on the mirror case $\sin(3\pi x)\sin(\pi y)$ —
fast in $x$, cutting straight across the strips — accuracy degrades sharply. The
natural fixes are to align the cut with the low-frequency axis or to decompose in
both directions. (Speed is unaffected either way — the decomposition is parallel
regardless of which case is solved.)

**Why the 1D speed-up is overhead-bound.** The 1D subdomains are below parity
(DD ≈0.9–1.0× vanilla) not because the parallelism fails but because the
per-subdomain work is too small to cover the fixed cost of decomposition —
process spawn, one interface exchange per Schwarz round, and the single-thread
pin on each worker. That cost is essentially constant, so the speed-up improves
as the network grows. Holding the problem fixed at $\sin(4\pi x)$ and scaling the
network depth (matched 6000-step budget) makes this explicit:

| Network depth | Vanilla params | DD params ($K{=}2$) | **DD vs. vanilla** |
|---|---:|---:|---:|
| 3 *(reported above)* | 4 654 | 4 418 | 0.94× |
| 5 | 9 166 | 8 642 | 0.99× |
| 8 | 15 934 | 14 978 | **1.06×** |

The ratio rises monotonically and crosses parity once the subdomains are heavy
enough. This is the same effect that puts 2D — where each strip carries a
genuinely larger workload — comfortably above parity, and it confirms that the
right lever for a wall-clock win is *more* work per subdomain, not less.

The measured parallelism is real but, on a laptop, bounded by the number of
**performance cores**. The Schwarz round is a synchronization **barrier** — each
round waits for the slowest subdomain — so the clean ≈1.5× speed-up at $K=2$
depends on both subdomains landing on performance cores. Using more subdomains
than performance cores would push workers onto the slower efficiency cores and
stall the barrier; this is why $K$ is fixed at **2** for these experiments.

The intended deployment is **homogeneous multi-node HPC**, where every subdomain
receives an equal core or node and the barrier cost vanishes — the regime that
matters for the LPB/COSMO application. The results here establish that the solver
is correct and genuinely parallel; large-scale strong/weak scaling belongs on a
cluster.

---

## Reproducing the results

Requirements: **Python 3.13**, **PyTorch ≥ 2.9**, **NumPy**.

```bash
# 1D — sequential-vs-parallel speed-up table
python3.13 Phase1_PINN_1D/dd_parallel_mp.py    --prob sin4  --Ks 2

# 1D — vanilla PINN vs. true-parallel DD, head-to-head
python3.13 Phase1_PINN_1D/dd_parallel_mp.py    --prob sin4  --vs-vanilla

# 2D — sequential-vs-parallel speed-up table
python3.13 Phase2_PINN_2D/dd_parallel_mp_2d.py --prob sin13 --Ks 2

# 2D — vanilla PINN vs. true-parallel DD, head-to-head
python3.13 Phase2_PINN_2D/dd_parallel_mp_2d.py --prob sin13 --vs-vanilla

# sweep every problem (vanilla + sequential DD + true-parallel DD) in one table
python3.13 Phase1_PINN_1D/run_all_1d.py       --Ks 2
python3.13 Phase2_PINN_2D/run_all_problems.py --Ks 2
```

Each script prints its own measured table for the host machine, so absolute
timings will vary while the qualitative conclusions hold. The scripts **must be
run from the command line**, not from a notebook (see below).

---

## Implementation notes: true parallelism on macOS

Naïve attempts at parallel PINN training fail on a Mac for subtle reasons. The
solver is designed around four of them:

1. **Processes, not threads.** Python's GIL serializes these small tensor
   operations, so `threading` yields ≈0 speed-up. Parallelism uses
   `torch.multiprocessing` with one OS process per subdomain.
2. **Module-scope workers.** macOS uses the `spawn` start method, which re-imports
   the target in each child process. A worker defined inside a Jupyter notebook
   cannot be pickled and the run hangs. The worker therefore lives at module scope
   in a `.py` file — the reason this project ships as scripts, not notebooks.
3. **No oversubscription.** Each worker pins `torch.set_num_threads(1)` so the
   per-process BLAS pools do not fight over cores.
4. **Persistent workers, minimal IPC.** Each process builds its model once and
   keeps it across all Schwarz rounds; only the interface values cross the pipe
   each round, so communication overhead is negligible.

---

## Roadmap

| Phase | Scope | Status |
|---|---|---|
| Phase 1 | Vanilla PINN vs. DD-PINN on 1D Poisson + true-parallel benchmark | ✅ Complete |
| Phase 2 | Vanilla PINN vs. DD-PINN on 2D Poisson + true-parallel benchmark | ✅ Complete |
| Next | Additive/asynchronous Schwarz (remove the round barrier) · multi-node cluster scaling · 3D PINN · Linearized Poisson–Boltzmann / COSMO | ⏳ Planned |

---

## Tech stack

Python · PyTorch · NumPy · `torch.multiprocessing`

---

## References

1. M. Raissi, P. Perdikaris, G. E. Karniadakis. *Physics-Informed Neural
   Networks: A Deep Learning Framework for Solving Forward and Inverse Problems
   Involving Nonlinear Partial Differential Equations.* Journal of Computational
   Physics, 2019.
2. S. Wang, S. Sankaran, H. Wang, P. Perdikaris. *An Expert's Guide to Training
   Physics-Informed Neural Networks.* 2023.
