# Domain Decomposition Accelerated Neural Networks (DD-ANNs)

### Summer Research Internship Program (SRIP) 2026 — IIT Gandhinagar

**Students:** Chitiveli Hemcharan Varma (IIT Gandhinagar) · Krishna (VIT Vellore)
**Supervisor:** Dr. Abhinav Jha
**Institute:** Indian Institute of Technology Gandhinagar

> Combining Physics-Informed Neural Networks (PINNs) with classical domain
> decomposition to build **scalable, parallel** mesh-free PDE solvers — aimed at
> the electrostatic models of computational chemistry (Linearized
> Poisson–Boltzmann, COSMO).

---

## Project overview

Physics-Informed Neural Networks solve PDEs by minimizing the equation residual
through automatic differentiation, with no mesh. They are flexible but suffer
from **spectral bias** (slow to learn high frequencies) and grow expensive as the
domain or solution complexity increases.

This project attacks both problems with **domain decomposition (DD)**: split the
domain into subdomains, train a small PINN on each, and couple them with a
classical **overlapping Schwarz** iteration that exchanges interface data between
neighbours. Each subdomain is a smaller, lower-frequency, *independent* problem —
so DD both mitigates spectral bias **and** exposes parallelism: the subdomains
can train simultaneously on separate cores or nodes.

The ultimate application is solving the **Linearized Poisson–Boltzmann (LPB)**
equation and the **COSMO** model used in biomolecular simulation and solvation-
energy calculations, where domains are large and a single global PINN does not
scale.

---

## Repository structure

```
Phase1_PINN_1D/
  dd_parallel_mp.py          # TRUE-parallel K-subdomain DD (torch.multiprocessing)
Phase2_PINN_2D/
  dd_parallel_mp_2d.py       # TRUE-parallel K-strip DD (torch.multiprocessing)
References/                  # key reference papers
README.md
```

Each script is **self-contained**: it implements vanilla PINN and DD-PINN from
scratch, runs them under matched capacity and a matched optimization budget, and
prints the **real measured** results — no cached or hand-edited numbers. The
domain is split into `K` overlapping subdomains, one OS process per subdomain;
`K` defaults to **2**.

---

## Methods compared

| | Method | Idea |
|---|---|---|
| **A** | **Vanilla PINN** | one global network over the whole domain |
| **B** | **DD-PINN** | split the domain into **overlapping** subdomains, a smaller PINN on each, coupled by an **overlapping Schwarz** outer iteration that exchanges interface (Dirichlet) data each round |

Both methods enforce boundary conditions **exactly** with a distance function
(`u = lift + (distance)·Nθ`), so the loss is the pure PDE residual — there is no
boundary-penalty term to balance.

**Why the overlap is essential.** With a hard BC, a subdomain network evaluated
*at its own boundary* returns the imposed value by construction. If two
subdomains only *touched* (no overlap), the transmitted interface value would be
self-referential and could never update — the classic ill-posedness of
non-overlapping Dirichlet–Dirichlet coupling. With overlap, each subdomain reads
its interface value from the **neighbour's interior** (a genuine PDE value), so
the Schwarz iteration converges **geometrically — faster with larger overlap**.

---

## Results (real, reproduced by the scripts)

All numbers are measured by the scripts on CPU, under matched network capacity
and a matched optimization budget.

Throughout, **DD uses K = 2 subdomains** (one global network vs. two smaller ones
of matched total capacity). Timing for every run is in the
[parallel-execution section](#real-parallel-execution-measured-not-inferred).

### 1D Poisson  −u″ = f on [0,1]

One width-47 network vs. two width-32 networks (≈4.7k vs ≈4.4k params), 6000
optimization steps each (15 Schwarz rounds × 400 steps for DD).

| Problem | Vanilla L2 | DD L2 (K=2) |
|---|---|---|
| sin(πx) | 1.89e-03 | 1.46e-03 |
| **sin(4πx)** *(high-freq)* | **2.45e+00** | **5.22e-01** |
| eˣ | 3.36e-04 | 1.06e-03 |

On smooth problems the global PINN is already excellent and DD is comparable. On
the **high-frequency** `sin(4πx)`, the global PINN is crippled by spectral bias
while DD is **~5× more accurate** (2.45 → 0.52) — each subdomain sees a lower
effective frequency. **This is DD's main value in 1D: accuracy, not speed.**

### 2D Poisson  −Δu = f on [0,1]²

One `2-64-64-64-1` network vs. two `2-64-64-1` strips (≈8.6k vs ≈8.8k params),
4500 steps (vanilla) vs. 12 Schwarz rounds × 400 steps (DD).

| Problem | Vanilla L2 | DD L2 (K=2) |
|---|---|---|
| sin(πx)sin(πy) | 1.24e-04 | 4.96e-03 |
| sin(πx)sin(3πy) | 2.44e-02 | 1.57e-02 |

Each strip is half the domain and trained independently. On the anisotropic
`sin(πx)sin(3πy)` DD is more accurate (2.4e-02 → 1.6e-02); on the very smooth
`sin(πx)sin(πy)` the single global net still wins on accuracy. **The 2D payoff is
speed** — see the next section.

---

## Real parallel execution (measured, not inferred)

The tables above report **single-process** training times only — real, but
sequential. **Every number in this section is a true measured parallel run**, not a
`max`-based estimate. [`dd_parallel_mp.py`](Phase1_PINN_1D/dd_parallel_mp.py) and
[`dd_parallel_mp_2d.py`](Phase2_PINN_2D/dd_parallel_mp_2d.py) spawn **one actual OS
process per subdomain** via `torch.multiprocessing`, run them **simultaneously**, and
wrap a real `time.perf_counter()` around the concurrent section
(`speed-up = sequential / parallel`).

> **Proof it really runs in parallel:** a measured `par < seq` is impossible unless
> the work genuinely overlapped — 1D goes **seq 15.69 s → par 9.78 s** and 2D goes
> **seq 46.21 s → par 29.82 s** (both K=2, this machine). The discarded
> "parallel-equivalent" estimate (`max` over subdomains) is a *different* computation
> that never actually runs concurrently; it is **not** used anywhere below.

All numbers below are fresh runs on **Apple M3 (4 performance + 4 efficiency
cores)**, `K = 2`.

### DD parallel speed-up: two subdomains in two processes vs. one process

`speed-up = sequential DD / true-parallel DD` — same work, same result, the only
difference is whether the two subdomains run concurrently.

| Problem | DD L2 | seq (s) | par (s) | **measured speed-up** |
|---|---:|---:|---:|---:|
| 1D sin(4πx), 15×400, width 32 | 5.22e-01 | 15.69 | 9.78 | **1.60×** |
| 2D sin(πx)sin(3πy), 12×400 | 1.57e-02 | 46.21 | 29.82 | **1.55×** |

Running the two subdomains in parallel is **~1.5–1.6× faster** than running them
back-to-back — real, measured concurrency on two performance cores.

### Head-to-head: vanilla PINN (full machine) vs true-parallel DD (K=2)

The vanilla PINN is one global network with the **whole machine** available; DD
trains its two subdomains in parallel processes (`--vs-vanilla`).

| Problem | Vanilla L2 | DD L2 | Vanilla wall (s) | DD wall (s) | DD vs vanilla |
|---|---:|---:|---:|---:|---:|
| 1D sin(πx) | 1.89e-03 | 1.46e-03 | 8.54 | 9.64 | 0.89× |
| 1D sin(4πx) | 2.45e+00 | 5.22e-01 | 8.57 | 9.51 | 0.90× |
| 1D eˣ | 3.36e-04 | 1.06e-03 | 8.59 | 9.51 | 0.90× |
| **2D sin(πx)sin(πy)** | 1.24e-04 | 4.96e-03 | 41.52 | 30.02 | **1.38× faster** |
| **2D sin(πx)sin(3πy)** | 2.44e-02 | 1.57e-02 | 41.47 | 35.02 | **1.18× faster** |

- **In 1D the problem is too small** for parallelism to pay off: DD lands ~0.9×
  vanilla on wall-clock. Its win there is **accuracy** (sin(4πx): 2.45 → 0.52).
- **In 2D each subdomain is heavy enough** that true-parallel DD beats the global
  PINN on wall-clock (1.18–1.38×) — and on the harder `sin(πx)sin(3πy)` it is more
  accurate too. **Speed without giving up accuracy** — the goal of the project.

### Honest reading of the speed-up

- **The parallelism is real**, but on this hardware it is **bounded by the number
  of *performance* cores**. The M3 has 4 performance + 4 efficiency cores, and the
  Jacobi Schwarz round is a **barrier** — every round waits for the *slowest*
  subdomain. At **K = 2** both subdomains land on performance cores, so the
  speed-up is clean (~1.5–1.6×). Splitting into more subdomains than performance
  cores would push workers onto the slow efficiency cores and stall the barrier —
  which is exactly why **K is fixed at 2** on this laptop.
- **The real win is on homogeneous multi-node HPC**, where every subdomain gets
  an equal core/node and the barrier cost vanishes — the deployment target for the
  LPB/COSMO application. The laptop numbers prove the machinery is correct and
  parallel; large-K *scaling* belongs on a cluster.

### Why naive parallelism fails on a Mac (and how this code fixes it)

1. **Threads don't help** — `ThreadPoolExecutor`/`threading` serialize on the
   Python GIL for these small ops → ~0 speed-up. **Use processes.**
2. **`spawn` can't pickle notebook functions** — macOS uses the `spawn` start
   method, which re-imports the target in the child. A worker defined *inside a
   Jupyter notebook* cannot be pickled → hangs / `PicklingError`. The worker
   therefore lives at **module scope in a `.py` file** and is imported.
3. **Oversubscription** — N processes each spawning their own BLAS threads fight
   over the cores. Each worker pins `torch.set_num_threads(1)`.
4. **IPC** — workers are **persistent** (model stays in the process across all
   rounds); only the interface values cross the pipe each round, so communication
   is negligible.

---

## How to run

The project runs on **CPU** (for these small networks the CPU beats MPS/GPU —
kernel-launch overhead dominates for tiny tensors). PyTorch is installed under
**Python 3.13**.

```bash
# true-parallel benchmarks (run as scripts; spawn-based MP must not be
# launched from inside a notebook). K defaults to 2; override with --Ks.
python3.13 Phase1_PINN_1D/dd_parallel_mp.py     --prob sin4  --Ks 2
python3.13 Phase2_PINN_2D/dd_parallel_mp_2d.py  --prob sin13 --Ks 2
```

The `.py` benchmarks print a `K · L2 · seq · par · speed-up` table for the
machine they run on. (Run from a terminal — `spawn`-based multiprocessing must
not be launched from a notebook cell that *defines* the worker.)

---

## Progress

| Phase | Task | Status |
|-------|------|--------|
| Phase 1 | PINN vs DD-PINN on 1D PDEs + true-parallel scaling | ✅ Complete |
| Phase 2 | PINN vs DD-PINN on 2D PDEs + true-parallel scaling | ✅ Complete |
| Next | Asynchronous/additive Schwarz (remove the round barrier); cluster scaling; 3D PINN; LPB / COSMO | ⏳ Upcoming |

---

## Key concepts implemented

- PDE residual minimization via automatic differentiation
- Exact (hard) boundary conditions via distance functions — pure residual loss
- Overlapping Schwarz domain decomposition with interface Dirichlet transmission
- Generalization from 2 to **K subdomains** (1D and 2D)
- **True multiprocess parallel training** (`torch.multiprocessing`, persistent
  workers, one process per subdomain) with measured speed-up vs. an identical
  sequential baseline
- Capacity- and budget-matched benchmarking; profiling of sequential, parallel,
  and inference time

---

## Tech stack

Python · PyTorch · NumPy · `torch.multiprocessing`

---

## References

1. Raissi, Perdikaris, Karniadakis — *Physics-Informed Neural Networks* (2019)
2. Wang, Sankaran, Wang, Perdikaris — *An Expert's Guide to Training PINNs* (2023)
