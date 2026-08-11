# Single SYK with an engineered kernel

Quick guide to running the single-SYK equilibrium solver and real-time KBE
evolution with the "engineered kernel" — a tuned deformation added to the
equilibrium self-energy that cancels the soft-mode (h=2) overlap, so the
prepared state is dominated by the first subleading operator (h1) instead.
The tuning itself (finding the right kernel couplings) is done separately in
`syk_matsubara_f00.ipynb` / `syk_matsubara_kernel.ipynb`; this doc only
covers *using* an already-tuned kernel to prepare a state and quench it.

Everything below drives one script, `syk_batch_tools.py`, which has two
subcommands: `eq-one` (equilibrium solve) and `kbe-one` (real-time KBE
evolution). All the bash scripts and `run_local.sh` are just parameter
sweeps / wrappers around these two.

## Requirements

`numpy`, `pandas`, `scipy`. Nothing exotic — any environment with those
three works, locally or on the cluster (`module load python/3.11.6-gcc-11.4.0`
on Lawrencium already provides them).

## Two ways to run

**Locally** (`run_local.sh`): runs one `eq-one` + one `kbe-one` pair directly
with `python3`, no SLURM. Good for testing a single (J4, beta, kernel) point
before committing to a cluster sweep. Edit the parameters at the top of the
file, then:

```bash
bash run_local.sh
# or, if your default python3 doesn't have numpy/pandas/scipy:
PYTHON=/path/to/python3 bash run_local.sh
```

**On the cluster** (`submit_eq.sh` + `submit_kbe.sh`): loop over arrays of
parameters (multiple betas, kernel tuples, etc.) and `sbatch` one job per
combination. Run on the Lawrencium login node:

```bash
bash submit_eq.sh    # step 1 — must run first
bash submit_kbe.sh    # step 2 — reads the manifest submit_eq.sh produced
```

## The workflow: equilibrium → KBE

1. **`eq-one` prepares the initial state.** It solves the finite-temperature
   equilibrium Green's function self-consistently for a given `(J4, beta)`,
   with the engineered kernel turned on (`kernel_lambda/c/cutoff` nonzero).
   Every converged run is recorded as a row in `eq_runs/syk_eq_manifest.csv`,
   with the `.npz` file it produced.

2. **`kbe-one` quenches it.** It loads one of those saved equilibrium files
   as the pre-quench state (`t<0`), then evolves it forward in real time
   under the Kadanoff-Baym equations as the coupling ramps `J4_i -> J4_f`.
   The key point: **the kernel used to select the initial state
   (`eq_kernel_lambda/c/cutoff`) is independent of the kernel actually
   present during the real-time evolution (`kernel_lambda/c/cutoff`)**. For
   normal quench production runs, the evolution kernel is always trivial —
   `0, 0, 0` — because the engineered deformation is only a trick for
   *preparing* the state; nothing forces it to remain on during the
   dynamics. So a typical `kbe-one` call sets `--kernel-lambda 0
   --kernel-c 0` but a *nonzero* `--eq-kernel-lambda/-c/-cutoff` matching
   whatever `eq-one` run you want to quench from.

   If `--eq-file` isn't given explicitly, `kbe-one` searches the eq
   manifest for a converged row matching `(J4_i, beta, eq_kernel_lambda,
   eq_kernel_c, eq_kernel_cutoff)` and picks the finest-resolution match.
   **This means step 1 must have actually produced a matching, converged
   equilibrium file before step 2 can find it** — same `J4`, same `beta`,
   same kernel triple.

## Parameter glossary

**Physical**
- `J4` (`J4_i`/`J4_f` for KBE) — the quartic SYK coupling. A quench ramps
  `J4_i -> J4_f`; `J4_i == J4_f` runs a stationarity check instead of a real
  quench.
- `beta` — inverse temperature of the initial equilibrium ensemble.
- `J2` (`J2_i`/`J2_f`) — optional quadratic coupling, default 0 (pure SYK4).

**Engineered kernel** — `kernel_lambda`, `kernel_c`, `kernel_cutoff`
- `kernel_lambda` — overall strength of the kernel deformation. `0` = off.
  Both signs (`+lambda`, `-lambda`) are typically needed together, since
  downstream analysis often reads off the odd-in-lambda response
  `D_lambda = (G_+ - G_-)/(2*lambda)`.
- `kernel_c` — the tuned coefficient (`delta_star`) that cancels the h=2
  overlap at a given cutoff. Comes out of `syk_matsubara_kernel.ipynb`, not
  a free parameter you pick by hand.
- `kernel_cutoff` (`Lambda`) — the energy cutoff of the kernel regulator
  `[Lambda/(Lambda - i*omega)]^4`. `kernel_c` and `kernel_cutoff` are a
  **paired tuple** — each tuned `Lambda` has its own `delta_star(Lambda)`.
  Don't loop over them as independent arrays; loop over `(c, cutoff)` pairs.
  Defaults to `0.5 * max(|J4|, 1.0)` if left unset and `lambda != 0`.
- For `kbe-one` specifically, the `eq_kernel_*` versions of these three
  select which equilibrium file to quench from (see workflow above), while
  the bare `kernel_*` ones are what's actually applied during the evolution.

**Numerical grid (eq-one)**
- `dt`, `omega_max`, `Nw` — the real-time/real-frequency grid the
  self-consistency loop runs on. Finer `dt` and larger `Nw`/`omega_max` cost
  more but resolve sharper features; `submit_eq.sh` derives these from
  `(J4, beta)` rather than hardcoding them (see the script for the exact
  formulas).
- `tol` — convergence threshold on the self-energy update `||dF(t)||`.
- `dab_tol` / `require_dab_convergence` — a stricter secondary convergence
  check; only enforced if `require_dab_convergence` is set.
- `mixing` — how much of each new iterate to mix in (smaller = more stable,
  slower).

**KBE timing (kbe-one)**
- `t_pre`/`t_post` or `t_pre_factor`/`t_post_factor` — how far back before
  the quench and forward after it to evolve, either as absolute times or as
  multiples of `beta`.
- `n_corr`, `corr_tol` — predictor-corrector iterations (and their
  tolerance) used to self-consistently solve each time step.
- `progress_every` — print a progress line every N time steps.

## Tuned kernel values used in this repo's current sweeps

From the large-`beta*J` converged Matsubara tuning:

| `kernel_lambda` | `kernel_c` | `kernel_cutoff` factor (of J4) |
|---|---|---|
| `+0.005` / `-0.005` | `-0.043936` | `0.65` |
| `+0.005` / `-0.005` | `-0.053648` | `0.75` |

These are the values hardcoded into `submit_eq.sh` / `submit_kbe.sh` /
`run_local.sh`. If you retune the kernel in `syk_matsubara_kernel.ipynb`,
update all three together.

## Outputs

- `eq_runs/syk_eq_manifest.csv` + one `.npz` per converged equilibrium run.
- `kbe_runs/syk_kbe_manifest.csv` + one `.npz` per KBE run, storing the
  post-quench Green's function rows plus a copy of the equilibrium reference
  it quenched from.

`run_local.sh` writes both under `./local_runs/`; the cluster scripts write
under `/global/scratch/users/$USER/sykquench/`.
