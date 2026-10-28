"""beta J = 32, dt = 0.053033, lambda = +-0.0025, mu = 0, t_pre = 2 beta: delta-G runs (and the vectorised plain-G scheme) next to
the production runs at the same settings. Outputs go to syk_sims/deltaG_test/ (outside kbe_runs, so the analysis index never sees them).

    python run_test.py [t_post] [runs]     runs: comma list from dG+,dG-,dG0,G+,G-   (default: all, t_post = 32 = beta)
"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_solver import reference, residual, toeplitz_majorana, evolve
from common import SIMS, OUT, PROD, EQ0, BETA, DT, T_PRE

OUT.mkdir(exist_ok=True)

t_post = float(sys.argv[1]) if len(sys.argv) > 1 else 32.0
runs = sys.argv[2].split(",") if len(sys.argv) > 2 else ["dG+", "dG-", "dG0", "G+", "G-"]
with np.load(EQ0) as z:
    omega, A0 = z["omega_real"], z["A"]
A = {}
for s, f in PROD.items():
    with np.load(f) as z:
        assert np.array_equal(z["omega_real"], omega)
        A[s] = z["A"]
A[0] = A0

t0 = time.time()
ref = reference(omega, A0, BETA, T_PRE, t_post, DT)
t, n0, g0, dg = ref
print(f"grid: Nt={len(t)}, n0={n0}, post steps={len(t) - 1 - n0}; reference {time.time() - t0:.1f} s", flush=True)
# spectral increment vs difference of the stored values
print(f"increment check: max|dg - diff(g0)| = {np.max(np.abs(dg[2:] - np.diff(g0)[1:])):.2e}, max|dg| = {np.max(np.abs(dg[1:])):.2e}", flush=True)
R = None
if any(r.startswith("dG") for r in runs):
    t0 = time.time(); R = residual(toeplitz_majorana(g0), DT, n0, dg)
    tR = time.time() - t0
    print(f"residual: {tR:.1f} s; max|R1| = {np.max(np.abs(R[0])):.2e}, median|R1| over post rows = {np.median(np.abs(R[0][n0 + 1:, :n0])):.2e}", flush=True)
tag = f"tpost_{t_post:g}"
for r in runs:
    mode, s = ("dG", r[2:]) if r.startswith("dG") else ("G", r[1:])
    lam = {"+": 1, "-": -1, "0": 0}[s]
    o = evolve(omega, A[lam], A0, BETA, T_PRE, t_post, DT, mode=mode, R=R, ref=ref, verbose=100)
    np.savez_compressed(OUT / f"b32_dt0p053_{tag}_{mode}_{'p' if lam > 0 else 'm' if lam < 0 else '0'}.npz",
                        t=o["t"], n0=o["n0"], X_post=o["X"][o["n0"]:, :], g0=o["g0"], iters=o["iters"], err=o["err"],
                        meta=json.dumps(dict(mode=mode, lam=0.0025 * lam, beta=BETA, dt=DT, t_pre=T_PRE, t_post=t_post, sec_per_step=o["sec_per_step"])))
    print(f"{r}: {o['sec_per_step']:.3f} s/step, iters median {np.median(o['iters'][n0 + 1:]):.0f} max {o['iters'][n0 + 1:].max()}, "
          f"final err max {np.nanmax(o['err'][n0 + 1:]):.1e}", flush=True)
