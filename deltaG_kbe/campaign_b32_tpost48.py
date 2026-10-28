"""beta J = 32 at the production dt = 0.0221, t_pre = 2 beta, t_post = 1.5 beta: delta-G runs at lambda = -+0.0025, -+0.00125 and 0.
The residual R (lambda-independent) is computed once and kept as its post-quench part only (R1[n > n0, :], R2[:, n > n0]).
Runs are sequential (about 5 GB each at Nt = 5069); a run whose output exists is skipped, so the script can be restarted.
Outputs: syk_sims/deltaG_test/b32_dt0p0221_tpost48/ (outside kbe_runs).

    python campaign_b32_tpost48.py [runs]     runs: comma list from m0025,p0025,m00125,p00125,0 (default: all, in that order)
"""
import sys, time, json
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from dg_solver import reference, residual, toeplitz_majorana, evolve

SIMS = Path("/Users/meabh/Desktop/CC Quench States/SYK Quenches/syk_sims")
OUT = SIMS / "deltaG_test" / "b32_dt0p0221_tpost48"
OUT.mkdir(parents=True, exist_ok=True)
BETA, DT, T_PRE, T_POST = 32.0, 0.022097086912079608, 64.0, 48.0
EQ0 = SIMS / "eq_runs/eq_J4_1_b_32_mu_0_lam_0_c_m0p5625_dt_0p0064_om_36_Nw_23041/syk_eq_J2_0_J4_1_beta_32_mu_0_klam_0_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_23041_tol_1em12_74d1bf7e.npz"
K = SIMS / "kbe_runs"
SRC = {   # A(omega) of the deformed equilibria, read from the production dt = 0.0221 KBE files (each carries its eq A)
    "p0025": (0.0025, "eqlam_0p0025"), "m0025": (-0.0025, "eqlam_m0p0025"), "p00125": (0.00125, "eqlam_0p00125"), "m00125": (-0.00125, "eqlam_m0p00125")}
RUNS = sys.argv[1].split(",") if len(sys.argv) > 1 else ["m0025", "p0025", "m00125", "p00125", "0"]


class Post:
    """R1[n, :n] / R2[:n, n] lookups on arrays holding only the post-quench rows / columns n > n0."""
    def __init__(self, a, n0, axis):
        self.a, self.n0, self.axis = a, n0, axis

    def __getitem__(self, idx):
        if self.axis == 0:
            n, sl = idx
            return self.a[n - self.n0 - 1, sl]
        sl, n = idx
        return self.a[sl, n - self.n0 - 1]


with np.load(EQ0) as z:
    omega, A0 = z["omega_real"], z["A"]
A = {"0": (0.0, A0)}
for key, (lam, tag) in SRC.items():
    f = [p for p in sorted((K / f"kbe_J4i_1_J4f_1_b_32_dt_0p022097086912079608_lam_0_c_0_{tag}_eqc_m0p5625_muf_0p0_eqmu_0").glob("*.npz"))
         if not p.name.endswith(".ckpt.npz")][0]
    with np.load(f, allow_pickle=False) as z:
        assert np.array_equal(z["omega_real"], omega)
        A[key] = (lam, z["A"])

t0 = time.time()
ref = reference(omega, A0, BETA, T_PRE, T_POST, DT)
t, n0, g0, dg = ref
Nt = len(t)
print(f"grid: Nt={Nt}, n0={n0}, post steps={Nt - 1 - n0}; reference {time.time() - t0:.1f} s", flush=True)

rfile = OUT / "R_post.npz"
if rfile.exists():
    with np.load(rfile) as z:
        R1p, R2p = z["R1p"], z["R2p"]
    print("residual loaded", flush=True)
else:
    t0 = time.time()
    R1, R2 = residual(toeplitz_majorana(g0), DT, n0, dg)
    R1p, R2p = R1[n0 + 1:, :].copy(), R2[:, n0 + 1:].copy()
    del R1, R2
    np.savez(rfile, R1p=R1p, R2p=R2p)
    print(f"residual: {time.time() - t0:.1f} s; max|R1| = {np.max(np.abs(R1p)):.2e}", flush=True)
R = (Post(R1p, n0, 0), Post(R2p, n0, 1))

for key in RUNS:
    lam, Aλ = A[key]
    out = OUT / f"b32_dt0p0221_tpost48_dG_{key}.npz"
    if out.exists():
        print(f"{key}: exists, skipped", flush=True)
        continue
    o = evolve(omega, Aλ, A0, BETA, T_PRE, T_POST, DT, mode="dG", R=R, ref=ref, verbose=200, corr_tol=1e-19, stagnation=True)
    np.savez_compressed(out, t=o["t"], n0=o["n0"], X_post=o["X"][o["n0"]:, :], g0=o["g0"], iters=o["iters"], err=o["err"],
                        meta=json.dumps(dict(mode="dG", lam=lam, beta=BETA, dt=DT, t_pre=T_PRE, t_post=T_POST, corr_tol=1e-19,
                                             stagnation=True, sec_per_step=o["sec_per_step"], eq0=str(EQ0))))
    print(f"{key} (lambda={lam:+g}): {o['sec_per_step']:.2f} s/step, iters median {np.median(o['iters'][n0 + 1:]):.0f} "
          f"max {o['iters'][n0 + 1:].max()}, final err max {np.nanmax(o['err'][n0 + 1:]):.1e}", flush=True)
    del o
print("done", flush=True)
