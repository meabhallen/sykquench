#!/usr/bin/env python3
"""Convert a stuck equilibrium .ckpt.npz directly into a converged eq_runs entry.

For a run whose delta_F residual converged long ago but whose d_ab residual
plateaus above an overly strict --dab-tol, so it loops to max_iter (or gets
requeued forever) without ever finishing -- and without ever landing a
manifest row, since a SIGUSR1/SIGTERM checkpoint save exits via SystemExit
before run_equilibrium_one's manifest write (see its docstring / the signal
handler in solve_equilibrium_greater_real_time).

Does NOT run any solver iterations -- you've already confirmed A/F_t in the
checkpoint are converged to your satisfaction, so this trusts them exactly as
saved rather than perturbing them with one more mixing update. The KBE step
(kbe-one, see syk_batch_tools.py's load of eq_file) only ever reads back
omega_real, A, mu, A_off, t_grid, Ggt_t from an eq npz -- never GR_w, K_R_w,
Ggt_off_t, or GRoff_w -- so this script does only the work needed to produce
those five fields, plus K_R_w/GR_w (cheap/free once Sigma_R_w exists) for
schema-completeness with normal eq_runs output. Specifically:
  - A, Ggt_t = -i*F_t: read straight from the checkpoint, no computation.
  - t_grid, omega_real: deterministic re-derivations from your run params.
  - A_off (needed only if mu != 0): requires exactly one forward transform
    of F_t to get Sigma_R_w -> a_w -> GRoff_w. mu == 0 -> A_off is exactly
    zero, no transform needed at all.
  - Ggt_off_t: one more transform, only run if mu != 0 (kept for schema
    completeness even though the KBE loader never reads it).

Usage:
    python finish_eq_checkpoint.py \
        --ckpt eq_runs/.../syk_eq_J2_0_J4_1_beta_36_..._<hash>.ckpt.npz \
        --J2 0 --J4 1 --beta 36 --mu 0 --dt 0.025 --omega-max 8 --Nw 4097 --tol 1e-10 \
        --out-dir eq_runs --manifest-dir eq_runs

The physical/grid parameters are the ones baked into the stuck file's own
name -- copy them straight out of it, e.g. for
    syk_eq_J2_0_J4_1_beta_36_mu_0_klam_0_kc_0_kcut_0_dt_0p025_om_8_Nw_4097_tol_1em10_<hash>.ckpt.npz
that's J2=0 J4=1 beta=36 mu=0 kernel-lambda=0 kernel-c=0 kernel-cutoff=0
dt=0.025 omega-max=8 Nw=4097 tol=1e-10 ('p' -> '.', 'm' -> '-').
--eta-ret isn't in the filename -- use whatever you passed to eq-one for
this run (default 1e-6 if you didn't set it).
"""
import argparse
import json
from pathlib import Path

import numpy as np

from syk_batch_tools import (
    atomic_savez_compressed,
    build_kernel_R_w,
    equilibrium_filename,
    omega_to_time,
    time_to_omega,
    update_manifest,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt", required=True, help="Path to the stuck .ckpt.npz.")
    p.add_argument("--J2", type=float, default=0.0)
    p.add_argument("--J4", type=float, required=True)
    p.add_argument("--beta", type=float, required=True)
    p.add_argument("--mu", type=float, default=0.0)
    p.add_argument("--dt", type=float, required=True)
    p.add_argument("--omega-max", type=float, default=8.0)
    p.add_argument("--Nw", type=int, default=4097)
    p.add_argument("--t-max", type=float, default=None)
    p.add_argument("--tol", type=float, required=True, help="Recorded for reference only -- not re-checked here.")
    p.add_argument("--eta-ret", type=float, default=1e-6)
    p.add_argument("--kernel-lambda", type=float, default=0.0)
    p.add_argument("--kernel-c", type=float, default=0.0)
    p.add_argument("--kernel-cutoff", type=float, default=None)
    p.add_argument("--out-dir", default="eq_runs")
    p.add_argument("--manifest-dir", default=None)
    p.add_argument("--keep-checkpoint", action="store_true", help="Don't delete the .ckpt.npz after a successful write.")
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else out_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "syk_eq_manifest.csv"

    with np.load(ckpt_path, allow_pickle=False) as ckpt:
        A = np.array(ckpt["A"], copy=True)
        F_t = np.array(ckpt["F_t"] if "F_t" in ckpt.files else ckpt["F_tau"], copy=True)
        it_done = int(np.asarray(ckpt["it_done"]).item())

    J2, J4, beta, mu = float(args.J2), float(args.J4), float(args.beta), float(args.mu)
    t_max = args.t_max if args.t_max is not None else max(80.0, 5.0 * beta)
    omega_real = np.linspace(-args.omega_max, args.omega_max, args.Nw)
    t_grid = np.arange(-t_max, t_max + 0.5 * args.dt, args.dt)
    nF = 1.0 / (np.exp(np.clip(beta * omega_real, -500, 500)) + 1.0)

    kernel_cutoff = args.kernel_cutoff
    if kernel_cutoff is None:
        kernel_cutoff = 0.5 * max(abs(J4), 1.0)
    K_R_w, kernel_cutoff = build_kernel_R_w(omega_real, J4, args.kernel_lambda, args.kernel_c, kernel_cutoff)

    Ggt_t = -1j * F_t

    print(f"Loaded checkpoint at it_done={it_done}")
    print(f"  F(0) = iG>(0) = {F_t[np.argmin(np.abs(t_grid))]} (should be approx +0.5)")
    print(f"  spectral sum = {np.trapz(A, omega_real) / (2 * np.pi):.6f} (should be approx 1)")
    print(f"  A evenness max |A(w)-A(-w)| = {np.max(np.abs(A - A[::-1])):.3e}")

    # Sigma_R_w (and hence GR_w) requires one forward transform of F_t --
    # needed regardless of mu, both for GR_w itself and, if mu != 0, as the
    # input to GRoff_w/A_off below.
    i0 = int(np.argmin(np.abs(t_grid)))
    theta = np.zeros_like(t_grid, dtype=float)
    theta[t_grid > 0] = 1.0
    theta[i0] = 0.5
    iSigma_gt_t = J2**2 * F_t + J4**2 * F_t**3
    iSigma_R_t = theta * (iSigma_gt_t + iSigma_gt_t[::-1])
    iSigma_R_w = time_to_omega(iSigma_R_t, t_grid, omega_real)
    Sigma_R_w = iSigma_R_w / 1j
    a_w = omega_real + 1j * args.eta_ret + K_R_w - Sigma_R_w
    GR_w = a_w / (a_w**2 - mu**2)

    if mu == 0.0:
        A_off = np.zeros_like(A)
        Ggt_off_t = np.zeros_like(Ggt_t)
        GRoff_w = np.zeros_like(A, dtype=complex)
    else:
        GRoff_w = -1j * mu / (a_w**2 - mu**2)
        A_off = 2.0 * np.real(GRoff_w)
        Ggt_off_w = (1.0 - nF) * A_off
        Ggt_off_t = omega_to_time(Ggt_off_w, omega_real, t_grid)
        print(f"  max |Goff^>| = {np.max(np.abs(Ggt_off_t)):.6f}")

    meta = {
        "kind": "equilibrium",
        "J2": J2, "J4": J4, "beta": beta, "mu": mu,
        "dt": float(args.dt), "omega_max": float(args.omega_max), "Nw": int(args.Nw),
        "t_max": float(t_max), "tol": float(args.tol),
        "dab_tol": None, "require_dab_convergence": False, "max_iter": None,
        "mixing": None, "eta_ret": float(args.eta_ret),
        "kernel_lambda": float(args.kernel_lambda), "kernel_c": float(args.kernel_c),
        "kernel_cutoff": float(kernel_cutoff),
        "enforce_even_A": True, "clip_negative_A": True, "normalize_A": True,
        "converged": True,
        "final_dab_sqrt_max": None,
        "note": "finished directly from a stuck checkpoint via finish_eq_checkpoint.py; "
                "d_ab was never re-checked, delta_F convergence trusted from the checkpoint as-is",
    }
    filename = equilibrium_filename(out_dir, meta)
    atomic_savez_compressed(
        filename,
        omega_real=omega_real, A=A, t_grid=t_grid, F_t=F_t, Ggt_t=Ggt_t,
        GR_w=GR_w, K_R_w=K_R_w, mu=np.array(mu),
        A_off=A_off, Ggt_off_t=Ggt_off_t, GRoff_w=GRoff_w,
        metadata_json=json.dumps(meta),
    )
    row = dict(meta)
    row.update({"filename": str(filename), "status": "ok"})
    update_manifest(manifest_path, row)

    if not args.keep_checkpoint:
        ckpt_path.unlink()

    print("Wrote:", filename)
    print("Manifest row added (status=ok) to:", manifest_path)
    print("Checkpoint removed:", not ckpt_path.exists())


if __name__ == "__main__":
    main()
