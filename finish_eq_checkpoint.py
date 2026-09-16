#!/usr/bin/env python3
"""Finish a stuck equilibrium .ckpt.npz into a proper converged eq_runs entry.

For a run whose delta_F residual converged long ago but whose d_ab residual
plateaus above an overly strict --dab-tol, so it loops to max_iter (or gets
requeued forever) without ever finishing -- and without ever landing a
manifest row, since a SIGUSR1/SIGTERM checkpoint save exits via SystemExit
before run_equilibrium_one's manifest write (see its docstring / the
signal handler in solve_equilibrium_greater_real_time).

This resumes the given .ckpt.npz directly (pointing at it explicitly, so the
usual hash-derived checkpoint lookup -- which would treat any changed
parameter, dab_tol included, as a different run and start over -- never
comes into play), with require_dab_convergence=False so it accepts on the
very next delta_F check with no further solving. Then writes the normal
output .npz plus a manifest row, status "ok", with the real final_dab_sqrt_max
recorded honestly (it just wasn't the convergence criterion for this row).

The physical/grid parameters (--J2 --J4 --beta --mu --dt --omega-max --Nw
--tol --kernel-lambda --kernel-c --kernel-cutoff) are the ones baked into the
stuck file's own name -- copy them straight out of it, e.g. for
    syk_eq_J2_0_J4_1_beta_36_mu_0_klam_0_kc_0_kcut_0_dt_0p025_om_8_Nw_4097_tol_1em10_<hash>.ckpt.npz
that's J2=0 J4=1 beta=36 mu=0 kernel-lambda=0 kernel-c=0 kernel-cutoff=0
dt=0.025 omega-max=8 Nw=4097 tol=1e-10 ('p' -> '.', 'm' -> '-').
--mixing/--eta-ret/--t-max aren't in the filename -- use whatever you passed
to eq-one for this run (defaults 0.05 / 1e-6 / None if you didn't).

Usage:
    python finish_eq_checkpoint.py \
        --ckpt eq_runs/syk_eq_J2_0_J4_1_beta_36_..._<hash>.ckpt.npz \
        --J2 0 --J4 1 --beta 36 --mu 0 --dt 0.025 --omega-max 8 --Nw 4097 --tol 1e-10 \
        --out-dir eq_runs
"""
import argparse
import json
from pathlib import Path

import numpy as np

from syk_batch_tools import (
    atomic_savez_compressed,
    equilibrium_filename,
    solve_equilibrium_greater_real_time,
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
    p.add_argument("--tol", type=float, required=True)
    p.add_argument("--dab-tol", type=float, default=1e-6, help="Recorded for reference only -- not the convergence criterion here.")
    p.add_argument("--mixing", type=float, default=0.05)
    p.add_argument("--eta-ret", type=float, default=1e-6)
    p.add_argument("--kernel-lambda", type=float, default=0.0)
    p.add_argument("--kernel-c", type=float, default=0.0)
    p.add_argument("--kernel-cutoff", type=float, default=None)
    p.add_argument("--out-dir", default="eq_runs")
    p.add_argument("--manifest-dir", default=None)
    args = p.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir = Path(args.manifest_dir) if args.manifest_dir else out_dir
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "syk_eq_manifest.csv"

    # Mirror run_equilibrium_one's own default resolution (syk_batch_tools.py
    # ~line 1814) so the manifest row records the actual cutoff used, since
    # solve_equilibrium_greater_real_time resolves None internally but does
    # not return the resolved value.
    kernel_cutoff = args.kernel_cutoff
    if kernel_cutoff is None:
        kernel_cutoff = 0.5 * max(abs(float(args.J4)), 1.0)

    (
        omega_real, A, t_grid, F_t, Ggt_t, GR_w, K_R_w, converged,
        final_dab_sqrt_max, A_off, Ggt_off_t, GRoff_w,
    ) = solve_equilibrium_greater_real_time(
        J2=args.J2, J4=args.J4, beta=args.beta, mu=args.mu,
        omega_max=args.omega_max, Nw=args.Nw, t_max=args.t_max, dt=args.dt,
        max_iter=10_000,  # irrelevant: converges on the very next delta_F check
        tol=args.tol,
        dab_tol=args.dab_tol,
        require_dab_convergence=False,  # accept on delta_F alone -- the whole point
        mixing=args.mixing,
        eta_ret=args.eta_ret,
        kernel_lambda=args.kernel_lambda,
        kernel_c=args.kernel_c,
        kernel_cutoff=kernel_cutoff,
        checkpoint_path=ckpt_path,  # explicit path -- bypasses hash-derived lookup
        checkpoint_every=10_000,
    )

    if not converged:
        raise RuntimeError(
            "Did not converge within 10000 resumed iterations -- delta_F was "
            "not actually below --tol yet. Increase max_iter in this script "
            "and rerun, or double-check --tol matches the original run."
        )

    meta = {
        "kind": "equilibrium",
        "J2": float(args.J2), "J4": float(args.J4), "beta": float(args.beta),
        "mu": float(args.mu), "dt": float(args.dt), "omega_max": float(args.omega_max),
        "Nw": int(args.Nw), "t_max": args.t_max, "tol": float(args.tol),
        "dab_tol": float(args.dab_tol), "require_dab_convergence": False,
        "max_iter": 10_000, "mixing": float(args.mixing), "eta_ret": float(args.eta_ret),
        "kernel_lambda": float(args.kernel_lambda), "kernel_c": float(args.kernel_c),
        "kernel_cutoff": float(kernel_cutoff),
        "enforce_even_A": True, "clip_negative_A": True, "normalize_A": True,
        "converged": True,
        "final_dab_sqrt_max": None if not np.isfinite(final_dab_sqrt_max) else float(final_dab_sqrt_max),
    }
    filename = equilibrium_filename(out_dir, meta)
    atomic_savez_compressed(
        filename,
        omega_real=omega_real, A=A, t_grid=t_grid, F_t=F_t, Ggt_t=Ggt_t,
        GR_w=GR_w, K_R_w=K_R_w, mu=np.array(float(args.mu)),
        A_off=A_off, Ggt_off_t=Ggt_off_t, GRoff_w=GRoff_w,
        metadata_json=json.dumps(meta),
    )
    row = dict(meta)
    row.update({"filename": str(filename), "status": "ok"})
    update_manifest(manifest_path, row)

    print(f"final_dab_sqrt_max = {final_dab_sqrt_max:.3e} (not the gate here, recorded for reference)")
    print("Wrote:", filename)
    print("Manifest row added (status=ok) to:", manifest_path)
    print("Original checkpoint removed:", not ckpt_path.exists())


if __name__ == "__main__":
    main()
