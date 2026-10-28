"""Loader for delta-G KBE outputs (kbe-one --delta-g): what the Gamma_S / Gamma_G analysis needs, read from dG directly.

A delta-G file stores dGgt_post = G - G_ref on the post-quench rows (all columns) and g0[k] = G_ref(k dt), the lambda = mu = 0
equilibrium (g0[0] = -i/2). Rebuilding G = G_ref + dG in double precision would throw away the lower round-off floor, so work
with dG: along a cut G(T + tau/2, T - tau/2) the reference is the constant G_ref(tau) = g0[k_tau], so

    cut_dG(T) = cut_G(T) - G_ref(tau)        (constant offset; derivatives, +-lambda differences and |D - C| fits are unaffected).

    run = load_run(path)
    x, dG = cut(run, tau_over_beta)          # x = T/beta, dG along the cut (same T grid and indexing as syk_cache.cut)
    x, G = cut(run, tau_over_beta, add_ref=True)   # the plain G cut (loses the floor gain)
"""
import json
from pathlib import Path

import numpy as np


def load_run(path):
    """dict with path, meta, beta, dt, lam (eq kernel lambda), mu (eq mu), t_row, t_col, n0, dG (post rows), g0, S (or None),
    corr_err, corr_iters, ref_hash."""
    with np.load(path, allow_pickle=False) as z:
        if "dGgt_post" not in z.files:
            raise ValueError(f"{path} is not a delta-G output (no dGgt_post); use syk_cache for plain runs.")
        meta = json.loads(str(z["metadata_json"]))
        n0 = int(z["n0"])
        run = dict(path=str(path), meta=meta, beta=float(meta["beta"]), dt=float(meta["dt"]),
                   lam=float(meta["eq_kernel_lambda"]), mu=float(meta.get("eq_mu") or 0.0), n0=n0,
                   t_row=np.array(z["t_row_post"]), t_col=np.array(z["t_col_all"]), dG=np.array(z["dGgt_post"]),
                   g0=np.array(z["g0"]), ref_hash=meta.get("ref_hash"),
                   corr_err=np.array(z["corr_final_err_post"]) if "corr_final_err_post" in z.files else None,
                   corr_iters=np.array(z["corr_iters_used_post"]) if "corr_iters_used_post" in z.files else None, S=None)
        if "Goff_post" in z.files:                     # S(t) = -2 Re Goff(t, t), as syk_cache does for mu > 0 runs
            idx = np.arange(z["Goff_post"].shape[0])
            run["S"] = -2.0 * np.asarray(z["Goff_post"])[idx, n0 + idx].real
    return run


def cut(run, tau_over_beta, add_ref=False):
    """(T/beta, dG(T + tau/2, T - tau/2)) on T = 0, dt, ..., with syk_cache's indexing; add_ref=True adds G_ref(tau)."""
    beta, dt, tr, tc = run["beta"], run["dt"], run["t_row"], run["t_col"]
    tau = tau_over_beta * beta
    T = np.arange(0.0, tr[-1] - tau / 2, dt)
    i = np.clip(np.rint((T + tau / 2 - tr[0]) / dt).astype(int), 0, run["dG"].shape[0] - 1)
    j = np.clip(np.rint((T - tau / 2 - tc[0]) / dt).astype(int), 0, run["dG"].shape[1] - 1)
    out = run["dG"][i, j]
    if add_ref:
        k = (i + run["n0"]) - j                      # full-grid index difference: constant along the cut
        g0 = run["g0"]
        out = out + np.where(k >= 0, g0[np.abs(k)], -np.conj(g0[np.abs(k)]))
    return T / beta, out
