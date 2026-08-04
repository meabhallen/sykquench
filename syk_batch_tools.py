"""
Batch tools for single-SYK equilibrium and real-time KBE runs.

This file is designed so each step can be run separately on a cluster:

    python syk_batch_tools.py eq-one  --J4 1.0 --beta 30 --dt 0.025 --omega-max 8 --Nw 4097 --tol 1e-10
    python syk_batch_tools.py kbe-one --J4-i 1.0 --J4-f 1.0 --beta 30 --dt 0.025 --t-pre-factor 2 --t-post-factor 1 --n-corr 8 --corr-tol 1e-10

Or import it in a notebook / job script:

    from syk_batch_tools import run_equilibrium_one, run_kbe_one, run_equilibrium_grid, run_kbe_grid

The filenames include protocol parameters plus a short hash. CSV manifests record all metadata.

KBE output uses a compact format: it saves only post-quench rows G^>(t1,t2) with
 t1 >= 0, the full t2 axis, and one copy of the equilibrium relative-time GF.
The purely pre-quench block is redundant and can be reconstructed as G_eq(t1-t2).
"""

from __future__ import annotations

from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import os
import signal
import traceback
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Compatibility with newer NumPy: np.trapz was removed; np.trapezoid is the replacement.
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
import pandas as pd
from scipy.interpolate import interp1d


# ============================================================
# General helper utilities
# ============================================================

def sf(x: Any) -> str:
    """Safe float-ish string for filenames."""
    try:
        return f"{float(x):.12g}".replace("-", "m").replace(".", "p")
    except Exception:
        return str(x).replace("-", "m").replace(".", "p").replace("/", "_")


def param_hash(params: Dict[str, Any], n: int = 8) -> str:
    """Short stable hash of a parameter dictionary."""
    s = json.dumps(params, sort_keys=True, default=str)
    return hashlib.sha1(s.encode()).hexdigest()[:n]


def atomic_savez_compressed(filename: os.PathLike | str, **kwargs: Any) -> None:
    """Save an npz atomically by writing a temporary file then moving it into place."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + ".tmp")
    # Use an open file handle so numpy does not append an extra .npz extension.
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **kwargs)
    os.replace(tmp, filename)


CHECKPOINT_SUFFIX = ".ckpt.npz"


def checkpoint_path_for(npz_path: os.PathLike | str) -> Path:
    """Derive the checkpoint path for a given output npz path.

    Strips .npz then appends .ckpt.npz to avoid double extensions.
    e.g. /runs/syk_eq_abc.npz  ->  /runs/syk_eq_abc.ckpt.npz
    """
    p = Path(npz_path)
    stem = p.name[:-4] if p.name.endswith(".npz") else p.name
    return p.parent / (stem + CHECKPOINT_SUFFIX)


def atomic_savez_checkpoint(path: os.PathLike | str, **arrays: Any) -> None:
    """Save a checkpoint npz atomically (write tmp then rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)


def update_manifest(manifest_path: os.PathLike | str, row: Dict[str, Any]) -> None:
    """Append/update one row in a CSV manifest, keyed by filename."""
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame([row])
    if manifest_path.exists():
        old = pd.read_csv(manifest_path)
        out = pd.concat([old, new], ignore_index=True)
        out = out.drop_duplicates(subset=["filename"], keep="last")
    else:
        out = new
    out.to_csv(manifest_path, index=False)


def load_manifest(path: os.PathLike | str) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def grid_from_dict(d: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """Convert dict of parameter lists into a list of parameter dictionaries."""
    keys = list(d.keys())
    vals = [list(d[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def infer_dt_from_t(t: np.ndarray) -> float:
    t = np.asarray(t).squeeze()
    return float(np.median(np.diff(t)))


# ============================================================
# Real-time / real-frequency equilibrium self-consistency
# ============================================================

def _trapz_weights(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    dx = x[1] - x[0]
    w = np.ones_like(x, dtype=float) * dx
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def omega_to_time(X_w: np.ndarray, omega: np.ndarray, t: np.ndarray, chunk: int = 512) -> np.ndarray:
    """x(t) = int dω/(2π) exp(-iωt) X(ω)."""
    omega = np.asarray(omega)
    t = np.asarray(t)
    X_w = np.asarray(X_w)
    w_omega = _trapz_weights(omega) / (2 * np.pi)
    X_weighted = X_w * w_omega
    out = np.empty(len(t), dtype=complex)
    for a in range(0, len(t), chunk):
        tt = t[a : a + chunk]
        phase = np.exp(-1j * np.outer(tt, omega))
        out[a : a + chunk] = phase @ X_weighted
    return out


def time_to_omega(x_t: np.ndarray, t: np.ndarray, omega: np.ndarray, chunk: int = 512) -> np.ndarray:
    """X(ω) = int dt exp(+iωt) x(t)."""
    omega = np.asarray(omega)
    t = np.asarray(t)
    x_t = np.asarray(x_t)
    w_t = _trapz_weights(t)
    x_weighted = x_t * w_t
    out = np.empty(len(omega), dtype=complex)
    for a in range(0, len(omega), chunk):
        ww = omega[a : a + chunk]
        phase = np.exp(1j * np.outer(ww, t))
        out[a : a + chunk] = phase @ x_weighted
    return out


def trapz_weights(x: np.ndarray) -> np.ndarray:
    dx = x[1] - x[0]
    w = np.ones_like(x, dtype=float) * dx
    w[0] *= 0.5
    w[-1] *= 0.5
    return w


def real_spectral_to_imag_time(
    omega: np.ndarray,
    A: np.ndarray,
    beta: float,
    Ntau: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert real-frequency spectral function A(omega) into Euclidean G_E(tau).

    Uses G_E(tau) = int dω/(2π) exp(-ω tau)/(1 + exp(-βω)) A(ω).
    """
    omega = np.asarray(omega)
    A = np.asarray(A)
    tauE = (np.arange(Ntau) + 0.5) * beta / Ntau
    w_omega = trapz_weights(omega) / (2 * np.pi)
    W = omega[None, :]
    T = tauE[:, None]
    log_kernel = -W * T - np.logaddexp(0.0, -beta * W)
    kernel = np.exp(log_kernel)
    G_tauE = kernel @ (A * w_omega)
    return tauE, G_tauE


def _init_eq_A_F(
    omega_real: np.ndarray,
    nF: np.ndarray,
    t_grid: np.ndarray,
    J2: float,
    J4: float,
    A_init: Optional[np.ndarray],
    omega_init: Optional[np.ndarray],
    G_t_init: Optional[np.ndarray],
    t_init: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    """Initialise A and F_t for the equilibrium solver."""
    if G_t_init is not None and t_init is not None:
        dt_init = t_init[1] - t_init[0]
        phase = np.exp(1j * np.outer(omega_real, t_init))
        Ggt_w = np.real((phase @ (-G_t_init)) * dt_init)
        denom = np.clip(1.0 - nF, 1e-6, None)
        A = Ggt_w / denom
    elif A_init is not None:
        A = np.interp(omega_real, omega_init if omega_init is not None else omega_real, A_init)
    else:
        width0 = max(J2, J4, 0.5)
        A = 2.0 * width0 / (omega_real**2 + width0**2)

    A = 0.5 * (A + A[::-1])
    A = np.clip(A, 0.0, None)
    sum_A = np.trapz(A, omega_real) / (2 * np.pi)
    if sum_A > 0:
        A /= sum_A
    else:
        width0 = max(J2, J4, 0.5)
        A = 2.0 * width0 / (omega_real**2 + width0**2)
        A = 0.5 * (A + A[::-1])
        A = np.clip(A, 0.0, None)
        A /= np.trapz(A, omega_real) / (2 * np.pi)

    F_w = (1.0 - nF) * A
    F_t = omega_to_time(F_w, omega_real, t_grid)
    return A, F_t


def build_kernel_R_w(
    omega: np.ndarray,
    J4: float,
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
) -> Tuple[np.ndarray, float]:
    """Causal retarded realization of the regulated tuned-kernel source.

        K^R(w) = lambda [w^3/J4^2 + c w] R^R_Lambda(w),
        R^R_Lambda(w) = [Lambda/(Lambda - i w)]^4.

    R(0)=1, its only pole is at w=-i Lambda (lower half plane), it decays
    as 1/w^4, and K^R(-w)=-conj[K^R(w)] as required for a Majorana
    quadratic kernel. This is a stable effective-action regulator; a later
    Hamiltonian/ancilla realization would need its own retuning.

    Shared by the equilibrium solver and the KBE evolver so both always use
    an identical kernel for a given (kernel_lambda, kernel_c, kernel_cutoff).
    Returns (K_R_w, kernel_cutoff) with kernel_cutoff resolved to a float.
    """
    J_kernel = abs(float(J4))
    if kernel_lambda != 0.0 and J_kernel == 0.0:
        raise ValueError("The cubic tuned kernel requires nonzero J4.")
    if kernel_cutoff is None:
        kernel_cutoff = 4.0 * max(J_kernel, 1.0)
    kernel_cutoff = float(kernel_cutoff)
    if kernel_cutoff <= 0.0:
        raise ValueError("kernel_cutoff must be positive.")

    if kernel_lambda == 0.0:
        K_R_w = np.zeros_like(omega, dtype=complex)
    else:
        regulator_R = (kernel_cutoff / (kernel_cutoff - 1j * omega)) ** 4
        K_R_w = float(kernel_lambda) * (
            omega**3 / J_kernel**2 + float(kernel_c) * omega
        ) * regulator_R
    return K_R_w, kernel_cutoff


def build_kernel_R_mat(
    t: np.ndarray, omega: np.ndarray, K_R_w: np.ndarray
) -> Optional[np.ndarray]:
    """Causal (lower-triangular) real-time kernel K^R(t_i, t_j) = K^R(t_i - t_j)

    for t_i >= t_j, built by Fourier-transforming K_R_w onto the needed
    relative-time range and evaluating at each grid pair by interpolation.
    Returns None (fast path) if K_R_w is identically zero, i.e. kernel off.
    """
    if not np.any(K_R_w):
        return None
    t = np.asarray(t)
    rel_max = float(t[-1] - t[0])
    t_rel_1d = np.linspace(0.0, rel_max, 4 * len(t) + 1)
    K_R_rel_1d = omega_to_time(K_R_w, omega, t_rel_1d)
    K_R_of_rel = interp1d(t_rel_1d, K_R_rel_1d, kind="cubic", fill_value="extrapolate")
    t_rel_mat = t[:, None] - t[None, :]
    K_R_mat = np.zeros_like(t_rel_mat, dtype=complex)
    lower = t_rel_mat >= 0.0
    K_R_mat[lower] = K_R_of_rel(t_rel_mat[lower])
    return K_R_mat


def _build_kernel_R_t(t: np.ndarray, omega: np.ndarray, K_R_w: np.ndarray) -> np.ndarray:
    """Causal retarded kernel K^R(t) on a relative-time grid t (zero for t<0).

    Same Fourier-transform + interpolation as build_kernel_R_mat, but returns
    only the 1D causal trace instead of the full O(Nt^2) pairwise-difference
    matrix -- what's needed for a time-translation-invariant (equilibrium)
    residual check.
    """
    if not np.any(K_R_w):
        return np.zeros_like(t, dtype=complex)
    t = np.asarray(t)
    rel_max = float(t[-1] - t[0])
    t_rel_1d = np.linspace(0.0, rel_max, 4 * len(t) + 1)
    K_R_rel_1d = omega_to_time(K_R_w, omega, t_rel_1d)
    K_R_of_rel = interp1d(t_rel_1d, K_R_rel_1d, kind="cubic", fill_value="extrapolate")
    K_R_t = np.zeros_like(t, dtype=complex)
    causal = t >= 0.0
    K_R_t[causal] = K_R_of_rel(t[causal])
    return K_R_t


def solve_equilibrium_greater_real_time(
    J2: float,
    J4: float,
    beta: float,
    omega_max: float = 8.0,
    Nw: int = 4097,
    t_max: Optional[float] = None,
    dt: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-9,
    dab_tol: float = 1e-6,
    require_dab_convergence: bool = False,
    mixing: float = 0.05,
    eta_ret: float = 1e-6,
    enforce_even_A: bool = True,
    clip_negative_A: bool = True,
    normalize_A: bool = True,
    verbose_every: int = 50,
    compute_kbe_dab_every: int = 50,
    kbe_dab_t_cut: Optional[float] = None,
    kbe_dab_edge_skip: int = 4,
    kbe_dab_power: float = 0.5,
    omega_init: Optional[np.ndarray] = None,
    A_init: Optional[np.ndarray] = None,
    t_init: Optional[np.ndarray] = None,
    G_t_init: Optional[np.ndarray] = None,
    # ── Tuned quadratic source ────────────────────────────────────────────────
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_path: Optional[Path] = None,  # path for .ckpt.npz; None = no checkpointing
    checkpoint_every: int = 100,             # iterations between checkpoint saves
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, bool, float]:
    """
    Returns: omega_real, A, t_grid, F_t, Ggt_t, GR_w, K_R_w, converged, final_dab_sqrt_max.

    Convention: F_t = i G^>(t), Ggt_t = G^>(t) = -i F_t.

    Convergence is a dual gate, matching the MQ/KKTZ equilibrium solvers:
    the mixing residual delta_F must be below tol, and -- only if
    require_dab_convergence is set -- the real-time KBE self-consistency
    residual d_ab (from calc_kbe_d_ab_syk_equilibrium) must also have
    max(d_ab**kbe_dab_power) below dab_tol. d_ab is (re)computed every
    compute_kbe_dab_every iterations and whenever delta_F first passes tol.

    Checkpoint files are saved to checkpoint_path (a .ckpt.npz alongside the
    output npz) every checkpoint_every iterations. If checkpoint_path exists
    on entry the solver resumes from that iteration. SIGUSR1/SIGTERM save the
    last fully completed iteration before exiting. The checkpoint is deleted
    only after actual convergence.
    """
    if t_max is None:
        t_max = max(80.0, 5.0 * beta)

    omega_real = np.linspace(-omega_max, omega_max, Nw)
    t_grid   = np.arange(-t_max, t_max + 0.5 * dt, dt)
    nF         = 1.0 / (np.exp(np.clip(beta * omega_real, -500, 500)) + 1.0)

    # Causal retarded realization of the regulated source; see build_kernel_R_w
    # for the definition. Shared with evolve_syk4_kbe so eq and (optional)
    # KBE runs always agree on the kernel for the same parameters.
    K_R_w, kernel_cutoff = build_kernel_R_w(
        omega_real, J4, kernel_lambda, kernel_c, kernel_cutoff
    )

    # ── Initialise or resume ─────────────────────────────────────────────────
    it_start = 0
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ckpt     = np.load(checkpoint_path, allow_pickle=True)
            A        = np.asarray(ckpt["A"])
            # "F_t" is the current key; "F_tau" is accepted for checkpoints
            # written by the pre-rename version of this script.
            F_t      = np.asarray(ckpt["F_t"] if "F_t" in ckpt.files else ckpt["F_tau"])
            it_start = int(ckpt["it_done"]) + 1
            print(f"Resumed equilibrium solver from checkpoint at iteration {it_start}.")
        except Exception as e:
            print(f"Warning: could not load eq checkpoint ({e}), starting fresh.")
            it_start = 0
            A, F_t = _init_eq_A_F(
                omega_real, nF, t_grid, J2, J4,
                A_init, omega_init, G_t_init, t_init,
            )
    else:
        A, F_t = _init_eq_A_F(
            omega_real, nF, t_grid, J2, J4,
            A_init, omega_init, G_t_init, t_init,
        )

    # Keep references to the last fully completed iteration. The signal handler
    # saves these arrays immediately when Slurm sends SIGUSR1/SIGTERM. Because
    # the iteration below replaces A and F_t rather than mutating them in
    # place, these references remain internally consistent during an interrupt.
    checkpoint_state: Dict[str, Any] = {
        "A": A,
        "F_t": F_t,
        "it_done": it_start - 1,
    }

    def _checkpoint_then_exit(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        if checkpoint_path is not None and checkpoint_state["it_done"] >= 0:
            atomic_savez_checkpoint(
                checkpoint_path,
                A=checkpoint_state["A"],
                F_t=checkpoint_state["F_t"],
                it_done=np.array(checkpoint_state["it_done"]),
            )
            print(
                f"\n[{signal_name}] Equilibrium checkpoint saved at "
                f"iter={checkpoint_state['it_done']}; exiting for requeue.",
                flush=True,
            )
        else:
            print(
                f"\n[{signal_name}] No completed equilibrium iteration to checkpoint; exiting.",
                flush=True,
            )
        # SystemExit is deliberately not caught by run_equilibrium_one's
        # `except Exception`, so an interrupted run is not marked as failed.
        raise SystemExit(128 + signum)

    for checkpoint_signal in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(checkpoint_signal, _checkpoint_then_exit)

    i0    = int(np.argmin(np.abs(t_grid)))
    theta = np.zeros_like(t_grid, dtype=float)
    theta[t_grid > 0] = 1.0
    theta[i0]           = 0.5

    print("\nReal-time equilibrium self-consistency")
    print(f"J2={J2}, J4={J4}, beta={beta}")
    print(
        "kernel: "
        f"lambda={kernel_lambda}, c={kernel_c}, "
        f"Lambda={kernel_cutoff}"
    )
    print(
        "kernel Majorana-symmetry error =",
        np.max(np.abs(K_R_w[::-1] + np.conj(K_R_w))),
    )
    print(f"omega in [{omega_real[0]:.2f}, {omega_real[-1]:.2f}], Nw={Nw}")
    print(f"t in [{t_grid[0]:.2f}, {t_grid[-1]:.2f}], Nt={len(t_grid)}, dt={dt}")
    if it_start > 0:
        print(f"(Resuming from iteration {it_start})")
    print(f"{'Iter':>5}  {'||dF(t)||':>13}  {'F(0)':>14}  {'sum A':>10}  {'min Araw':>11}  {'max A':>11}")
    print("-" * 76)

    GR_w = None
    converged = False
    last_dab = None
    last_dab_sqrt_max = np.inf
    for it in range(it_start, max_iter):
        iSigma_gt_t = J2**2 * F_t + J4**2 * F_t**3
        iSigma_R_t  = theta * (iSigma_gt_t + iSigma_gt_t[::-1])
        iSigma_R_w    = time_to_omega(iSigma_R_t, t_grid, omega_real)
        Sigma_R_w     = iSigma_R_w / (1j)
        GR_w          = 1.0 / (
            omega_real + 1j * eta_ret + K_R_w - Sigma_R_w
        )
        A_raw         = -2.0 * np.imag(GR_w)
        A_new         = A_raw.copy()
        if enforce_even_A:
            A_new = 0.5 * (A_new + A_new[::-1])
        if clip_negative_A:
            A_new = np.clip(A_new, 0.0, None)
        sum_A = np.trapz(A_new, omega_real) / (2 * np.pi)
        if normalize_A:
            if not np.isfinite(sum_A) or sum_A <= 0:
                raise RuntimeError(f"Bad spectral sum at iter {it}: {sum_A}")
            A_new = A_new / sum_A
            sum_A = np.trapz(A_new, omega_real) / (2 * np.pi)

        F_w_new   = (1.0 - nF) * A_new
        F_t_new = omega_to_time(F_w_new, omega_real, t_grid)
        delta_F   = np.mean(np.abs(F_t_new - F_t))
        F_t     = (1.0 - mixing) * F_t + mixing * F_t_new
        A         = A_new
        Ggt_current = -1j * F_t

        # Mark this as the latest fully completed, restart-safe iteration.
        checkpoint_state["A"] = A
        checkpoint_state["F_t"] = F_t
        checkpoint_state["it_done"] = it

        delta_ok = delta_F < tol
        need_dab = False
        if compute_kbe_dab_every and (it % compute_kbe_dab_every == 0 or delta_ok):
            need_dab = True
        if require_dab_convergence and delta_ok:
            need_dab = True

        if need_dab:
            last_dab = calc_kbe_d_ab_syk_equilibrium(
                t=t_grid,
                Ggt=Ggt_current,
                J2=J2,
                J4=J4,
                kernel_lambda=kernel_lambda,
                kernel_c=kernel_c,
                kernel_cutoff=kernel_cutoff,
                omega=omega_real,
                t_cut=kbe_dab_t_cut,
                edge_skip=kbe_dab_edge_skip,
                return_details=False,
            )
            last_dab_sqrt_max = float(last_dab ** kbe_dab_power)

        if it % verbose_every == 0 or delta_ok:
            msg = (
                f"{it:5d}  {delta_F:13.3e}  {F_t[i0].real:14.8f}  "
                f"{sum_A:10.6f}  {np.nanmin(A_raw):11.3e}  {np.nanmax(A):11.3e}"
            )
            if last_dab is not None and need_dab:
                msg += f"  KBE d_ab^{kbe_dab_power:g}={last_dab_sqrt_max:.3e} tol={dab_tol:.3e}"
            if require_dab_convergence:
                msg += f"  require_dab={require_dab_convergence}"
            print(msg)

        # ── Checkpoint ────────────────────────────────────────────────────────
        if checkpoint_path is not None and (it + 1) % checkpoint_every == 0:
            atomic_savez_checkpoint(
                checkpoint_path,
                A=A,
                F_t=F_t,
                it_done=np.array(it),
                dab_tol=dab_tol,
                require_dab_convergence=bool(require_dab_convergence),
                last_dab_sqrt_max=last_dab_sqrt_max,
            )
            print(f"  [eq checkpoint saved at iter={it}]")

        if delta_ok:
            dab_ok = (not require_dab_convergence) or (last_dab_sqrt_max < dab_tol)
            if dab_ok:
                converged = True
                print(f"\nConverged at iteration {it}.")
                break
    else:
        print(f"\nDid not converge within {max_iter} iterations.")

    # Delete restart state only after genuine convergence. If max_iter is
    # reached, save and retain the latest completed iteration rather than
    # silently discarding it.
    if converged and checkpoint_path is not None and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("Eq checkpoint removed (converged).")
    elif not converged and checkpoint_path is not None and checkpoint_state["it_done"] >= 0:
        atomic_savez_checkpoint(
            checkpoint_path,
            A=checkpoint_state["A"],
            F_t=checkpoint_state["F_t"],
            it_done=np.array(checkpoint_state["it_done"]),
        )
        print(
            f"Eq checkpoint retained at iter={checkpoint_state['it_done']} "
            f"(not converged): {checkpoint_path}"
        )

    F_w     = (1.0 - nF) * A
    F_t   = omega_to_time(F_w, omega_real, t_grid)
    Ggt_t = -1j * F_t

    print("\nFinal checks:")
    print("  F(0) = iG>(0) =", F_t[i0])
    print("  should be approx +0.5")
    print("  spectral sum =", np.trapz(A, omega_real) / (2 * np.pi))
    print("  A evenness max |A(w)-A(-w)| =", np.max(np.abs(A - A[::-1])))
    print("  max |G>| =", np.max(np.abs(Ggt_t)))
    print(f"  converged = {converged}, final dab^{kbe_dab_power:g} = {last_dab_sqrt_max:.3e}")

    return omega_real, A, t_grid, F_t, Ggt_t, GR_w, K_R_w, converged, last_dab_sqrt_max


# ============================================================
# Single SYK4 KBE quench: J4_i -> J4_f, J2_i -> J2_f
# ============================================================

def greater_from_spectral(omega: np.ndarray, A: np.ndarray, beta: float, t_eval: np.ndarray) -> np.ndarray:
    """Build equilibrium G^>(t) from A(omega)."""
    omega = np.asarray(omega)
    A = np.asarray(A)
    dw = omega[1] - omega[0]
    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    Ggt_omega = -1j * (1.0 - nF) * A
    Gt = np.array([
        np.sum(Ggt_omega * np.exp(-1j * omega * t)) * dw / (2 * np.pi)
        for t in t_eval
    ], dtype=complex)
    return Gt


def trap_weights(m: int, dt: float) -> np.ndarray:
    w = np.ones(m, dtype=float) * dt
    w[0] *= 0.5
    if m > 1:
        w[-1] *= 0.5
    return w


def sigma_greater_syk4(G: np.ndarray, J2_of_t: np.ndarray, J4_of_t: np.ndarray) -> np.ndarray:
    """Sigma^> = J2(t1)J2(t2) G^> - J4(t1)J4(t2) [G^>]^3."""
    JJ2 = J2_of_t[:, None] * J2_of_t[None, :]
    JJ4 = J4_of_t[:, None] * J4_of_t[None, :]
    return JJ2 * G - JJ4 * G**3


def rhs_t1(
    G: np.ndarray, S: np.ndarray, i: int, j: int, dt: float,
    K_R_mat: Optional[np.ndarray] = None,
) -> complex:
    """KBE RHS for d_{t1} G^>(t1,t2).

    K_R_mat, if given, is the causal (lower-triangular) real-time kernel
    K^R(t_a,t_b) built from the tuned frequency-space kernel. It has no
    greater/lesser component (it's a fixed c-number regulator, not a
    fluctuating self-energy), so it only ever enters alongside Sigma_R here,
    never alongside Sigma^> in the I2 term below. Sign: minus, matching the
    equilibrium solver's GR_w = 1/(omega + i*eta + K_R_w - Sigma_R_w), i.e.
    Sigma_R_total = Sigma_R - K_R (see calc_kbe_d_ab_syk_equilibrium).
    """
    k = np.arange(i + 1)
    w = trap_weights(len(k), dt)
    Sigma_R = S[i, k] + S[k, i]
    if K_R_mat is not None:
        Sigma_R = Sigma_R - K_R_mat[i, k]
    I1 = np.sum(w * Sigma_R * G[k, j])

    k = np.arange(j + 1)
    w = trap_weights(len(k), dt)
    G_A = -(G[k, j] + G[j, k])
    I2 = np.sum(w * S[i, k] * G_A)
    return -1j * (I1 + I2)


def rhs_t2(
    G: np.ndarray, S: np.ndarray, i: int, j: int, dt: float,
    K_R_mat: Optional[np.ndarray] = None,
) -> complex:
    """KBE RHS for d_{t2} G^>(t1,t2).

    K^A(t_a,t_b) = conj(K^R(t_b,t_a)) is the standard retarded/advanced
    conjugate relation; with K_R_mat[j,k] valid for j >= k it supplies
    exactly the K^A(t_k,t_j) contribution needed here. Sign: minus, matching
    rhs_t1 (Sigma_A_total = Sigma_A - K^A; see calc_kbe_d_ab_syk_equilibrium).
    """
    k = np.arange(i + 1)
    w = trap_weights(len(k), dt)
    G_R = G[i, k] + G[k, i]
    I1 = np.sum(w * G_R * S[k, j])

    k = np.arange(j + 1)
    w = trap_weights(len(k), dt)
    Sigma_A = -(S[k, j] + S[j, k])
    if K_R_mat is not None:
        Sigma_A = Sigma_A - np.conj(K_R_mat[j, k])
    I2 = np.sum(w * G[i, k] * Sigma_A)
    return +1j * (I1 + I2)


def enforce_majorana_slice(G: np.ndarray, n: int) -> None:
    """Majorana consistency G^>(t_j,t_n) = -conj[G^>(t_n,t_j)] and G(t,t)=-i/2."""
    for j in range(n):
        r = 0.5 * (G[n, j] - np.conj(G[j, n]))
        G[n, j] = r
        G[j, n] = -np.conj(r)
    G[n, n] = -0.5j


def _init_G(Nt: int, n0: int, G_eq, t: np.ndarray) -> np.ndarray:
    """Initialise the full G matrix from the equilibrium GF interpolant."""
    G = np.zeros((Nt, Nt), dtype=complex)
    for i in range(n0 + 1):
        for j in range(n0 + 1):
            G[i, j] = G_eq(t[i] - t[j])
    for i in range(n0 + 1):
        G[i, i] = -0.5j
        for j in range(i):
            G[i, j] = -np.conj(G[j, i])
    return G


def evolve_syk4_kbe(
    omega: np.ndarray,
    A: np.ndarray,
    beta_i: float,
    J2_i: float,
    J2_f: float,
    J4_i: float,
    J4_f: float,
    t_pre: float,
    t_post: float,
    dt: float = 0.05,
    n_corr: int = 4,
    corr_tol: float = 1e-10,
    iterations: int = 50,
    return_diagnostics: bool = False,
    # ── Tuned quadratic source (static: same K throughout the run) ─────────────
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    # ── Checkpointing ─────────────────────────────────────────────────────────
    checkpoint_path: Optional[Path] = None,  # path for .ckpt.npz; None = no checkpointing
    checkpoint_every: int = 50,              # time steps between checkpoint saves
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    KBE evolution on a square real-time grid.

    Checkpoint files are saved to checkpoint_path every checkpoint_every steps.
    If checkpoint_path exists on entry the evolution resumes from that step.
    The checkpoint is deleted automatically on successful completion.

    The tuned kernel (kernel_lambda/c/cutoff) is static: present throughout
    the whole run if nonzero, absent throughout if left at the default zero.
    It is NOT quenched by J4_i -> J4_f. The two intended uses are:
      - Quench (default): leave kernel_lambda=0 here. The deformation only
        shaped the *initial* state via the eq-one run this was seeded from;
        pure SYK governs the relaxation being fit for Delta.
      - Equilibrium/stationarity check: pass the SAME kernel_lambda/c/cutoff
        used to build the eq file, with J4_i == J4_f, to verify the deformed
        background is truly stationary under its own real-time equations.
    """
    if kernel_lambda != 0.0 and J4_i != J4_f:
        print(
            "WARNING: kernel_lambda != 0 with J4_i != J4_f. The tuned kernel "
            "is static (not quenched) -- it will be present identically "
            "before and after t=0. This combination has no clean physical "
            "reading as either a pure quench or a stationarity check; make "
            "sure that's really what you want."
        )
    K_R_w, kernel_cutoff = build_kernel_R_w(
        omega, J4_i, kernel_lambda, kernel_c, kernel_cutoff
    )

    t = np.arange(-t_pre, t_post + 0.5 * dt, dt)
    n0 = int(np.argmin(np.abs(t)))
    t[n0] = 0.0
    Nt = len(t)

    K_R_mat = build_kernel_R_mat(t, omega, K_R_w)

    t_rel_max  = t_pre + t_post
    t_grid = np.linspace(-t_rel_max, t_rel_max, 4 * Nt + 1)
    G_eq_t = greater_from_spectral(omega, A, beta_i, t_grid)
    G_eq     = interp1d(t_grid, G_eq_t, kind="cubic", fill_value="extrapolate")

    corr_final_err  = np.full(Nt, np.nan, dtype=float)
    corr_iters_used = np.zeros(Nt, dtype=int)

    # ── Resume from checkpoint or initialise fresh ───────────────────────────
    n_start = n0 + 1
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            ckpt    = np.load(checkpoint_path, allow_pickle=True)
            G       = np.asarray(ckpt["G"]).astype(complex)
            n_start = int(ckpt["n_done"]) + 1
            corr_final_err[:n_start]  = ckpt["corr_final_err"][:n_start]
            corr_iters_used[:n_start] = ckpt["corr_iters_used"][:n_start]
            print(f"Resumed KBE from checkpoint at n={n_start - 1} / {Nt - 1}"
                  f"  (t={t[n_start - 1]:.3f})")
        except Exception as e:
            print(f"Warning: could not load KBE checkpoint ({e}), starting fresh.")
            n_start = n0 + 1
            G = _init_G(Nt, n0, G_eq, t)
    else:
        G = _init_G(Nt, n0, G_eq, t)

    J2_of_t = np.where(t < 0.0, J2_i, J2_f)
    J4_of_t = np.where(t < 0.0, J4_i, J4_f)

    # Keep references to the last fully completed time step. The signal
    # handler saves these immediately when Slurm sends SIGUSR1/SIGTERM, so an
    # interrupted run is never more than one checkpoint_every window stale.
    checkpoint_state: Dict[str, Any] = {"n_done": n_start - 1}

    def _checkpoint_then_exit(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        n_done = checkpoint_state["n_done"]
        if checkpoint_path is not None and n_done >= n0:
            atomic_savez_checkpoint(
                checkpoint_path,
                G=G,
                n_done=np.array(n_done),
                n0=np.array(n0),
                corr_final_err=corr_final_err,
                corr_iters_used=corr_iters_used,
            )
            print(
                f"\n[{signal_name}] KBE checkpoint saved at n={n_done}, "
                f"t={t[n_done]:.3f}; exiting for requeue.",
                flush=True,
            )
        else:
            print(
                f"\n[{signal_name}] No completed KBE step beyond the "
                "equilibrium point to checkpoint; exiting.",
                flush=True,
            )
        # SystemExit is deliberately not caught by run_kbe_one's
        # `except Exception`, so an interrupted run is not marked as failed.
        raise SystemExit(128 + signum)

    for checkpoint_signal in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(checkpoint_signal, _checkpoint_then_exit)

    # ── Main time-stepping loop ───────────────────────────────────────────────
    for n in range(n_start, Nt):
        S      = sigma_greater_syk4(G, J2_of_t, J4_of_t)
        d1_old = np.array([rhs_t1(G, S, n - 1, j, dt, K_R_mat) for j in range(n)], dtype=complex)
        d2_old = np.array([rhs_t2(G, S, i, n - 1, dt, K_R_mat) for i in range(n)], dtype=complex)

        G[n, :n] = G[n - 1, :n] + dt * d1_old
        G[:n, n] = G[:n, n - 1] + dt * d2_old
        G[n, n]  = -0.5j
        enforce_majorana_slice(G, n)

        err     = np.inf
        it_used = 0
        for it in range(n_corr):
            row_old = G[n, :n].copy()
            col_old = G[:n, n].copy()
            S      = sigma_greater_syk4(G, J2_of_t, J4_of_t)
            d1_new = np.array([rhs_t1(G, S, n, j, dt, K_R_mat) for j in range(n)], dtype=complex)
            d2_new = np.array([rhs_t2(G, S, i, n, dt, K_R_mat) for i in range(n)], dtype=complex)

            G[n, :n] = G[n - 1, :n] + 0.5 * dt * (d1_old + d1_new)
            G[:n, n] = G[:n, n - 1] + 0.5 * dt * (d2_old + d2_new)
            G[n, n]  = -0.5j
            enforce_majorana_slice(G, n)

            err = max(
                float(np.max(np.abs(G[n, :n] - row_old))),
                float(np.max(np.abs(G[:n, n] - col_old))),
            )
            it_used = it + 1
            if err < corr_tol:
                break

        corr_final_err[n]  = err
        corr_iters_used[n] = it_used
        checkpoint_state["n_done"] = n

        if iterations and (n - n0) % iterations == 0:
            print(f"evolved to t = {t[n]:.3f}, corr err = {err:.3e}, iters = {it_used}")

        # ── Checkpoint ────────────────────────────────────────────────────────
        if checkpoint_path is not None and (n - n0) % checkpoint_every == 0:
            atomic_savez_checkpoint(
                checkpoint_path,
                G=G,
                n_done=np.array(n),
                n0=np.array(n0),
                corr_final_err=corr_final_err,
                corr_iters_used=corr_iters_used,
            )
            print(f"  [KBE checkpoint saved at n={n}, t={t[n]:.3f}]")

    # Remove checkpoint on successful completion
    if checkpoint_path is not None and Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()
        print("KBE checkpoint removed (run complete).")

    if return_diagnostics:
        diagnostics = {
            "corr_final_err":  corr_final_err,
            "corr_iters_used": corr_iters_used,
            "n0":              np.array(n0),
        }
        return t, G, diagnostics
    return t, G


# ============================================================
# Equilibrium-KBE self-consistency residual (scalar, single Majorana)
# ============================================================

def _theta_plus_minus(t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, int]:
    t = np.asarray(t)
    i0 = int(np.argmin(np.abs(t)))

    theta_p = np.zeros_like(t, dtype=float)
    theta_m = np.zeros_like(t, dtype=float)

    theta_p[t > 0.0] = 1.0
    theta_m[t < 0.0] = 1.0

    theta_p[i0] = 0.5
    theta_m[i0] = 0.5

    return theta_p, theta_m, i0


def _central_derivative_1d(X: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t)
    dt = t[1] - t[0]

    dX = np.empty_like(X, dtype=complex)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def _conv_1d_same(A: np.ndarray, B: np.ndarray, t: np.ndarray) -> np.ndarray:
    """C(t) = int du A(t-u) B(u), for scalar arrays on a symmetric uniform grid."""
    w = _trapz_weights(t)
    return np.convolve(np.asarray(B) * w, np.asarray(A), mode="same")


def calc_kbe_d_ab_syk_equilibrium(
    t: np.ndarray,
    Ggt: np.ndarray,
    J2: float,
    J4: float,
    *,
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    omega: Optional[np.ndarray] = None,
    t_cut: Optional[float] = None,
    edge_skip: int = 4,
    return_details: bool = False,
) -> float | Dict[str, Any]:
    """
    Real-time KBE residual d_ab for equilibrium single SYK4 (+ SYK2).

    Scalar (single Majorana, no L/R matrix indices) analog of
    calc_kbe_d_ab_mq_equilibrium in mq_batch_tools.py: build G^R/G^A/G^< from
    G^> via theta-function step conventions and the Majorana relation
    G^<(t) = -G^>(-t), evaluate Sigma from G, evaluate both variants of the
    KBE update equation as a residual.

    Checks both equilibrium KBE equations:
        i d_t G^>(t) = Sigma^R * G^> + Sigma^> * G^A
        i d_t G^>(t) = G^R * Sigma^> + G^> * Sigma^A
    (convolutions over relative time), where Sigma^>(t) = J2^2 G^>(t) -
    J4^2 G^>(t)^3.

    If kernel_lambda != 0, the static tuned kernel (see build_kernel_R_w) is
    folded into Sigma^R/Sigma^A as Sigma^R -= K^R(t), Sigma^A -= K^A(t) =
    conj(K^R(-t)) (matching the frequency-domain Dyson equation the
    equilibrium solver actually converges: GR_w = 1/(omega + i*eta + K_R_w
    - Sigma_R_w)), so this can check whether a kernel-deformed equilibrium
    solution is a true fixed point of the real-time KBE dynamics. Same sign
    convention as evolve_syk4_kbe's rhs_t1/rhs_t2.

    Returns d_ab = 0.5 * (mean_t |res_t1|^2 + mean_t |res_t2|^2).
    """
    t = np.asarray(t)
    Ggt = np.asarray(Ggt, dtype=complex)
    if Ggt.shape != t.shape:
        raise ValueError(f"Ggt must have shape {t.shape}, got {Ggt.shape}")

    theta_p, theta_m, i0 = _theta_plus_minus(t)

    # Lesser Green's function: G^<(t) = -G^>(-t)
    Glt = -Ggt[::-1]

    GR = theta_p * (Ggt - Glt)
    GA = theta_m * (Glt - Ggt)

    # Greater self-energy: Sigma^>(t) = J2^2 G^>(t) - J4^2 G^>(t)^3
    Sgt = J2**2 * Ggt - J4**2 * Ggt**3

    # Lesser self-energy: Sigma^<(t) = -Sigma^>(-t)
    Slt = -Sgt[::-1]

    SR = theta_p * (Sgt - Slt)
    SA = theta_m * (Slt - Sgt)

    if kernel_lambda != 0.0:
        if omega is None:
            raise ValueError(
                "omega is required to build the tuned kernel when kernel_lambda != 0."
            )
        K_R_w, _ = build_kernel_R_w(omega, J4, kernel_lambda, kernel_c, kernel_cutoff)
        K_R_t = _build_kernel_R_t(t, omega, K_R_w)
        # K^A(t) = conj(K^R(-t)); the grid is symmetric about t=0, so
        # reversal maps t -> -t (same convention used elsewhere in this
        # file, e.g. Glt/Slt above).
        K_A_t = np.conj(K_R_t[::-1])
        # Minus, not plus: GR_w = 1/(omega + i*eta + K_R_w - Sigma_R_w) means
        # Sigma_R_total = Sigma_R - K_R in the standard Dyson form.
        SR = SR - K_R_t
        SA = SA - K_A_t

    dG = _central_derivative_1d(Ggt, t)

    conv_t1 = _conv_1d_same(SR, Ggt, t) + _conv_1d_same(Sgt, GA, t)
    conv_t2 = _conv_1d_same(GR, Sgt, t) + _conv_1d_same(Ggt, SA, t)

    res_t1 = 1j * dG - conv_t1
    res_t2 = 1j * dG - conv_t2

    mask = np.ones(len(t), dtype=bool)
    if edge_skip is not None and edge_skip > 0:
        mask[:edge_skip] = False
        mask[-edge_skip:] = False
    if t_cut is not None:
        mask &= np.abs(t) <= t_cut

    d_t1 = float(np.mean(np.abs(res_t1[mask]) ** 2))
    d_t2 = float(np.mean(np.abs(res_t2[mask]) ** 2))
    d_ab = 0.5 * (d_t1 + d_t2)

    if not return_details:
        return d_ab

    return {
        "d_ab": d_ab,
        "d_t1": d_t1,
        "d_t2": d_t2,
        "res_t1": res_t1,
        "res_t2": res_t2,
        "Glt": Glt,
        "GR": GR,
        "GA": GA,
        "Sgt": Sgt,
        "Slt": Slt,
        "SR": SR,
        "SA": SA,
        "conv_t1": conv_t1,
        "conv_t2": conv_t2,
        "mask": mask,
    }


# ============================================================
# File naming and run wrappers
# ============================================================

def equilibrium_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"syk_eq_J2_{sf(meta['J2'])}_J4_{sf(meta['J4'])}_beta_{sf(meta['beta'])}"
        f"_klam_{sf(meta.get('kernel_lambda', 0.0))}"
        f"_kc_{sf(meta.get('kernel_c', 0.0))}"
        f"_kcut_{sf(meta.get('kernel_cutoff', 0.0))}"
        f"_dt_{sf(meta['dt'])}_om_{sf(meta['omega_max'])}_Nw_{int(meta['Nw'])}"
        f"_tol_{sf(meta['tol'])}_{h}.npz"
    )


def kbe_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"syk_kbe_J2_{sf(meta['J2_i'])}_to_{sf(meta['J2_f'])}"
        f"_J4_{sf(meta['J4_i'])}_to_{sf(meta['J4_f'])}"
        f"_beta_{sf(meta['beta'])}_dt_{sf(meta['dt'])}"
        f"_tpre_{sf(meta['t_pre'])}_tpost_{sf(meta['t_post'])}"
        f"_ncorr_{int(meta['n_corr'])}_ctol_{sf(meta['corr_tol'])}"
        f"_it_{int(meta['iterations'])}_{h}.npz"
    )


def run_equilibrium_one(
    J4: float,
    beta: float,
    *,
    J2: float = 0.0,
    out_dir: os.PathLike | str = "eq_runs",
    dt: float = 0.05,
    omega_max: float = 8.0,
    Nw: int = 4097,
    t_max: Optional[float] = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
    dab_tol: float = 1e-6,
    require_dab_convergence: bool = False,
    mixing: float = 0.05,
    eta_ret: float = 1e-6,
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    enforce_even_A: bool = True,
    clip_negative_A: bool = True,
    normalize_A: bool = True,
    verbose_every: int = 50,
    overwrite: bool = False,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "syk_eq_manifest.csv"

    J_kernel = abs(float(J4))
    if kernel_cutoff is None:
        kernel_cutoff = 4.0 * max(J_kernel, 1.0)

    meta = {
        "kind": "equilibrium",
        "J2": float(J2),
        "J4": float(J4),
        "beta": float(beta),
        "dt": float(dt),
        "omega_max": float(omega_max),
        "Nw": int(Nw),
        "t_max": None if t_max is None else float(t_max),
        "tol": float(tol),
        "dab_tol": float(dab_tol),
        "require_dab_convergence": bool(require_dab_convergence),
        "max_iter": int(max_iter),
        "mixing": float(mixing),
        "eta_ret": float(eta_ret),
        "kernel_lambda": float(kernel_lambda),
        "kernel_c": float(kernel_c),
        "kernel_cutoff": float(kernel_cutoff),
        "enforce_even_A": bool(enforce_even_A),
        "clip_negative_A": bool(clip_negative_A),
        "normalize_A": bool(normalize_A),
    }
    filename = equilibrium_filename(out_dir, meta)

    if filename.exists() and not overwrite:
        print(f"Skipping existing equilibrium file: {filename}")
        return filename

    checkpoint_path = checkpoint_path_for(filename)

    print("\n============================================================")
    print("Solving equilibrium")
    print(json.dumps(meta, indent=2))
    print("Saving to:", filename)
    print("============================================================")

    try:
        omega_real, A, t_grid, F_t, Ggt_t, GR_w, K_R_w, converged, final_dab_sqrt_max = solve_equilibrium_greater_real_time(
            J2=J2,
            J4=J4,
            beta=beta,
            omega_max=omega_max,
            Nw=Nw,
            t_max=t_max,
            dt=dt,
            max_iter=max_iter,
            tol=tol,
            dab_tol=dab_tol,
            require_dab_convergence=require_dab_convergence,
            mixing=mixing,
            eta_ret=eta_ret,
            kernel_lambda=kernel_lambda,
            kernel_c=kernel_c,
            kernel_cutoff=kernel_cutoff,
            enforce_even_A=enforce_even_A,
            clip_negative_A=clip_negative_A,
            normalize_A=normalize_A,
            verbose_every=verbose_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=100,
        )
        meta["converged"] = bool(converged)
        meta["final_dab_sqrt_max"] = None if not np.isfinite(final_dab_sqrt_max) else float(final_dab_sqrt_max)
        atomic_savez_compressed(
            filename,
            omega_real=omega_real,
            A=A,
            t_grid=t_grid,
            F_t=F_t,
            Ggt_t=Ggt_t,
            GR_w=GR_w,
            K_R_w=K_R_w,
            metadata_json=json.dumps(meta),
        )
        row = dict(meta)
        row.update({"filename": str(filename), "status": "ok"})
        update_manifest(manifest_path, row)
        print("Saved:", filename)
        return filename
    except Exception as e:
        traceback.print_exc()
        row = dict(meta)
        row.update({"filename": str(filename), "status": "failed", "error": repr(e)})
        update_manifest(manifest_path, row)
        return None


def run_equilibrium_grid(eq_grid: Dict[str, Iterable[Any]], *, out_dir: os.PathLike | str = "eq_runs", overwrite: bool = False) -> List[Optional[Path]]:
    paths = []
    for p in grid_from_dict(eq_grid):
        paths.append(run_equilibrium_one(out_dir=out_dir, overwrite=overwrite, **p))
    return paths


def find_eq_file(
    eq_manifest: pd.DataFrame,
    J4: float,
    beta: float,
    *,
    J2: float = 0.0,
    kernel_lambda: float = 0.0,
    prefer_smallest_dt: bool = True,
) -> pd.Series:
    if eq_manifest.empty:
        raise FileNotFoundError("Equilibrium manifest is empty.")
    good = eq_manifest[
        (eq_manifest["status"] == "ok")
        & np.isclose(eq_manifest["J2"].astype(float), float(J2))
        & np.isclose(eq_manifest["J4"].astype(float), float(J4))
        & np.isclose(eq_manifest["beta"].astype(float), float(beta))
    ].copy()
    if "kernel_lambda" in good.columns:
        good = good[
            np.isclose(
                good["kernel_lambda"].fillna(0.0).astype(float),
                float(kernel_lambda),
            )
        ]
    elif not np.isclose(kernel_lambda, 0.0):
        good = good.iloc[0:0]
    if len(good) == 0:
        raise FileNotFoundError(
            f"No equilibrium file found for J2={J2}, J4={J4}, beta={beta}, "
            f"kernel_lambda={kernel_lambda}"
        )
    if prefer_smallest_dt:
        good = good.sort_values(["dt", "tol", "Nw"], ascending=[True, True, False])
    else:
        good = good.sort_values(["tol", "Nw"], ascending=[True, False])
    return good.iloc[0]


def run_kbe_one(
    J4_i: float,
    J4_f: float,
    beta: float,
    *,
    J2_i: float = 0.0,
    J2_f: float = 0.0,
    dt: float = 0.05,
    t_pre: Optional[float] = None,
    t_post: Optional[float] = None,
    t_pre_factor: Optional[float] = 2.0,
    t_post_factor: Optional[float] = 1.0,
    n_corr: int = 4,
    corr_tol: float = 1e-10,
    iterations: int = 50,
    # ── Tuned quadratic source (static: same K throughout the run) ─────────────
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    eq_dir: os.PathLike | str = "eq_runs",
    out_dir: os.PathLike | str = "kbe_runs",
    eq_file: Optional[os.PathLike | str] = None,
    overwrite: bool = False,
    save_diagnostics: bool = True,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "syk_kbe_manifest.csv"

    if t_pre is None:
        if t_pre_factor is None:
            raise ValueError("Need either t_pre or t_pre_factor.")
        t_pre = float(t_pre_factor) * float(beta)
    if t_post is None:
        if t_post_factor is None:
            raise ValueError("Need either t_post or t_post_factor.")
        t_post = float(t_post_factor) * float(beta)

    if eq_file is None:
        eq_manifest = load_manifest(Path(eq_dir) / "syk_eq_manifest.csv")
        eq_row = find_eq_file(eq_manifest, J2=J2_i, J4=J4_i, beta=beta)
        eq_file = eq_row["filename"]
    eq_file = Path(eq_file)

    J_kernel = abs(float(J4_i))
    if kernel_cutoff is None:
        kernel_cutoff = 4.0 * max(J_kernel, 1.0)

    meta = {
        "kind": "kbe",
        "J2_i": float(J2_i),
        "J2_f": float(J2_f),
        "J4_i": float(J4_i),
        "J4_f": float(J4_f),
        "beta": float(beta),
        "dt": float(dt),
        "t_pre": float(t_pre),
        "t_post": float(t_post),
        "t_pre_factor": None if t_pre_factor is None else float(t_pre_factor),
        "t_post_factor": None if t_post_factor is None else float(t_post_factor),
        "n_corr": int(n_corr),
        "corr_tol": float(corr_tol),
        "iterations": int(iterations),
        "kernel_lambda": float(kernel_lambda),
        "kernel_c": float(kernel_c),
        "kernel_cutoff": float(kernel_cutoff),
        "eq_source": str(eq_file),
        "save_diagnostics": bool(save_diagnostics),
        # KBE files now save only rows with t1 >= 0 plus one equilibrium GF copy.
        # The purely pre-quench block is redundant because G_prepre(t1,t2)=G_eq(t1-t2).
        "save_format": "post_quench_rows_plus_eq_gf",
        "full_Ggt_saved": False,
    }
    filename = kbe_filename(out_dir, meta)
    if filename.exists() and not overwrite:
        print(f"Skipping existing KBE file: {filename}")
        return filename

    checkpoint_path = checkpoint_path_for(filename)

    print("\n============================================================")
    print("Running KBE")
    print(json.dumps(meta, indent=2))
    print("Saving to:", filename)
    print("============================================================")

    try:
        eq_data = np.load(eq_file, allow_pickle=True)
        omega_real = eq_data["omega_real"]
        A = eq_data["A"]

        # Save one reusable copy of the equilibrium greater GF.
        # This is enough to reconstruct the purely pre-quench block via
        # G_prepre(t1,t2) = G_eq(t1 - t2), so we do not save that block again.
        # "t_grid"/"Ggt_t" are the current keys; "tau_grid"/"Ggt_tau" are
        # accepted for equilibrium files written by the pre-rename version
        # of this script (the arrays themselves are unchanged, only the
        # in-file key names differ).
        if "t_grid" in eq_data.files and "Ggt_t" in eq_data.files:
            t_eq = eq_data["t_grid"]
            Ggt_eq_t = eq_data["Ggt_t"]
        elif "tau_grid" in eq_data.files and "Ggt_tau" in eq_data.files:
            t_eq = eq_data["tau_grid"]
            Ggt_eq_t = eq_data["Ggt_tau"]
        else:
            # Fallback for older equilibrium files that only contain omega_real and A.
            t_eq = np.arange(-(t_pre + t_post), (t_pre + t_post) + 0.5 * dt, dt)
            Ggt_eq_t = greater_from_spectral(omega_real, A, beta, t_eq)

        if save_diagnostics:
            t_kbe, Ggt, diagnostics = evolve_syk4_kbe(
                omega_real,
                A,
                beta,
                J2_i=J2_i,
                J2_f=J2_f,
                J4_i=J4_i,
                J4_f=J4_f,
                t_pre=t_pre,
                t_post=t_post,
                dt=dt,
                n_corr=n_corr,
                corr_tol=corr_tol,
                iterations=iterations,
                kernel_lambda=kernel_lambda,
                kernel_c=kernel_c,
                kernel_cutoff=kernel_cutoff,
                return_diagnostics=True,
                checkpoint_path=checkpoint_path,
                checkpoint_every=50,
            )
            n0 = int(np.asarray(diagnostics["n0"]).item())
            t_kbe[n0] = 0.0

            # Keep only rows with t1 >= 0.
            # Columns keep the full t2 axis, so this includes post-post and post-pre/mixed data.
            # The omitted pre-pre block is just equilibrium data and can be reconstructed from t_eq/Ggt_eq_t.
            t_row_post = t_kbe[n0:].copy()
            t_col_all = t_kbe.copy()
            Ggt_post = Ggt[n0:, :].copy()

            atomic_savez_compressed(
                filename,
                # t_kbe is kept as a tiny compatibility alias for the full original time axis.
                # There is intentionally no full Ggt array saved.
                t_kbe=t_col_all,
                t_row_post=t_row_post,
                t_col_all=t_col_all,
                Ggt_post=Ggt_post,
                t_eq=t_eq,
                Ggt_eq_t=Ggt_eq_t,
                omega_real=omega_real,
                A=A,
                corr_final_err_post=diagnostics["corr_final_err"][n0:],
                corr_iters_used_post=diagnostics["corr_iters_used"][n0:],
                n0=np.array(n0),
                metadata_json=json.dumps(meta),
            )
        else:
            t_kbe, Ggt = evolve_syk4_kbe(
                omega_real,
                A,
                beta,
                J2_i=J2_i,
                J2_f=J2_f,
                J4_i=J4_i,
                J4_f=J4_f,
                t_pre=t_pre,
                t_post=t_post,
                dt=dt,
                n_corr=n_corr,
                corr_tol=corr_tol,
                iterations=iterations,
                kernel_lambda=kernel_lambda,
                kernel_c=kernel_c,
                kernel_cutoff=kernel_cutoff,
                return_diagnostics=False,
                checkpoint_path=checkpoint_path,
                checkpoint_every=50,
            )
            n0 = int(np.argmin(np.abs(t_kbe)))
            t_kbe[n0] = 0.0

            t_row_post = t_kbe[n0:].copy()
            t_col_all = t_kbe.copy()
            Ggt_post = Ggt[n0:, :].copy()

            atomic_savez_compressed(
                filename,
                # t_kbe is kept as a tiny compatibility alias for the full original time axis.
                # There is intentionally no full Ggt array saved.
                t_kbe=t_col_all,
                t_row_post=t_row_post,
                t_col_all=t_col_all,
                Ggt_post=Ggt_post,
                t_eq=t_eq,
                Ggt_eq_t=Ggt_eq_t,
                omega_real=omega_real,
                A=A,
                n0=np.array(n0),
                metadata_json=json.dumps(meta),
            )

        row = dict(meta)
        row.update({"filename": str(filename), "status": "ok"})
        update_manifest(manifest_path, row)
        print("Saved:", filename)
        return filename
    except Exception as e:
        traceback.print_exc()
        row = dict(meta)
        row.update({"filename": str(filename), "status": "failed", "error": repr(e)})
        update_manifest(manifest_path, row)
        return None


def run_kbe_grid(kbe_grid: Dict[str, Iterable[Any]], *, eq_dir: os.PathLike | str = "eq_runs", out_dir: os.PathLike | str = "kbe_runs", overwrite: bool = False) -> List[Optional[Path]]:
    paths = []
    for p in grid_from_dict(kbe_grid):
        paths.append(run_kbe_one(eq_dir=eq_dir, out_dir=out_dir, overwrite=overwrite, **p))
    return paths


def load_kbe_dataframe(kbe_dir: os.PathLike | str = "kbe_runs") -> pd.DataFrame:
    """
    Load all successful KBE runs from the manifest into a DataFrame with arrays.

    New KBE files save only the post-quench rows:
        t_row_post : t1 axis with t1 >= 0
        t_col_all  : full t2 axis
        Ggt_post   : G^>(t1,t2) with t1 >= 0 and all t2
        t_eq     : relative-time equilibrium axis
        Ggt_eq_t : one reusable equilibrium G^>(t)

    Legacy files containing a full Ggt matrix are still supported. For legacy files,
    this loader creates Ggt_post by slicing the full matrix, but keeps Ggt as well.
    """
    kbe_dir = Path(kbe_dir)
    manifest = load_manifest(kbe_dir / "syk_kbe_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        if row.get("status") != "ok":
            continue
        filename = Path(row["filename"])
        if not filename.exists():
            print("Missing:", filename)
            continue
        data = np.load(filename, allow_pickle=True)
        keys = set(data.files)

        entry: Dict[str, Any] = {
            "Ji": float(row["J4_i"]),
            "Jf": float(row["J4_f"]),
            "J2_i": float(row["J2_i"]),
            "J2_f": float(row["J2_f"]),
            "beta": float(row["beta"]),
            "dt": float(row["dt"]),
            "t_pre": float(row["t_pre"]),
            "t_post": float(row["t_post"]),
            "n_corr": int(row["n_corr"]),
            "corr_tol": float(row["corr_tol"]),
            "iterations": int(row["iterations"]),
            "filename": str(filename),
            "eq_source": row["eq_source"],
        }

        if "Ggt_post" in keys:
            # New compact format: full Ggt is intentionally absent.
            t_col_all = data["t_col_all"] if "t_col_all" in keys else data["t_kbe"]
            n0 = int(np.asarray(data["n0"]).item()) if "n0" in keys else int(np.argmin(np.abs(t_col_all)))
            entry.update({
                "save_format": "post_quench_rows_plus_eq_gf",
                "n0": n0,
                "t_kbe": t_col_all,       # compatibility alias for the full time axis
                "t_col_all": t_col_all,
                "t_row_post": data["t_row_post"],
                "Ggt_post": data["Ggt_post"],
                # "t_eq"/"Ggt_eq_t" are current keys; "tau_eq"/"Ggt_eq_tau" are
                # accepted for KBE files written by the pre-rename version of
                # this script.
                "t_eq": data["t_eq"] if "t_eq" in keys else (data["tau_eq"] if "tau_eq" in keys else None),
                "Ggt_eq_t": data["Ggt_eq_t"] if "Ggt_eq_t" in keys else (data["Ggt_eq_tau"] if "Ggt_eq_tau" in keys else None),
                "Ggt": None,
            })
            if "corr_final_err_post" in keys:
                entry["corr_final_err_post"] = data["corr_final_err_post"]
            if "corr_iters_used_post" in keys:
                entry["corr_iters_used_post"] = data["corr_iters_used_post"]
        else:
            # Legacy full-matrix format.
            t_kbe = data["t_kbe"]
            n0 = int(np.asarray(data["n0"]).item()) if "n0" in keys else int(np.argmin(np.abs(t_kbe)))
            entry.update({
                "save_format": "full_Ggt_legacy",
                "n0": n0,
                "t_kbe": t_kbe,
                "t_col_all": t_kbe,
                "t_row_post": t_kbe[n0:],
                "Ggt": data["Ggt"],
                "Ggt_post": data["Ggt"][n0:, :],
                "t_eq": data["t_eq"] if "t_eq" in keys else (data["tau_eq"] if "tau_eq" in keys else None),
                "Ggt_eq_t": data["Ggt_eq_t"] if "Ggt_eq_t" in keys else (data["Ggt_eq_tau"] if "Ggt_eq_tau" in keys else None),
            })
            if "corr_final_err" in keys:
                entry["corr_final_err_post"] = data["corr_final_err"][n0:]
            if "corr_iters_used" in keys:
                entry["corr_iters_used_post"] = data["corr_iters_used"][n0:]

        rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================
# CLI for cluster-style one-run jobs
# ============================================================

def _parse_bool(s: str) -> bool:
    if str(s).lower() in {"1", "true", "yes", "y"}:
        return True
    if str(s).lower() in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Cannot parse boolean: {s}")


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch utilities for single-SYK equilibrium and KBE runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    peq = sub.add_parser("eq-one", help="Run one equilibrium solve.")
    peq.add_argument("--J4", type=float, required=True)
    peq.add_argument("--beta", type=float, required=True)
    peq.add_argument("--J2", type=float, default=0.0)
    peq.add_argument("--out-dir", default="eq_runs")
    peq.add_argument("--dt", type=float, default=0.05)
    peq.add_argument("--omega-max", type=float, default=8.0)
    peq.add_argument("--Nw", type=int, default=4097)
    peq.add_argument("--t-max", type=float, default=None)
    peq.add_argument("--max-iter", type=int, default=2000)
    peq.add_argument("--tol", type=float, default=1e-9)
    peq.add_argument("--dab-tol", type=float, default=1e-6, help="Tolerance for max(d_ab**0.5). Printed always when d_ab is computed; only required if --require-dab-convergence is set.")
    peq.add_argument("--require-dab-convergence", action="store_true", help="Require max(d_ab**0.5) < --dab-tol before marking equilibrium converged.")
    peq.add_argument("--mixing", type=float, default=0.05)
    peq.add_argument("--eta-ret", type=float, default=1e-6)
    peq.add_argument("--kernel-lambda", type=float, default=0.0)
    peq.add_argument("--kernel-c", type=float, default=0.0)
    peq.add_argument("--kernel-cutoff", type=float, default=None)
    peq.add_argument("--verbose-every", type=int, default=50)
    peq.add_argument("--overwrite", action="store_true")

    pkbe = sub.add_parser("kbe-one", help="Run one KBE evolution from an equilibrium npz.")
    pkbe.add_argument("--J4-i", type=float, required=True)
    pkbe.add_argument("--J4-f", type=float, required=True)
    pkbe.add_argument("--beta", type=float, required=True)
    pkbe.add_argument("--J2-i", type=float, default=0.0)
    pkbe.add_argument("--J2-f", type=float, default=0.0)
    pkbe.add_argument("--dt", type=float, default=0.05)
    pkbe.add_argument("--t-pre", type=float, default=None)
    pkbe.add_argument("--t-post", type=float, default=None)
    pkbe.add_argument("--t-pre-factor", type=float, default=2.0)
    pkbe.add_argument("--t-post-factor", type=float, default=1.0)
    pkbe.add_argument("--n-corr", type=int, default=4)
    pkbe.add_argument("--corr-tol", type=float, default=1e-10)
    pkbe.add_argument("--iterations", type=int, default=50)
    pkbe.add_argument("--kernel-lambda", type=float, default=0.0)
    pkbe.add_argument("--kernel-c", type=float, default=0.0)
    pkbe.add_argument("--kernel-cutoff", type=float, default=None)
    pkbe.add_argument("--eq-dir", default="eq_runs")
    pkbe.add_argument("--out-dir", default="kbe_runs")
    pkbe.add_argument("--eq-file", default=None)
    pkbe.add_argument("--no-diagnostics", action="store_true")
    pkbe.add_argument("--overwrite", action="store_true")

    pgrid = sub.add_parser("grid", help="Run eq-grid or kbe-grid from a JSON config file.")
    pgrid.add_argument("--kind", choices=["eq", "kbe"], required=True)
    pgrid.add_argument("--config", required=True, help="JSON file containing a dict of parameter lists.")
    pgrid.add_argument("--eq-dir", default="eq_runs")
    pgrid.add_argument("--out-dir", default=None)
    pgrid.add_argument("--overwrite", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd == "eq-one":
        run_equilibrium_one(
            J4=args.J4,
            beta=args.beta,
            J2=args.J2,
            out_dir=args.out_dir,
            dt=args.dt,
            omega_max=args.omega_max,
            Nw=args.Nw,
            t_max=args.t_max,
            max_iter=args.max_iter,
            tol=args.tol,
            dab_tol=args.dab_tol,
            require_dab_convergence=args.require_dab_convergence,
            mixing=args.mixing,
            eta_ret=args.eta_ret,
            kernel_lambda=args.kernel_lambda,
            kernel_c=args.kernel_c,
            kernel_cutoff=args.kernel_cutoff,
            verbose_every=args.verbose_every,
            overwrite=args.overwrite,
        )
    elif args.cmd == "kbe-one":
        run_kbe_one(
            J4_i=args.J4_i,
            J4_f=args.J4_f,
            beta=args.beta,
            J2_i=args.J2_i,
            J2_f=args.J2_f,
            dt=args.dt,
            t_pre=args.t_pre,
            t_post=args.t_post,
            t_pre_factor=args.t_pre_factor,
            t_post_factor=args.t_post_factor,
            n_corr=args.n_corr,
            corr_tol=args.corr_tol,
            iterations=args.iterations,
            kernel_lambda=args.kernel_lambda,
            kernel_c=args.kernel_c,
            kernel_cutoff=args.kernel_cutoff,
            eq_dir=args.eq_dir,
            out_dir=args.out_dir,
            eq_file=args.eq_file,
            overwrite=args.overwrite,
            save_diagnostics=not args.no_diagnostics,
        )
    elif args.cmd == "grid":
        with open(args.config) as f:
            cfg = json.load(f)
        if args.kind == "eq":
            run_equilibrium_grid(cfg, out_dir=args.out_dir or "eq_runs", overwrite=args.overwrite)
        else:
            run_kbe_grid(cfg, eq_dir=args.eq_dir, out_dir=args.out_dir or "kbe_runs", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
