"""Checkpointing + lean-output utilities for the KKTZ quartic-coupled SYK KBE runs.

This reuses the matrix-KBE plumbing from mq_kbe_patch.py, but replaces the
Maldacena-Qi self-energy by the Kim-Klebanov-Tarnopolsky-Zhao quartic
two-flavor self-energy.  The optional ``mu`` is retained as a small quadratic
source for Q=i chi_1 chi_2, useful for selecting/probing the Q channel.
"""
from __future__ import annotations

import json
import os
import signal
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from mq_kbe_patch import (
    _build_GR0_mq_seed,
    _central_derivative_1d,
    _conv_mat_same,
    _omega_to_time_mat,
    _theta_plus_minus,
    _time_to_omega_mat,
    enforce_equal_time_majorana,
    fill_equilibrium_block,
    h_mu_matrix,
    rhs_t1_matrix,
    rhs_t2_matrix,
)


def _json_safe(x: Any) -> str:
    def default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, (np.ndarray,)):
            return o.tolist()
        return str(o)
    return json.dumps(x, sort_keys=True, default=default)


def atomic_savez(path: str | Path, *, compressed: bool = False, **kwargs) -> None:
    """Atomic npz save: write temporary file, then os.replace into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    saver = np.savez_compressed if compressed else np.savez
    saver(tmp, **kwargs)
    # np.savez appends .npz if the filename does not end in .npz. Our tmp suffix does not.
    tmp_actual = Path(str(tmp) + ".npz") if not tmp.exists() else tmp
    os.replace(tmp_actual, path)


def _pair_prefactor_from_time_array(x, prefix_shape):
    """Return x(t1)x(t2) for two-time arrays, or x(t)^2 for one-time arrays."""
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return float(arr) ** 2
    if len(prefix_shape) == 1:
        if arr.shape[0] != prefix_shape[0]:
            raise ValueError(f"time-dependent coupling length {arr.shape[0]} incompatible with shape {prefix_shape}")
        return arr ** 2
    if len(prefix_shape) == 2:
        if arr.shape[0] != prefix_shape[0] or arr.shape[0] != prefix_shape[1]:
            raise ValueError(f"time-dependent coupling length {arr.shape[0]} incompatible with shape {prefix_shape}")
        return arr[:, None] * arr[None, :]
    raise ValueError(f"Expected one- or two-time Green's function prefix shape, got {prefix_shape}")


def _alpha_linear_and_quadratic(alpha, prefix_shape):
    """Coefficients replacing alpha and alpha^2 in the two-time self-energy.

    For constant alpha this is exact. For an alpha quench, terms linear in alpha
    are symmetrized as (alpha(t1)+alpha(t2))/2 and quadratic terms use
    alpha(t1) alpha(t2). Most KKTZ runs should keep alpha_i=alpha_f and quench
    only the small Q-source mu, in which case there is no ambiguity.
    """
    arr = np.asarray(alpha, dtype=float)
    if arr.ndim == 0:
        a = float(arr)
        return a, a * a
    if len(prefix_shape) == 1:
        if arr.shape[0] != prefix_shape[0]:
            raise ValueError(f"time-dependent alpha length {arr.shape[0]} incompatible with shape {prefix_shape}")
        return arr, arr ** 2
    if len(prefix_shape) == 2:
        if arr.shape[0] != prefix_shape[0] or arr.shape[0] != prefix_shape[1]:
            raise ValueError(f"time-dependent alpha length {arr.shape[0]} incompatible with shape {prefix_shape}")
        a1 = arr[:, None]
        a2 = arr[None, :]
        return 0.5 * (a1 + a2), a1 * a2
    raise ValueError(f"Expected one- or two-time Green's function prefix shape, got {prefix_shape}")


def _kktz_cubic_polynomial(G, alpha):
    """Cubic KKTZ flavor polynomial P_ab[G] without the overall J^2/sign."""
    X = np.asarray(G)
    if X.shape[-2:] != (2, 2):
        raise ValueError(f"Expected trailing matrix shape (2,2), got {X.shape}")
    prefix = X.shape[:-2]
    a, a2 = _alpha_linear_and_quadratic(alpha, prefix)

    g11 = X[..., 0, 0]
    g12 = X[..., 0, 1]
    g21 = X[..., 1, 0]
    g22 = X[..., 1, 1]

    P = np.empty_like(X, dtype=np.result_type(X, complex))
    P[..., 0, 0] = (
        g11**3
        + 3.0 * a * g11 * (g12**2 + g21**2)
        + 3.0 * a2 * g11 * g22**2
        + 6.0 * a2 * g22 * g12 * g21
    )
    P[..., 1, 1] = (
        g22**3
        + 3.0 * a * g22 * (g12**2 + g21**2)
        + 3.0 * a2 * g22 * g11**2
        + 6.0 * a2 * g11 * g12 * g21
    )
    P[..., 0, 1] = (
        g12**3
        + 3.0 * a * g12 * (g11**2 + g22**2)
        + 3.0 * a2 * g12 * g21**2
        + 6.0 * a2 * g11 * g22 * g21
    )
    P[..., 1, 0] = (
        g21**3
        + 3.0 * a * g21 * (g11**2 + g22**2)
        + 3.0 * a2 * g21 * g12**2
        + 6.0 * a2 * g11 * g22 * g12
    )
    return P


def i_sigma_greater_kktz_from_F(F, J, alpha, *, keep_cross_self_energy=True):
    """Return i Sigma^>[F] for equilibrium iterations with F=iG^>.

    Since G^> = -iF and Sigma^> = -J^2 P[G^>], one has
    i Sigma^> = +J^2 P[F] for a cubic polynomial P.
    """
    X = np.asarray(F)
    prefix = X.shape[:-2]
    jj = _pair_prefactor_from_time_array(J, prefix)
    out = _kktz_cubic_polynomial(X, alpha)
    if not keep_cross_self_energy:
        out[..., 0, 1] = 0.0
        out[..., 1, 0] = 0.0
    return jj[..., None, None] * out if np.ndim(jj) else jj * out


def sigma_greater_kktz(G, J, alpha, *, keep_cross_self_energy=True):
    """KKTZ greater self-energy Sigma^> for a 2x2 flavor Green's function.

    Constant-alpha equations implemented here are eq. (3.16) of
    Kim-Klebanov-Tarnopolsky-Zhao, analytically continued to the greater
    self-energy convention used by the KBE code:

        Sigma^>_ab = -J(t1)J(t2) P_ab[G^>].

    For alpha=0 this reduces to the original MQ/coupled-SYK KBE convention
    ``Sigma^>_ab = -J(t1)J(t2) (G^>_ab)^3``.
    """
    X = np.asarray(G)
    prefix = X.shape[:-2]
    jj = _pair_prefactor_from_time_array(J, prefix)
    out = -_kktz_cubic_polynomial(X, alpha)
    if not keep_cross_self_energy:
        out[..., 0, 1] = 0.0
        out[..., 1, 0] = 0.0
    return jj[..., None, None] * out if np.ndim(jj) else jj * out

def calc_kbe_d_ab_kktz_equilibrium(
    t,
    Ggt,
    J,
    alpha,
    mu,
    *,
    t_cut=None,
    edge_skip=4,
    keep_cross_self_energy=True,
    return_details=False,
):
    """Real-time KBE residual d_ab for equilibrium KKTZ.

    This is the same diagnostic structure as ``calc_kbe_d_ab_mq_equilibrium``
    in ``mq_kbe_patch_v4.py``, but with the KKTZ two-flavor greater
    self-energy

        Sigma^>_ab(t) = -J^2 P_ab[G^>(t), alpha]

    instead of the MQ componentwise cubic self-energy.

    Returns d_ab = 0.5 * (d_t1_ab + d_t2_ab), where each component is a
    mean-squared residual.  The printed convergence diagnostic usually uses
    d_ab**0.5, matching the MQ wrapper convention.
    """
    t = np.asarray(t)
    Ggt = np.asarray(Ggt, dtype=complex)

    if Ggt.shape != (len(t), 2, 2):
        raise ValueError(f"Ggt must have shape {(len(t), 2, 2)}, got {Ggt.shape}")

    theta_p, theta_m, _i0 = _theta_plus_minus(t)
    h = h_mu_matrix(mu)

    # Lesser Green's function: G^<_ab(t) = -G^>_ba(-t)
    Glt = -np.transpose(Ggt[::-1], (0, 2, 1))

    GR = theta_p[:, None, None] * (Ggt - Glt)
    GA = theta_m[:, None, None] * (Glt - Ggt)

    # KKTZ greater self-energy.  For alpha=0 this exactly reduces to MQ.
    Sgt = sigma_greater_kktz(
        Ggt,
        J,
        alpha,
        keep_cross_self_energy=keep_cross_self_energy,
    )

    # Lesser self-energy: Sigma^<_ab(t) = -Sigma^>_ba(-t)
    Slt = -np.transpose(Sgt[::-1], (0, 2, 1))

    SR = theta_p[:, None, None] * (Sgt - Slt)
    SA = theta_m[:, None, None] * (Slt - Sgt)

    dG = _central_derivative_1d(Ggt, t)

    conv_t1 = _conv_mat_same(SR, Ggt, t) + _conv_mat_same(Sgt, GA, t)
    conv_t2 = _conv_mat_same(GR, Sgt, t) + _conv_mat_same(Ggt, SA, t)

    hG = np.einsum("ac,tcb->tab", h, Ggt)
    Gh = np.einsum("tac,cb->tab", Ggt, h)

    res_t1 = 1j * dG - hG - conv_t1
    res_t2 = 1j * dG - Gh - conv_t2

    mask = np.ones(len(t), dtype=bool)
    if edge_skip is not None and edge_skip > 0:
        mask[:edge_skip] = False
        mask[-edge_skip:] = False
    if t_cut is not None:
        mask &= np.abs(t) <= t_cut
    if not np.any(mask):
        raise ValueError("Residual mask is empty; reduce edge_skip or increase t_cut.")

    d_t1_ab = np.mean(np.abs(res_t1[mask])**2, axis=0)
    d_t2_ab = np.mean(np.abs(res_t2[mask])**2, axis=0)
    d_ab = 0.5 * (d_t1_ab + d_t2_ab)

    if not return_details:
        return d_ab

    return {
        "d_ab": d_ab,
        "d_t1_ab": d_t1_ab,
        "d_t2_ab": d_t2_ab,
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


def calc_O1_from_kbe_Ggreater(t, Ggt_2time, J_ref=1.0, time_is_dimensionless=False):
    """
    Compute O1_a(t)/N = -1/2 (d1^2 d2 - d1 d2^2) G^>_aa |_{t1=t2=t}.
    This is the same finite-difference stencil used in mq_coupled_quench_plot.ipynb.
    """
    t = np.asarray(t, dtype=float)
    G = np.asarray(Ggt_2time)
    Nt = len(t)
    if G.shape != (Nt, Nt, 2, 2):
        raise ValueError(f"Expected G shape {(Nt, Nt, 2, 2)}, got {G.shape}")
    dt = t[1] - t[0]
    if not np.allclose(np.diff(t), dt, rtol=1e-10, atol=1e-12):
        raise ValueError("Uniform t grid required.")

    c_d1 = np.array([-0.5, 0.0, 0.5]) / dt
    c_d2 = np.array([1.0, -2.0, 1.0]) / dt**2
    O1 = np.full((Nt, 2), np.nan + 0j, dtype=complex)

    for n in range(1, Nt - 1):
        sl = slice(n - 1, n + 2)
        for a in (0, 1):
            patch = G[sl, sl, a, a]
            d112 = np.einsum("i,j,ij->", c_d2, c_d1, patch)
            d122 = np.einsum("i,j,ij->", c_d1, c_d2, patch)
            O1[n, a] = -0.5 * (d112 - d122)

    if time_is_dimensionless:
        O1 = J_ref**3 * O1

    return {
        "O1_L": O1[:, 0],
        "O1_R": O1[:, 1],
        "O1_avg": 0.5 * (O1[:, 0] + O1[:, 1]),
        "O1_sum": O1[:, 0] + O1[:, 1],
        "O1_L_over_J3": O1[:, 0] / J_ref**3,
        "O1_R_over_J3": O1[:, 1] / J_ref**3,
        "O1_avg_over_J3": 0.5 * (O1[:, 0] + O1[:, 1]) / J_ref**3,
        "O1_sum_over_J3": (O1[:, 0] + O1[:, 1]) / J_ref**3,
    }


def post_observables_from_G(t, G, *, J_ref=1.0, include_pre=False) -> Dict[str, np.ndarray]:
    """Extract equal-time LR and O1 from full two-time G; default returns post-quench only."""
    t = np.asarray(t, dtype=float)
    G = np.asarray(G)
    n0 = int(np.argmin(np.abs(t)))
    idx = np.arange(len(t))

    B_LR = -np.real(G[idx, idx, 0, 1])
    Re_GLR_equal = np.real(G[idx, idx, 0, 1])
    GLL_equal = G[idx, idx, 0, 0]
    GRR_equal = G[idx, idx, 1, 1]
    O1 = calc_O1_from_kbe_Ggreater(t, G, J_ref=J_ref, time_is_dimensionless=False)

    sl = slice(None) if include_pre else slice(n0, None)
    t_out = t[sl]
    # make post time exactly start at 0 for downstream fitting
    if not include_pre:
        t_out = t_out - t[n0]

    out = {
        "t": t_out,
        "B_LR": B_LR[sl],
        "Re_GLR_equal": Re_GLR_equal[sl],
        "GLL_equal": GLL_equal[sl],
        "GRR_equal": GRR_equal[sl],
    }
    for k, v in O1.items():
        out[k] = np.asarray(v)[sl]
    return out


def save_kbe_lean(
    path: str | Path,
    t,
    G,
    *,
    J_i: float,
    J_f: float,
    alpha_i: float,
    alpha_f: float,
    mu_i: float,
    mu_f: float,
    beta: float,
    dt: float,
    t_pre: float,
    t_post: float,
    Nw: int | None = None,
    include_postpost_G: bool = False,
    compressed: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    """Save non-redundant physics output. By default does NOT save the pre-quench G block."""
    t = np.asarray(t)
    G = np.asarray(G)
    n0 = int(np.argmin(np.abs(t)))
    obs = post_observables_from_G(t, G, J_ref=J_f, include_pre=False)
    meta = {
        "J_i": J_i,
        "J_f": J_f,
        "alpha_i": alpha_i,
        "alpha_f": alpha_f,
        "mu_i": mu_i,
        "mu_f": mu_f,
        "beta": beta,
        "dt": dt,
        "t_pre": t_pre,
        "t_post": t_post,
        "Nw": Nw,
        "n0_original": n0,
        "saved_pre_quench_G": False,
        "saved_full_G": False,
        "notes": "Lean KKTZ KBE output: post-quench equal-time observables and O1 only. Use checkpoint file to resume full two-time evolution.",
    }
    if extra_metadata:
        meta.update(extra_metadata)

    payload = dict(obs)
    payload.update({
        "J_i": J_i,
        "J_f": J_f,
        "alpha_i": alpha_i,
        "alpha_f": alpha_f,
        "mu_i": mu_i,
        "mu_f": mu_f,
        "beta": beta,
        "dt": dt,
        "t_pre": t_pre,
        "t_post": t_post,
        "metadata_json": _json_safe(meta),
    })
    if Nw is not None:
        payload["Nw"] = int(Nw)
    if include_postpost_G:
        payload["G_postpost"] = G[n0:, n0:, :, :]
        meta["saved_postpost_G"] = True
        payload["metadata_json"] = _json_safe(meta)

    atomic_savez(path, compressed=compressed, **payload)


def save_equilibrium_result(
    path: str | Path,
    *,
    omega,
    t,
    Ggt,
    GR_w,
    F_t,
    J: float,
    alpha: float,
    mu: float,
    beta: float,
    omega_max: float,
    Nw: int,
    dt: float,
    tol: float,
    converged: bool,
    it: int,
    compressed: bool = True,
    extra_metadata: Optional[Dict[str, Any]] = None,
):
    meta = {
        "J": J,
        "alpha": alpha,
        "mu": mu,
        "beta": beta,
        "omega_max": omega_max,
        "Nw": Nw,
        "dt": dt,
        "tol": tol,
        "converged": bool(converged),
        "it": int(it),
    }
    if extra_metadata:
        meta.update(extra_metadata)
    atomic_savez(
        path,
        compressed=compressed,
        omega=omega,
        t=t,
        t_eq=t,
        Ggt=Ggt,
        Ggt_eq=Ggt,
        GR_w=GR_w,
        F_t=F_t,
        J=J,
        alpha=alpha,
        mu=mu,
        beta=beta,
        omega_max=omega_max,
        Nw=Nw,
        dt=dt,
        tol=tol,
        metadata_json=_json_safe(meta),
    )


def solve_equilibrium_kktz_matrix_F_checkpointed(
    J,
    alpha,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-9,
    dab_tol=1e-6,
    require_dab_convergence=False,
    mixing=0.05,
    eta_ret=1e-6,
    verbose_every=25,
    compute_kbe_dab_every=0,
    kbe_dab_t_cut=None,
    kbe_dab_edge_skip=4,
    kbe_dab_power=0.5,
    init_mode="free",
    init_omega=None,
    init_GR_LL=None,
    init_GR_RR=None,
    init_GR_w=None,
    init_offdiag="zero",
    project_A=True,
    clip_A=True,
    normalize_A=True,
    project_B=True,
    pin_equal_time=True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    final_path: str | Path | None = None,
):
    """Checkpointed KKTZ equilibrium solver using the quartic two-flavor self-energy."""
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]
    Nt = len(t)
    omega = np.linspace(-omega_max, omega_max, Nw)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)
    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    one_minus_nF = 1.0 - nF
    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5

    start_it = 0
    GR_w = None
    ckpt = Path(checkpoint_path) if checkpoint_path else None

    if ckpt and resume and ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        F_t = z["F_t"]
        start_it = int(z["it"]) + 1
        print(f"Resuming equilibrium from {ckpt} at iteration {start_it}")
    else:
        if init_GR_w is not None:
            GR0 = np.asarray(init_GR_w, dtype=complex).copy()
            if GR0.shape != (Nw, 2, 2):
                raise ValueError(
                    f"init_GR_w must have shape {(Nw, 2, 2)}, got {GR0.shape}"
                )
        else:
            GR0 = _build_GR0_mq_seed(
                omega=omega,
                eta_ret=eta_ret,
                mu=mu,
                init_mode=init_mode,
                init_omega=init_omega,
                init_GR_LL=init_GR_LL,
                init_GR_RR=init_GR_RR,
                init_offdiag=init_offdiag,
            )
        A_LL_init = -2.0 * np.imag(GR0[:, 0, 0])
        A_RR_init = -2.0 * np.imag(GR0[:, 1, 1])
        B_LR_init = 2 * np.real(GR0[:, 0, 1])
        B_RL_init = 2 * np.real(GR0[:, 1, 0])
        F_w = np.zeros((Nw, 2, 2), dtype=complex)
        F_w[:, 0, 0] = one_minus_nF * A_LL_init
        F_w[:, 1, 1] = one_minus_nF * A_RR_init
        F_w[:, 0, 1] = 1j * one_minus_nF * B_LR_init
        F_w[:, 1, 0] = 1j * one_minus_nF * B_RL_init
        F_t = _omega_to_time_mat(F_w, omega, t)

    converged = False
    delta = np.inf
    last_dab = None
    last_dab_sqrt_max = np.inf
    last_it = start_it - 1

    # Keep references to the last fully completed iteration. The signal
    # handler saves these immediately when Slurm sends SIGUSR1/SIGTERM, so an
    # interrupted run always has a checkpoint no more than one iteration stale.
    checkpoint_state: Dict[str, Any] = {"F_t": F_t, "it_done": start_it - 1}

    def _checkpoint_then_exit(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        if ckpt is not None and checkpoint_state["it_done"] >= 0:
            atomic_savez(
                ckpt,
                compressed=False,
                it=checkpoint_state["it_done"],
                F_t=checkpoint_state["F_t"],
                omega=omega,
                t=t,
                J=J,
                alpha=alpha,
                mu=mu,
                beta=beta,
                omega_max=omega_max,
                Nw=Nw,
                dt=dt,
                tol=tol,
                delta=delta,
                dab_tol=dab_tol,
                require_dab_convergence=bool(require_dab_convergence),
                last_dab_sqrt_max=last_dab_sqrt_max,
                metadata_json=_json_safe(
                    {"protocol": "equilibrium", "it": checkpoint_state["it_done"]}
                ),
            )
            print(
                f"\n[{signal_name}] KKTZ equilibrium checkpoint saved at "
                f"iter={checkpoint_state['it_done']}; exiting for requeue.",
                flush=True,
            )
        else:
            print(
                f"\n[{signal_name}] No completed KKTZ equilibrium iteration to "
                "checkpoint; exiting.",
                flush=True,
            )
        raise SystemExit(128 + signum)

    for checkpoint_signal in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(checkpoint_signal, _checkpoint_then_exit)

    for it in range(start_it, max_iter):
        last_it = it
        iSgt = i_sigma_greater_kktz_from_F(F_t, J, alpha)
        iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))
        iSR_t = theta[:, None, None] * (iSgt + iSgt_flipped)
        iSR_w = _time_to_omega_mat(iSR_t, t, omega)
        SR_w = -1j * iSR_w

        GR_w = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

        A_LL_new = -2.0 * np.imag(GR_w[:, 0, 0])
        A_RR_new = -2.0 * np.imag(GR_w[:, 1, 1])
        B_LR_new = 2 * np.real(GR_w[:, 0, 1])
        B_RL_new = 2 * np.real(GR_w[:, 1, 0])

        if project_A:
            A_LL_new = 0.5 * (A_LL_new + A_LL_new[::-1])
            A_RR_new = 0.5 * (A_RR_new + A_RR_new[::-1])
        if clip_A:
            A_LL_new = np.clip(A_LL_new, 0.0, None)
            A_RR_new = np.clip(A_RR_new, 0.0, None)
        if normalize_A:
            sum_LL = np.trapz(A_LL_new, omega) / (2 * np.pi)
            sum_RR = np.trapz(A_RR_new, omega) / (2 * np.pi)
            if sum_LL > 0:
                A_LL_new = A_LL_new / sum_LL
            if sum_RR > 0:
                A_RR_new = A_RR_new / sum_RR
        if project_B:
            B_LR_new = 0.5 * (B_LR_new - B_LR_new[::-1])
            B_RL_new = 0.5 * (B_RL_new - B_RL_new[::-1])

        F_w_new = np.zeros((Nw, 2, 2), dtype=complex)
        F_w_new[:, 0, 0] = one_minus_nF * A_LL_new
        F_w_new[:, 1, 1] = one_minus_nF * A_RR_new
        F_w_new[:, 0, 1] = 1j * one_minus_nF * B_LR_new
        F_w_new[:, 1, 0] = 1j * one_minus_nF * B_RL_new
        F_t_new = _omega_to_time_mat(F_w_new, omega, t)

        if pin_equal_time:
            F_t_new[i0, 0, 0] = 0.5
            F_t_new[i0, 1, 1] = 0.5
            F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
            F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])

        delta = np.max(np.abs(F_t_new - F_t))
        F_t = (1.0 - mixing) * F_t + mixing * F_t_new
        Ggt_current = -1j * F_t
        checkpoint_state["F_t"] = F_t
        checkpoint_state["it_done"] = it

        delta_ok = delta < tol
        need_dab = False
        if compute_kbe_dab_every and (it % compute_kbe_dab_every == 0 or delta_ok):
            need_dab = True
        if require_dab_convergence and delta_ok:
            need_dab = True

        if need_dab:
            last_dab = calc_kbe_d_ab_kktz_equilibrium(
                t=t,
                Ggt=Ggt_current,
                J=J,
                alpha=alpha,
                mu=mu,
                t_cut=kbe_dab_t_cut,
                edge_skip=kbe_dab_edge_skip,
                keep_cross_self_energy=True,
                return_details=False,
            )
            last_dab_sqrt_max = float(np.max(np.asarray(last_dab) ** kbe_dab_power))

        if verbose_every and (it % verbose_every == 0 or delta_ok):
            msg = f"{it:5d}  max|dF|={delta:.3e}  F_LL(0)={F_t[i0,0,0].real:.8f}  B=ImF_LR(0)={np.imag(F_t[i0,0,1]):.8f}"
            if last_dab is not None and need_dab:
                msg += f"  KKTZ d_ab^{kbe_dab_power:g}={np.asarray(last_dab)**kbe_dab_power}  max={last_dab_sqrt_max:.3e} tol={dab_tol:.3e}"
            if require_dab_convergence:
                msg += f"  require_dab={require_dab_convergence}"
            print(msg, flush=True)

        if ckpt and checkpoint_every and (it % checkpoint_every == 0):
            atomic_savez(
                ckpt,
                compressed=False,
                it=it,
                F_t=F_t,
                omega=omega,
                t=t,
                J=J,
                alpha=alpha,
                mu=mu,
                beta=beta,
                omega_max=omega_max,
                Nw=Nw,
                dt=dt,
                tol=tol,
                delta=delta,
                dab_tol=dab_tol,
                require_dab_convergence=bool(require_dab_convergence),
                last_dab_sqrt_max=last_dab_sqrt_max,
                metadata_json=_json_safe({"protocol": "equilibrium", "it": it}),
            )

        if delta_ok:
            dab_ok = (not require_dab_convergence) or (last_dab_sqrt_max < dab_tol)
            if dab_ok:
                converged = True
                break

    if GR_w is None:
        # Happens only if max_iter was already completed by a resumed checkpoint.
        iSgt = i_sigma_greater_kktz_from_F(F_t, J, alpha)
        iSR_t = theta[:, None, None] * (iSgt + np.transpose(iSgt[::-1], (0, 2, 1)))
        SR_w = -1j * _time_to_omega_mat(iSR_t, t, omega)
        GR_w = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

    Ggt = -1j * F_t
    if final_path is not None:
        save_equilibrium_result(
            final_path,
            omega=omega,
            t=t,
            Ggt=Ggt,
            GR_w=GR_w,
            F_t=F_t,
            J=J,
            alpha=alpha,
            mu=mu,
            beta=beta,
            omega_max=omega_max,
            Nw=Nw,
            dt=dt,
            tol=tol,
            converged=converged,
            it=last_it,
            extra_metadata={
                "dab_tol": float(dab_tol),
                "require_dab_convergence": bool(require_dab_convergence),
                "final_delta": float(delta),
                "final_dab_sqrt_max": None if not np.isfinite(last_dab_sqrt_max) else float(last_dab_sqrt_max),
            },
        )

    if converged and ckpt is not None and ckpt.exists():
        ckpt.unlink()
        print("KKTZ equilibrium checkpoint removed (converged).")
    elif not converged and ckpt is not None and checkpoint_state["it_done"] >= 0:
        print(
            f"KKTZ equilibrium checkpoint retained at iter={checkpoint_state['it_done']} "
            f"(not converged): {ckpt}"
        )

    return omega, t, Ggt, GR_w, F_t


def evolve_kktz_kbe_matrix_checkpointed(
    t_eq,
    Ggt_eq,
    *,
    J_i=1.0,
    J_f=1.0,
    alpha_i=0.0,
    alpha_f=0.0,
    mu_i=0.075,
    mu_f=0.0,
    beta=np.nan,
    t_pre=80.0,
    t_post=40.0,
    dt=0.1,
    n_corr=6,
    corr_tol=1e-10,
    print_every=50,
    keep_cross_self_energy=True,
    checkpoint_path: str | Path | None = None,
    checkpoint_every: int = 25,
    resume: bool = True,
    final_path: str | Path | None = None,
    Nw: int | None = None,
    save_full_final_G: bool = False,
    include_postpost_G: bool = False,
):
    """Checkpointed KKTZ two-time KBE evolution.

    Checkpoint file stores the full in-progress G so the job can resume.
    Final file is lean by default: post-quench equal-time LR and O1 only.
    """
    t = np.arange(-t_pre, t_post + 0.5 * dt, dt)
    n0 = int(np.argmin(np.abs(t)))
    t = t - t[n0]
    Nt = len(t)
    ckpt = Path(checkpoint_path) if checkpoint_path else None

    J_of_t = np.where(t < 0.0, J_i, J_f)
    alpha_of_t = np.where(t < 0.0, alpha_i, alpha_f)
    mu_of_t = np.where(t < 0.0, mu_i, mu_f)
    h_list = np.array([h_mu_matrix(m) for m in mu_of_t])

    if ckpt and resume and ckpt.exists():
        z = np.load(ckpt, allow_pickle=True)
        G = z["G"]
        n_last = int(z["n_last"])
        if G.shape != (Nt, Nt, 2, 2):
            raise ValueError(f"Checkpoint G shape {G.shape} incompatible with requested {(Nt, Nt, 2, 2)}")
        print(f"Resuming KBE from {ckpt}; last completed index n={n_last}, t={t[n_last]:.6g}")
    else:
        max_needed_t = np.max(np.abs(t[: n0 + 1, None] - t[None, : n0 + 1]))
        if max_needed_t > np.max(np.abs(t_eq)):
            print("WARNING: t_eq does not cover needed equilibrium time range; interpolation will extrapolate.")
        G = fill_equilibrium_block(t, n0, t_eq, Ggt_eq)
        n_last = n0

    # Keep a reference to the last fully completed time-step index. The
    # signal handler saves G at that index immediately when Slurm sends
    # SIGUSR1/SIGTERM, so an interrupted run is never more than one
    # checkpoint_every window stale.
    checkpoint_state: Dict[str, Any] = {"n_last": n_last}

    def _checkpoint_then_exit(signum: int, _frame: Any) -> None:
        signal_name = signal.Signals(signum).name
        n_done = checkpoint_state["n_last"]
        if ckpt is not None and n_done > n0:
            atomic_savez(
                ckpt,
                compressed=False,
                G=G,
                t=t,
                n0=n0,
                n_last=n_done,
                J_i=J_i,
                J_f=J_f,
                alpha_i=alpha_i,
                alpha_f=alpha_f,
                mu_i=mu_i,
                mu_f=mu_f,
                beta=beta,
                dt=dt,
                t_pre=t_pre,
                t_post=t_post,
                Nw=-1 if Nw is None else int(Nw),
                metadata_json=_json_safe(
                    {"protocol": "kbe", "n_last": n_done, "t_last": float(t[n_done])}
                ),
            )
            print(
                f"\n[{signal_name}] KKTZ KBE checkpoint saved at "
                f"n={n_done}, t={t[n_done]:.6g}; exiting for requeue.",
                flush=True,
            )
        else:
            print(
                f"\n[{signal_name}] No completed KKTZ KBE step beyond the "
                "equilibrium block to checkpoint; exiting.",
                flush=True,
            )
        raise SystemExit(128 + signum)

    for checkpoint_signal in (signal.SIGUSR1, signal.SIGTERM):
        signal.signal(checkpoint_signal, _checkpoint_then_exit)

    for n in range(n_last + 1, Nt):
        S = sigma_greater_kktz(G, J_of_t, alpha_of_t, keep_cross_self_energy=keep_cross_self_energy)

        d1_old = np.array([rhs_t1_matrix(G, S, h_list, n - 1, j, dt) for j in range(n)])
        d2_old = np.array([rhs_t2_matrix(G, S, h_list, i, n - 1, dt) for i in range(n)])
        ddiag_old = rhs_t1_matrix(G, S, h_list, n - 1, n - 1, dt) + rhs_t2_matrix(G, S, h_list, n - 1, n - 1, dt)

        G[n, :n] = G[n - 1, :n] + dt * d1_old
        G[:n, n] = G[:n, n - 1] + dt * d2_old
        G[n, n] = G[n - 1, n - 1] + dt * ddiag_old
        enforce_equal_time_majorana(G, n)

        last_err = np.inf
        for _ in range(n_corr):
            row_old = G[n, :n].copy()
            col_old = G[:n, n].copy()
            diag_old = G[n, n].copy()

            S = sigma_greater_kktz(G, J_of_t, alpha_of_t, keep_cross_self_energy=keep_cross_self_energy)
            d1_new = np.array([rhs_t1_matrix(G, S, h_list, n, j, dt) for j in range(n)])
            d2_new = np.array([rhs_t2_matrix(G, S, h_list, i, n, dt) for i in range(n)])
            ddiag_new = rhs_t1_matrix(G, S, h_list, n, n, dt) + rhs_t2_matrix(G, S, h_list, n, n, dt)

            G[n, :n] = G[n - 1, :n] + 0.5 * dt * (d1_old + d1_new)
            G[:n, n] = G[:n, n - 1] + 0.5 * dt * (d2_old + d2_new)
            G[n, n] = G[n - 1, n - 1] + 0.5 * dt * (ddiag_old + ddiag_new)
            enforce_equal_time_majorana(G, n)

            last_err = max(
                np.max(np.abs(G[n, :n] - row_old)),
                np.max(np.abs(G[:n, n] - col_old)),
                np.max(np.abs(G[n, n] - diag_old)),
            )
            if last_err < corr_tol:
                break

        checkpoint_state["n_last"] = n

        if print_every and (n - n0) % print_every == 0:
            B = -np.real(G[n, n, 0, 1])
            print(f"evolved to t={t[n]:.3f}; n={n}/{Nt-1}; B={B:.8e}; corr_err={last_err:.3e}", flush=True)

        if ckpt and checkpoint_every and ((n - n0) % checkpoint_every == 0 or n == Nt - 1):
            atomic_savez(
                ckpt,
                compressed=False,
                G=G,
                t=t,
                n0=n0,
                n_last=n,
                J_i=J_i,
                J_f=J_f,
                alpha_i=alpha_i,
                alpha_f=alpha_f,
                mu_i=mu_i,
                mu_f=mu_f,
                beta=beta,
                dt=dt,
                t_pre=t_pre,
                t_post=t_post,
                Nw=-1 if Nw is None else int(Nw),
                metadata_json=_json_safe({"protocol": "kbe", "n_last": n, "t_last": float(t[n])}),
            )

    if final_path is not None:
        if save_full_final_G:
            atomic_savez(
                final_path,
                compressed=True,
                t_kbe=t,
                Ggt=G,
                J_i=J_i,
                J_f=J_f,
                alpha_i=alpha_i,
                alpha_f=alpha_f,
                mu_i=mu_i,
                mu_f=mu_f,
                beta=beta,
                dt=dt,
                t_pre=t_pre,
                t_post=t_post,
                Nw=-1 if Nw is None else int(Nw),
                metadata_json=_json_safe({"saved_full_G": True}),
            )
        else:
            save_kbe_lean(
                final_path,
                t,
                G,
                J_i=J_i,
                J_f=J_f,
                alpha_i=alpha_i,
                alpha_f=alpha_f,
                mu_i=mu_i,
                mu_f=mu_f,
                beta=beta,
                dt=dt,
                t_pre=t_pre,
                t_post=t_post,
                Nw=Nw,
                include_postpost_G=include_postpost_G,
                compressed=True,
            )

    if ckpt is not None and ckpt.exists():
        ckpt.unlink()
        print("KKTZ KBE checkpoint removed (run complete).")

    return t, G
