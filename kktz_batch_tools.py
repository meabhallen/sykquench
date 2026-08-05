#!/usr/bin/env python3
"""Batch tools for KKTZ quartic-coupled SYK equilibrium and real-time KBE runs.

Designed to mirror the single-SYK workflow in syk_batch_tools.py:

    python kktz_batch_tools.py eq-one  --J 1.0 --mu 0.001 --beta 20 --dt 0.005 --omega-max 8 --Nw 4097 --tol 1e-10
    python kktz_batch_tools.py kbe-one --J-i 1.0 --J-f 1.0 --mu-i 0.001 --mu-f 0.0 --beta 20 --dt 0.005

Equilibrium output saves the full relative-time 2x2 Green's function G^>_ab(t).
KBE output saves the full two-time matrix G^>_ab(t1,t2), shape (Nt,Nt,2,2), by default.
Checkpoints also store the full in-progress G so a requeued/timed-out job can resume.

Self-contained: the matrix Green's-function plumbing (formerly mq_kbe_patch.py)
and the checkpointed KKTZ solvers (formerly kktz_checkpoint_utils.py) live in
this file.
"""
from __future__ import annotations

from pathlib import Path
from itertools import product
import argparse
import hashlib
import json
import os
import traceback
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
if not hasattr(np, "trapz") and hasattr(np, "trapezoid"):
    np.trapz = np.trapezoid
import pandas as pd
from scipy.interpolate import interp1d
import signal


# ============================================================
# Helper utilities
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


def atomic_savez(filename: os.PathLike | str, *, compressed: bool = True, **kwargs: Any) -> None:
    """Atomic npz save by writing to a temp file then renaming."""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    tmp = filename.with_name(filename.name + ".tmp")
    saver = np.savez_compressed if compressed else np.savez
    with open(tmp, "wb") as f:
        saver(f, **kwargs)
    os.replace(tmp, filename)


def checkpoint_path_for(npz_path: os.PathLike | str) -> Path:
    p = Path(npz_path)
    stem = p.name[:-4] if p.name.endswith(".npz") else p.name
    return p.parent / (stem + ".ckpt.npz")


def update_manifest(manifest_path: os.PathLike | str, row: Dict[str, Any]) -> None:
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


def load_manifests_recursive(root: os.PathLike | str, manifest_name: str) -> pd.DataFrame:
    """Load manifest_name from root and all subdirectories; useful when each Slurm job has its own OUT_DIR."""
    root = Path(root)
    frames: List[pd.DataFrame] = []
    for p in sorted(root.rglob(manifest_name)):
        try:
            df = pd.read_csv(p)
            if len(df):
                frames.append(df)
        except Exception as e:
            print(f"Warning: could not read manifest {p}: {e}")
    if not frames:
        direct = root / manifest_name
        return load_manifest(direct)
    out = pd.concat(frames, ignore_index=True)
    if "filename" in out.columns:
        out = out.drop_duplicates(subset=["filename"], keep="last")
    return out


def grid_from_dict(d: Dict[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    keys = list(d.keys())
    vals = [list(d[k]) for k in keys]
    return [dict(zip(keys, combo)) for combo in product(*vals)]


def _json_safe(meta: Dict[str, Any]) -> str:
    def default(o: Any):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return str(o)
    return json.dumps(meta, sort_keys=True, default=default)


def _read_npz_scalar(z: np.lib.npyio.NpzFile, key: str, default: Any = None) -> Any:
    if key not in z.files:
        return default
    arr = z[key]
    try:
        return arr.item()
    except Exception:
        return arr



# ============================================================
# Physics / matrix Green's-function plumbing (formerly mq_kbe_patch.py)
# ============================================================

def _trap_weights_uniform_from_dx(n, dx):
    w = np.ones(n, dtype=float) * dx
    if n:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w

def _trap_weights_grid(x):
    x = np.asarray(x)
    dx = x[1] - x[0]
    return _trap_weights_uniform_from_dx(len(x), dx)

''' Fourier transforms, same as single SYK'''

def _time_to_omega_mat(X_t, t, omega, chunk=384):
    t = np.asarray(t)
    omega = np.asarray(omega)
    wt = _trap_weights_grid(t)
    Xt = np.asarray(X_t) * wt[:, None, None]
    out = np.empty((len(omega), 2, 2), dtype=complex)
    for a in range(0, len(omega), chunk):
        phase = np.exp(1j * np.outer(omega[a:a+chunk], t))
        out[a:a+chunk] = np.einsum("ot,tab->oab", phase, Xt, optimize=True)
    return out


def _omega_to_time_mat(X_w, omega, t, chunk=384):
    omega = np.asarray(omega)
    t = np.asarray(t)
    ww = _trap_weights_grid(omega) / (2.0 * np.pi)
    Xw = np.asarray(X_w) * ww[:, None, None]
    out = np.empty((len(t), 2, 2), dtype=complex)
    for a in range(0, len(t), chunk):
        phase = np.exp(-1j * np.outer(t[a:a+chunk], omega))
        out[a:a+chunk] = np.einsum("to,oab->tab", phase, Xw, optimize=True)
    return out


def h_mu_matrix(mu):
    # h in i d_t G = h G + ... ; h = i * mu_matrix, real antisymmetric matrix appearing in d psi / dt = mu_matrix psi
    return 1j * np.array([[0.0, mu], [-mu, 0.0]], dtype=complex)

def get_G_Sigma_mq_matsubara(
    J, mu, beta,
    q=4,
    N_half=4*512,
    max_iter=2000,
    tol=1e-10,
    mixing=0.03,iterations=50,
):
    """
    Imaginary-time SD self-consistency for the Maldacena-Qi model.

    Self-energy (diagonal only):
        Sigma_LL(tau) = J^2 * [2 G_LL(tau)]^{q-1}
        Sigma_LR = 0

    Dyson (2x2 matrix inversion with mu contact term):
        D(iw) = (iw - Sigma_LL)^2 - mu^2
        G_LL(iw) = (iw - Sigma_LL) / D
        G_LR(iw) = -i*mu / D

    The self-consistency loop runs on G_LL only.
    G_LR is computed from G_LL at convergence (no separate loop).

    Fourier convention: G(tau) = (1/beta) sum_n G(iw_n) exp(+i w_n tau)
    => free G_LL(0+) = +1/2  (matches MQ paper section 5.1)

    Returns
    -------
    tau_grid    : imaginary times in [0, beta)
    G_LL_tau    : real array
    G_LR_tau    : complex array (purely imaginary at convergence)
    omega_n     : Matsubara frequencies
    Sigma_LL_iw : self-energy in Matsubara space
    """
    N        = 2 * N_half
    n_arr    = np.arange(-N_half, N_half)
    omega_n  = (2 * n_arr + 1) * np.pi / beta
    tau_grid = np.linspace(0, beta, N, endpoint=False)
    dtau     = beta / N

    # Fourier matrices with the +i w tau convention
    M_iw2tau = np.exp(+1j * np.outer(omega_n, tau_grid)).T / beta  # (N_tau, N_freq) #. T means transpose, np.outer takes every combo 
    M_tau2iw = np.exp(-1j * np.outer(tau_grid, omega_n)).T * dtau  # (N_freq, N_tau)

    G_LL_iw = 1.0 / (1j * omega_n)   # free propagator seed

    print(f"\nMQ Matsubara SD: J={J}, mu={mu}, beta={beta}, q={q}")
    print(f"{'Iter':>5}  {'||delta G_LL||':>15}  {'G_LL(b/2)':>12}  {'G_LL(0+)':>10}")
    print("-" * 50)

    for it in range(max_iter):
        G_LL_tau  = np.real(M_iw2tau @ G_LL_iw)

        # Diagonal self-energy: SYK4 on each dot independently
        Sigma_LL_tau = J**2 * ( G_LL_tau)**(q - 1)
        Sigma_LL_iw  = M_tau2iw @ Sigma_LL_tau

        # 2x2 Dyson: mu enters as off-diagonal mass, not self-energy
        A_iw     = 1j * omega_n - Sigma_LL_iw
        D_iw     = A_iw**2 - mu**2
        G_LL_new = A_iw / D_iw

        delta    = np.mean(np.abs(G_LL_new - G_LL_iw))
        G_LL_iw  = (1.0 - mixing) * G_LL_iw + mixing * G_LL_new

        if it % iterations == 0 or delta < tol:
            G_now = np.real(M_iw2tau @ G_LL_iw)
            print(f"{it:5d}  {delta:15.3e}  {G_now[N_half]:12.6f}  {G_now[1]:10.5f}")

        if delta < tol:
            print(f"\nConverged at iteration {it}.")
            break
    else:
        print(f"\nDid not converge within {max_iter} iterations.")

    # Final G_LL and Sigma_LL
    G_LL_tau     = np.real(M_iw2tau @ G_LL_iw)
    Sigma_LL_tau = J**2 * ( G_LL_tau)**(q - 1)
    Sigma_LL_iw  = M_tau2iw @ Sigma_LL_tau

    # G_LR from Dyson: no loop needed, fully determined by G_LL
    A_iw     = 1j * omega_n - Sigma_LL_iw
    D_iw     = A_iw**2 - mu**2
    G_LR_iw  = (-1j * mu) / D_iw
    G_LR_tau = M_iw2tau @ G_LR_iw   # purely imaginary at convergence

    print(f"\nFinal checks:")
    print(f"  G_LL(0+)      = {G_LL_tau[1]:.6f}  (should be ~+0.5)")
    print(f"  G_LL(beta/2)  = {G_LL_tau[N_half]:.6f}")
    print(f"  max|Re G_LR|  = {np.max(np.abs(np.real(G_LR_tau))):.2e}  (should be ~0)")
    print(f"  iG_LR(0+)     = {np.imag(G_LR_tau[1]):.6f}")

    return tau_grid, G_LL_tau, G_LR_tau, omega_n, Sigma_LL_iw


def _fd_factor(beta, omega):
    # 1/(exp(-beta omega)+1) = 1 - nF, stable
    x = np.clip(-beta * omega, -500, 500)
    return 1.0 / (np.exp(x) + 1.0)


def wightman_from_GR_mq(omega, GR, beta):
    """Frequency-space G^>(omega) from the MQ/Milekhin component FDT."""
    fac = _fd_factor(beta, omega)
    Ggt_w = np.zeros_like(GR, dtype=complex)

    # diagonal FDT, eq. (102)
    Ggt_w[:, 0, 0] = fac * (2j * np.imag(GR[:, 0, 0]))
    Ggt_w[:, 1, 1] = fac * (2j * np.imag(GR[:, 1, 1]))

    # off-diagonal FDT: eq. (103)
    Ggt_w[:, 0, 1] = fac * (1 * np.real(GR[:, 0, 1]))
    Ggt_w[:, 1, 0] = fac * (1 * np.real(GR[:, 0, 1]))
    return Ggt_w

def retarded_self_energy_from_greater(Sgt, t):
    i0 = int(np.argmin(np.abs(t)))
    theta = np.zeros_like(t, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5
    Sless = -np.transpose(Sgt[::-1], (0, 2, 1))
    return  theta[:, None, None] * (Sgt - Sless)  # note -i factor, -1j *theta[:, None, None] * (Sgt - Sless)
def _interp_complex_to_grid(x_old, y_old, x_new, kind="linear"):
    """
    Interpolate a complex-valued function y_old(x_old) onto x_new.
    """
    x_old = np.asarray(x_old)
    y_old = np.asarray(y_old)

    f_re = interp1d(
        x_old, np.real(y_old),
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    f_im = interp1d(
        x_old, np.imag(y_old),
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return f_re(x_new) + 1j * f_im(x_new)


def _build_GR0_mq_seed(
    omega,
    eta_ret,
    mu,
    *,
    init_mode="free",
    init_omega=None,
    init_GR_LL=None,
    init_GR_RR=None,
    init_offdiag="zero",
):
    """
    Build initial 2x2 retarded Green function GR0[k,a,b].

    init_mode options:
        "free":
            GR0 = [(w+i eta0) I]^{-1}

        "uncoupled_syk":
            Use supplied single-SYK GR for LL/RR.

    init_offdiag options for init_mode="uncoupled_syk":
        "zero":
            GR_LR = GR_RL = 0

        "linear_mu":
            GR_LR ~ g_L h_LR g_R, linear in mu

        "dyson_mu":
            GR0 = [diag(g_L^{-1}, g_R^{-1}) - h]^{-1},
            i.e. resum the bilinear mu in the seed.
    """
    omega = np.asarray(omega)
    Nw = len(omega)

    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)
    eta0 = max(eta_ret, 0.05)

    GR0 = np.zeros((Nw, 2, 2), dtype=complex)

    if init_mode == "free":
        for k, w in enumerate(omega):
            GR0[k] = np.linalg.inv((w + 1j * eta0) * eye)
        return GR0

    if init_mode != "uncoupled_syk":
        raise ValueError(
            "init_mode must be 'free' or 'uncoupled_syk', "
            f"got {init_mode!r}"
        )

    if init_omega is None or init_GR_LL is None:
        raise ValueError(
            "For init_mode='uncoupled_syk', pass init_omega and init_GR_LL."
        )

    if init_GR_RR is None:
        init_GR_RR = init_GR_LL

    gL = _interp_complex_to_grid(init_omega, init_GR_LL, omega)
    gR = _interp_complex_to_grid(init_omega, init_GR_RR, omega)

    if init_offdiag == "zero":
        GR0[:, 0, 0] = gL
        GR0[:, 1, 1] = gR
        GR0[:, 0, 1] = 0.0
        GR0[:, 1, 0] = 0.0

    elif init_offdiag == "linear_mu":
        GR0[:, 0, 0] = gL
        GR0[:, 1, 1] = gR

        # Since Dyson is D = D0 - h, inverse to first order is
        # G ≈ G0 + G0 h G0.
        GR0[:, 0, 1] = gL * h[0, 1] * gR
        GR0[:, 1, 0] = gR * h[1, 0] * gL

    elif init_offdiag == "dyson_mu":
        for k in range(Nw):
            D = np.array(
                [
                    [1.0 / gL[k], 0.0],
                    [0.0, 1.0 / gR[k]],
                ],
                dtype=complex,
            ) - h
            GR0[k] = np.linalg.inv(D)

    else:
        raise ValueError(
            "init_offdiag must be 'zero', 'linear_mu', or 'dyson_mu', "
            f"got {init_offdiag!r}"
        )

    return GR0

def solve_equilibrium_mq_real_time_matrix_v2(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-8,
    mixing=0.02,
    eta_ret=1e-4,
    verbose_every=25,
    Ggt_init=None,
    t_init=None,
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ using FDT.

    Returns:
        omega, t, Ggt_t, GR_w, Ggt_w
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    # Use an odd number of time points centered on zero.
    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]

    omega = np.linspace(-omega_max, omega_max, Nw)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)

    # Initial guess: noninteracting coupled Majoranas, broadened by eta_ret.
    if Ggt_init is not None and t_init is not None:
        Ggt = np.empty((len(t), 2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                f = interp1d(t_init, Ggt_init[:, a, b], kind="cubic",
                             bounds_error=False, fill_value="extrapolate")
                Ggt[:, a, b] = f(t)
    else:
        GR0 = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            #GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye - h)
            GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye +h)
        Ggt_w0 = wightman_from_GR_mq(omega, GR0, beta)
        Ggt = _omega_to_time_mat(Ggt_w0, omega, t)
    print("Initial Ggt_w_LR at k=Nw//2+100:", Ggt_w0[Nw//2+100, 0, 1])
    print("Initial Ggt_LR t=0:", Ggt[len(t)//2, 0, 1])
    print("Initial Ggt_LR t=dt:", Ggt[len(t)//2+1, 0, 1])
    print("Initial GR0_LR at k=Nw//2+100:", GR0[Nw//2+100, 0, 1])

    GR = None
    Ggt_w = None
    for it in range(max_iter):
        Sgt = -J**2 * Ggt**3
        #Sgt = -J**2 * Ggt**3 + h
        Sigma_R_t = retarded_self_energy_from_greater(Sgt, t)
        Sigma_R_w = _time_to_omega_mat(Sigma_R_t, t, omega)

        GR_new = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_new[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - Sigma_R_w[k])

        Ggt_w_new = wightman_from_GR_mq(omega, GR_new, beta)
        Ggt_new = _omega_to_time_mat(Ggt_w_new, omega, t)

        # enforce exact equal-time Majorana normalization on the diagonal only;
        # do not touch LR/RL.
        Ggt_new[i0, 0, 0] = -0.5j
        Ggt_new[i0, 1, 1] = -0.5j

        delta = np.max(np.abs(Ggt_new - Ggt))
        Ggt = (1.0 - mixing) * Ggt + mixing * Ggt_new
        GR = GR_new
        Ggt_w = Ggt_w_new

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            B = -np.real(Ggt[i0, 0, 1])
            print(f"{it:5d}  max|dG|={delta:.3e}  iGLL0={1j*Ggt[i0,0,0]:.8f}  B=Re GLR0={-B:.8f}")
        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    return omega, t, Ggt, GR, Ggt_w

def _ds_residual_mq(F_t, iSgt, t, omega, eta_ret, h, eye, one_minus_nF, i0):
    """
    Compute the DS/convergence residual d_ab as in Eq. (104) of the paper.

    The paper defines (Eq. 104, translated to real-time / F = iG^> language):

        d_ab = (1/Npoints) * || d_tau F_ab(tau) - (iSigma^R * F)_ab(tau) - (F * iSigma^A)_ab(tau)
                                - i h_ac F_cb(tau) + delta_ab delta(tau) ||_2

    In real time with F = iG^>:
        i d_t G^>_ab = h_ac G^>_cb + (Sigma^R * G^>)_ab + (Sigma^> * G^A)_ab     [Eq. 53, left branch]

    Multiplying through by i:
        -d_t F_ab = i h_ac F_cb + (iSigma^R * F)_ab + (iSigma^> * iG^A)_ab

    In equilibrium, the FDT ties together iG^> = F and G^A, so we check
    self-consistency in frequency space where the DS equation is algebraic:

        [(omega - h - Sigma^R) G^R]_ab = delta_ab   (Dyson equation)

    The most natural residual matching Eq. (104) is therefore the frequency-domain
    DS discrepancy for the diagonal spectral function A_aa = -2 Im G^R_aa
    (which should integrate to 1/(2pi) by the sum rule) and the time-domain
    discrepancy of the DS equation of motion for F, measured per-point as:

        d_aa  = (1/Nt) * || d_t F_aa - [iSR * F]_aa ||_2   (diagonal)
        d_LR  = (1/Nt) * || d_t F_LR - i*mu*F_RL - [iSR * F]_LR ||_2   (off-diagonal)

    Here we evaluate a simpler but faithful proxy: the RMS of
        residual_ab(t) = (d/dt - i*h) F_ab(t)  -  (iSigma^R * F)_ab(t)
    computed in frequency space where convolution is multiplication:
        residual_ab(omega) = (-i*omega - h) F_ab(omega) - iSigma^R(omega) F_ab(omega)
                           = [(-i*omega)*eye - h - iSR_w] F_w  +  F_w  (from the Dyson identity = eye)
    This reduces to checking how well (omega*eye - h - SR_w) G_R = eye, which is
    exact by construction. Instead we directly measure the DS equation violation
    in time: for each ab component we compute

        d_ab = (1/Nt) * sum_t |d_t F_ab(t) - i*h_ac F_cb(t) - (iSR * F)_ab(t)|

    which we approximate via finite differences for d_t and trapezoidal convolution
    for (iSR * F).  To keep cost low we evaluate it only at the *current* iterate.
    """
    Nt = len(t)
    dt = t[1] - t[0]

    # Finite-difference time derivative of F_t: shape (Nt, 2, 2)
    # Use centred differences in the interior, one-sided at boundaries.
    dF = np.empty_like(F_t)
    dF[1:-1] = (F_t[2:] - F_t[:-2]) / (2 * dt)
    dF[0]    = (F_t[1]  - F_t[0])   / dt
    dF[-1]   = (F_t[-1] - F_t[-2])  / dt

    # i h F term: (i h)_ac F_cb  -- h is already i*mu_matrix so i*h = i*(i*mu_mat) = -mu_mat
    # actually h = h_mu_matrix(mu) = i * [[0,mu],[-mu,0]], so
    # the contribution to d_t F from the mu term is  i*h @ F = i*(i*mu_mat) @ F = -mu_mat @ F.
    # But the full EOM is  d_t F = i*h @ F + (iSR * F),
    # so residual = d_t F - i*h @ F - (iSR * F).
    ih = 1j * h  # shape (2,2) -- this is -mu_matrix, but let's keep it general

    # (iSR * F) convolution in time: (iSR * F)_ab(t1) = int dt' iSR_ac(t1-t') F_cb(t')
    # We only need the diagonal strip so approximate by a simple running trapezoid sum.
    # For the residual we use only the causal (t1>=t') part since iSR is retarded.
    w = _trap_weights_uniform_from_dx(Nt, dt)
    iSR_t = iSgt.copy()  # shape (Nt, 2, 2) -- we reuse the already computed iSgt here
    # Build iSigma^R in time (retarded part only)
    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5
    iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))
    iSR_t_full = theta[:, None, None] * (iSgt + iSgt_flipped)

    conv = np.zeros((Nt, 2, 2), dtype=complex)
    for n in range(Nt):
        # causal sum: t' from t[0] to t[n] where t[n] >= t'
        # iSR(t[n]-t') is nonzero for t[n]-t' >= 0, i.e. t' <= t[n]
        n_causal = n + 1
        for m in range(n_causal):
            dt_rel = t[n] - t[m]
            # find index in iSR_t_full closest to dt_rel
            idx = int(round((dt_rel - t[0]) / dt))
            idx = max(0, min(Nt - 1, idx))
            conv[n] += w[m] * (iSR_t_full[idx] @ F_t[m])

    residual = dF - (ih[None, :, :] @ F_t) - conv
    Npoints = Nt

    # d_ab per component (Eq. 104 spirit: RMS over time points)
    d = {}
    for a, aname in enumerate(['L', 'R']):
        for b, bname in enumerate(['L', 'R']):
            d[f'd_{aname}{bname}'] = np.sqrt(np.mean(np.abs(residual[:, a, b])**2))

    return d


def _enforce_spectral(A, omega):
    """
    Enforce the three exact symmetry constraints on a diagonal SYK spectral function:
      1. Even:        A(omega) = A(-omega)
      2. Non-negative: A(omega) >= 0
      3. Sum rule:    int A(omega) dw/(2pi) = 1

    These are exact properties of A_aa = -2 Im G^R_aa for any single SYK or MQ diagonal
    component.  Enforcing them at each iteration mirrors what solve_equilibrium_greater_real_time
    does in the single-SYK notebook and prevents the spectral function drifting negative
    or off the sum rule, which would destabilise the iteration.
    """
    # 1. Symmetrise (A should be exactly even; any asymmetry is numerical noise)
    A = 0.5 * (A + A[::-1])
    # 2. Clip (causality / positivity of the spectral function)
    A = np.clip(A, 0.0, None)
    # 3. Normalise (sum rule: fermion anti-commutator = 1 in units where {psi,psi}=1)
    dw = omega[1] - omega[0]
    sum_A = (np.sum(A) - 0.5 * A[0] - 0.5 * A[-1]) * dw / (2.0 * np.pi)  # trapezoid
    if sum_A > 0:
        A = A / sum_A
    return A


def _enforce_odd(B):
    """
    Enforce exact odd symmetry on the off-diagonal "spectral" function
    B_LR(omega) = Re G^R_LR(omega).

    This follows from G^R_LR(omega)^* = -G^R_LR(-omega) (consequence of the Z4
    symmetry psi_L -> -psi_R, psi_R -> psi_L combined with the retarded GF reality
    condition), which forces Re G^R_LR to be odd and Im G^R_LR to be even.
    Any even contamination in Re G^R_LR is numerical noise and should be projected out.
    """
    return 0.5 * (B - B[::-1])

def _residual_even(B):
    """
    Checking even contamination in Re G^R_LR.
    """
    return 0.5 * (B + B[::-1])

def solve_equilibrium_mq_matrix_F(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-9,
    mixing=0.05,
    eta_ret=1e-6,
    verbose_every=25,
    compute_kbe_dab_every=50,
    kbe_dab_t_cut=None,
    kbe_dab_edge_skip=4,
    kbe_dab_power=0.5,
    return_dab_history=False,    
    init_mode="free",
    init_omega=None,
    init_GR_LL=None,
    init_GR_RR=None,
    init_offdiag="zero",
    project_A=True,
    clip_A=True,
    normalize_A=True,
    project_B=True,
    pin_equal_time=True,    
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ q=4.

    Same as before, but now also computes the actual real-time KBE residual
    using calc_kbe_d_ab_mq_equilibrium every `compute_kbe_dab_every` iterations.

    The printed quantity is d_ab**kbe_dab_power. Default kbe_dab_power=0.5
    prints the RMS residual, since calc_kbe_d_ab_mq_equilibrium returns mean |res|^2.

    Returns:
        omega, t, Ggt, GR_w, F_t

    If return_dab_history=True:
        omega, t, Ggt, GR_w, F_t, dab_history
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]
    Nt = len(t)
    Nw_ = Nw

    omega = np.linspace(-omega_max, omega_max, Nw_)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)

    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    one_minus_nF = 1.0 - nF

    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5

    # ---- initial guess ----
    GR0 = _build_GR0_mq_seed(
    omega=omega,
    eta_ret=eta_ret,
    mu=mu,
    init_mode=init_mode,
    init_omega=init_omega,
    init_GR_LL=init_GR_LL,
    init_GR_RR=init_GR_RR,
    init_offdiag=init_offdiag,)

    #A_LL_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 0, 0]), omega)
    #A_RR_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 1, 1]), omega)
    #B_LR_init = _enforce_odd(np.real(GR0[:, 0, 1]))
    #B_RL_init = _enforce_odd(np.real(GR0[:, 1, 0]))
    A_LL_init = -2.0 * np.imag(GR0[:, 0, 0])
    A_RR_init = -2.0 * np.imag(GR0[:, 1, 1])
    B_LR_init = 2*np.real(GR0[:, 0, 1])
    B_RL_init = 2*np.real(GR0[:, 1, 0])
    

    F_w = np.zeros((Nw_, 2, 2), dtype=complex)
    F_w[:, 0, 0] = one_minus_nF * A_LL_init
    F_w[:, 1, 1] = one_minus_nF * A_RR_init
    F_w[:, 0, 1] = 1j * one_minus_nF * B_LR_init
    F_w[:, 1, 0] = 1j * one_minus_nF * B_RL_init

    F_t = _omega_to_time_mat(F_w, omega, t)

    GR_w = None
    delta = np.inf
    dab_history = []

    for it in range(max_iter):

        # i Sigma^>_ab(t) = J^2 F_ab(t)^3
        iSgt = J**2 * F_t**3

        # i Sigma^R_ab(t) = theta(t)[i Sigma^>_ab(t) + i Sigma^>_ba(-t)]
        iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))
        iSR_t = theta[:, None, None] * (iSgt + iSgt_flipped)

        iSR_w = _time_to_omega_mat(iSR_t, t, omega)
        SR_w = -1j * iSR_w

        # Dyson equation
        GR_w = np.empty((Nw_, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

        # FDT
        #A_LL_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 0, 0]), omega)
        #A_RR_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 1, 1]), omega)
        #B_LR_new = _enforce_odd(np.real(GR_w[:, 0, 1]))
        #B_LR_res = _residual_even(np.real(GR_w[:, 0, 1]))
        #B_RL_new = _enforce_odd(np.real(GR_w[:, 1, 0]))
        A_LL_new = -2.0 * np.imag(GR_w[:, 0, 0])
        A_RR_new = -2.0 * np.imag(GR_w[:, 1, 1])
        B_LR_new = 2*np.real(GR_w[:, 0, 1])
        B_RL_new = 2*np.real(GR_w[:, 1, 0])

        if project_A:
            A_LL_new = 0.5 * (A_LL_new + A_LL_new[::-1])
            A_RR_new = 0.5 * (A_RR_new + A_RR_new[::-1])

        if clip_A:
            A_LL_new = np.clip(A_LL_new, 0.0, None)
            A_RR_new = np.clip(A_RR_new, 0.0, None)

        if normalize_A:
            sum_LL = np.trapz(A_LL_new, omega) / (2*np.pi)
            sum_RR = np.trapz(A_RR_new, omega) / (2*np.pi)
            if sum_LL > 0:
                A_LL_new = A_LL_new / sum_LL
            if sum_RR > 0:
                A_RR_new = A_RR_new / sum_RR

        if project_B:
            B_LR_new = 0.5 * (B_LR_new - B_LR_new[::-1])
            B_RL_new = 0.5 * (B_RL_new - B_RL_new[::-1])

        F_w_new = np.zeros((Nw_, 2, 2), dtype=complex)
        F_w_new[:, 0, 0] = one_minus_nF * A_LL_new
        F_w_new[:, 1, 1] = one_minus_nF * A_RR_new
        F_w_new[:, 0, 1] = 1j * one_minus_nF * B_LR_new
        F_w_new[:, 1, 0] = 1j * one_minus_nF * B_RL_new

        F_t_new = _omega_to_time_mat(F_w_new, omega, t)

        # Equal-time pins
        #F_t_new[i0, 0, 0] = 0.5
        #F_t_new[i0, 1, 1] = 0.5
        #F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
        #F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])
        if pin_equal_time:
            F_t_new[i0, 0, 0] = 0.5
            F_t_new[i0, 1, 1] = 0.5
            F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
            F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])

        delta = np.max(np.abs(F_t_new - F_t))

        # Mix in time domain
        F_t = (1.0 - mixing) * F_t + mixing * F_t_new

        # Convert current mixed iterate to G^>
        Ggt_current = -1j * F_t

        # ---- actual real-time KBE residual d_ab ----
        do_kbe_dab = (
            compute_kbe_dab_every is not None
            and compute_kbe_dab_every > 0
            and (it % compute_kbe_dab_every == 0 or delta < tol)
        )

        dab_info = None
        if do_kbe_dab:
            dab_info = calc_kbe_d_ab_mq_equilibrium(
                t=t,
                Ggt=Ggt_current,
                J=J,
                mu=mu,
                t_cut=kbe_dab_t_cut,
                edge_skip=kbe_dab_edge_skip,
                keep_cross_self_energy=True,
                return_details=True,
            )

            d_ab = dab_info["d_ab"]
            d_t1_ab = dab_info["d_t1_ab"]
            d_t2_ab = dab_info["d_t2_ab"]

            d_print = d_ab**kbe_dab_power
            d_t1_print = d_t1_ab**kbe_dab_power
            d_t2_print = d_t2_ab**kbe_dab_power

            dab_history.append({
                "it": it,
                "delta": delta,
                "d_ab": d_ab.copy(),
                "d_t1_ab": d_t1_ab.copy(),
                "d_t2_ab": d_t2_ab.copy(),
                "d_print": d_print.copy(),
                "d_t1_print": d_t1_print.copy(),
                "d_t2_print": d_t2_print.copy(),
            })

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            B = np.imag(F_t[i0, 0, 1])
            msg = (
                f"{it:5d}  max|dF|={delta:.3e}  "
                f"F_LL(0)={F_t[i0,0,0].real:.8f}  "
                f"B=ImF_LR(0)={B:.8f}"
            )

            if dab_info is not None:
                d_print = dab_info["d_ab"]**kbe_dab_power
                msg += (
                    f"\n       KBE d_ab^{kbe_dab_power:g} = "
                    f"[[{d_print[0,0]:.3e}, {d_print[0,1]:.3e}], "
                    f"[{d_print[1,0]:.3e}, {d_print[1,1]:.3e}]]"
                )

            print(msg)

        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    Ggt = -1j * F_t

    print(f"\nFinal: G^>_LL(0) = {Ggt[i0,0,0]:.6f}  (should be -0.5i)")
    print(f"Final: G^>_LR(0) = {Ggt[i0,0,1]:.6f}  (should be purely real)")
    B_final = np.imag(F_t[i0, 0, 1])
    print(f"Final: B = Im F_LR(0) = -Re G^>_LR(0) = {B_final:.8f}")

    A_LL_final = -2.0 * np.imag(GR_w[:, 0, 0])
    A_RR_final = -2.0 * np.imag(GR_w[:, 1, 1])
    dw = omega[1] - omega[0]

    def _trap_sum(A):
        return (np.sum(A) - 0.5*A[0] - 0.5*A[-1]) * dw / (2.0*np.pi)

    print(f"Final: int A_LL dw/(2pi) = {_trap_sum(A_LL_final):.6f}  (should be 1)")
    print(f"Final: int A_RR dw/(2pi) = {_trap_sum(A_RR_final):.6f}  (should be 1)")

    B_LR_final = np.real(GR_w[:, 0, 1])
    print(
        "Final: B_LR odd-symmetry max|B(w)+B(-w)| = "
        f"{np.max(np.abs(B_LR_final + B_LR_final[::-1])):.2e}  (should be ~0)"
    )
    print(f"Final: mean|dF| = {delta:.3e}  (converged when < tol={tol:.0e})")

    if dab_history:
        last = dab_history[-1]
        print(f"Final: KBE d_ab^{kbe_dab_power:g} =")
        print(last["d_print"])

    if return_dab_history:
        return omega, t, Ggt, GR_w, F_t, dab_history

    return omega, t, Ggt, GR_w, F_t

def solve_equilibrium_mq_matrix_Fv1(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-9,
    mixing=0.05,
    eta_ret=1e-6,
    verbose_every=25,
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ q=4.

    Works in terms of F_ab = i * G^>_ab.

    Conventions:
        F_aa(t=0)  = +0.5       [real; from G^>_aa(0) = -i/2]
        F_LR(t)    = purely imaginary at all t
                     [since G^>_LR(t) is real: G^>_LR(t)* = G^>_LR(-t) by Z4 + Majorana]
        F_LR(t=0)  = 0          [real part is zero; imaginary part = B = <psi_L psi_R>]

        i*Sigma^>_ab  = J^2 * F_ab^3
        i*Sigma^R_ab(t) = theta(t)[i*Sigma^>_ab(t) + i*Sigma^>_ba(-t)]
                          (off-diagonal: ba-transpose on the time-reversed term)

        FDT diagonal (eq. 102):
            G^>_aa(omega) = 2i Im G^R_aa(omega) * (1-nF)
            F_aa(omega)   = (1-nF) * A_aa(omega)
            A_aa          = -2 Im G^R_aa  [real, EVEN, NON-NEGATIVE, int dw/(2pi)=1]
            Stabilising:  enforce even + clip>=0 + normalise (as in single-SYK notebook)

        FDT off-diagonal (eq. 103):
            G^>_LR(omega) = Re G^R_LR(omega) * (1-nF)     [no factor of 2]
            F_LR(omega)   = 1j * (1-nF) * B_LR(omega)     [purely imaginary in omega]
            B_LR          = Re G^R_LR                     [real, ODD]
            Stabilising:  enforce odd (anti-symmetrise)

    Returns:
        omega, t, Ggt, GR_w, F_t
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]
    Nt = len(t)
    Nw_ = Nw

    omega = np.linspace(-omega_max, omega_max, Nw_)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)  # h = i * [[0,mu],[-mu,0]]

    # Fermi factor: nF(omega) = 1/(exp(beta*omega)+1)
    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    one_minus_nF = 1.0 - nF  # shape (Nw,)

    # theta function for retarded
    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5

    # ---- initial guess: noninteracting F from GR0 ----
    # We start from the free (non-interacting but coupled) retarded Green function,
    # which is the 2x2 matrix inverse of (omega + i*eta - h).
    # From GR0 we extract:
    #   A_aa(omega) = -2 Im GR0_aa(omega)  [diagonal spectral function]
    #   B_LR(omega) =  2 Re GR0_LR(omega)  [off-diagonal "spectral" function for FDT eq.(103)]
    # and then set F_w via FDT:
    #   F_aa(omega) = (1-nF) * A_aa(omega)
    #   F_LR(omega) = (1-nF) * B_LR(omega)
    #
    # This is analogous to what solve_equilibrium_greater_real_time does for the
    # single SYK, where the initial A(omega) is a Lorentzian (the free spectral
    # function).  The MQ version must use the 2x2 matrix to correctly seed the
    # off-diagonal components.
    GR0 = np.empty((Nw_, 2, 2), dtype=complex)
    for k, w in enumerate(omega):
        GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye - 0*h)

    # F_ab(omega) initial guess from FDT
    #   Diagonal:    F_aa(omega) = (1-nF) * A_aa(omega),  A_aa = -2 Im G^R_aa  [real, even, >=0]
    #   Off-diagonal: F_LR(omega) = 1j * (1-nF) * B_LR(omega),  B_LR = Re G^R_LR [real, odd]
    #     The factor of 1j makes F_LR(t) purely imaginary in time, consistent with
    #     G^>_LR(t) being real (paper eq 11 + Majorana symmetry).  No factor of 2 for off-diag.
    A_LL_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 0, 0]), omega)
    A_RR_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 1, 1]), omega)
    B_LR_init = _enforce_odd(np.real(GR0[:, 0, 1]))
    B_RL_init = _enforce_odd(np.real(GR0[:, 1, 0]))

    F_w = np.zeros((Nw_, 2, 2), dtype=complex)
    F_w[:, 0, 0] = one_minus_nF * A_LL_init
    F_w[:, 1, 1] = one_minus_nF * A_RR_init
    F_w[:, 0, 1] = 1j * one_minus_nF * B_LR_init
    F_w[:, 1, 0] = 1j * one_minus_nF * B_RL_init

    # F_t = i G^> in time domain
    F_t = _omega_to_time_mat(F_w, omega, t)

    GR_w = None

    for it in range(max_iter):

        # i Sigma^>_ab(t) = J^2 * F_ab(t)^3
        # Derivation: G^> = -iF  and Sigma^> = -J^2 (G^>)^3
        iSgt = J**2 * F_t**3 #- np.array([[0.0, mu], [-mu, 0.0]], dtype=complex)

        # ------------------------------------------------------------------ #
        # Build i Sigma^R(t) = theta(t) * [i Sigma^>(t) - i Sigma^<(t)]
        #
        # The lesser self-energy satisfies (from the Majorana/Keldysh identity):
        #     i Sigma^<_ab(t) = -i Sigma^>_ba(-t)
        #
        # So:
        #     i Sigma^>(t) - i Sigma^<(t) = i Sigma^>_ab(t) + i Sigma^>_ba(-t)
        #
        # For DIAGONAL components (a=b): Sigma^>_aa(-t) is just iSgt[::-1] evaluated at -t.
        # For OFF-DIAGONAL components: we need Sigma^>_ba(-t), i.e. the TRANSPOSED
        # component evaluated at -t.  This is why we need iSgt_flipped, not iSgt[::-1].
        #
        # iSgt_flipped[n, a, b] = iSgt[Nt-1-n, b, a]  = i Sigma^>_ba(-t[n])
        # ------------------------------------------------------------------ #
        iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))   # shape (Nt, 2, 2)
        iSR_t = theta[:, None, None] * (iSgt + iSgt_flipped)

        # Fourier to frequency: i Sigma^R(omega)
        iSR_w = _time_to_omega_mat(iSR_t, t, omega)

        # Sigma^R(omega) = iSR_w / i = -i * iSR_w
        SR_w = -1j * iSR_w

        # Dyson equation: G^R = (omega + i*eta - h - Sigma^R)^{-1}
        GR_w = np.empty((Nw_, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

        # FDT to get new F_w  (Eqs. 102-103 of paper)
        #
        #   Diagonal (eq. 102):
        #     G^>_aa(omega) = 2i Im G^R_aa(omega) * (1-nF)
        #     F_aa(omega) = i*G^>_aa = (1-nF)*(-2 Im G^R_aa) = (1-nF)*A_aa
        #     A_aa = -2 Im G^R_aa  [real, even, non-negative, int dw/(2pi) = 1]
        #
        #   Off-diagonal (eq. 103):
        #     G^>_LR(omega) = Re G^R_LR(omega) * (1-nF)     [NO factor of 2]
        #     F_LR(omega) = i*G^>_LR = i*(1-nF)*Re G^R_LR   [purely imaginary in omega]
        #     => F_LR(t) = IFT(i*real_odd) = purely imaginary in time  ✓
        #     B_LR = Re G^R_LR  [real, odd]
        #
        # After extracting A_aa and B_LR, apply the same stabilising projections
        # that solve_equilibrium_greater_real_time uses for the single SYK:
        #   A_aa: symmetrise (even), clip (>=0), normalise (sum rule)
        #   B_LR: anti-symmetrise (odd)

        A_LL_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 0, 0]), omega)
        A_RR_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 1, 1]), omega)
        B_LR_new = _enforce_odd(np.real(GR_w[:, 0, 1]))
        B_LR_res = _residual_even(np.real(GR_w[:, 0, 1]))
        B_RL_new = _enforce_odd(np.real(GR_w[:, 1, 0]))

        F_w_new = np.zeros((Nw_, 2, 2), dtype=complex)
        F_w_new[:, 0, 0] = one_minus_nF * A_LL_new
        F_w_new[:, 1, 1] = one_minus_nF * A_RR_new
        F_w_new[:, 0, 1] = 1j * one_minus_nF * B_LR_new
        F_w_new[:, 1, 0] = 1j * one_minus_nF * B_RL_new

        F_t_new = _omega_to_time_mat(F_w_new, omega, t)

        # Equal-time pins:
        #   Diagonal:    F_aa(t=0) = +0.5  [from G^>_aa(0) = -i/2]
        #   Off-diagonal: F_LR(t=0) = 0    [F_LR(t) is purely imaginary, so Re F_LR(0) = 0;
        #                                    the imaginary part carries the physical B value]
        F_t_new[i0, 0, 0] = 0.5
        F_t_new[i0, 1, 1] = 0.5
        # Off-diagonal: zero the real part at t=0 (it should be exactly zero by symmetry)
        F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
        F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])

        # ---- convergence metric (mean absolute change, matching single-SYK convention) ----
        delta = np.mean(np.abs(F_t_new - F_t))

        # mix in time domain
        F_t = (1.0 - mixing) * F_t + mixing * F_t_new

        # ---- DS self-consistency residual (Eq. 104 analogue) ----

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            # B = -Re G^>_LR(0) = Im F_LR(0)  [since F_LR(t) purely imaginary]
            B = np.imag(F_t[i0, 0, 1])
            #dab_str = f"  sc_LL={d_LL:.2e}  sc_LR={d_LR:.2e}" if np.isfinite(d_LL) else ""
            #print(f"{it:5d}  mean|dF|={delta:.3e}  F_LL(0)={F_t[i0,0,0].real:.8f}  B=ImF_LR(0)={B:.8f}{dab_str}")
            print(f"{it:5d}  mean|dF|={delta:.3e}  F_LL(0)={F_t[i0,0,0].real:.8f} B=ImF_LR(0)={B:.8f}" f"residual_even_B={np.max(np.abs(B_LR_res)):.3e}")

        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    # convert back to G^> = -i F
    Ggt = -1j * F_t

    # Final diagnostics
    print(f"\nFinal: G^>_LL(0) = {Ggt[i0,0,0]:.6f}  (should be -0.5i)")
    print(f"Final: G^>_LR(0) = {Ggt[i0,0,1]:.6f}  (should be purely real)")
    B_final = np.imag(F_t[i0, 0, 1])
    print(f"Final: B = Im F_LR(0) = -Re G^>_LR(0) = {B_final:.8f}")

    # Spectral sum-rule and symmetry checks
    A_LL_final = -2.0 * np.imag(GR_w[:, 0, 0])
    A_RR_final = -2.0 * np.imag(GR_w[:, 1, 1])
    dw = omega[1] - omega[0]
    def _trap_sum(A):
        return (np.sum(A) - 0.5*A[0] - 0.5*A[-1]) * dw / (2.0*np.pi)
    print(f"Final: int A_LL dw/(2pi) = {_trap_sum(A_LL_final):.6f}  (should be 1)")
    print(f"Final: int A_RR dw/(2pi) = {_trap_sum(A_RR_final):.6f}  (should be 1)")
    B_LR_final = np.real(GR_w[:, 0, 1])
    print(f"Final: B_LR odd-symmetry max|B(w)+B(-w)| = {np.max(np.abs(B_LR_final + B_LR_final[::-1])):.2e}  (should be ~0)")
    print(f"Final: mean|dF| = {delta:.3e}  (converged when < tol={tol:.0e})")

    return omega, t, Ggt, GR_w, F_t


# ---------- KBE evolution ----------

def lesser_mat(G, i, j):
    return -G[j, i].T


def retarded_mat(G, i, k):
    return G[i, k] - lesser_mat(G, i, k)


def advanced_mat(G, k, j):
    return lesser_mat(G, k, j) - G[k, j]


def lesser_sigma_mat(S, i, j):
    return -S[j, i].T


def retarded_sigma_mat(S, i, k):
    return S[i, k] - lesser_sigma_mat(S, i, k)


def advanced_sigma_mat(S, k, j):
    return lesser_sigma_mat(S, k, j) - S[k, j]


def sigma_greater_mq(G, J_of_t, keep_cross_self_energy=True):
    JJ = J_of_t[:, None] * J_of_t[None, :]
    S = -JJ[:, :, None, None] * G**3
    if not keep_cross_self_energy:
        S[:, :, 0, 1] = 0.0
        S[:, :, 1, 0] = 0.0
    return S


def _trap_weights_segment(m, dt):
    return _trap_weights_uniform_from_dx(m, dt)


def rhs_t1_matrix(G, S, h_list, i, j, dt):
    conv = np.zeros((2, 2), dtype=complex)
    ks = np.arange(i + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (retarded_sigma_mat(S, i, k) @ G[k, j])

    ks = np.arange(j + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (S[i, k] @ advanced_mat(G, k, j))

    return -1j * (h_list[i] @ G[i, j] + conv)


def rhs_t2_matrix(G, S, h_list, i, j, dt):
    conv = np.zeros((2, 2), dtype=complex)
    ks = np.arange(i + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (retarded_mat(G, i, k) @ S[k, j])

    ks = np.arange(j + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (G[i, k] @ advanced_sigma_mat(S, k, j))

    return +1j * (G[i, j] @ h_list[j] + conv)


def enforce_equal_time_majorana(G, n):
    # Only exact equal-time constraints. Do not re-project off-diagonal time slices.
    G[n, n, 0, 0] = -0.5j
    G[n, n, 1, 1] = -0.5j
    # ensure anti-Hermitian equal-time matrix
    off = 0.5 * (G[n, n, 0, 1] - np.conj(G[n, n, 1, 0]))
    G[n, n, 0, 1] = off
    G[n, n, 1, 0] = -np.conj(off)


def fill_equilibrium_block(t, n0, t_eq, Ggt_eq):
    G = np.zeros((len(t), len(t), 2, 2), dtype=complex)
    interps = [[interp1d(t_eq, Ggt_eq[:, a, b], kind="cubic",
                         bounds_error=False, fill_value="extrapolate")
                for b in range(2)] for a in range(2)]
    for i in range(n0 + 1):
        for j in range(n0 + 1):
            tau = t[i] - t[j]
            for a in range(2):
                for b in range(2):
                    G[i, j, a, b] = interps[a][b](tau)
    for n in range(n0 + 1):
        enforce_equal_time_majorana(G, n)
    return G


def evolve_mq_kbe_matrix_v2(
    t_eq,
    Ggt_eq,
    beta=None,  # unused; kept for call compatibility
    J_i=1.0,
    J_f=1.0,
    mu_i=0.075,
    mu_f=0.0,
    t_pre=80.0,
    t_post=40.0,
    dt=0.1,
    n_corr=6,
    corr_tol=1e-10,
    print_every=50,
    keep_cross_self_energy=True,
):
    """Full 2x2 KBE evolution initialized directly from equilibrium Ggt_eq."""
    t = np.arange(-t_pre, t_post + 0.5 * dt, dt)
    n0 = int(np.argmin(np.abs(t)))
    t = t - t[n0]
    Nt = len(t)

    if np.max(np.abs(t[:n0+1, None] - t[None, :n0+1])) > np.max(np.abs(t_eq)):
        print("WARNING: t_eq does not cover the needed equilibrium tau range; interpolation will extrapolate.")

    G = fill_equilibrium_block(t, n0, t_eq, Ggt_eq)

    J_of_t = np.where(t < 0.0, J_i, J_f)
    mu_of_t = np.where(t < 0.0, mu_i, mu_f)
    h_list = np.array([h_mu_matrix(m) for m in mu_of_t])

    for n in range(n0 + 1, Nt):
        S = sigma_greater_mq(G, J_of_t, keep_cross_self_energy=keep_cross_self_energy)

        d1_old = np.array([rhs_t1_matrix(G, S, h_list, n - 1, j, dt) for j in range(n)])
        d2_old = np.array([rhs_t2_matrix(G, S, h_list, i, n - 1, dt) for i in range(n)])
        ddiag_old = rhs_t1_matrix(G, S, h_list, n - 1, n - 1, dt) + rhs_t2_matrix(G, S, h_list, n - 1, n - 1, dt)

        G[n, :n] = G[n - 1, :n] + dt * d1_old
        G[:n, n] = G[:n, n - 1] + dt * d2_old
        G[n, n] = G[n - 1, n - 1] + dt * ddiag_old
        enforce_equal_time_majorana(G, n)

        for _ in range(n_corr):
            row_old = G[n, :n].copy()
            col_old = G[:n, n].copy()
            diag_old = G[n, n].copy()

            S = sigma_greater_mq(G, J_of_t, keep_cross_self_energy=keep_cross_self_energy)
            d1_new = np.array([rhs_t1_matrix(G, S, h_list, n, j, dt) for j in range(n)])
            d2_new = np.array([rhs_t2_matrix(G, S, h_list, i, n, dt) for i in range(n)])
            ddiag_new = rhs_t1_matrix(G, S, h_list, n, n, dt) + rhs_t2_matrix(G, S, h_list, n, n, dt)

            G[n, :n] = G[n - 1, :n] + 0.5 * dt * (d1_old + d1_new)
            G[:n, n] = G[:n, n - 1] + 0.5 * dt * (d2_old + d2_new)
            G[n, n] = G[n - 1, n - 1] + 0.5 * dt * (ddiag_old + ddiag_new)
            enforce_equal_time_majorana(G, n)

            err = max(
                np.max(np.abs(G[n, :n] - row_old)),
                np.max(np.abs(G[:n, n] - col_old)),
                np.max(np.abs(G[n, n] - diag_old)),
            )
            if err < corr_tol:
                break

        if print_every and (n - n0) % print_every == 0:
            B = -np.real(G[n, n, 0, 1])
            print(f"evolved to t={t[n]:.3f}; B={B:.8f}")

    return t, G


def _central_derivative_1d(X, t):
    """
    Central derivative along axis 0.
    X shape: (Nt,) or (Nt,2,2)
    """
    t = np.asarray(t)
    dt = t[1] - t[0]

    dX = np.empty_like(X, dtype=complex)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def _theta_plus_minus(t):
    t = np.asarray(t)
    i0 = int(np.argmin(np.abs(t)))

    theta_p = np.zeros_like(t, dtype=float)
    theta_m = np.zeros_like(t, dtype=float)

    theta_p[t > 0.0] = 1.0
    theta_m[t < 0.0] = 1.0

    theta_p[i0] = 0.5
    theta_m[i0] = 0.5

    return theta_p, theta_m, i0


def _conv_1d_same(A, B, t):
    """
    Discrete convolution

        C(t) = int du A(t-u) B(u)

    for scalar arrays on a symmetric uniform grid.
    """
    w = _trap_weights_grid(t)
    return np.convolve(B * w, A, mode="same")


def _conv_mat_same(A, B, t):
    """
    Matrix convolution

        C_ab(t) = int du A_ac(t-u) B_cb(u)

    A, B shape: (Nt,2,2)
    Returns shape: (Nt,2,2)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    w = _trap_weights_grid(t)

    Nt = len(t)
    C = np.zeros((Nt, 2, 2), dtype=complex)

    for a in range(2):
        for b in range(2):
            tmp = np.zeros(Nt, dtype=complex)
            for c in range(2):
                tmp += np.convolve(B[:, c, b] * w, A[:, a, c], mode="same")
            C[:, a, b] = tmp

    return C

def calc_kbe_d_ab_mq_equilibrium(
    t,
    Ggt,
    J,
    mu,
    *,
    t_cut=None,
    edge_skip=4,
    keep_cross_self_energy=True,
    return_details=False,
):
    """
    Real-time KBE residual d_ab for equilibrium coupled SYK / MQ.

    Input:
        t    : relative-time grid
        Ggt  : G^>_ab(t), shape (Nt,2,2)
        J    : SYK coupling
        mu   : MQ bilinear coupling

    Checks both equilibrium KBE equations:

        i d_t G^>(t) =
            h G^>(t)
            + Sigma^R * G^>
            + Sigma^> * G^A

    and

        i d_t G^>(t) =
            G^>(t) h
            + G^R * Sigma^>
            + G^> * Sigma^A

    where:
        h = h_mu_matrix(mu) = i [[0, mu], [-mu, 0]]

    and all products/convolutions are matrix products in L/R space.

    Returns:
        d_ab = 0.5 * (d_t1_ab + d_t2_ab)

    with:
        d_t1_ab = mean_t |res_t1_ab(t)|^2
        d_t2_ab = mean_t |res_t2_ab(t)|^2
    """
    t = np.asarray(t)
    Ggt = np.asarray(Ggt, dtype=complex)

    if Ggt.shape != (len(t), 2, 2):
        raise ValueError(f"Ggt must have shape {(len(t), 2, 2)}, got {Ggt.shape}")

    theta_p, theta_m, i0 = _theta_plus_minus(t)

    h = h_mu_matrix(mu)

    # Lesser Green's function:
    # G^<_ab(t) = -G^>_ba(-t)
    Glt = -np.transpose(Ggt[::-1], (0, 2, 1))

    GR = theta_p[:, None, None] * (Ggt - Glt)
    GA = theta_m[:, None, None] * (Glt - Ggt)

    # Greater self-energy:
    # Sigma^>_ab(t) = -J^2 G^>_ab(t)^3
    Sgt = -J**2 * Ggt**3

    if not keep_cross_self_energy:
        Sgt[:, 0, 1] = 0.0
        Sgt[:, 1, 0] = 0.0

    # Lesser self-energy:
    # Sigma^<_ab(t) = -Sigma^>_ba(-t)
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

# ============================================================
# KKTZ self-energy + checkpointed solvers + lean-output helpers (formerly kktz_checkpoint_utils.py)
# ============================================================

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
    (defined above), but with the KKTZ two-flavor greater
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


# ============================================================
# Filenames
# ============================================================

def equilibrium_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"kktz_eq_J_{sf(meta['J'])}_alpha_{sf(meta['alpha'])}_mu_{sf(meta['mu'])}"
        f"_beta_{sf(meta['beta'])}_dt_{sf(meta['dt'])}"
        f"_om_{sf(meta['omega_max'])}_Nw_{int(meta['Nw'])}"
        f"_tol_{sf(meta['tol'])}_{h}.npz"
    )


def kbe_filename(out_dir: os.PathLike | str, meta: Dict[str, Any]) -> Path:
    out_dir = Path(out_dir)
    h = param_hash(meta)
    return out_dir / (
        f"kktz_kbe_J_{sf(meta['J_i'])}_to_{sf(meta['J_f'])}"
        f"_alpha_{sf(meta['alpha_i'])}_to_{sf(meta['alpha_f'])}"
        f"_mu_{sf(meta['mu_i'])}_to_{sf(meta['mu_f'])}"
        f"_beta_{sf(meta['beta'])}_dt_{sf(meta['dt'])}"
        f"_tpre_{sf(meta['t_pre'])}_tpost_{sf(meta['t_post'])}"
        f"_ncorr_{int(meta['n_corr'])}_ctol_{sf(meta['corr_tol'])}"
        f"_it_{int(meta['iterations'])}_{h}.npz"
    )


# ============================================================
# Equilibrium wrappers
# ============================================================

def _solve_or_load_mu0_seed(
    *,
    J: float,
    alpha: float,
    beta: float,
    out_dir: Path,
    dt: float,
    omega_max: float,
    Nw: int,
    t_max: Optional[float],
    max_iter: int,
    seed_tol: float,
    mixing: float,
    eta_ret: float,
    verbose_every: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
    project: bool,
) -> tuple[np.ndarray, np.ndarray]:
    seed_meta = {
        "kind": "equilibrium_seed_mu0",
        "J": float(J),
        "alpha": float(alpha),
        "mu": 0.0,
        "beta": float(beta),
        "dt": float(dt),
        "omega_max": float(omega_max),
        "Nw": int(Nw),
        "t_max": None if t_max is None else float(t_max),
        "tol": float(seed_tol),
        "max_iter": int(max_iter),
        "mixing": float(mixing),
        "eta_ret": float(eta_ret),
        "project": bool(project),
    }
    seed_file = equilibrium_filename(out_dir, seed_meta)
    seed_ckpt = checkpoint_path_for(seed_file)

    if seed_file.exists() and not overwrite:
        print(f"Loading existing mu=0 seed: {seed_file}")
        z = np.load(seed_file, allow_pickle=True)
        return z["omega"], z["GR_w"]

    print("Solving mu=0 seed for weak-KKTZ initialisation...")
    omega0, _t0, _Ggt0, GR_w0, _F_t0 = solve_equilibrium_kktz_matrix_F_checkpointed(
        J=J,
        alpha=alpha,
        mu=0.0,
        beta=beta,
        omega_max=omega_max,
        Nw=Nw,
        t_max=t_max,
        dt=dt,
        max_iter=max_iter,
        tol=seed_tol,
        mixing=mixing,
        eta_ret=eta_ret,
        verbose_every=verbose_every,
        checkpoint_path=seed_ckpt,
        checkpoint_every=checkpoint_every,
        resume=resume,
        final_path=seed_file,
        project_A=project,
        clip_A=project,
        normalize_A=project,
        project_B=project,
        pin_equal_time=project,
        compute_kbe_dab_every=0,
        init_mode="free",
        init_offdiag="zero",
    )
    return omega0, GR_w0


def _solve_or_load_sb_seed(
    *,
    J: float,
    alpha: float,
    sb_seed_mu: float,
    beta: float,
    out_dir: Path,
    dt: float,
    omega_max: float,
    Nw: int,
    t_max: Optional[float],
    max_iter: int,
    seed_tol: float,
    mixing: float,
    eta_ret: float,
    verbose_every: int,
    checkpoint_every: int,
    resume: bool,
    overwrite: bool,
    project: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve/load a small-Q-source state used only to select an SSB branch."""
    seed_meta = {
        "kind": "equilibrium_sb_seed",
        "J": float(J),
        "alpha": float(alpha),
        "mu": float(sb_seed_mu),
        "beta": float(beta),
        "dt": float(dt),
        "omega_max": float(omega_max),
        "Nw": int(Nw),
        "t_max": None if t_max is None else float(t_max),
        "tol": float(seed_tol),
        "max_iter": int(max_iter),
        "mixing": float(mixing),
        "eta_ret": float(eta_ret),
        "project": bool(project),
    }
    seed_file = equilibrium_filename(out_dir, seed_meta)
    seed_ckpt = checkpoint_path_for(seed_file)

    if seed_file.exists() and not overwrite:
        print(f"Loading existing symmetry-breaking seed: {seed_file}")
        z = np.load(seed_file, allow_pickle=True)
        return z["omega"], z["GR_w"]

    print(
        f"Solving symmetry-breaking seed at alpha={alpha:g}, "
        f"mu_seed={sb_seed_mu:g}..."
    )
    omega0, _t0, _Ggt0, GR_w0, _F_t0 = solve_equilibrium_kktz_matrix_F_checkpointed(
        J=J,
        alpha=alpha,
        mu=sb_seed_mu,
        beta=beta,
        omega_max=omega_max,
        Nw=Nw,
        t_max=t_max,
        dt=dt,
        max_iter=max_iter,
        tol=seed_tol,
        mixing=mixing,
        eta_ret=eta_ret,
        verbose_every=verbose_every,
        checkpoint_path=seed_ckpt,
        checkpoint_every=checkpoint_every,
        resume=resume,
        final_path=seed_file,
        project_A=project,
        clip_A=project,
        normalize_A=project,
        project_B=project,
        pin_equal_time=project,
        compute_kbe_dab_every=0,
        init_mode="free",
        init_offdiag="linear_mu",
    )
    return omega0, GR_w0


def run_equilibrium_one(
    J: float,
    alpha: float,
    mu: float,
    beta: float,
    *,
    out_dir: os.PathLike | str = "eq_runs",
    dt: float = 0.1,
    omega_max: float = 8.0,
    Nw: int = 4097,
    t_max: Optional[float] = None,
    max_iter: int = 2000,
    tol: float = 1e-9,
    dab_tol: float = 1e-6,
    require_dab_convergence: bool = False,
    mixing: float = 0.05,
    eta_ret: float = 1e-6,
    verbose_every: int = 25,
    compute_kbe_dab_every: int = 50,
    checkpoint_every: int = 25,
    seed_from_mu0: bool = False,
    sb_seed_mu: float = 0.0,
    seed_tol: Optional[float] = None,
    project: bool = True,
    overwrite: bool = False,
    resume: bool = True,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "kktz_eq_manifest.csv"

    meta = {
        "kind": "kktz_equilibrium",
        "J": float(J),
        "alpha": float(alpha),
        "mu": float(mu),
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
        "compute_kbe_dab_every": int(compute_kbe_dab_every),
        "checkpoint_every": int(checkpoint_every),
        "seed_from_mu0": bool(seed_from_mu0),
        "project": bool(project),
    }
    if abs(sb_seed_mu) > 0:
        meta["sb_seed_mu"] = float(sb_seed_mu)
    filename = equilibrium_filename(out_dir, meta)
    checkpoint_path = checkpoint_path_for(filename)

    if filename.exists() and not overwrite:
        print(f"Skipping existing equilibrium file: {filename}")
        return filename

    init_kwargs: Dict[str, Any] = dict(init_mode="free", init_offdiag="zero")
    use_sb_seed = alpha < 0 and abs(mu) < 1e-15 and abs(sb_seed_mu) > 0
    if use_sb_seed:
        st = float(seed_tol if seed_tol is not None else tol)
        omega0, GR_w0 = _solve_or_load_sb_seed(
            J=J,
            alpha=alpha,
            sb_seed_mu=sb_seed_mu,
            beta=beta,
            out_dir=out_dir,
            dt=dt,
            omega_max=omega_max,
            Nw=Nw,
            t_max=t_max,
            max_iter=max(200, min(max_iter, 2000)),
            seed_tol=st,
            mixing=mixing,
            eta_ret=eta_ret,
            verbose_every=verbose_every,
            checkpoint_every=checkpoint_every,
            resume=resume,
            overwrite=overwrite,
            project=project,
        )
        if len(omega0) != Nw or not np.allclose(
            omega0, np.linspace(-omega_max, omega_max, Nw)
        ):
            raise ValueError("Symmetry-breaking seed frequency grid does not match target grid.")
        init_kwargs = dict(init_GR_w=GR_w0)
    elif seed_from_mu0 and abs(mu) > 0:
        st = float(seed_tol if seed_tol is not None else max(tol, 1e-5))
        omega0, GR_w0 = _solve_or_load_mu0_seed(
            J=J,
            alpha=alpha,
            beta=beta,
            out_dir=out_dir,
            dt=dt,
            omega_max=omega_max,
            Nw=Nw,
            t_max=t_max,
            max_iter=max(200, min(max_iter, 2000)),
            seed_tol=st,
            mixing=mixing,
            eta_ret=eta_ret,
            verbose_every=verbose_every,
            checkpoint_every=checkpoint_every,
            resume=resume,
            overwrite=False,
            project=project,
        )
        init_kwargs = dict(
            init_mode="uncoupled_syk",
            init_omega=omega0,
            init_GR_LL=GR_w0[:, 0, 0],
            init_GR_RR=GR_w0[:, 1, 1],
            init_offdiag="linear_mu",
        )

    print("\n============================================================")
    print("Solving KKTZ equilibrium")
    print(json.dumps(meta, indent=2))
    print("Saving to:", filename)
    print("Checkpoint:", checkpoint_path)
    print("============================================================")

    try:
        solve_equilibrium_kktz_matrix_F_checkpointed(
            J=J,
            alpha=alpha,
            mu=mu,
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
            verbose_every=verbose_every,
            compute_kbe_dab_every=compute_kbe_dab_every,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
            final_path=filename,
            project_A=project,
            clip_A=project,
            normalize_A=project,
            project_B=project,
            pin_equal_time=project,
            **init_kwargs,
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


# ============================================================
# KBE wrappers
# ============================================================

def find_eq_file(
    eq_dir: os.PathLike | str,
    *,
    J: float,
    alpha: float,
    mu: float,
    beta: float,
    dt: Optional[float] = None,
    omega_max: Optional[float] = None,
    Nw: Optional[int] = None,
    prefer_smallest_dt: bool = True,
) -> Path:
    manifest = load_manifests_recursive(eq_dir, "kktz_eq_manifest.csv")
    if manifest.empty:
        raise FileNotFoundError(f"No kktz_eq_manifest.csv files found under {eq_dir}")

    good = manifest[
        (manifest["status"] == "ok")
        & np.isclose(manifest["J"].astype(float), float(J))
        & np.isclose(manifest["alpha"].astype(float), float(alpha))
        & np.isclose(manifest["mu"].astype(float), float(mu))
        & np.isclose(manifest["beta"].astype(float), float(beta))
    ].copy()
    if dt is not None and "dt" in good.columns:
        good = good[np.isclose(good["dt"].astype(float), float(dt))]
    if omega_max is not None and "omega_max" in good.columns:
        good = good[np.isclose(good["omega_max"].astype(float), float(omega_max))]
    if Nw is not None and "Nw" in good.columns:
        good = good[good["Nw"].astype(int) == int(Nw)]

    if len(good) == 0:
        raise FileNotFoundError(
            f"No equilibrium file found for J={J}, alpha={alpha}, mu={mu}, beta={beta}, "
            f"dt={dt}, omega_max={omega_max}, Nw={Nw} under {eq_dir}"
        )
    sort_cols: List[str] = []
    ascending: List[bool] = []
    for col, asc in [("dt", True), ("tol", True), ("Nw", False), ("omega_max", False)]:
        if col in good.columns:
            sort_cols.append(col)
            ascending.append(asc)
    if sort_cols:
        good = good.sort_values(sort_cols, ascending=ascending)
    return Path(good.iloc[0]["filename"])


def run_kbe_one(
    J_i: float,
    J_f: float,
    alpha_i: float,
    alpha_f: float,
    mu_i: float,
    mu_f: float,
    beta: float,
    *,
    dt: float = 0.1,
    t_pre: Optional[float] = None,
    t_post: Optional[float] = None,
    t_pre_factor: Optional[float] = 2.0,
    t_post_factor: Optional[float] = 1.0,
    n_corr: int = 6,
    corr_tol: float = 1e-10,
    iterations: int = 50,
    eq_dir: os.PathLike | str = "eq_runs",
    out_dir: os.PathLike | str = "kbe_runs",
    eq_file: Optional[os.PathLike | str] = None,
    eq_dt: Optional[float] = None,
    eq_omega_max: Optional[float] = None,
    eq_Nw: Optional[int] = None,
    keep_cross_self_energy: bool = True,
    checkpoint_every: int = 25,
    overwrite: bool = False,
    resume: bool = True,
    compressed: bool = True,
) -> Optional[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "kktz_kbe_manifest.csv"

    if t_pre is None:
        if t_pre_factor is None:
            raise ValueError("Need either t_pre or t_pre_factor.")
        t_pre = float(t_pre_factor) * float(beta)
    if t_post is None:
        if t_post_factor is None:
            raise ValueError("Need either t_post or t_post_factor.")
        t_post = float(t_post_factor) * float(beta)

    if eq_file is None:
        eq_file = find_eq_file(
            eq_dir,
            J=J_i,
            alpha=alpha_i,
            mu=mu_i,
            beta=beta,
            dt=eq_dt,
            omega_max=eq_omega_max,
            Nw=eq_Nw,
        )
    eq_file = Path(eq_file)

    meta = {
        "kind": "kktz_kbe",
        "J_i": float(J_i),
        "J_f": float(J_f),
        "alpha_i": float(alpha_i),
        "alpha_f": float(alpha_f),
        "mu_i": float(mu_i),
        "mu_f": float(mu_f),
        "beta": float(beta),
        "dt": float(dt),
        "t_pre": float(t_pre),
        "t_post": float(t_post),
        "t_pre_factor": None if t_pre_factor is None else float(t_pre_factor),
        "t_post_factor": None if t_post_factor is None else float(t_post_factor),
        "n_corr": int(n_corr),
        "corr_tol": float(corr_tol),
        "iterations": int(iterations),
        "checkpoint_every": int(checkpoint_every),
        "keep_cross_self_energy": bool(keep_cross_self_energy),
        "eq_source": str(eq_file),
        "eq_dt_filter": None if eq_dt is None else float(eq_dt),
        "eq_omega_max_filter": None if eq_omega_max is None else float(eq_omega_max),
        "eq_Nw_filter": None if eq_Nw is None else int(eq_Nw),
        "save_format": "full_two_time_Ggt",
        "Ggt_shape": "(Nt,Nt,2,2)",
        "saved_bosonic_primaries": False,
    }
    filename = kbe_filename(out_dir, meta)
    checkpoint_path = checkpoint_path_for(filename)

    if filename.exists() and not overwrite:
        print(f"Skipping existing KBE file: {filename}")
        return filename

    print("\n============================================================")
    print("Running KKTZ KBE")
    print(json.dumps(meta, indent=2))
    print("Saving full two-time GF to:", filename)
    print("Checkpoint:", checkpoint_path)
    print("============================================================")

    try:
        z = np.load(eq_file, allow_pickle=True)
        if "t_eq" in z.files:
            t_eq = z["t_eq"]
        elif "t" in z.files:
            t_eq = z["t"]
        else:
            raise KeyError(f"Equilibrium file {eq_file} contains no t_eq/t array")

        if "Ggt_eq" in z.files:
            Ggt_eq = z["Ggt_eq"]
        elif "Ggt" in z.files:
            Ggt_eq = z["Ggt"]
        else:
            raise KeyError(f"Equilibrium file {eq_file} contains no Ggt_eq/Ggt array")

        Nw_loaded = int(_read_npz_scalar(z, "Nw", -1))
        omega_loaded = z["omega"] if "omega" in z.files else None
        GR_w = z["GR_w"] if "GR_w" in z.files else None
        F_t = z["F_t"] if "F_t" in z.files else None

        t_kbe, Ggt = evolve_kktz_kbe_matrix_checkpointed(
            t_eq,
            Ggt_eq,
            J_i=J_i,
            J_f=J_f,
            alpha_i=alpha_i,
            alpha_f=alpha_f,
            mu_i=mu_i,
            mu_f=mu_f,
            beta=beta,
            t_pre=t_pre,
            t_post=t_post,
            dt=dt,
            n_corr=n_corr,
            corr_tol=corr_tol,
            print_every=iterations,
            keep_cross_self_energy=keep_cross_self_energy,
            checkpoint_path=checkpoint_path,
            checkpoint_every=checkpoint_every,
            resume=resume,
            final_path=None,  # Save below so metadata/eq arrays match the batch manifest.
            Nw=Nw_loaded,
            save_full_final_G=False,
        )

        n0 = int(np.argmin(np.abs(t_kbe)))
        t_kbe[n0] = 0.0
        idx = np.arange(len(t_kbe))
        B_LR_equal = -np.real(Ggt[idx, idx, 0, 1])

        payload: Dict[str, Any] = {
            "t_kbe": t_kbe,
            "Ggt": Ggt,
            "n0": np.array(n0),
            "B_LR_equal": B_LR_equal,
            "t_eq": t_eq,
            "Ggt_eq": Ggt_eq,
            "J_i": np.array(J_i),
            "J_f": np.array(J_f),
            "alpha_i": np.array(alpha_i),
            "alpha_f": np.array(alpha_f),
            "mu_i": np.array(mu_i),
            "mu_f": np.array(mu_f),
            "beta": np.array(beta),
            "dt": np.array(dt),
            "t_pre": np.array(t_pre),
            "t_post": np.array(t_post),
            "Nw": np.array(Nw_loaded),
            "metadata_json": _json_safe(meta),
        }
        if omega_loaded is not None:
            payload["omega"] = omega_loaded
        if GR_w is not None:
            payload["GR_w_eq"] = GR_w
        if F_t is not None:
            payload["F_t_eq"] = F_t

        atomic_savez(filename, compressed=compressed, **payload)

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


def load_kktz_kbe_dataframe(kbe_dir: os.PathLike | str = "kbe_runs") -> pd.DataFrame:
    manifest = load_manifests_recursive(kbe_dir, "kktz_kbe_manifest.csv")
    rows: List[Dict[str, Any]] = []
    for _, row in manifest.iterrows():
        if row.get("status") != "ok":
            continue
        filename = Path(row["filename"])
        if not filename.exists():
            print("Missing:", filename)
            continue
        z = np.load(filename, allow_pickle=True)
        entry: Dict[str, Any] = dict(row)
        entry.update({
            "filename": str(filename),
            "t_kbe": z["t_kbe"],
            "Ggt": z["Ggt"],
            "n0": int(z["n0"]),
            "B_LR_equal": z["B_LR_equal"] if "B_LR_equal" in z.files else None,
            "t_eq": z["t_eq"] if "t_eq" in z.files else None,
            "Ggt_eq": z["Ggt_eq"] if "Ggt_eq" in z.files else None,
        })
        rows.append(entry)
    return pd.DataFrame(rows)


# ============================================================
# CLI
# ============================================================

def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Batch utilities for KKTZ quartic-coupled SYK equilibrium and KBE runs.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    peq = sub.add_parser("eq-one", help="Run one KKTZ equilibrium solve.")
    peq.add_argument("--J", type=float, required=True)
    peq.add_argument("--alpha", type=float, required=True, help="KKTZ quartic coupling alpha")
    peq.add_argument("--mu", type=float, required=True, help="Optional quadratic Q source; set 0 for pure KKTZ")
    peq.add_argument("--beta", type=float, required=True)
    peq.add_argument("--out-dir", default="eq_runs")
    peq.add_argument("--dt", type=float, default=0.1)
    peq.add_argument("--omega-max", type=float, default=8.0)
    peq.add_argument("--Nw", type=int, default=4097)
    peq.add_argument("--t-max", type=float, default=None)
    peq.add_argument("--max-iter", type=int, default=2000)
    peq.add_argument("--tol", type=float, default=1e-9, help="Fixed-point tolerance for max|F_new - F_old|.")
    peq.add_argument("--dab-tol", type=float, default=1e-6, help="Tolerance for max(d_ab**0.5). Printed always when d_ab is computed; only required if --require-dab-convergence is set.")
    peq.add_argument("--require-dab-convergence", action="store_true", help="Require max(d_ab**0.5) < --dab-tol before marking equilibrium converged.")
    peq.add_argument("--mixing", type=float, default=0.05)
    peq.add_argument("--eta-ret", type=float, default=1e-6)
    peq.add_argument("--verbose-every", type=int, default=25)
    peq.add_argument("--compute-kbe-dab-every", type=int, default=50)
    peq.add_argument("--checkpoint-every", type=int, default=25)
    peq.add_argument("--seed-from-mu0", action="store_true")
    peq.add_argument(
        "--sb-seed-mu",
        type=float,
        default=0.0,
        help=(
            "For alpha<0 and physical mu=0, first solve at this small Q-source "
            "and use the full solution to seed a symmetry-broken branch. "
            "Its sign selects the branch."
        ),
    )
    peq.add_argument("--seed-tol", type=float, default=None)
    peq.add_argument("--no-project", action="store_true")
    peq.add_argument("--no-resume", action="store_true")
    peq.add_argument("--overwrite", action="store_true")

    pkbe = sub.add_parser("kbe-one", help="Run one KKTZ KBE evolution from an equilibrium npz.")
    pkbe.add_argument("--J-i", type=float, required=True)
    pkbe.add_argument("--J-f", type=float, default=None, help="Defaults to J-i")
    pkbe.add_argument("--alpha-i", type=float, required=True)
    pkbe.add_argument("--alpha-f", type=float, default=None, help="Defaults to alpha-i")
    pkbe.add_argument("--mu-i", type=float, required=True)
    pkbe.add_argument("--mu-f", type=float, required=True)
    pkbe.add_argument("--beta", type=float, required=True)
    pkbe.add_argument("--dt", type=float, default=0.1)
    pkbe.add_argument("--t-pre", type=float, default=None)
    pkbe.add_argument("--t-post", type=float, default=None)
    pkbe.add_argument("--t-pre-factor", type=float, default=2.0)
    pkbe.add_argument("--t-post-factor", type=float, default=1.0)
    pkbe.add_argument("--n-corr", type=int, default=6)
    pkbe.add_argument("--corr-tol", type=float, default=1e-10)
    pkbe.add_argument("--iterations", type=int, default=50, help="Print every this many post-quench steps.")
    pkbe.add_argument("--eq-dir", default="eq_runs")
    pkbe.add_argument("--out-dir", default="kbe_runs")
    pkbe.add_argument("--eq-file", default=None)
    pkbe.add_argument("--eq-dt", type=float, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--eq-omega-max", type=float, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--eq-Nw", type=int, default=None, help="Optional filter for equilibrium file lookup.")
    pkbe.add_argument("--checkpoint-every", type=int, default=25)
    pkbe.add_argument("--drop-cross-self-energy", action="store_true")
    pkbe.add_argument("--uncompressed", action="store_true", help="Use faster/larger np.savez instead of np.savez_compressed.")
    pkbe.add_argument("--no-resume", action="store_true")
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
            J=args.J,
            alpha=args.alpha,
            mu=args.mu,
            beta=args.beta,
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
            verbose_every=args.verbose_every,
            compute_kbe_dab_every=args.compute_kbe_dab_every,
            checkpoint_every=args.checkpoint_every,
            seed_from_mu0=args.seed_from_mu0,
            sb_seed_mu=args.sb_seed_mu,
            seed_tol=args.seed_tol,
            project=not args.no_project,
            overwrite=args.overwrite,
            resume=not args.no_resume,
        )
    elif args.cmd == "kbe-one":
        run_kbe_one(
            J_i=args.J_i,
            J_f=args.J_f if args.J_f is not None else args.J_i,
            alpha_i=args.alpha_i,
            alpha_f=args.alpha_f if args.alpha_f is not None else args.alpha_i,
            mu_i=args.mu_i,
            mu_f=args.mu_f,
            beta=args.beta,
            dt=args.dt,
            t_pre=args.t_pre,
            t_post=args.t_post,
            t_pre_factor=args.t_pre_factor,
            t_post_factor=args.t_post_factor,
            n_corr=args.n_corr,
            corr_tol=args.corr_tol,
            iterations=args.iterations,
            eq_dir=args.eq_dir,
            out_dir=args.out_dir,
            eq_file=args.eq_file,
            eq_dt=args.eq_dt,
            eq_omega_max=args.eq_omega_max,
            eq_Nw=args.eq_Nw,
            keep_cross_self_energy=not args.drop_cross_self_energy,
            checkpoint_every=args.checkpoint_every,
            overwrite=args.overwrite,
            resume=not args.no_resume,
            compressed=not args.uncompressed,
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
