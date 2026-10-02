"""
ho_solver.py -- next-order KBE time propagation for the SYK4 quench.

Upgrades BOTH pieces of syk_batch_tools.evolve_syk4_kbe's discretization,
which are currently both trapezoidal (globally 2nd order):

  1. The memory/collision-integral quadrature inside rhs_t1/rhs_t2 (sums
     over history index k) -- upgraded from trap_weights (2nd order) to
     ho_quadrature.ho_weights (validated 4th order).

  2. The local "diagonal advancement" corrector that solves for each new
     row/column G[n,:n], G[:n,n] -- upgraded from AM1/trapezoidal (2nd
     order) to AM2 (validated 3rd order), with a single AM1 startup step
     (see ho_multistep.py's docstring for why this pairing is order-
     consistent while a naive AM3 bootstrap is not).

Physics (sigma_greater_syk4, kernel construction, equilibrium seeding,
Majorana enforcement) is reused unmodified from syk_batch_tools.py via
import -- nothing there is duplicated or edited, so this cannot corrupt
that file or its behavior. Deliberately does NOT reimplement
checkpointing/signal handling; this is for validation/comparison, not
production runs.
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import syk_batch_tools as sbt  # noqa: E402  (path insert must come first)

from ho_quadrature import ho_weights  # noqa: E402
from ho_multistep import am1_step, am2_step  # noqa: E402


def rhs_t1_ho(G, S, i, j, dt, K_R_mat=None):
    """Same physics as syk_batch_tools.rhs_t1, quadrature upgraded to ho_weights."""
    k = np.arange(i + 1)
    w = ho_weights(len(k), dt)
    Sigma_R = S[i, k] + S[k, i]
    if K_R_mat is not None:
        Sigma_R = Sigma_R - K_R_mat[i, k]
    I1 = np.sum(w * Sigma_R * G[k, j])

    k = np.arange(j + 1)
    w = ho_weights(len(k), dt)
    G_A = -(G[k, j] + G[j, k])
    I2 = np.sum(w * S[i, k] * G_A)
    return -1j * (I1 + I2)


def rhs_t2_ho(G, S, i, j, dt, K_R_mat=None):
    """Same physics as syk_batch_tools.rhs_t2, quadrature upgraded to ho_weights."""
    k = np.arange(i + 1)
    w = ho_weights(len(k), dt)
    G_R = G[i, k] + G[k, i]
    I1 = np.sum(w * G_R * S[k, j])

    k = np.arange(j + 1)
    w = ho_weights(len(k), dt)
    Sigma_A = -(S[k, j] + S[j, k])
    if K_R_mat is not None:
        Sigma_A = Sigma_A - np.conj(K_R_mat[j, k])
    I2 = np.sum(w * G[i, k] * Sigma_A)
    return +1j * (I1 + I2)


def evolve_syk4_kbe_ho(
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
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
    return_diagnostics: bool = False,
) -> Tuple[np.ndarray, np.ndarray] | Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    K_R_w, kernel_cutoff = sbt.build_kernel_R_w(
        omega, J4_i, kernel_lambda, kernel_c, kernel_cutoff
    )

    t, n0 = sbt.kbe_time_grid(t_pre, t_post, dt)  # quench exactly on the uniform grid
    Nt = len(t)

    K_R_mat = sbt.build_kernel_R_mat(t, omega, K_R_w)

    t_rel_max = t_pre + t_post
    t_grid = np.linspace(-t_rel_max, t_rel_max, 4 * Nt + 1)
    G_eq_t = sbt.greater_from_spectral(omega, A, beta_i, t_grid)
    G_eq = interp1d(t_grid, G_eq_t, kind="cubic", fill_value="extrapolate")

    corr_final_err = np.full(Nt, np.nan, dtype=float)
    corr_iters_used = np.zeros(Nt, dtype=int)

    G = sbt._init_G(Nt, n0, G_eq, t)

    J2_of_t = np.where(t < 0.0, J2_i, J2_f)
    J4_of_t = np.where(t < 0.0, J4_i, J4_f)
    JJ2 = np.outer(J2_of_t, J2_of_t)
    JJ4 = np.outer(J4_of_t, J4_of_t)

    n_start = n0 + 1
    # NOTE: d1_prev1/d1_prev2 below (the RHS at rows n-1, n-2) are recomputed
    # from scratch every step, over the FULL current j-range (0..n-1) each
    # time -- they can NOT be cached/reused from when those rows were
    # themselves finalized, since at that point the column range was
    # narrower (the memory integral's upper limit grows every step, so
    # rhs_t1_ho(G,S,n-2,j,...) genuinely depends on columns added since row
    # n-2 was last touched).

    for n in range(n_start, Nt):
        S = sbt.sigma_greater_syk4(G, JJ2, JJ4)

        d1_prev1 = np.array([rhs_t1_ho(G, S, n - 1, j, dt, K_R_mat) for j in range(n)], dtype=complex)
        d2_prev1 = np.array([rhs_t2_ho(G, S, i, n - 1, dt, K_R_mat) for i in range(n)], dtype=complex)

        use_am2 = (n - n_start) >= 1  # step 1 (n==n_start): AM1 startup; step 2+: AM2
        if use_am2:
            d1_prev2 = np.array([rhs_t1_ho(G, S, n - 2, j, dt, K_R_mat) for j in range(n)], dtype=complex)
            d2_prev2 = np.array([rhs_t2_ho(G, S, i, n - 2, dt, K_R_mat) for i in range(n)], dtype=complex)

        # predictor: explicit Euler (only sets the initial iterate; final
        # accuracy is set by the corrector once converged, not this guess)
        G[n, :n] = G[n - 1, :n] + dt * d1_prev1
        G[:n, n] = G[:n, n - 1] + dt * d2_prev1
        G[n, n] = -0.5j
        sbt.enforce_majorana_slice(G, n)

        err = np.inf
        it_used = 0
        for it in range(n_corr):
            row_old = G[n, :n].copy()
            col_old = G[:n, n].copy()
            S = sbt.sigma_greater_syk4(G, JJ2, JJ4)
            d1_new = np.array([rhs_t1_ho(G, S, n, j, dt, K_R_mat) for j in range(n)], dtype=complex)
            d2_new = np.array([rhs_t2_ho(G, S, i, n, dt, K_R_mat) for i in range(n)], dtype=complex)

            if use_am2:
                G[n, :n] = am2_step(G[n - 1, :n], d1_new, d1_prev1, d1_prev2, dt)
                G[:n, n] = am2_step(G[:n, n - 1], d2_new, d2_prev1, d2_prev2, dt)
            else:
                G[n, :n] = am1_step(G[n - 1, :n], d1_new, d1_prev1, dt)
                G[:n, n] = am1_step(G[:n, n - 1], d2_new, d2_prev1, dt)
            G[n, n] = -0.5j
            sbt.enforce_majorana_slice(G, n)

            err = max(
                float(np.max(np.abs(G[n, :n] - row_old))),
                float(np.max(np.abs(G[:n, n] - col_old))),
            )
            it_used = it + 1
            if err < corr_tol:
                break

        corr_final_err[n] = err
        corr_iters_used[n] = it_used

    if return_diagnostics:
        return t, G, {
            "corr_final_err": corr_final_err,
            "corr_iters_used": corr_iters_used,
            "n0": n0,
        }
    return t, G
