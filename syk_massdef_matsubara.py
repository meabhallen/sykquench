"""
Matsubara (imaginary-time) Schwinger-Dyson solver for the mass-deformed SYK4
model of Nosaka & Numasawa, JHEP08(2020)081 [arXiv:1912.12302]:

    H = H_SYK4 + H_M,
    H_SYK4 = sum_{i<j<k<l} J_{ijkl} psi_i psi_j psi_k psi_l,
    H_M    = i*mu * sum_k s_k psi_{2k-1} psi_{2k}   (mass/spin term, sec. 2.2 of the paper)

paired against an (optional) engineered retarded kernel K_iw exactly as
already used for `solve_equilibrium_matsubara` in syk_matsubara_kernel.ipynb
(K_iw enters the diagonal sector only, added alongside Sigma).

Derivation of the modified SDE (see chat writeup / README section below for
the full derivation): H_M is a *deterministic* bilinear, not part of the
random disorder average, so it never gets dressed by the self-energy -- the
only change to the Schwinger-Dyson content is that the single diagonal
propagator G(tau) = (1/N) sum_i <T psi_i(tau) psi_i(0)> is now accompanied by
an off-diagonal partner

    Goff(tau) = <T psi_{2k}(tau) psi_{2k-1}(0)>   (independent of k, s_k=+1 WLOG)

and the pair (G, Goff) mix through H_M like a 2-level system. Writing

    b(iwn) = s_n - Sigma(iwn) - K(iwn),   s_n = -i*wn,
    Sigma(tau) = J2^2 G(tau) + J4^2 G(tau)^3      (diagonal sector only, unchanged),

inverting the resulting 2x2 Dyson matrix

    G^{-1}(iwn) = [[ b,      i*mu ],
                   [ -i*mu,  b    ]]     (Sigma_off = -i*mu*delta(tau-tau'), exact/bare)

gives

    G(iwn)    =  b / (b^2 - mu^2)
    Goff(iwn) = -i*mu / (b^2 - mu^2)

which is *exactly* the mu=0 solver's `G_target = 1/b` with the substitution
1/b -> b/(b^2-mu^2), plus the new Goff channel that falls out for free (no
extra self-consistency loop needed: Goff never feeds back into Sigma). At
mu=0 this reduces identically to the existing `solve_equilibrium_matsubara`.

Sanity-checked two independent ways (see syk_massdef_benchmark.py):
  - J4=0 exact two-level solution (closed form, no SDE iteration needed).
  - mu=0 regression against the un-deformed solver.
  - conformal-limit mass-gap scaling E_gap ~ mu^{1/(1-2*Delta)} = mu^2 (q=4,
    Delta=1/4), paper eq. (3.22)-(3.23).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


# ============================================================
# Fermionic Matsubara grid + transforms (identical to
# syk_matsubara_kernel.ipynb's cell, lifted into an importable module)
# ============================================================

def fermionic_matsubara_grid(beta: float, N_tau: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if N_tau <= 0 or N_tau % 2 != 0:
        raise ValueError("N_tau must be a positive even integer.")

    j = np.arange(N_tau)
    tau = (j + 0.5) * beta / N_tau

    n = np.arange(-N_tau // 2, N_tau // 2)
    omega_n = 2.0 * np.pi * (n + 0.5) / beta

    return tau, omega_n, n


def fermion_tau_to_iw(f_tau: np.ndarray, beta: float) -> np.ndarray:
    # F(iw_n) = int_0^beta dtau exp(-i w_n tau) F(tau).
    f_tau = np.asarray(f_tau, dtype=np.complex128)
    N_tau = f_tau.size

    j = np.arange(N_tau)
    n = np.arange(-N_tau // 2, N_tau // 2)

    twisted_tau = f_tau * np.exp(-1j * np.pi * j / N_tau)

    return (
        (beta / N_tau)
        * np.exp(-1j * np.pi * (n + 0.5) / N_tau)
        * np.fft.fftshift(np.fft.fft(twisted_tau))
    )


def fermion_iw_to_tau(f_iw: np.ndarray, beta: float) -> np.ndarray:
    # F(tau) = (1/beta) sum_n exp(+i w_n tau) F(iw_n).
    f_iw = np.asarray(f_iw, dtype=np.complex128)
    N_tau = f_iw.size

    j = np.arange(N_tau)
    n = np.arange(-N_tau // 2, N_tau // 2)

    twisted_iw = f_iw * np.exp(+1j * np.pi * n / N_tau)

    return (
        (N_tau / beta)
        * np.exp(+1j * np.pi * (j + 0.5) / N_tau)
        * np.fft.ifft(np.fft.ifftshift(twisted_iw))
    )


# ============================================================
# mu=0 solver, kept verbatim for regression testing against the
# mass-deformed solver below.
# ============================================================

def solve_equilibrium_matsubara(
    beta: float,
    J4: float,
    *,
    J2: float = 0.0,
    N_tau: int = 2048,
    K_iw: Optional[np.ndarray] = None,
    G_iw_init: Optional[np.ndarray] = None,
    mixing: float = 0.10,
    tol: float = 1e-11,
    max_iter: int = 20_000,
    verbose_every: int = 10**9,
) -> Dict[str, Any]:
    """Undeformed (mu=0) solver -- identical to syk_matsubara_kernel.ipynb."""
    tau, omega_n, n = fermionic_matsubara_grid(beta, N_tau)
    s_n = -1j * omega_n

    K_iw = np.zeros(N_tau, dtype=np.complex128) if K_iw is None else np.asarray(K_iw, dtype=np.complex128)
    G_iw = 1.0 / s_n if G_iw_init is None else np.asarray(G_iw_init, dtype=np.complex128).copy()

    converged = False
    relative_update = np.inf
    for iteration in range(max_iter):
        G_tau = fermion_iw_to_tau(G_iw, beta)
        Sigma_tau = J2**2 * G_tau + J4**2 * G_tau**3
        Sigma_iw = fermion_tau_to_iw(Sigma_tau, beta)

        G_target = 1.0 / (s_n - Sigma_iw - K_iw)
        relative_update = float(np.linalg.norm(G_target - G_iw) / max(np.linalg.norm(G_target), 1e-30))
        G_iw = (1.0 - mixing) * G_iw + mixing * G_target

        if iteration % verbose_every == 0 or relative_update < tol:
            print(f"  beta={beta:g} equilibrium iter={iteration:6d}  relative update={relative_update:.3e}")
        if relative_update < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(f"Equilibrium solver did not converge (final update={relative_update:.3e}).")

    G_tau = fermion_iw_to_tau(G_iw, beta)
    Sigma_tau = J2**2 * G_tau + J4**2 * G_tau**3
    Sigma_iw = fermion_tau_to_iw(Sigma_tau, beta)

    return {
        "beta": float(beta), "J2": float(J2), "J4": float(J4), "N_tau": int(N_tau),
        "tau": tau, "u": tau / beta, "omega_n": omega_n, "n": n, "s_n": s_n, "K_iw": K_iw,
        "G_tau": G_tau, "G_iw": G_iw, "Sigma_tau": Sigma_tau, "Sigma_iw": Sigma_iw,
        "iterations": int(iteration + 1), "relative_update": relative_update,
    }


# ============================================================
# Mass-deformed solver
# ============================================================

def solve_equilibrium_matsubara_massdef(
    beta: float,
    J4: float,
    mu: float = 0.0,
    *,
    J2: float = 0.0,
    N_tau: int = 2048,
    K_iw: Optional[np.ndarray] = None,
    G_iw_init: Optional[np.ndarray] = None,
    mixing: float = 0.10,
    tol: float = 1e-11,
    max_iter: int = 20_000,
    verbose_every: int = 10**9,
) -> Dict[str, Any]:
    """Self-consistent solve including the H_M = i*mu*sum_k s_k psi_{2k-1}psi_{2k}
    mass/spin deformation (Nosaka-Numasawa sec. 2-3), optionally alongside an
    engineered retarded kernel K_iw (same convention as `solve_equilibrium_matsubara`).

    Returns G_tau/G_iw (diagonal correlator, same object as the mu=0 solver)
    plus Goff_tau/Goff_iw (the off-diagonal correlator <psi_{2k}(tau)psi_{2k-1}(0)>).
    Only G feeds Sigma, so the iteration is exactly the mu=0 fixed point with a
    different final map from b(iwn) to G(iwn); mu=0 reproduces
    `solve_equilibrium_matsubara` exactly (same b, since 1/b == b/(b^2-0)).
    """
    tau, omega_n, n = fermionic_matsubara_grid(beta, N_tau)
    s_n = -1j * omega_n
    mu = float(mu)

    K_iw = np.zeros(N_tau, dtype=np.complex128) if K_iw is None else np.asarray(K_iw, dtype=np.complex128)
    G_iw = 1.0 / s_n if G_iw_init is None else np.asarray(G_iw_init, dtype=np.complex128).copy()

    converged = False
    relative_update = np.inf
    for iteration in range(max_iter):
        G_tau = fermion_iw_to_tau(G_iw, beta)
        Sigma_tau = J2**2 * G_tau + J4**2 * G_tau**3
        Sigma_iw = fermion_tau_to_iw(Sigma_tau, beta)

        b_iw = s_n - Sigma_iw - K_iw
        G_target = b_iw / (b_iw**2 - mu**2)
        relative_update = float(np.linalg.norm(G_target - G_iw) / max(np.linalg.norm(G_target), 1e-30))
        G_iw = (1.0 - mixing) * G_iw + mixing * G_target

        if iteration % verbose_every == 0 or relative_update < tol:
            print(f"  beta={beta:g} mu={mu:g} equilibrium iter={iteration:6d}  relative update={relative_update:.3e}")
        if relative_update < tol:
            converged = True
            break

    if not converged:
        raise RuntimeError(f"Mass-deformed equilibrium solver did not converge (final update={relative_update:.3e}).")

    G_tau = fermion_iw_to_tau(G_iw, beta)
    Sigma_tau = J2**2 * G_tau + J4**2 * G_tau**3
    Sigma_iw = fermion_tau_to_iw(Sigma_tau, beta)
    b_iw = s_n - Sigma_iw - K_iw

    Goff_iw = -1j * mu / (b_iw**2 - mu**2)
    Goff_tau = fermion_iw_to_tau(Goff_iw, beta)

    return {
        "beta": float(beta), "J2": float(J2), "J4": float(J4), "mu": mu, "N_tau": int(N_tau),
        "tau": tau, "u": tau / beta, "omega_n": omega_n, "n": n, "s_n": s_n, "K_iw": K_iw,
        "G_tau": G_tau, "G_iw": G_iw, "Goff_tau": Goff_tau, "Goff_iw": Goff_iw,
        "Sigma_tau": Sigma_tau, "Sigma_iw": Sigma_iw, "b_iw": b_iw,
        "iterations": int(iteration + 1), "relative_update": relative_update,
    }


# ============================================================
# Exact free-theory (J4=0) reference solution -- a single Majorana pair
# precessing under H_M alone, diagonalized exactly (H_M = -(mu/2) sigma_z in
# the {|S=+1>, |S=-1>} basis). See derivation in the module docstring's
# accompanying chat writeup.
# ============================================================

def exact_free_massdef_G(tau: np.ndarray, beta: float, mu: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact G(tau), Goff(tau) for J4=J2=0 (pure mass term), 0 < tau < beta, in
    this module's own sign convention (matched empirically against
    fermion_iw_to_tau applied to the bare G_iw=1/s_n seed: that seed gives
    G(tau)=-1/2 for H=0, i.e. this codebase's G(tau) = -<T psi(tau)psi(0)>,
    the overall extra minus sign relative to the bare operator average that
    `solve_equilibrium_matsubara`'s Sigma=+J^2*G^3 convention already bakes
    in; carried through here so mu>0 stays consistent with mu=0).

    Undoing that sign on the textbook two-level-system solution (H_M =
    -(mu/2)*sigma_z on the single-pair {S=+-1} qubit, psi1=sigma_x/sqrt2,
    psi2=sigma_y/sqrt2, S=-2i*psi1*psi2=sigma_z) gives, in THIS module's
    convention:
        G(tau)    = -cosh(mu*(beta/2-tau)) / (2cosh(mu*beta/2))
        Goff(tau) =  i*sinh(mu*(beta/2-tau)) / (2cosh(mu*beta/2))
    Verified against the solver's own J4=0 output (see syk_massdef_benchmark.py).
    """
    tau = np.asarray(tau, dtype=float)
    mu = float(mu)
    denom = 2.0 * np.cosh(mu * beta / 2.0)
    G = -np.cosh(mu * (beta / 2.0 - tau)) / denom
    Goff = 1j * np.sinh(mu * (beta / 2.0 - tau)) / denom
    return G, Goff


def spin_expectation(result: Dict[str, Any]) -> complex:
    """<S_k> = <-2i psi_{2k-1} psi_{2k}>, expressed via this module's Goff(tau)
    (defined with the same overall sign convention as G(tau), see
    exact_free_massdef_G). Empirically calibrated against the exact J4=0
    two-level solution (<S_k> = tanh(mu*beta/2)): <S_k> = -2i*Goff(0)."""
    Goff0 = result["Goff_tau"][0]
    return -2j * Goff0
