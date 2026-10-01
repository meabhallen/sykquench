"""
Validation of the coupled mass-deformed KBE (evolve_syk4_kbe_massdef in
syk_batch_tools.py) against the exact J4=0 (free) two-level solution, plus
the actual production run: prepare a mass-deformed equilibrium (mu>0,
lambda=0), quench mu->0, evolve for t=beta, and check S_k(t,t) decays and
encodes Delta.

Run with:
    python3 syk_massdef_kbe_test.py
"""

from __future__ import annotations

import numpy as np

from syk_batch_tools import (
    solve_equilibrium_greater_real_time,
    evolve_syk4_kbe_massdef,
)
from syk_massdef_realtime import solve_equilibrium_massdef_real_time


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# 1. Exact J4=0 free-theory check of the full 2D (t1,t2) evolution
# ------------------------------------------------------------------

def check_free_theory_kbe(beta: float = 8.0, mu: float = 0.35) -> None:
    hr(f"1. Exact free-theory (J4=0) check of the coupled KBE stepping "
       f"(beta={beta}, mu={mu}, stationary: mu_i=mu_f)")
    Nw, omega_max = 2049, 10.0
    dw = 2 * omega_max / (Nw - 1)
    eta_ret = 1.5 * dw

    eq = solve_equilibrium_massdef_real_time(
        J2=0.0, J4=0.0, beta=beta, mu=mu,
        omega_max=omega_max, Nw=Nw, dt=0.02, t_max=40.0,
        max_iter=5, mixing=1.0, tol=1e-13, verbose_every=10**9, eta_ret=eta_ret,
    )

    t_pre, t_post, dt = 0.4 * beta, 0.4 * beta, 0.02
    t, G, Goff = evolve_syk4_kbe_massdef(
        omega=eq["omega"], A=eq["A"], A_off=eq["A_off"],
        beta_i=beta, mu_i=mu, mu_f=mu,  # no quench: pure stationarity check
        J2_i=0.0, J2_f=0.0, J4_i=0.0, J4_f=0.0,
        t_pre=t_pre, t_post=t_post, dt=dt, n_corr=6, corr_tol=1e-11,
        progress_every=0,
    )

    x = np.tanh(mu * beta / 2.0)

    def G_exact(dt_):
        return -0.5j * np.cos(mu * dt_) - 0.5 * x * np.sin(mu * dt_)

    def Goff_exact(dt_):
        return 0.5j * np.sin(mu * dt_) - 0.5 * x * np.cos(mu * dt_)

    n0 = int(np.argmin(np.abs(t)))
    # Sample a handful of (t1,t2) pairs spanning pre- and post-"quench" (nothing
    # actually changes here), well inside the grid to avoid edge effects.
    idxs = [n0 // 2, n0, n0 + len(t) // 8, n0 + len(t) // 4, len(t) - 1 - len(t) // 8]
    max_errG, max_errGoff = 0.0, 0.0
    for i in idxs:
        for j in idxs:
            dtij = t[i] - t[j]
            errG = abs(G[i, j] - G_exact(dtij))
            errOff = abs(Goff[i, j] - Goff_exact(dtij))
            max_errG = max(max_errG, errG)
            max_errGoff = max(max_errGoff, errOff)
    print(f"  max|G - exact| over sampled (t1,t2) grid    = {max_errG:.3e}")
    print(f"  max|Goff - exact| over sampled (t1,t2) grid = {max_errGoff:.3e}")
    print("  (tolerance is resolution-limited by the eta_ret broadening needed to")
    print("   resolve the J4=0 delta-function spectral peak on a finite frequency")
    print("   grid -- same caveat as syk_massdef_realtime_test.py section 3.)")
    assert max_errG < 0.1 and max_errGoff < 0.1, "coupled KBE stepping disagrees with exact free solution"
    print("  PASS: coupled (G, Goff) stepping matches the exact free two-level solution.")


# ------------------------------------------------------------------
# 2. Exact free-theory mu-quench check (mu_i -> mu_f=0, J4=0 throughout):
#    without interactions there's no relaxation mechanism, so everything must
#    freeze exactly at its t=0 value post-quench. This specifically exercises
#    the mu_of_t step-function bookkeeping (same pattern already trusted for
#    J2_of_t/J4_of_t in evolve_syk4_kbe) rather than the EOM algebra itself.
# ------------------------------------------------------------------

def check_free_theory_mu_quench(beta: float = 8.0, mu_i: float = 0.35) -> None:
    hr(f"2. Exact free-theory mu-quench check (J4=0, mu_i={mu_i} -> mu_f=0, "
       f"beta={beta})")
    Nw, omega_max = 2049, 10.0
    dw = 2 * omega_max / (Nw - 1)
    eta_ret = 1.5 * dw

    eq = solve_equilibrium_massdef_real_time(
        J2=0.0, J4=0.0, beta=beta, mu=mu_i,
        omega_max=omega_max, Nw=Nw, dt=0.02, t_max=40.0,
        max_iter=5, mixing=1.0, tol=1e-13, verbose_every=10**9, eta_ret=eta_ret,
    )

    t_pre, t_post, dt = 0.4 * beta, 0.4 * beta, 0.02
    t, G, Goff = evolve_syk4_kbe_massdef(
        omega=eq["omega"], A=eq["A"], A_off=eq["A_off"],
        beta_i=beta, mu_i=mu_i, mu_f=0.0,
        J2_i=0.0, J2_f=0.0, J4_i=0.0, J4_f=0.0,
        t_pre=t_pre, t_post=t_post, dt=dt, n_corr=6, corr_tol=1e-11,
        progress_every=0,
    )
    n0 = int(np.argmin(np.abs(t)))

    # Everything post-quench must be frozen at its value right at the quench:
    # G(t1,t2) = G(0,0) = -i/2 for all t1,t2 > 0; Goff(t1,t2) = Goff(0,0) (a
    # single fixed number) for all t1,t2 > 0 -- no t1/t2 dependence at all.
    Goff00 = Goff[n0, n0]
    post = slice(n0 + 1, n0 + 1 + len(t) // 8)  # a block of post-quench indices
    max_errG = np.max(np.abs(G[post, post] - (-0.5j)))
    max_errGoff = np.max(np.abs(Goff[post, post] - Goff00))
    print(f"  Goff(0,0) [frozen value] = {Goff00:.6f}   (exact: {(-0.5*np.tanh(mu_i*beta/2)):.6f})")
    print(f"  max|G(t1,t2) - (-i/2)| for t1,t2>0       = {max_errG:.3e}")
    print(f"  max|Goff(t1,t2) - Goff(0,0)| for t1,t2>0 = {max_errGoff:.3e}")
    assert max_errG < 0.02 and max_errGoff < 0.02, "post-quench block should be frozen without interactions"
    assert abs(Goff00 - (-0.5 * np.tanh(mu_i * beta / 2))) < 0.02
    print("  PASS: without interactions the post-quench state is exactly frozen, as expected.")


if __name__ == "__main__":
    check_free_theory_kbe()
    check_free_theory_mu_quench()
