"""
Mass-deformation quench: prepare a thermal equilibrium of H_SYK4 + mu*H_M
(lambda=0, i.e. no engineered kernel) at inverse temperature beta, then quench
mu -> 0 and evolve under pure SYK4 for t=beta, tracking the equal-time spin
expectation value

    <S_k(t,t)> = -2 * Goff(t,t)

(see the dGoff_dt.../dG_dt... docstrings in syk_batch_tools.py for the
derivation of this relation, cross-checked there against the exact free
two-level thermal result <S_k>=tanh(mu*beta/2)).

Run with:
    python3 syk_massdef_quench_run.py
"""

from __future__ import annotations

import time

import numpy as np

from syk_batch_tools import evolve_syk4_kbe_massdef
from syk_massdef_realtime import solve_equilibrium_massdef_real_time


def run_quench(beta: float, mu: float, J4: float = 1.0, dt: float = 0.1,
                t_pre_factor: float = 2.0, n_corr: int = 4, corr_tol: float = 1e-6):
    print("\n" + "=" * 70)
    print(f"beta={beta}, mu={mu}, J4={J4}, dt={dt}  (lambda=0, quench mu->0)")
    print("=" * 70)

    t0 = time.time()
    eq = solve_equilibrium_massdef_real_time(
        J2=0.0, J4=J4, beta=beta, mu=mu,
        omega_max=10.0, Nw=4097, dt=0.02, t_max=max(80.0, 5.0 * beta),
        max_iter=3000, tol=1e-8, mixing=0.1, verbose_every=500,
    )
    print(f"[eq] converged={eq['converged']}  ({time.time()-t0:.1f}s)")
    assert eq["converged"]

    t_pre = t_pre_factor * beta
    t_post = beta
    t1 = time.time()
    t, G, Goff, diag = evolve_syk4_kbe_massdef(
        omega=eq["omega"], A=eq["A"], A_off=eq["A_off"],
        beta_i=beta, mu_i=mu, mu_f=0.0,
        J2_i=0.0, J2_f=0.0, J4_i=J4, J4_f=J4,
        t_pre=t_pre, t_post=t_post, dt=dt,
        n_corr=n_corr, corr_tol=corr_tol, progress_every=50,
        return_diagnostics=True,
    )
    print(f"[kbe] done  ({time.time()-t1:.1f}s), max corr err = {np.nanmax(diag['corr_final_err']):.3e}, "
          f"max corr iters used = {int(np.nanmax(diag['corr_iters_used']))}")

    n0 = int(diag["n0"])
    t_post_arr = t[n0:]
    S_diag = np.array([-2.0 * Goff[n, n] for n in range(n0, len(t))])
    return t_post_arr, S_diag, eq, t


def fit_late_time_rate(t_post: np.ndarray, S: np.ndarray, beta: float,
                        lo_frac: float = 0.3, hi_frac: float = 0.95):
    y = np.abs(S.real)
    lo, hi = lo_frac * beta, hi_frac * beta
    mask = (t_post >= lo) & (t_post <= hi) & (y > 0)
    if mask.sum() < 3:
        return None
    slope, intercept = np.polyfit(t_post[mask], np.log(y[mask]), 1)
    resid = np.log(y[mask]) - (slope * t_post[mask] + intercept)
    r2 = 1.0 - np.sum(resid**2) / np.sum((np.log(y[mask]) - np.log(y[mask]).mean())**2)
    return -slope, r2, mask


if __name__ == "__main__":
    results = {}
    for beta in [5.0, 10.0]:
        t_post, S, eq, t_full = run_quench(beta=beta, mu=0.005, J4=1.0, dt=0.1)
        results[beta] = (t_post, S)

        print(f"\n  S_k(0,0) at quench      = {S[0].real:+.6e}")
        print(f"  S_k(t,t) at t=0.25*beta = {S[int(0.25*len(S))].real:+.6e}")
        print(f"  S_k(t,t) at t=0.5*beta  = {S[int(0.5*len(S))].real:+.6e}")
        print(f"  S_k(t,t) at t=beta      = {S[-1].real:+.6e}")

        fit = fit_late_time_rate(t_post, S, beta)
        if fit is not None:
            rate, r2, mask = fit
            print(f"  fitted late-time decay rate lambda = {rate:.5f}  (R^2={r2:.4f}, "
                  f"window t in [{t_post[mask][0]:.2f},{t_post[mask][-1]:.2f}])")
            print(f"  lambda*beta/pi = {rate*beta/np.pi:.4f}  "
                  f"(candidate predictions: 4*Delta=1.0 [T=0 boundary-state formula, eq 6.5], "
                  f"2*Delta=0.5 [thermal single-particle-type falloff]; q=4 => Delta=1/4)")
        else:
            print("  Could not fit a late-time rate (S_k(t,t) too small/noisy in the fit window).")

    if 5.0 in results and 10.0 in results:
        r5 = fit_late_time_rate(*results[5.0], beta=5.0)
        r10 = fit_late_time_rate(*results[10.0], beta=10.0)
        if r5 and r10:
            print("\n" + "=" * 70)
            print("beta-scaling check (most robust check available: does the decay rate")
            print("scale as 1/beta, the universal signature of conformal/Delta-driven")
            print("relaxation, independent of the T=0-vs-thermal subtlety noted above?)")
            print("=" * 70)
            print(f"  lambda(beta=5)  = {r5[0]:.5f}   lambda(beta=5)*5   = {r5[0]*5:.4f}")
            print(f"  lambda(beta=10) = {r10[0]:.5f}   lambda(beta=10)*10 = {r10[0]*10:.4f}")
            print(f"  ratio lambda(5)/lambda(10) = {r5[0]/r10[0]:.4f}  (expect ~2.0 if rate ~ 1/beta)")
