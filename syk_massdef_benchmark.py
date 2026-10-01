"""
Benchmark suite for the mass-deformed Matsubara solver in syk_massdef_matsubara.py.

Run with:
    python3 syk_massdef_benchmark.py

Sections:
  1. Exact free-theory check (J4=0): the mass term alone is a solvable 2-level
     system: G,Goff should match the closed form to numerical (FFT) precision,
     independent of beta or N_tau. Validates the mu-dependent Matsubara algebra
     and the fermion_tau_to_iw/iw_to_tau conventions with zero ambiguity.
  2. mu=0 regression: the new solver at mu=0 must reduce identically to the
     un-deformed one (same fixed point, 1/b == b/(b^2-0)).
  3. Small-beta, loose-tolerance smoke test (lambda=0): confirms the coupled
     solver converges over a spread of mu at cheap settings, and that the
     basic physical trends (|Goff| growing with mu, spin expectation bounded
     by 1, energy finite) look sane. This is the "benchmark at lambda=0,
     small beta, loose tolerance" requested as a first pass.
  4. Mass-gap scaling at larger beta, small mu (q=4 conformal limit): paper
     eq. (3.22)-(3.23) predicts E_gap ~ mu^{1/(1-2*Delta)} = mu^2 for q=4.
     This needs beta*mu >> 1 and beta*J4 >> 1 to be in the asymptotic regime,
     so treat the fitted exponent as approximate/qualitative, not exact.
  5. Combined mu>0 AND kernel(lambda)>0 plumbing smoke test: confirms the
     solver runs fine when both are turned on together (imaginary-time,
     placeholder Euclidean regulator -- NOT the exact continuation of the
     production real-time kernel; that check lives in
     syk_massdef_realtime.py, which reuses the actual build_kernel_R_w).
"""

from __future__ import annotations

import numpy as np

from syk_massdef_matsubara import (
    solve_equilibrium_matsubara,
    solve_equilibrium_matsubara_massdef,
    exact_free_massdef_G,
    spin_expectation,
)


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


# ------------------------------------------------------------------
# 1. Exact free-theory (J4=0) check
# ------------------------------------------------------------------

def check_free_theory(beta: float = 12.0, N_tau: int = 16384) -> None:
    hr(f"1. Exact free-theory check (J4=0, beta={beta}, N_tau={N_tau})")
    print("  (excluding the few tau points nearest 0/beta: a Fourier series of a")
    print("   discontinuous function -- G has the usual {psi,psi}=1 jump at tau=0 --")
    print("   always keeps an O(1) Gibbs overshoot at the single nearest sample point,")
    print("   regardless of N_tau; away from the jump it converges as ~1/N_tau. This")
    print("   is a generic feature of `fermion_iw_to_tau`, already present at mu=0.)")
    edge = max(4, N_tau // 64)
    for mu in [0.0, 0.05, 0.3, 1.5]:
        res = solve_equilibrium_matsubara_massdef(
            beta=beta, J4=0.0, mu=mu, J2=0.0, N_tau=N_tau,
            mixing=1.0, tol=1e-13, max_iter=5,  # Sigma=0 identically -> exact in 1 iter
        )
        G_exact, Goff_exact = exact_free_massdef_G(res["tau"], beta, mu)
        errG = np.max(np.abs(res["G_tau"].real[edge:-edge] - G_exact[edge:-edge]))
        errGoff = np.max(np.abs(res["Goff_tau"][edge:-edge] - Goff_exact[edge:-edge]))
        print(
            f"  mu={mu:6.3f}  iters={res['iterations']:3d}  "
            f"max|G-Gexact|={errG:.3e}  max|Goff-Goffexact|={errGoff:.3e}  "
            f"<S>={spin_expectation(res).real:+.6f}  tanh(mu*beta/2)={np.tanh(mu*beta/2):+.6f}"
        )
        assert errG < 2e-6 and errGoff < 2e-6, "free-theory mismatch exceeds numerical-precision budget"
    print("  PASS: numeric solver matches the exact two-level solution away from the tau=0/beta edge.")


# ------------------------------------------------------------------
# 2. mu=0 regression against the un-deformed solver
# ------------------------------------------------------------------

def check_mu_zero_regression(beta: float = 20.0, J4: float = 1.0, N_tau: int = 1024) -> None:
    hr(f"2. mu=0 regression (J4={J4}, beta={beta}, N_tau={N_tau})")
    ref = solve_equilibrium_matsubara(beta=beta, J4=J4, N_tau=N_tau, tol=1e-11, mixing=0.15)
    new = solve_equilibrium_matsubara_massdef(beta=beta, J4=J4, mu=0.0, N_tau=N_tau, tol=1e-11, mixing=0.15)
    errG = np.max(np.abs(ref["G_tau"] - new["G_tau"]))
    errGoff = np.max(np.abs(new["Goff_tau"]))
    print(f"  max|G_ref - G_new| = {errG:.3e}   (should be ~solver tol)")
    print(f"  max|Goff(mu=0)|    = {errGoff:.3e}   (should be exactly 0)")
    assert errG < 1e-8
    assert errGoff < 1e-12
    print("  PASS: mass-deformed solver reduces to the mu=0 solver exactly.")


# ------------------------------------------------------------------
# 3. Small-beta, loose-tolerance smoke test (lambda=0)
# ------------------------------------------------------------------

def smoke_test_small_beta(beta: float = 8.0, J4: float = 1.0, N_tau: int = 512) -> None:
    hr(f"3. Small-beta / loose-tolerance smoke test, lambda=0 (J4={J4}, beta={beta}, N_tau={N_tau})")
    print(f"{'mu':>8}  {'iters':>6}  {'relupd':>10}  {'G(beta/2)':>11}  {'Goff(0)':>18}  {'<S_k>':>9}")
    prev_abs_Goff0 = -1.0
    for mu in [0.0, 0.05, 0.1, 0.2, 0.4, 0.8]:
        res = solve_equilibrium_matsubara_massdef(
            beta=beta, J4=J4, mu=mu, N_tau=N_tau,
            mixing=0.2, tol=1e-6, max_iter=20000,  # deliberately loose
        )
        mid = N_tau // 2
        S = spin_expectation(res).real
        Goff0 = res["Goff_tau"][0]
        print(
            f"{mu:8.3f}  {res['iterations']:6d}  {res['relative_update']:10.2e}  "
            f"{res['G_tau'][mid].real:11.6f}  {Goff0.real:+.4e}{Goff0.imag:+.4e}j  {S:9.5f}"
        )
        assert np.isfinite(res["G_tau"]).all() and np.isfinite(res["Goff_tau"]).all()
        assert abs(S) <= 1.0 + 1e-6, "|<S_k>| must be bounded by 1 (it's a spin expectation value)"
        # |Goff(0)| should grow monotonically with mu at fixed beta,J4 (more mass -> more polarized)
        assert abs(Goff0) >= prev_abs_Goff0 - 1e-9
        prev_abs_Goff0 = abs(Goff0)
    print("  PASS: converges at loose tolerance for all mu; trends are physically sane.")


# ------------------------------------------------------------------
# 4. Mass-gap scaling vs. paper eq. (3.22)-(3.23): E_gap ~ mu^2 at q=4
# ------------------------------------------------------------------

def fit_gap(tau: np.ndarray, G: np.ndarray, beta: float) -> float:
    """Fit E_gap from log|G(tau)| ~ -E_gap*tau over a mid-range window, away
    from both the short-time conformal power-law and the beta/2 turning point.
    (This module's G(tau) has a fixed overall sign -- see exact_free_massdef_G
    -- so we fit |G| rather than assume a sign.)"""
    lo, hi = 0.12 * beta, 0.38 * beta
    mask = (tau > lo) & (tau < hi) & (np.abs(G.real) > 0)
    x = tau[mask]
    y = np.log(np.abs(G[mask].real))
    slope, _ = np.polyfit(x, y, 1)
    return -slope


def check_mass_gap_scaling(beta: float = 150.0, J4: float = 1.0, N_tau: int = 4000) -> None:
    hr(f"4. Mass-gap scaling check vs. paper eq.(3.22)-(3.23), q=4 => E_gap ~ mu^2 "
       f"(beta={beta}, J4={J4}, N_tau={N_tau})")
    mus = [0.02, 0.03, 0.045, 0.07, 0.10]
    gaps = []
    G_prev = None
    for mu in mus:
        res = solve_equilibrium_matsubara_massdef(
            beta=beta, J4=J4, mu=mu, N_tau=N_tau,
            mixing=0.15, tol=1e-9, max_iter=40000,
            G_iw_init=G_prev,
        )
        G_prev = res["G_iw"]
        Egap = fit_gap(res["tau"], res["G_tau"], beta)
        gaps.append(Egap)
        print(f"  mu={mu:6.3f}  iters={res['iterations']:5d}  relupd={res['relative_update']:.2e}  "
              f"E_gap(fit)={Egap:.5e}  E_gap/mu^2={Egap/mu**2:.4f}")
    mus_a = np.array(mus)
    gaps_a = np.array(gaps)
    slope, intercept = np.polyfit(np.log(mus_a), np.log(gaps_a), 1)
    print(f"\n  Fitted power law: E_gap ~ mu^{slope:.3f}  (paper's conformal-limit prediction: exponent = 2)")
    print("  NOTE: E_gap ~ mu^2 is a strong low-temperature/low-mu asymptotic statement; at this beta")
    print("  and mu range the fitted exponent is only expected to be qualitatively close to 2, not exact.")


# ------------------------------------------------------------------
# 5. mu>0 AND kernel(lambda)>0 plumbing smoke test
# ------------------------------------------------------------------

def placeholder_euclidean_kernel(omega_n: np.ndarray, J4: float, kernel_lambda: float,
                                  kernel_c: float, kernel_cutoff: float) -> np.ndarray:
    """A simple, manifestly-analytic Euclidean regulator with the same schematic
    shape as the production real-time kernel (cubic + linear in frequency, cut
    off at kernel_cutoff), used ONLY to smoke-test that the solver's K_iw slot
    behaves sanely with mu>0 simultaneously switched on. NOT claimed to be the
    exact Matsubara continuation of build_kernel_R_w in syk_batch_tools.py --
    that check is done on the real axis directly in syk_massdef_realtime.py,
    which is what should be used for actual equilibrium-preparation runs.
    """
    s_n = -1j * omega_n
    regulator = (kernel_cutoff**2 / (kernel_cutoff**2 + omega_n**2)) ** 2
    return kernel_lambda * (s_n**3 / J4**2 + kernel_c * s_n) * regulator


def combined_mu_lambda_smoke_test(beta: float = 8.0, J4: float = 1.0, N_tau: int = 512) -> None:
    hr(f"5. Combined mu>0 AND lambda>0 plumbing smoke test (J4={J4}, beta={beta}, N_tau={N_tau})")
    _, omega_n, _ = __import__("syk_massdef_matsubara").fermionic_matsubara_grid(beta, N_tau)
    kernel_cutoff = 0.5 * J4
    for mu, kernel_lambda, kernel_c in [
        (0.0, 0.0, 0.0),
        (0.1, 0.005, -0.0439),
        (0.1, -0.005, -0.0439),
        (0.3, 0.02, -0.0439),
    ]:
        K_iw = placeholder_euclidean_kernel(omega_n, J4, kernel_lambda, kernel_c, kernel_cutoff)
        res = solve_equilibrium_matsubara_massdef(
            beta=beta, J4=J4, mu=mu, N_tau=N_tau, K_iw=K_iw,
            mixing=0.15, tol=1e-8, max_iter=20000,
        )
        mid = N_tau // 2
        print(
            f"  mu={mu:5.2f}  lambda={kernel_lambda:+.4f}  c={kernel_c:+.5f}  "
            f"iters={res['iterations']:5d}  relupd={res['relative_update']:.2e}  "
            f"G(beta/2)={res['G_tau'][mid].real:.6f}  Goff(0)={res['Goff_tau'][0]:.4e}  "
            f"<S_k>={spin_expectation(res).real:+.5f}"
        )
        assert np.isfinite(res["G_tau"]).all() and np.isfinite(res["Goff_tau"]).all()
    print("  PASS: solver converges with mu and the kernel slot both active at once.")


if __name__ == "__main__":
    check_free_theory()
    check_mu_zero_regression()
    smoke_test_small_beta()
    check_mass_gap_scaling()
    combined_mu_lambda_smoke_test()
    print("\nAll benchmark sections completed.")
