"""
Smoke tests + cross-validation for the real-time mass-deformed equilibrium
solver in syk_massdef_realtime.py -- the production-style (build_kernel_R_w
using the actual tuned engineered kernel) extension of
`solve_equilibrium_greater_real_time`.

Run with:
    python3 syk_massdef_realtime_test.py

Sections:
  1. mu>0, lambda=0: quick convergence smoke test over a few mu values.
  2. mu>0 AND lambda>0 together (README's tuned kernel_c/kernel_cutoff pairs):
     confirms "equilibrium preparation" runs fine with both switched on.
  3. Exact free-theory (J4=0) real-time check: the pure mass term is an exactly
     solvable 2-level system (same closed form used in
     syk_massdef_benchmark.py, continued to real time), used here to validate
     BOTH the diagonal and off-diagonal real-time greater functions built by
     solve_equilibrium_massdef_real_time. Because J4=0 makes the spectral
     function a genuine delta function (no interacting continuum to broaden
     it), this needs `eta_ret` set comparable to the frequency-grid spacing
     to be resolved by the trapezoidal frequency integral -- a purely
     numerical artifact of this singular test case, absent once J4>0 provides
     its own O(J4) continuum broadening (see the default eta_ret=1e-6 used
     everywhere else in this file and in production).
  4. Cross-validation against the Matsubara solver (diagonal sector only,
     lambda=0): converts the converged real-time A_diag(w) to Euclidean
     G(tau) via `real_spectral_to_imag_time` and compares to
     syk_massdef_matsubara.solve_equilibrium_matsubara_massdef at the same
     (beta, mu, J4). (The off-diagonal sector isn't cross-checked this way:
     `real_spectral_to_imag_time` assumes a real, even-under-tau->beta-tau
     Euclidean function, which is the diagonal G(tau) but not Goff(tau)
     -- Goff(tau) is purely imaginary and odd under tau->beta-tau. That
     channel is validated directly in real time instead, in section 3.)
"""

from __future__ import annotations

import numpy as np

from syk_batch_tools import real_spectral_to_imag_time
from syk_massdef_realtime import solve_equilibrium_massdef_real_time
from syk_massdef_matsubara import solve_equilibrium_matsubara_massdef


def hr(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def smoke_test_mu_only(beta: float = 8.0, J4: float = 1.0) -> None:
    hr(f"1. Real-time solver, mu>0, lambda=0 (beta={beta}, J4={J4})")
    for mu in [0.0, 0.1, 0.3, 0.6]:
        res = solve_equilibrium_massdef_real_time(
            J2=0.0, J4=J4, beta=beta, mu=mu,
            omega_max=10.0, Nw=2049, dt=0.05, t_max=60.0,
            max_iter=1500, tol=1e-7, mixing=0.15, verbose_every=10**9,
        )
        i0 = int(np.argmin(np.abs(res["t"])))
        print(
            f"  mu={mu:4.2f}  converged={res['converged']}  "
            f"F_diag(0)={res['F_t'][i0].real:.6f}  "
            f"max|Ggt_off|={np.max(np.abs(res['Ggt_off_t'])):.6f}"
        )
        assert res["converged"], f"real-time solver failed to converge at mu={mu}"
    print("  PASS.")


def combined_mu_lambda_test(beta: float = 8.0, J4: float = 1.0) -> None:
    hr(f"2. Real-time solver, mu>0 AND lambda>0 together (beta={beta}, J4={J4})")
    # README's tuned (kernel_c, kernel_cutoff-factor) pairs.
    tuned = [(-0.043936, 0.65), (-0.053648, 0.75)]
    for mu in [0.1, 0.3]:
        for kernel_c, cutoff_factor in tuned:
            for kernel_lambda in [0.005, -0.005]:
                res = solve_equilibrium_massdef_real_time(
                    J2=0.0, J4=J4, beta=beta, mu=mu,
                    omega_max=10.0, Nw=2049, dt=0.05, t_max=60.0,
                    max_iter=1500, tol=1e-7, mixing=0.15, verbose_every=10**9,
                    kernel_lambda=kernel_lambda, kernel_c=kernel_c,
                    kernel_cutoff=cutoff_factor * J4,
                )
                i0 = int(np.argmin(np.abs(res["t"])))
                print(
                    f"  mu={mu:4.2f}  lambda={kernel_lambda:+.4f}  c={kernel_c:+.6f}  "
                    f"Lambda={cutoff_factor*J4:.3f}  converged={res['converged']}  "
                    f"F_diag(0)={res['F_t'][i0].real:.6f}  "
                    f"max|Ggt_off|={np.max(np.abs(res['Ggt_off_t'])):.6f}"
                )
                assert res["converged"], "real-time solver failed with mu and kernel both on"
    print("  PASS: equilibrium preparation runs fine with mu and the engineered kernel both active.")


def check_free_theory_real_time(beta: float = 8.0, mu: float = 0.35) -> None:
    hr(f"3. Exact free-theory (J4=0) real-time check, diagonal + off-diagonal "
       f"(beta={beta}, mu={mu})")
    Nw, omega_max = 2049, 10.0
    dw = 2 * omega_max / (Nw - 1)
    eta_ret = 1.5 * dw  # resolve the J4=0 delta-function spectral peak on this grid; see docstring
    res = solve_equilibrium_massdef_real_time(
        J2=0.0, J4=0.0, beta=beta, mu=mu,
        omega_max=omega_max, Nw=Nw, dt=0.02, t_max=40.0,
        max_iter=5, mixing=1.0, tol=1e-13, verbose_every=10**9, eta_ret=eta_ret,
    )
    t = res["t"]
    x = np.tanh(mu * beta / 2.0)
    Gdiag_exact = -0.5j * np.cos(mu * t) - 0.5 * np.sin(mu * t) * x
    Goff_exact = 0.5j * np.sin(mu * t) - 0.5 * np.cos(mu * t) * x
    mask = np.abs(t) < 0.35 * beta  # stay well away from the pre/post-quench-window edges
    errG = np.max(np.abs(res["Ggt_t"][mask] - Gdiag_exact[mask]))
    errGoff = np.max(np.abs(res["Ggt_off_t"][mask] - Goff_exact[mask]))
    print(f"  eta_ret={eta_ret:.4f} (~1.5x grid spacing dw={dw:.4f}, needed only for this")
    print("   singular non-interacting test case -- J4>0 provides its own O(J4) broadening)")
    print(f"  max|Ggt_diag - exact| = {errG:.3e}")
    print(f"  max|Ggt_off  - exact| = {errGoff:.3e}")
    assert errG < 0.05 and errGoff < 0.05, "real-time free-theory mismatch"
    print("  PASS: both channels match the exact two-level real-time solution.")


def cross_validate_diag_against_matsubara(beta: float = 20.0, J4: float = 1.0, mu: float = 0.2) -> None:
    hr(f"4. Cross-validation (diagonal sector): real-time (FDT) vs. Matsubara solver, "
       f"lambda=0 (beta={beta}, J4={J4}, mu={mu})")
    rt = solve_equilibrium_massdef_real_time(
        J2=0.0, J4=J4, beta=beta, mu=mu,
        omega_max=12.0, Nw=4097, dt=0.02, t_max=200.0,
        max_iter=4000, tol=1e-9, mixing=0.1, verbose_every=10**9,
    )
    assert rt["converged"]

    mb = solve_equilibrium_matsubara_massdef(
        beta=beta, J4=J4, mu=mu, N_tau=4096, mixing=0.15, tol=1e-11, max_iter=40000,
        verbose_every=10**9,
    )

    tauE_diag, G_E_diag = real_spectral_to_imag_time(rt["omega"], rt["A"], beta, Ntau=4096)
    G_mb_interp = np.interp(tauE_diag, mb["tau"], mb["G_tau"].real)

    ref_idx = len(tauE_diag) // 4  # avoid tau=beta/2 and the tau=0/beta edges
    ratio_diag = G_mb_interp[ref_idx] / G_E_diag[ref_idx]

    edge = len(tauE_diag) // 20
    rel_err_diag = np.max(np.abs(G_mb_interp[edge:-edge] - ratio_diag * G_E_diag[edge:-edge])) / np.max(
        np.abs(G_mb_interp)
    )

    print(f"  ratio(Matsubara/real-time) at tau={tauE_diag[ref_idx]:.3f} = {ratio_diag:.6f}")
    print(f"  max relative deviation from a pure rescaling = {rel_err_diag:.3e}")
    print("  (a fixed ratio close to +-1 here means the two independent solvers -- Matsubara")
    print("   fixed point vs. real-frequency FDT self-consistency -- agree on the whole shape")
    print("   of the mass-deformed diagonal correlator, a strong, solver-independent check.)")
    assert abs(abs(ratio_diag) - 1.0) < 0.01
    assert rel_err_diag < 0.01, "diagonal-sector shapes disagree beyond expected discretization error"
    print("  PASS.")


if __name__ == "__main__":
    smoke_test_mu_only()
    combined_mu_lambda_test()
    check_free_theory_real_time()
    cross_validate_diag_against_matsubara()
    print("\nAll real-time benchmark sections completed.")
