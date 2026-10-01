"""
Thin dict-based wrapper around syk_batch_tools.solve_equilibrium_greater_real_time,
which now has the mass-deformation (mu) machinery merged directly into it
(see its docstring for the full derivation). Kept only so the validation
scripts written against a dict interface (syk_massdef_realtime_test.py,
syk_massdef_kbe_test.py, syk_massdef_quench_run.py) don't need to be
rewritten -- this is no longer a second, independent implementation.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from syk_batch_tools import solve_equilibrium_greater_real_time


def solve_equilibrium_massdef_real_time(
    J2: float,
    J4: float,
    beta: float,
    mu: float = 0.0,
    omega_max: float = 8.0,
    Nw: int = 4097,
    t_max: Optional[float] = None,
    dt: float = 0.05,
    max_iter: int = 2000,
    tol: float = 1e-9,
    mixing: float = 0.05,
    eta_ret: float = 1e-6,
    verbose_every: int = 100,
    kernel_lambda: float = 0.0,
    kernel_c: float = 0.0,
    kernel_cutoff: Optional[float] = None,
) -> Dict[str, Any]:
    (
        omega_real, A, t_grid, F_t, Ggt_t, GR_w, K_R_w, converged, _final_dab_sqrt_max,
        A_off, Ggt_off_t, GRoff_w,
    ) = solve_equilibrium_greater_real_time(
        J2=J2, J4=J4, beta=beta, mu=mu,
        omega_max=omega_max, Nw=Nw, t_max=t_max, dt=dt,
        max_iter=max_iter, tol=tol, mixing=mixing, eta_ret=eta_ret,
        verbose_every=verbose_every,
        kernel_lambda=kernel_lambda, kernel_c=kernel_c, kernel_cutoff=kernel_cutoff,
    )
    return {
        "omega": omega_real, "A": A, "t": t_grid, "F_t": F_t, "Ggt_t": Ggt_t, "GR_w": GR_w,
        "A_off": A_off, "Ggt_off_t": Ggt_off_t, "GRoff_w": GRoff_w,
        "K_R_w": K_R_w, "kernel_cutoff": kernel_cutoff, "mu": mu, "beta": beta, "J2": J2, "J4": J4,
        "converged": converged,
    }
