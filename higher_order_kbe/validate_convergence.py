"""
validate_convergence.py -- empirically measure the convergence order of
the existing 2nd-order KBE solver (syk_batch_tools.evolve_syk4_kbe) versus
the new higher-order one (ho_solver.evolve_syk4_kbe_ho), on a real
equilibrium seed from local_runs/eq_runs.

Method: Richardson-style successive-refinement differencing. For a scheme
whose error scales as C*h^p, halving h each time makes successive
differences in the result shrink by a factor of ~2^p:

    order ~= log2( |X(h) - X(2h)| / |X(h/2) - X(h)| )

This needs no separate high-accuracy reference run -- just each scheme run
at a geometric sequence of step sizes.
"""

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import syk_batch_tools as sbt  # noqa: E402
from ho_solver import evolve_syk4_kbe_ho  # noqa: E402

EQ_FILE = (
    "/Users/meabh/Desktop/CC Quench States/SYK Quenches/local_runs/eq_runs/"
    "syk_eq_J2_0_J4_1_beta_5_klam_0p005_kc_m0p043936_kcut_0p65_dt_0p5_om_36_"
    "Nw_4001_tol_0p001_0cb37ae0.npz"
)
BETA = 5.0
T_PRE = 1.0
T_POST = 1.0
DTS = [0.16, 0.08, 0.04, 0.02]  # geometric, ratio 2, for Richardson differencing
# (coarser than an earlier attempt at [0.08,...,0.005]: at dt<~0.01 the NEW
# scheme's own discretization error dropped below ~1e-7, close enough to
# corr_tol/float noise that the Richardson ratio became noise-dominated and
# unreliable -- staying coarser keeps genuine discretization error dominant.)
PROBE_TIMES = [0.32, 0.8]  # FIXED physical times (both exact multiples of every dt above, safely inside t_post)


def probe(G, t, n0, dt):
    """G(t_target, 0) at fixed PHYSICAL times, not fixed grid-index offsets --
    a grid-index-relative probe (e.g. "one step past t=0") compares a
    DIFFERENT physical time at each dt, since that step's duration is dt
    itself; its own O(dt) approach to the boundary then swamps the actual
    numerical error and gives a meaningless "order". Every PROBE_TIMES entry
    must be an exact integer multiple of every dt in DTS."""
    vals = []
    for t_target in PROBE_TIMES:
        k = t_target / dt
        assert abs(k - round(k)) < 1e-6, f"t_target={t_target} not a multiple of dt={dt}"
        i = n0 + int(round(k))
        vals.append(G[i, n0])
    return np.array(vals)


def run_scheme(name, fn, **kwargs):
    results = []
    print(f"\n=== {name} ===")
    for dt in DTS:
        t0 = time.time()
        t, G = fn(dt=dt, **kwargs)
        n0 = int(np.argmin(np.abs(t)))
        p = probe(G, t, n0, dt)
        results.append(p)
        print(f"  dt={dt:<7} Nt={len(t):<5} probe={p}  ({time.time()-t0:.1f}s)")
    return results


def empirical_orders(results):
    """log2 of successive-difference ratios, per probe component."""
    diffs = [results[i + 1] - results[i] for i in range(len(results) - 1)]
    orders = []
    for k in range(len(diffs) - 1):
        num = np.abs(diffs[k])
        den = np.abs(diffs[k + 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            orders.append(np.log2(num / den))
    return orders


def main():
    d = np.load(EQ_FILE)
    omega, A = d["omega_real"], d["A"]

    common = dict(
        omega=omega, A=A, beta_i=BETA,
        J2_i=0.0, J2_f=0.0, J4_i=1.0, J4_f=1.0,
        t_pre=T_PRE, t_post=T_POST,
        n_corr=12, corr_tol=1e-13,
    )

    old_results = run_scheme(
        "OLD (syk_batch_tools.evolve_syk4_kbe, trapezoidal, 2nd order)",
        lambda dt, **kw: sbt.evolve_syk4_kbe(dt=dt, **kw), **common,
    )
    ho_results = run_scheme(
        "NEW (ho_solver.evolve_syk4_kbe_ho, AM2 + 4th-order quadrature)",
        lambda dt, **kw: evolve_syk4_kbe_ho(dt=dt, **kw), **common,
    )

    print("\n--- empirical order (per probe component, per refinement pair) ---")
    print("OLD:", empirical_orders(old_results))
    print("NEW:", empirical_orders(ho_results))


if __name__ == "__main__":
    main()
