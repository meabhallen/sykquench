"""
ho_multistep.py -- Adams-Moulton corrector formulas used to advance one new
time point given the RHS ("d1"/"d2" in syk_batch_tools' notation) evaluated
at the current point and, for the higher-order formulas, at previously
finalized points. Matches the indexing convention already used in
syk_batch_tools.evolve_syk4_kbe: solving for y_n given y_{n-1} and RHS
values f_n (iterated), f_{n-1}, f_{n-2}, f_{n-3} (already finalized, fixed).

    AM1 (trapezoidal, order 2, already what syk_batch_tools uses):
        y_n = y_{n-1} + h/2  * (f_n + f_{n-1})

    AM2 (order 3):
        y_n = y_{n-1} + h/12 * (5 f_n + 8 f_{n-1} - f_{n-2})

    AM3 (order 4):
        y_n = y_{n-1} + h/24 * (9 f_n + 19 f_{n-1} - 5 f_{n-2} + f_{n-3})

STARTUP CAVEAT (found by testing, not just assumed): a multistep method's
global order is capped by its WORST startup step, because a one-time local
error of size O(h^{q+1}) at step 1 does not decay away in a non-dissipative
system -- it persists as an O(h^{q+1}) contribution to the error at every
later time, competing directly against the bulk method's O(h^p) accumulated
error. For a target global order p, every step (startup included) needs
local error O(h^{p+1}), i.e. "step order" q >= p-1.

  - AM2 (p=3) needs startup order q>=2. A single AM1 step has q=2 (local
    error O(h^3)) -- exactly sufficient. Confirmed empirically below: AM2
    with one AM1 startup step cleanly achieves global order ~3.

  - AM3 (p=4) needs startup order q>=3 at EVERY startup step. The naive
    "AM1 for step 1, AM2 for step 2" bootstrap gives step 1 only q=2,
    which is NOT sufficient -- confirmed empirically below: this bootstrap
    caps the achievable order at ~3, not 4. AM3 is implemented here and is
    algebraically correct (a decaying-ODE test alone would have hidden
    this, since decay damps the startup error away -- caught by re-testing
    on a non-decaying oscillatory ODE instead), but is NOT used by
    ho_solver.py's default order=3 configuration until a matching-order
    startup (e.g. a genuine one-step 4th-order scheme for step 1, or a
    fully implicit small system solved for the first 2 points at once) is
    implemented. Treat order=4 in ho_solver.py as experimental/unvalidated.
"""

import numpy as np


def am1_step(y_prev, f_n, f_prev, h):
    return y_prev + 0.5 * h * (f_n + f_prev)


def am2_step(y_prev, f_n, f_prev1, f_prev2, h):
    return y_prev + (h / 12.0) * (5.0 * f_n + 8.0 * f_prev1 - f_prev2)


def am3_step(y_prev, f_n, f_prev1, f_prev2, f_prev3, h):
    return y_prev + (h / 24.0) * (9.0 * f_n + 19.0 * f_prev1 - 5.0 * f_prev2 + f_prev3)


def _integrate(order, h, T, omega=1.3):
    """Integrate y' = i*omega*y, y(0)=1 (non-decaying, so startup errors
    don't get masked by damping) with the given AM order and the naive
    AM1->AM2->AM3 bootstrap, iterating each step's corrector to full
    self-consistency (matching syk_batch_tools' own fixed-point loop)."""
    def f(y):
        return 1j * omega * y

    n_steps = int(round(T / h))
    ys = [1.0 + 0j]
    fs = [f(1.0 + 0j)]
    for n in range(1, n_steps + 1):
        y_prev = ys[-1]
        y_n = y_prev
        for _ in range(80):
            f_n = f(y_n)
            if order == 1 or n == 1:
                y_new = am1_step(y_prev, f_n, fs[-1], h)
            elif order == 2 or n == 2:
                y_new = am2_step(y_prev, f_n, fs[-1], fs[-2], h)
            else:
                y_new = am3_step(y_prev, f_n, fs[-1], fs[-2], fs[-3], h)
            if abs(y_new - y_n) < 1e-15:
                y_n = y_new
                break
            y_n = y_new
        ys.append(y_n)
        fs.append(f(y_n))
    return ys[-1]


def _self_test() -> None:
    T = 2.0
    omega = 1.3
    exact = np.exp(1j * omega * T)
    hs = [0.2, 0.1, 0.05, 0.025]

    expected = {1: (1.8, 2.2), 2: (2.7, 3.3)}
    for order, (lo, hi) in expected.items():
        errs = [abs(_integrate(order, h, T, omega) - exact) for h in hs]
        emp = np.polyfit(np.log(hs), np.log(errs), 1)[0]
        assert lo < emp < hi, f"AM{order}: empirical order {emp}, expected ~{order+1}. errs={errs}"
        print(f"AM{order} (+ its startup): empirical order {emp:.2f} -- validated.")

    # Document (not assert-fail on) the known AM3 startup limitation.
    errs = [abs(_integrate(3, h, T, omega) - exact) for h in hs]
    emp = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    print(f"AM3 (naive AM1/AM2 startup): empirical order {emp:.2f} "
          f"(capped by startup, NOT ~4 -- known limitation, see module docstring).")
    assert 2.5 < emp < 3.5, (
        f"AM3 naive-bootstrap order changed to {emp}; either the startup got fixed "
        "(update ho_solver.py to use order=4) or something else broke."
    )

    print("ho_multistep: all self-tests passed.")


if __name__ == "__main__":
    _self_test()
