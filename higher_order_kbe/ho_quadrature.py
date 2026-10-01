"""
ho_quadrature.py -- 4th-order end-corrected trapezoidal quadrature weights,
as a drop-in replacement for syk_batch_tools.trap_weights.

Derivation (Euler-Maclaurin):

    int_0^{(m-1)h} f dx = T - (h^2/12) [f'(b) - f'(a)] + O(h^4)

where T is the plain composite trapezoidal sum and B_2 = 1/6 is the second
Bernoulli number (B_2/2! = 1/12). Estimating f'(a), f'(b) via the standard
one-sided 2nd-order-accurate 3-point finite difference

    f'(a) ~= (-3 f_0 + 4 f_1 - f_2) / (2h) + O(h^2)
    f'(b) ~= (3 f_{m-1} - 4 f_{m-2} + f_{m-3}) / (2h) + O(h^2)

and substituting turns the O(h^2) derivative correction into an O(h^4)
correction to the integral (matching the next term the Euler-Maclaurin
series would have produced anyway), giving a quadrature rule with global
error O(h^4) -- two orders better than plain trapezoidal (O(h^2)).

This only modifies 3 weights at each end; interior weights are the
ordinary trapezoidal value h. Requires m >= 6 so the two end corrections
don't overlap; smaller m falls back to plain trapezoidal (order doesn't
matter for windows that short).
"""

import numpy as np


def ho_weights(m: int, h: float) -> np.ndarray:
    """4th-order end-corrected trapezoidal weights for m uniformly spaced
    points covering [0, (m-1)h]. Falls back to plain trapezoidal for m < 6."""
    if m < 1:
        raise ValueError("m must be positive.")
    w = np.full(m, h, dtype=float)
    w[0] *= 0.5
    if m > 1:
        w[-1] *= 0.5
    if m < 6:
        return w

    # Boundary correction weights (additive), from -(h/24)*(3 f0 -4 f1 +f2)
    # at the left end and the mirror-image correction at the right end.
    left_corr = np.array([-3.0, 4.0, -1.0]) * (h / 24.0)
    w[0:3] += left_corr
    w[-1:-4:-1] += left_corr  # mirror: w[-1]+=left_corr[0], w[-2]+=left_corr[1], w[-3]+=left_corr[2]
    return w


def _self_test() -> None:
    """Verify: exact on monomials up to x^3 (degree 3), and empirically
    4th order (error ~ h^4) on x^4 and on a transcendental function."""
    rng = np.random.default_rng(0)

    for m in (6, 7, 8, 9, 15, 16, 30, 31):
        h = 0.037
        x = np.arange(m) * h
        w = ho_weights(m, h)

        # exact-on-cubics check
        for k in range(4):
            f = x**k
            exact = ((m - 1) * h) ** (k + 1) / (k + 1)
            got = np.sum(w * f)
            assert np.isclose(got, exact, atol=1e-9, rtol=1e-9), (
                f"m={m} k={k}: got {got}, exact {exact}"
            )

    # Empirical order checks on x^4 and exp(x), refining h at FIXED total
    # domain length L (m grows as ~L/h) -- refining h and m independently
    # (e.g. fixed m, shrinking h) shrinks the domain too and gives a
    # meaningless "order" dominated by the shrinking integration range.
    L = 6.0
    hs = [0.3, 0.15, 0.075, 0.0375]

    errs = []
    for h in hs:
        m = int(round(L / h)) + 1
        x = np.arange(m) * h
        w = ho_weights(m, h)
        f = x**4
        exact = x[-1] ** 5 / 5
        errs.append(abs(np.sum(w * f) - exact))
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 3.7 < order < 4.3, f"x^4 empirical order {order}, expected ~4"

    errs = []
    for h in hs:
        m = int(round(L / h)) + 1
        x = np.arange(m) * h
        w = ho_weights(m, h)
        f = np.exp(x)
        exact = np.exp(x[-1]) - 1.0
        errs.append(abs(np.sum(w * f) - exact))
    order = np.polyfit(np.log(hs), np.log(errs), 1)[0]
    assert 3.7 < order < 4.3, f"exp(x) empirical order {order}, expected ~4"

    # sanity: plain-trapezoidal fallback for small m still integrates a line exactly
    for m in range(1, 6):
        h = 0.1
        x = np.arange(m) * h
        w = ho_weights(m, h)
        if m >= 2:
            f = 2.0 * x + 1.0
            exact = ((m - 1) * h) * (1.0 + (2.0 * (m - 1) * h + 2.0) / 2.0) - 0  # trapz of line is exact regardless
            got = np.sum(w * f)
            # trapezoidal is always exact for straight lines
            exact = np.trapezoid(f, x)
            assert np.isclose(got, exact), f"m={m}: fallback trapz not exact on a line"

    print("ho_quadrature: all self-tests passed "
          f"(exact on cubics, empirical order {order:.2f} on exp(x)).")


if __name__ == "__main__":
    _self_test()
