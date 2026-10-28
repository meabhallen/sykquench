"""delta-G formulation of the production SYK4 KBE stepper (syk_batch_tools.evolve_syk4_kbe), mu = 0, no kernel after the quench.

Write G = G_ref + dG, with G_ref(t1, t2) = G_0(t1 - t2) the undeformed (lambda = 0) equilibrium at the same beta on the whole two-time
grid. The production scheme

    G[n, j] = G[n-1, j] + dt/2 (F[n, j] + F[n-1, j])        (AM1 corrector, iterated to corr_tol; F from trapezoidal memory sums)

becomes, exactly,

    dG[n, j] = dG[n-1, j] + dt/2 (dF[n, j] + dF[n-1, j]) + R[n, j],
    dF = F[G_ref + dG] - F[G_ref],   R[n, j] = dt/2 (F_ref[n, j] + F_ref[n-1, j]) - (G_ref[n, j] - G_ref[n-1, j]).

dF is evaluated with the product rule (XY - X_r Y_r = X_r dY + dX Y_full) and dS = -J4^2 dG (3 G_r^2 + 3 G_r dG + dG^2), so no O(1)
quantities are subtracted and the rounding is relative to |dG| (~1e-4 |G| here), not |G|. R is lambda-independent: it is the
discrete residual of the undeformed equilibrium (dt error plus the memory cut at t_0 = -t_pre), the source of the lambda^0 artefact,
and it is computed once per grid and is bitwise identical for every lambda, so it cancels exactly in G(+lambda) - G(-lambda).
The equilibrium increment G_0(k dt) - G_0((k-1) dt) is summed spectrally with the factor (e^{-i w dt} - 1), not differenced.

mode="G" runs the same vectorised code on G itself (G_ref = 0, R = 0): the production scheme with a different summation order.
The pre-quench block is built exactly as evolve_syk4_kbe builds it (cubic interpolation of greater_from_spectral on 4 Nt + 1 points).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import syk_batch_tools as sbt  # noqa: E402


def spectral_greater(omega, A, beta, s, extra=None, chunk=96):
    """sum_w -i (1 - nF(w)) A(w) extra(w) e^{-i w s} dw / 2 pi on the array s (the Riemann sum of sbt.greater_from_spectral)."""
    dw = omega[1] - omega[0]
    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    c = -1j * (1.0 - nF) * A * dw / (2 * np.pi)
    if extra is not None:
        c = c * extra
    s = np.asarray(s, dtype=float)
    out = np.empty(len(s), dtype=complex)
    for a in range(0, len(s), chunk):
        out[a:a + chunk] = np.exp(-1j * np.outer(s[a:a + chunk], omega)) @ c
    return out


def toeplitz_majorana(g):
    """M[i, j] = g[i - j] (i >= j), -conj(g[j - i]) (i < j); g[0] is the diagonal."""
    n = len(g)
    K = np.arange(n)[:, None] - np.arange(n)[None, :]
    return np.where(K >= 0, g[np.abs(K)], -np.conj(g[np.abs(K)]))


def upper_trap(Nt, dt):
    """W[k, j] = trap_weights(j + 1, dt)[k] for k <= j, else 0."""
    W = np.triu(np.full((Nt, Nt), dt))
    W[0, :] *= 0.5
    W[np.diag_indices(Nt)] *= 0.5
    W[0, 0] = 0.5 * dt
    return W


def reference(omega, A0, beta, t_pre, t_post, dt, spectral_increment=False):
    """Grid, g0[k] = G_0(k dt) (g0[0] = -i/2, the production diagonal), and the increments dg[k] = g0[k] - g0[k-1].

    The increments must be those of the stored reference array: then G_ref + dG obeys the production recursion exactly. The
    floating-point difference of two neighbouring g0 values is exact (Sterbenz), so np.diff is the right choice. Summing the
    increment separately (spectral_increment=True, factor e^{-i w dt} - 1) is closer to the continuum increment but differs from
    the array's by up to ~1e-15 (independent phase rounding of the two sums, |w s| up to ~3500), which acts as a spurious source."""
    t, n0 = sbt.kbe_time_grid(t_pre, t_post, dt)
    Nt = len(t)
    k = np.arange(Nt) * dt
    g0 = spectral_greater(omega, A0, beta, k)
    G00 = g0[0]
    g0[0] = -0.5j
    if spectral_increment:
        dg = spectral_greater(omega, A0, beta, k, extra=1.0 - np.exp(1j * omega * dt))   # e^{-iwk dt} - e^{-iw(k-1)dt}
        dg[1] += G00 + 0.5j          # g0[0] is -i/2, not G_0(0)
    else:
        dg = np.concatenate([[0.0], np.diff(g0)])
    dg[0] = 0.0
    return t, n0, g0, dg


def pre_quench_block(omega, A, A0, beta, t, n0, t_pre, t_post, g0, mode, s=1.0):
    """Initial G (mode "G") or dG = G - G_ref (mode "dG") on the pre-quench block, as evolve_syk4_kbe builds G there.

    s scales the pre-quench deviation from the reference: G_pre = G_ref + s (G_eq,lambda - G_ref), i.e. an effective lambda of
    s * lambda at linear order (s = 1 is the production state)."""
    Nt = len(t)
    t_rel_max = t_pre + t_post
    t_grid = np.linspace(-t_rel_max, t_rel_max, 4 * Nt + 1)
    iu = np.triu_indices(n0 + 1, 1)
    arg = t[iu[0]] - t[iu[1]]
    X = np.zeros((Nt, Nt), dtype=complex)
    if mode == "G" and s == 1.0:
        vals = interp1d(t_grid, sbt.greater_from_spectral(omega, A, beta, t_grid), kind="cubic", fill_value="extrapolate")(arg)
        X[np.diag_indices(n0 + 1)] = -0.5j
    else:
        diff = interp1d(t_grid, spectral_greater(omega, A - A0, beta, t_grid), kind="cubic", fill_value="extrapolate")(arg)
        ierr = interp1d(t_grid, sbt.greater_from_spectral(omega, A0, beta, t_grid), kind="cubic", fill_value="extrapolate")(arg) \
            - (-np.conj(g0[iu[1] - iu[0]]))      # cubic-interpolation error of the production lambda = 0 block (one-time, ~ulp)
        vals = s * (diff + ierr)
    X[iu] = vals
    X[iu[1], iu[0]] = -np.conj(vals)
    if mode == "G" and s != 1.0:
        X[:n0 + 1, :n0 + 1] += toeplitz_majorana(g0[:n0 + 1])
    return X


class Stepper:
    """Vectorised production stepper on X = G (Gr = 0) or X = dG (Gr = G_ref)."""

    def __init__(self, Gr, X, dt, n0, JJ=1.0):
        self.Gr, self.X, self.dt, self.n0, self.JJ = Gr, X, dt, n0, JJ
        self.Nt = X.shape[0]
        self.W = upper_trap(self.Nt, dt)
        self.Sr = -JJ * Gr ** 3
        self.dS = self._dS(Gr, X)
        self.diag = -0.5j - Gr[0, 0]          # dG[n, n] (0 in dG mode, -i/2 in G mode)
        self.stagnation = False               # True: also stop the corrector once its change stops falling

    def _dS(self, gr, x):
        return -self.JJ * x * (3 * gr ** 2 + 3 * gr * x + x ** 2)

    def _update_dS(self, n):
        Gr, X = self.Gr, self.X
        self.dS[n, :n] = self._dS(Gr[n, :n], X[n, :n])
        self.dS[:n, n] = self._dS(Gr[:n, n], X[:n, n])
        self.dS[n, n] = self._dS(Gr[n, n], X[n, n])

    def fixed(self, n):
        """Matrices that do not change during step n (only rows/columns < n)."""
        Gr, X, W = self.Gr[:n, :n], self.X[:n, :n], self.W[:n, :n]
        Gf = Gr + X
        GAd, GAf = -(X + X.T), -(Gf + Gf.T)
        self.MWd, self.MWf = W * GAd, W * GAf                       # I2 of the rows (k <= j)
        self.Qr, self.Qd = W.T * (Gr + Gr.T), W.T * (X + X.T)       # I1 of the columns (k <= i)
        self.Gf, self.Xn, self.Grn = Gf, X, Gr

    def old(self, n):
        """dF at row n-1 (j < n) and column n-1 (i < n): rhs_t1(., n-1, j), rhs_t2(., i, n-1) minus their reference values."""
        Sr, dS, dt = self.Sr, self.dS, self.dt
        w = sbt.trap_weights(n, dt)
        i = n - 1
        SRr, SRd = Sr[i, :n] + Sr[:n, i], dS[i, :n] + dS[:n, i]
        I1 = (w * SRr) @ self.Xn + (w * SRd) @ self.Gf
        I2 = Sr[i, :n] @ self.MWd + dS[i, :n] @ self.MWf
        d1 = -1j * (I1 + I2)
        SAd = -(dS[:n, i] + dS[i, :n]); SAf = -(Sr[:n, i] + Sr[i, :n]) + SAd
        J1 = self.Qr @ dS[:n, i] + self.Qd @ (Sr[:n, i] + dS[:n, i])
        J2 = self.Grn @ (w * SAd) + self.Xn @ (w * SAf)
        d2 = 1j * (J1 + J2)
        return d1, d2

    def new(self, n):
        """dF at row n (j < n) and column n (i < n), using the current row/column n of X."""
        Sr, dS, dt, X, Gr = self.Sr, self.dS, self.dt, self.X, self.Gr
        w1 = sbt.trap_weights(n + 1, dt)
        SRr, SRd = Sr[n, :n] + Sr[:n, n], dS[n, :n] + dS[:n, n]
        I1 = (w1[:n] * SRr) @ self.Xn + (w1[:n] * SRd) @ self.Gf \
            + w1[n] * (2 * Sr[n, n] * X[n, :n] + 2 * dS[n, n] * (Gr[n, :n] + X[n, :n]))
        I2 = Sr[n, :n] @ self.MWd + dS[n, :n] @ self.MWf
        d1 = -1j * (I1 + I2)
        SAd = -(dS[:n, n] + dS[n, :n]); SAf = -(Sr[:n, n] + Sr[n, :n]) + SAd
        J1 = self.Qr @ dS[:n, n] + self.Qd @ (Sr[:n, n] + dS[:n, n])
        J2 = self.Grn @ (w1[:n] * SAd) + self.Xn @ (w1[:n] * SAf) \
            + w1[n] * (Gr[:n, n] * (-2 * dS[n, n]) + X[:n, n] * (-2 * (Sr[n, n] + dS[n, n])))
        d2 = 1j * (J1 + J2)
        return d1, d2

    def _set(self, n, row, col):
        X = self.X
        X[n, :n], X[:n, n] = row, col
        X[n, n] = self.diag
        r = 0.5 * (X[n, :n] - np.conj(X[:n, n]))
        X[n, :n] = r
        X[:n, n] = -np.conj(r)

    def step(self, n, R1=None, R2=None, n_corr=50, corr_tol=1e-16):
        X, dt = self.X, self.dt
        r1 = 0.0 if R1 is None else R1
        r2 = 0.0 if R2 is None else R2
        self.fixed(n)
        d1o, d2o = self.old(n)
        self._set(n, X[n - 1, :n] + dt * d1o + r1, X[:n, n - 1] + dt * d2o + r2)
        err, it = np.inf, 0
        for it in range(1, n_corr + 1):
            ro, co = X[n, :n].copy(), X[:n, n].copy()
            self._update_dS(n)
            d1n, d2n = self.new(n)
            self._set(n, X[n - 1, :n] + 0.5 * dt * (d1o + d1n) + r1, X[:n, n - 1] + 0.5 * dt * (d2o + d2n) + r2)
            err_prev, err = err, max(np.max(np.abs(X[n, :n] - ro)), np.max(np.abs(X[:n, n] - co)))
            if err < corr_tol:
                break
            if self.stagnation and it >= 4 and err > 0.7 * err_prev:     # no longer contracting: at the arithmetic floor
                break
        self._update_dS(n)
        return err, it


def residual(Gr, dt, n0, dg):
    """R1[n, :n], R2[:n, n] for n > n0 from F[G_ref] (one evaluation per row/column, no iteration)."""
    Nt = Gr.shape[0]
    st = Stepper(np.zeros_like(Gr), Gr.copy(), dt, n0)
    F1 = np.zeros((Nt, Nt), dtype=complex); F2 = np.zeros((Nt, Nt), dtype=complex)
    for n in range(n0 + 1, Nt + 1):           # row/column n-1, all j/i <= n-1
        st.fixed(n)
        F1[n - 1, :n], F2[:n, n - 1] = st.old(n)
    R1 = np.zeros((Nt, Nt), dtype=complex); R2 = np.zeros((Nt, Nt), dtype=complex)
    for n in range(n0 + 1, Nt):
        k = n - np.arange(n)
        R1[n, :n] = 0.5 * dt * (F1[n, :n] + F1[n - 1, :n]) - dg[k]
        R2[:n, n] = 0.5 * dt * (F2[:n, n] + F2[:n, n - 1]) + np.conj(dg[k])
    return R1, R2


def evolve(omega, A, A0, beta, t_pre, t_post, dt, mode="dG", n_corr=50, corr_tol=None, R=None, ref=None, verbose=200, s=1.0,
           stagnation=False):
    """One run; returns dict(t, n0, X (G or dG), g0, iters, err, sec_per_step, R). s scales the pre-quench deviation (see
    pre_quench_block); stagnation=True also stops the corrector when its change stops falling."""
    t, n0, g0, dg = ref if ref is not None else reference(omega, A0, beta, t_pre, t_post, dt)
    Nt = len(t)
    if mode == "G":
        Gr = np.zeros((Nt, Nt), dtype=complex)
        corr_tol = 1e-16 if corr_tol is None else corr_tol
    else:
        Gr = toeplitz_majorana(g0)
        corr_tol = 1e-19 if corr_tol is None else corr_tol
        if R is None:
            t0 = time.time(); R = residual(Gr, dt, n0, dg)
            if verbose:
                print(f"residual R: {time.time() - t0:.1f} s, max|R| = {np.max(np.abs(R[0])):.3e}", flush=True)
    X = pre_quench_block(omega, A, A0, beta, t, n0, t_pre, t_post, g0, mode, s)
    st = Stepper(Gr, X, dt, n0)
    st.stagnation = stagnation
    iters = np.zeros(Nt, int); errs = np.full(Nt, np.nan)
    t0 = time.time()
    for n in range(n0 + 1, Nt):
        r1 = None if mode == "G" else R[0][n, :n]
        r2 = None if mode == "G" else R[1][:n, n]
        errs[n], iters[n] = st.step(n, r1, r2, n_corr, corr_tol)
        if verbose and (n - n0) % verbose == 0:
            print(f"  {mode} n={n - n0}/{Nt - 1 - n0} t={t[n]:.2f} err={errs[n]:.2e} iters={iters[n]} ({time.time() - t0:.0f} s)", flush=True)
    return dict(t=t, n0=n0, X=st.X, g0=g0, iters=iters, err=errs, sec_per_step=(time.time() - t0) / (Nt - 1 - n0), R=R, mode=mode)
