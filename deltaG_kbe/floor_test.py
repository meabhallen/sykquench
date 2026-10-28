"""Floor test: the production beta J = 32, dt = 0.053 runs with the pre-quench deviation scaled by s = 0.01
(G_pre = G_ref + s (G_eq,lambda - G_ref), effective lambda = +-2.5e-5 at linear order), evolved on G and on delta-G.
The paired signal is 100x smaller, so the plain-G scheme meets its round-off floor inside the data while delta-G should not.

    python floor_test.py run         # R (once) + G+, G-, dG+, dG- at s = 0.01   -> syk_sims/deltaG_test/*_s0p01_*.npz
    python floor_test.py analyse     # floors, flattening point, E h_loc; deltaG_floor_test_s0p01.png
"""
import sys, time, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from common import OUT, PROD, EQ0, BETA, DT, T_PRE  # noqa: E402
from dg_solver import reference, residual, toeplitz_majorana, evolve  # noqa: E402

S = 0.01
T_POST = 32.0
LAM = 0.0025
TAU = 1 / (1.5 * np.pi)
TOL = {"G": 1e-16, "dG": 3e-19 * S}       # G: production; dG: just above its arithmetic floor (~1-2e-19 at s = 1, scales with |dG|)


def fname(mode, sgn, s=S):
    tag = "s0p01" if s == S else "tpost_32"
    return OUT / (f"b32_dt0p053_{tag}_{mode}_{sgn}.npz" if s == S else f"b32_dt0p053_tpost_32_{mode}_{sgn}.npz")


def run():
    with np.load(EQ0) as z:
        omega, A0 = z["omega_real"], z["A"]
    A = {}
    for sg, f in PROD.items():
        with np.load(f) as z:
            A[sg] = z["A"]
    ref = reference(omega, A0, BETA, T_PRE, T_POST, DT)
    t, n0, g0, dg = ref
    t0 = time.time(); R = residual(toeplitz_majorana(g0), DT, n0, dg)
    print(f"residual {time.time() - t0:.0f} s", flush=True)
    for mode in ("G", "dG"):
        for sg in (+1, -1):
            o = evolve(omega, A[sg], A0, BETA, T_PRE, T_POST, DT, mode=mode, R=R, ref=ref, verbose=150, s=S,
                       corr_tol=TOL[mode], stagnation=True)
            it, er = o["iters"][n0 + 1:], o["err"][n0 + 1:]
            print(f"{mode}{'+' if sg > 0 else '-'}: {o['sec_per_step']:.3f} s/step, iters median {np.median(it):.0f} max {it.max()}, "
                  f"err median {np.median(er):.1e} max {er.max():.1e}", flush=True)
            np.savez_compressed(fname(mode, "p" if sg > 0 else "m"), t=o["t"], n0=o["n0"], X_post=o["X"][o["n0"]:, :], g0=o["g0"],
                                iters=o["iters"], err=o["err"],
                                meta=json.dumps(dict(mode=mode, lam=LAM * sg, s=S, corr_tol=TOL[mode], sec_per_step=o["sec_per_step"])))


def analyse():
    sys.path.insert(0, "/Users/meabh/Downloads/syk_convergence/clock_matched"); sys.path.insert(0, "/Users/meabh/Downloads")
    import unify as U
    from unify import _interp
    gc = U.gc_abs(TAU * BETA, BETA)
    (r_p1, _), (r_m1, _) = U.r_run(32, DT, LAM, 0.0), U.r_run(32, DT, -LAM, 0.0)
    c = (r_m1 - r_p1) / (2 * LAM)

    def load(mode, sg, s):
        with np.load(fname(mode, sg, s)) as z:
            return dict(t=z["t"], n0=int(z["n0"]), X=z["X_post"], g0=z["g0"], iters=z["iters"], meta=json.loads(str(z["meta"])))

    runs = {(m, sg, s): load(m, sg, s) for m in ("G", "dG") for sg in ("p", "m") for s in (S, 1.0)}
    t, n0 = runs[("dG", "p", S)]["t"], runs[("dG", "p", S)]["n0"]
    nrows = runs[("dG", "p", S)]["X"].shape[0]
    T = np.arange(0.0, t[n0 + nrows - 1] - TAU * BETA / 2, DT)
    i = np.clip(np.rint((T + TAU * BETA / 2 - t[n0]) / DT).astype(int), 0, nrows - 1)
    j = np.clip(np.rint((T - TAU * BETA / 2 - t[0]) / DT).astype(int), 0, len(t) - 1)
    x = T / BETA

    def paired(m, s):
        Dp, Dm = (runs[(m, sg, s)]["X"][i, j] / (s * LAM * gc) for sg in ("p", "m"))
        return 0.5 * (Dp - Dm), Dp, Dm

    def flat_x(xx, D):
        """first x > 0.4 where the local decay rate of the 21-pt-mean |dD/dx| (smoothed over 0.05) falls below 1.5 (signal ~3.8)"""
        sm = np.abs(np.convolve(np.gradient(D, xx), np.ones(21) / 21, mode="same"))
        h = U.local_h(xx, sm, smooth_x=0.05)
        k = (xx > 0.4) & (xx < xx[-1] - 0.03) & (h < 1.5)
        return xx[k][0] if k.any() else np.nan

    res, curves = [], {}
    for m, s in (("G", S), ("dG", S), ("dG", 1.0), ("G", 1.0)):
        O, Dp, Dm = paired(m, s)
        rs = s * c * LAM
        r_p, r_m = 1 - rs, 1 + rs
        xe = x * r_m; k = xe <= x[-1] * r_p; xe = xe[k]
        Dt = 0.5 * (_interp(x * r_p, Dp, xe) - Dm[k])
        d = np.abs(np.gradient(O, x)); sm = np.abs(np.convolve(np.gradient(O, x), np.ones(21) / 21, mode="same"))
        tail = x >= x[-1] - 0.04
        R = U.rates(xe, Dt)
        res.append(dict(run=f"{m}, s={s:g}", p2p=np.median(d[tail]), sm21=np.median(sm[tail]), x_flat=flat_x(x, O),
                        cut_C_E=R["xcC"] - TAU / 2, cut_d_E=R["xcd"] - TAU / 2,
                        sec=runs[(m, "p", s)]["meta"]["sec_per_step"], iters=np.median(runs[(m, "p", s)]["iters"][n0 + 1:])))
        curves[(m, s)] = (x, O, xe, Dt, R)
    # point-to-point noise of D along the cut from third differences (MAD; noise std of D3 = sqrt(20) sigma), expressed as |dD/dx|
    dx = x[1] - x[0]
    for r, key in zip(res, (("G", S), ("dG", S), ("dG", 1.0), ("G", 1.0))):
        O = curves[key][1]
        for lo, hi, nm in ((0.75, x[-1] - 0.01, "noise_late"), (0.3, 0.5, "noise_mid")):
            k = (x >= lo) & (x <= hi)
            d3 = np.diff(O[k], 3)
            sig = 1.4826 * np.median(np.abs(d3 - np.median(d3))) / np.sqrt(20)
            r[nm] = sig * np.sqrt(2) / dx
    print("point-to-point noise of D_odd as |dD/dx| (third-difference MAD), T/beta in [0.75, end] and [0.3, 0.5]:")
    for r in res:
        print(f"  {r['run']:10s} {r['noise_late']:.2e}  {r['noise_mid']:.2e}")
    print("paired odd response, |dD/dx| in units of s|lambda||G_c| (D normalised by s|lambda|):")
    for r in res:
        print(f"  {r['run']:10s} last 0.04: point-to-point {r['p2p']:.2e}, 21-pt mean {r['sm21']:.2e}; decay flattens at T/beta = {r['x_flat']:.3f}; "
              f"E cut_C at (t-tau/2)/beta_eff = {r['cut_C_E']:.3f} (cut_d {r['cut_d_E']:.3f}); {r['sec']:.3f} s/step, iters {r['iters']:.0f}")
    # where the s = 0.01 curves depart from the s = 1 delta-G signal (normalised): first x > 0.3 with a > 30% deviation of the 21-pt mean
    ref = np.abs(np.convolve(np.gradient(curves[("dG", 1.0)][1], x), np.ones(21) / 21, mode="same"))
    for m in ("G", "dG"):
        sm = np.abs(np.convolve(np.gradient(curves[(m, S)][1], x), np.ones(21) / 21, mode="same"))
        dev = np.abs(sm / ref - 1); k = (x > 0.3) & (x < x[-1] - 0.02) & (dev > 0.3)
        print(f"  {m}, s=0.01: 21-pt |dD/dx| departs by >30% from the s=1 dG signal at T/beta = {x[k][0] if k.any() else np.nan:.3f}; "
              f"median |ratio-1| over T/beta in [0.3, 0.6]: {np.median(dev[(x > 0.3) & (x < 0.6)]):.3f}")

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {("G", S): "C0", ("dG", S): "C3", ("dG", 1.0): "0.5"}
    LAB = {("G", S): r"$G$, $s=0.01$", ("dG", S): r"$\delta G$, $s=0.01$", ("dG", 1.0): r"$\delta G$, $s=1$ (production $\lambda$)"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(wspace=0.25))
    for key in (("dG", 1.0), ("G", S), ("dG", S)):
        xx, O, xe, Dt, R = curves[key]
        axs[0].semilogy(xx, np.abs(np.convolve(np.gradient(O, xx), np.ones(21) / 21, mode="same")), color=COL[key], lw=1.3, label=LAB[key])
        if key[1] == S:
            axs[1].semilogy(xx, np.abs(np.gradient(O, xx)), color=COL[key], lw=0.5, alpha=0.8, label=LAB[key])
        k = (xe > 0.3) & (xe <= R["xcC"]); off = TAU / 2
        axs[2].plot(xe[k] - off, R["hC"][k], color=COL[key], lw=1.5, label=LAB[key] + f" (cut {R['xcC'] - off:.2f})")
        k2 = (xe > R["xcC"]) & (xe < xe[-1] - 0.06)
        axs[2].plot(xe[k2] - off, R["hC"][k2], color=COL[key], lw=0.8, alpha=0.3)
    axs[0].set_title(r"paired odd response: $|\partial_xD_{\rm odd}|$, 21-pt mean", fontsize=10, loc="left")
    axs[1].set_title(r"same, point to point ($s=0.01$)", fontsize=10, loc="left")
    axs[2].set_title(r"E: $h_{\rm loc}$ of $|\tilde D-\tilde C|$ (faint: past each cut)", fontsize=10, loc="left")
    for ax in axs[:2]:
        ax.set_xlabel(r"$T/\beta$"); ax.set_ylim(1e-14, 1e-2); ax.legend(frameon=False, fontsize=8)
    axs[0].set_ylabel(r"units of $s|\lambda||G_c|$")
    axs[2].set_xlabel(r"$(t-\tau/2)/\beta_{\rm eff}$"); axs[2].set_ylim(3.0, 5.0); axs[2].set_xlim(0.3, 0.85)
    axs[2].axhline(3.7735, color="k", lw=0.5, ls=":"); axs[2].legend(frameon=False, fontsize=8)
    fig.suptitle(r"$\beta J=32$, $dt=0.053$: pre-quench deviation scaled by $s=0.01$ ($\lambda_{\rm eff}=\pm2.5\times10^{-5}$), $D$ normalised by $s|\lambda|$",
                 fontsize=11)
    fig.savefig(HERE / "deltaG_floor_test_s0p01.png", dpi=130, bbox_inches="tight")


if __name__ == "__main__":
    {"run": run, "analyse": analyse}[sys.argv[1]]()
