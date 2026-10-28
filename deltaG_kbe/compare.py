"""Compare the delta-G and vectorised-G test runs with the production runs at the same settings.

    python compare.py [t_post]      (default 32; the tag of the run_test.py outputs)

(1) early-time agreement of G and of the paired odd response; (2) late-time floors of |dD/dx| along the cut tau = beta/1.5pi and
where cut_C / cut_d end, for the plain pair, the single lambda<0 run and the clock-matched pair E; (3) h_loc of |Dt - C| past the old
cut; writes deltaG_floor_compare.png next to this file.
"""
import sys, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/Users/meabh/Downloads/syk_convergence/clock_matched")
sys.path.insert(0, "/Users/meabh/Downloads")
from common import OUT, PROD, BETA, DT  # noqa: E402
from dg_solver import toeplitz_majorana  # noqa: E402

t_post = sys.argv[1] if len(sys.argv) > 1 else "32"
TAU = 1 / (1.5 * np.pi)
LAM = 0.0025


def load_mine(mode, s):
    with np.load(OUT / f"b32_dt0p053_tpost_{t_post}_{mode}_{s}.npz") as z:
        t, n0, X, g0 = z["t"], int(z["n0"]), z["X_post"], z["g0"]
        meta = json.loads(str(z["meta"])); iters = z["iters"]
    return dict(t=t, n0=n0, X=X, g0=g0, meta=meta, iters=iters)


def load_prod(s):
    with np.load(PROD[s]) as z:
        return dict(t=z["t_kbe"], n0=int(z["n0"]), G=z["Ggt_post"])


def cut_idx(t, n0, nrows):
    tr, tc = t[n0:n0 + nrows], t
    T = np.arange(0.0, tr[-1] - TAU * BETA / 2, DT)
    i = np.clip(np.rint((T + TAU * BETA / 2 - tr[0]) / DT).astype(int), 0, nrows - 1)
    j = np.clip(np.rint((T - TAU * BETA / 2 - tc[0]) / DT).astype(int), 0, len(tc) - 1)
    return T / BETA, i, j


def floor_stats(x, D):
    d = np.abs(np.gradient(D, x)); sm = np.abs(np.convolve(np.gradient(D, x), np.ones(21) / 21, mode="same"))
    tail = x >= x[-1] - 0.04
    return np.median(d[tail]), np.median(sm[tail])


if __name__ == "__main__":
    import unify as U          # an.py helpers: gc_abs, local_h, fit_rates, cut_C, cut_d, r_run, _interp
    from unify import _interp
    gc = U.gc_abs(TAU * BETA, BETA)
    P = {s: load_prod(s) for s in (+1, -1)}
    M = {(m, s): load_mine(m, {1: "p", -1: "m"}[s]) for m in ("dG", "G") for s in (+1, -1)}
    nrows = min(P[1]["G"].shape[0], M[("dG", 1)]["X"].shape[0])
    n0 = P[1]["n0"]
    ncol = min(P[1]["G"].shape[1], M[("dG", 1)]["X"].shape[1])
    Gref_post = toeplitz_majorana(M[("dG", 1)]["g0"])[n0:n0 + nrows, :ncol]
    full = {("prod", s): P[s]["G"][:nrows, :ncol] for s in (+1, -1)}
    full.update({("G", s): M[("G", s)]["X"][:nrows, :ncol] for s in (+1, -1)})
    full.update({("dG", s): Gref_post + M[("dG", s)]["X"][:nrows, :ncol] for s in (+1, -1)})

    # (1) early-time agreement: first 40 post-quench rows
    print("(1) agreement with production, rows n0..n0+39 (relative to max|G| = %.3f):" % np.max(np.abs(full[("prod", 1)][:40])))
    for m in ("G", "dG"):
        for s in (+1, -1):
            d = np.max(np.abs(full[(m, s)][:40] - full[("prod", s)][:40]))
            print(f"    {m:2s} lambda={s * LAM:+g}: max|G - G_prod| = {d:.2e}")
        Op = 0.5 * (full[("prod", 1)][:40] - full[("prod", -1)][:40])
        Om = 0.5 * (M[(m, 1)]["X"][:40, :ncol] - M[(m, -1)]["X"][:40, :ncol]) if m == "dG" else 0.5 * (full[(m, 1)][:40] - full[(m, -1)][:40])
        print(f"    {m:2s} paired odd part: max|O - O_prod| = {np.max(np.abs(Om - Op)):.2e} (max|O| = {np.max(np.abs(Op)):.2e})")
    for m in ("G", "dG"):
        d = np.max(np.abs(full[(m, 1)] - full[("prod", 1)]), axis=1)
        print(f"    {m:2s} lambda=+: max|G - G_prod| by row: n-n0=100: {d[100]:.1e}, 300: {d[300]:.1e}, last: {d[-1]:.1e}" if nrows > 300 else "")

    # (2) floors along the cut
    x, i, j = cut_idx(P[1]["t"][:ncol], n0, nrows)
    cutD = {}
    for s in (+1, -1):
        cutD[("prod", s)] = full[("prod", s)][i, j] / (LAM * gc)
        cutD[("G", s)] = full[("G", s)][i, j] / (LAM * gc)
        cutD[("dG", s)] = M[("dG", s)]["X"][:, :ncol][i, j] / (LAM * gc)          # G_ref is constant along the cut
    (r_p, _), (r_m, _) = U.r_run(32, DT, LAM, 0.0), U.r_run(32, DT, -LAM, 0.0)
    rows, curves = [], {}
    for m in ("prod", "G", "dG"):
        pair = 0.5 * (cutD[(m, 1)] - cutD[(m, -1)])
        single = cutD[(m, -1)]
        xe = x * r_m; k = xe <= x[-1] * r_p; xe = xe[k]
        Dt = 0.5 * (_interp(x * r_p, cutD[(m, 1)], xe) - cutD[(m, -1)][k])
        for name, xx, D in (("plain pair", x, pair), ("single lambda<0", x * r_m, single), ("E", xe, Dt)):
            p2p, sm = floor_stats(xx, D)
            R = U.rates(xx, D)
            rows.append(dict(scheme=m, curve=name, p2p=p2p, sm21=sm, cut_C=R["xcC"], cut_d=R["xcd"]))
            curves[(m, name)] = (xx, D, R)
    print("(2) floors over the last 0.04 of data, |dD/dx| in units of |lambda||G_c| (cuts on the curve's own clock, before the -tau/2 shift):")
    for r in rows:
        print(f"    {r['scheme']:4s} {r['curve']:16s} point-to-point {r['p2p']:.2e}  21-pt mean {r['sm21']:.2e}  cut_C {r['cut_C']:.3f}  cut_d {r['cut_d']:.3f}")

    # figure
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    COL = {"prod": "0.45", "G": "C0", "dG": "C3"}
    LAB = {"prod": "production (cluster)", "G": "vectorised, G", "dG": r"vectorised, $\delta G$"}
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw=dict(wspace=0.25))
    for m in ("prod", "G", "dG"):
        for ax, name in zip(axs[:2], ("plain pair", "single lambda<0")):
            xx, D, R = curves[(m, name)]
            ax.semilogy(xx, np.abs(np.convolve(np.gradient(D, xx), np.ones(21) / 21, mode="same")), color=COL[m], lw=1.2, label=LAB[m])
        xx, D, R = curves[(m, "E")]
        off = 0.5 * (r_p + r_m) * TAU / 2
        k = (xx > 0.3) & (xx <= max(R["xcC"], 0.3)); ax = axs[2]
        ax.plot(xx[k] - off, R["hC"][k], color=COL[m], lw=1.4, label=LAB[m] + f" (cut {R['xcC'] - off:.2f})")
        k2 = (xx > R["xcC"]) & (xx < xx[-1] - 0.06)
        ax.plot(xx[k2] - off, R["hC"][k2], color=COL[m], lw=0.8, alpha=0.35)
    axs[0].set_title(r"plain $\pm\lambda$ pair: $|\partial_xD_{\rm odd}|$ (21-pt mean)", fontsize=10, loc="left")
    axs[1].set_title(r"single $\lambda<0$ run: $|\partial_xD|$ (21-pt mean)", fontsize=10, loc="left")
    axs[2].set_title(r"E: $h_{\rm loc}$ of $|\tilde D-\tilde C|$ (faint: past each cut)", fontsize=10, loc="left")
    for ax in axs[:2]:
        ax.set_xlabel(r"$T/\beta$ (own clock for the single run)"); ax.set_ylim(1e-15, 1e-2); ax.legend(frameon=False, fontsize=8)
    axs[2].set_xlabel(r"$(t-\tau/2)/\beta_{\rm eff}$"); axs[2].set_ylim(3.0, 5.0); axs[2].set_xlim(0.3, 0.85)
    axs[2].axhline(3.7735, color="k", lw=0.5, ls=":"); axs[2].legend(frameon=False, fontsize=8)
    fig.suptitle(r"$\beta J=32$, $dt=0.053$, $|\lambda|=0.0025$, $\mu=0$: production scheme on $G$ vs on $\delta G=G-G_{\rm ref}$", fontsize=11)
    fig.savefig(HERE / "deltaG_floor_compare.png", dpi=130, bbox_inches="tight")
    for m in ("G", "dG"):
        print(f"(4) {m}: {M[(m, 1)]['meta']['sec_per_step']:.3f} s/step, corrector iterations median {np.median(M[(m, 1)]['iters'][n0 + 1:]):.0f}")
