# δG formulation of the SYK4 KBE stepper (prototype)

Goal: lower the round-off floor that ends the late-time Γ_G curves (about (t−τ/2)/β_eff ≈ 0.65–0.7 at the production settings).
The idea is to evolve the deviation δG = G − G_ref from a fixed reference, so that floating-point rounding is relative to |δG|
(about 1e-4 |G| at |λ| = 0.0025) rather than |G|.

The prototype is μ = 0 only, with no kernel after the quench. Nothing in `syk_batch_tools.py` was changed.

## Method

- **Reference.** G_ref(t₁,t₂) = G₀(t₁−t₂) is the undeformed (λ = 0) equilibrium at the same β, a Toeplitz matrix on the whole
  two-time grid. It is summed from the λ = 0 eq file's A(ω) with the Riemann sum `greater_from_spectral` uses. Its diagonal is set
  to −i/2, as in production.
- **The exact rewrite.** The production scheme is G[n,j] = G[n−1,j] + dt/2 (F[n,j] + F[n−1,j]): an Euler predictor, then the AM1
  corrector iterated to corr_tol, with F from trapezoidal memory sums. Writing G = G_ref + δG turns it exactly into

  δG[n,j] = δG[n−1,j] + dt/2 (δF[n,j] + δF[n−1,j]) + R[n,j],

  where δF = F[G_ref+δG] − F[G_ref] and R = dt/2 (F_ref[n] + F_ref[n−1]) − (G_ref[n] − G_ref[n−1]).
- **No O(1) cancellation in δF.** δF is evaluated with the product rule, XY − X_rY_r = X_r δY + δX Y_full, and with
  δΣ = −J²δG(3G_r² + 3G_r δG + δG²). No O(1) quantities are ever subtracted.
- **R is λ-independent.** It is the discrete residual of the undeformed equilibrium: the dt error plus the memory cut at t₀ = −t_pre,
  i.e. the source of the λ⁰ artefact ε. It is computed once per grid, bitwise identical for every λ, so it cancels exactly in
  G(+λ) − G(−λ).
- **The increment in R is `np.diff(g0)` of the stored reference, not a separately summed spectral increment.** The decomposition is
  exact only if R uses the increments of the array that F_ref is built from. Subtracting two neighbouring doubles is exact
  (Sterbenz), so the difference is free of rounding.
  - The spectral increment (factor e^{−iω dt} − 1) is closer to the continuum value, but it differs from the array's increment by up
    to 9e-16. That comes from independent phase rounding of the two sums, |ωs| up to ~3500.
  - Used in R, that difference would be a spurious source of up to 9e-16 per step.
  - `reference(..., spectral_increment=True)` keeps the spectral version for checks only.
- **Pre-quench block.** It is built exactly as `evolve_syk4_kbe` builds it: cubic interpolation of `greater_from_spectral` on
  4Nt + 1 points spanning ±(t_pre + t_post). For δG the λ-difference is interpolated directly from A_λ − A₀ (the interpolation is
  linear in the data), plus the one-time interpolation error of the λ = 0 block.
- **Vectorised stepper.** The memory sums are matrix–vector products. Matrices that do not change within a step are built once per
  step, and Σ is updated only on row and column n. `mode="G"` runs the same code on G itself (G_ref = 0, R = 0): the production
  scheme with a different summation order.

## Files

- `dg_solver.py`: `reference`, `pre_quench_block`, `Stepper`, `residual` and `evolve`.
- `common.py`: paths. The test compares against the production βJ = 32, dt = 0.053033, λ = ±0.0025, μ = 0 runs, with
  t_pre = 2β and t_post = β.
- `run_test.py [t_post] [runs]`: the runs `dG+`, `dG-`, `dG0`, `G+` and `G-`. Outputs go to `syk_sims/deltaG_test/`, outside
  `kbe_runs`, so the analysis index never sees them.
- `compare.py [t_post]`: validation numbers and `deltaG_floor_compare.png`.
- `floor_test.py`: the s = 0.01 floor test (see below) and `deltaG_floor_test_s0p01.png`.
- `time_prod.py`: seconds per step of the production `evolve_syk4_kbe` on the short grid.

## Test 1: production settings (βJ = 32, dt = 0.053, |λ| = 0.0025, t_post = β; Nt = 1811, 603 post-quench steps)

| quantity | plain G (vectorised) | δG |
|---|---|---|
| s/step (laptop) | 0.53–0.55 | 0.68–0.69 |
| one-off residual R | none | 234 s, max \|R\| = 3.8e-6, median 4.9e-9 over post-quench rows |
| corrector tolerance | 1e-16 | 1e-19 |
| corrector iterations (median) | 9 | 9–10 |
| final corrector change | median 5.6e-17, max 8.9e-17 | median 3–5e-20, max 2–4e-19 |
| steps hitting the 50-iteration cap | 0 | 35–122 of 603 (from step ~130–240), stalling at 1–2e-19 |
| max \|G − G_prod\|, first 40 rows | 6e-16 to 2e-15 | 1.7e-15 to 2.7e-15 |
| max \|O − O_prod\|, paired odd part (\|O\| ≈ 2e-3) | 7.7e-16 | 1.2e-15 |
| max \|G − G_prod\| at later rows (n−n₀ = 100, 300, last) | ~5e-16 | ~2–3e-15 |

- **Correctness.** Both versions agree with the cluster runs at round-off level throughout. With the same t_post they also agree
  with each other to 3e-16. With a different t_post the pre-quench interpolation grid differs, and so does G, by ~1e-11.
- **The δG corrector reaches its own arithmetic floor at 1–2e-19**, against 6–9e-17 for G: about 2.5 decades lower. The 1e-19
  tolerance sits just below that floor, so some steps run to the cap. A tolerance of about 3e-19 × (|δG| scale), or a stagnation
  stop, avoids this; see test 2.
- **Cost per step is about 1.3× the vectorised G stepper**, plus the one-off R, which costs about 0.4 of a full run and is shared by
  every λ at that grid. Most of the extra comes from the capped steps.
- **No floor gain is visible in test 1.** At these settings the paired signal is still decaying at the end of the data
  (T/β ≈ 0.9), so production, G and δG coincide to the end: the floor is never reached. Test 1 validates correctness only.
- **The sole correction to the original design: the R increment.** It is `np.diff` of the stored reference, not a spectral sum
  (see Method).

## Test 2: floor test (scaled quench, s = 0.01)

Same grid, residual and run length as test 1, with the pre-quench deviation scaled:

- G_pre = G_ref + s (G_eq,λ − G_ref) with s = 0.01, i.e. effective λ = ±2.5e-5 at linear order.
- The same initial state for both schemes: they agree to 3e-17.
- D is normalised by s|λ|, so the paired signal is unchanged while the rounding, relative to it, is 100× larger.
- Corrector tolerance: 1e-16 for G; 3e-21 (= 3e-19 × s) for δG, with a stagnation stop (stop once the change no longer falls by
  30%). No δG step hit the cap: 10–12 iterations, final change median 3–7e-21.

| | G, s = 0.01 | δG, s = 0.01 | δG, s = 1 (reference signal) |
|---|---|---|---|
| decay of \|∂D_odd\| flattens at | T/β = 0.77 | not within the data (T/β ≤ 0.89) | not within the data |
| \|∂D_odd/dx\| over the last 0.04, point-to-point / 21-pt mean | 2.0e-9 / 1.4e-9 (floor) | 1.5e-10 / 1.5e-10 (signal) | 1.6e-10 / 1.7e-10 (signal) |
| 21-pt \|∂D\| departs > 30% from the s = 1 signal at | T/β = 0.76 | never | n/a |
| agreement with the s = 1 signal over T/β ∈ [0.3, 0.6] | 0.4% | 0.4% | n/a |
| point-to-point noise of D_odd, T/β ≥ 0.75 (third-difference MAD, as \|∂D/dx\|) | 7.2e-9 | 1.1e-11 | ≤ 4.6e-13 (signal curvature) |
| same in absolute \|∂G\| units | 3.6e-14 | 5.4e-17 | ≤ 2.3e-16 |
| E: h_loc of \|D̃−C̃\| clean to (cut_C) | (t−τ/2)/β_eff = 0.58 | ≥ 0.70 (end of data) | 0.67 |
| s/step | 0.53 | 0.55 | 0.69 |

- **Plain G.** Its noise is the same in absolute terms at s = 1 and s = 0.01 (4.2e-14 and 3.6e-14 in |∂G|), as expected for
  rounding relative to |G|.
- **δG.** Its noise at s = 0.01 is 5.4e-17, i.e. **2.8 decades lower**. At production λ the δG value is no larger than the
  signal's own curvature, which bounds the gain at ≥ 2.3 decades.
- **What that buys in time.** At h₁ ≈ 3.8 one decade is Δx ≈ 0.1, so the floor moves out by about 0.25 in T/β. At s = 0.01 the
  plain-G flattening at 0.77 would move to about 1.0, beyond the data.
- **Linear response checks out.** The δG s = 0.01 response tracks the production-λ response to 0.4% over T/β 0.3–0.6. The
  remaining differences in E at the transient (0.35–0.5) are the genuine O(λ) effects at |λ| = 0.0025.
- Figure: `deltaG_floor_test_s0p01.png`.

## Cost

On the short grid (Nt = 1246, the first 38 post-quench steps, same laptop):

| stepper | s/step |
|---|---|
| production `evolve_syk4_kbe` (`time_prod.py`) | 3.2 |
| vectorised G | 0.32–0.37 |
| δG | 0.31–0.40 |

- **The vectorised memory sums alone are about 9× faster than production.** Production loops over j in Python and recomputes Σ on
  the full Nt × Nt grid every corrector iteration.
- **δG costs 1.0–1.3× the vectorised G stepper**, plus the one-off R: 234 s at Nt = 1811, about 0.4 of one run, shared by all λ on
  that grid.

## Porting into syk_batch_tools.py

1. **Vectorise the existing stepper (worth doing on its own, about 9× faster, no physics change):**
   - replace the per-j `rhs_t1`/`rhs_t2` loops with the matrix–vector forms of `Stepper.fixed/old/new`;
   - update Σ only on row and column n.
   - Checkpointing is unaffected, since it stores only G.
2. **Add a δG mode:**
   - add `--delta-g --ref-eq-file <λ = 0 eq file>`;
   - build G_ref and R before the loop, caching R per (β, dt, t_pre, t_post) next to the eq file;
   - store and checkpoint δG instead of G;
   - save `Ggt_post` as δG plus `g0`, or G_ref + δG for compatibility, which costs back the gain in the saved file.
   The analysis then needs to read δG directly (`syk_cache.cut` on δG): the pair and E use δG₊ − δG₋, and single runs use δG.
3. **Use a corrector tolerance of about 3e-19 × (|δG|/1e-4) with the stagnation stop.**
4. **μ ≠ 0 (`evolve_syk4_kbe_massdef`) needs the same rewrite for Goff.**

## Caveats

- μ = 0 and no evolution kernel only. The mass-deformed evolver (μ ≠ 0) would need the same rewrite for Goff.
- Checkpointing is not implemented in the prototype.
- R needs the λ = 0 eq file on the same ω grid as the runs, and one extra pass per (β, dt, t_pre, t_post) grid. It is also a full
  Nt × Nt array: 300 MB at Nt = 4344.
- The gain applies to the deviation from the undeformed equilibrium. For a single run the λ⁰ artefact ε is still present: R is the
  same discretisation error as in production, only free of extra rounding. It cancels in pairs.
