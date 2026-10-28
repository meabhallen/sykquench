"""Paths and settings shared by run_test.py and compare.py."""
from pathlib import Path

SIMS = Path("/Users/meabh/Desktop/CC Quench States/SYK Quenches/syk_sims")
OUT = SIMS / "deltaG_test"
K = SIMS / "kbe_runs"
PROD = {+1: K / "kbe_J4i_1_J4f_1_b_32_dt_0p053033008588991064_lam_0_c_0_eqlam_0p0025_eqc_m0p5625_muf_0p0_eqmu_0/syk_kbe_J2_0_to_0_J4_1_to_1_mu_0_to_0_beta_32_dt_0p053033008589_tpre_64_tpost_32_ncorr_50_ctol_1em16_01e7899e.npz",
        -1: K / "kbe_J4i_1_J4f_1_b_32_dt_0p053033008588991064_lam_0_c_0_eqlam_m0p0025_eqc_m0p5625_muf_0p0_eqmu_0/syk_kbe_J2_0_to_0_J4_1_to_1_mu_0_to_0_beta_32_dt_0p053033008589_tpre_64_tpost_32_ncorr_50_ctol_1em16_3cd8bc72.npz"}
EQ0 = SIMS / "eq_runs/eq_J4_1_b_32_mu_0_lam_0_c_m0p5625_dt_0p0064_om_36_Nw_23041/syk_eq_J2_0_J4_1_beta_32_mu_0_klam_0_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_23041_tol_1em12_74d1bf7e.npz"
BETA, DT, T_PRE = 32.0, 0.053033008588991064, 64.0

