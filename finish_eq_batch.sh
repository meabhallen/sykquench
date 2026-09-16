#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0085.lr4,n0112.lr4,n0122.lr4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --job-name=finish_eq_ckpts
#SBATCH --output=/global/scratch/users/%u/sykquench/logs/finish_eq_%j.out
#SBATCH --error=/global/scratch/users/%u/sykquench/logs/finish_eq_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=meabh_allen@berkeley.edu

# Finishes every stuck eq checkpoint listed below via finish_eq_checkpoint.py,
# on a compute node instead of the login node. Each call is independent and
# guarded so one failure (e.g. an already-finished checkpoint that was
# already deleted by a prior successful run) doesn't stop the rest.

module load python/3.11.6-gcc-11.4.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

WORK_DIR=/global/scratch/users/$USER/sykquench
cd "$WORK_DIR"

run() {
    echo ""
    echo "============================================================"
    echo "$@"
    echo "============================================================"
    python3 finish_eq_checkpoint.py "$@" || echo "FAILED (see above) -- continuing with the rest."
}

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_36_mu_0p0025_lam_0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921/syk_eq_J2_0_J4_1_beta_36_mu_0p0025_klam_0p0025_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_25921_tol_1em12_2f2a506c.ckpt.npz \
    --J2 0 --J4 1 --beta 36 --mu 0.0025 \
    --kernel-lambda 0.0025 --kernel-c -0.5625 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 25921 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_36_mu_0p0025_lam_0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921 \
    --manifest-dir eq_runs

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_36_mu_0p0025_lam_m0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921/syk_eq_J2_0_J4_1_beta_36_mu_0p0025_klam_m0p0025_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_25921_tol_1em12_90e286f0.ckpt.npz \
    --J2 0 --J4 1 --beta 36 --mu 0.0025 \
    --kernel-lambda -0.0025 --kernel-c -0.5625 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 25921 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_36_mu_0p0025_lam_m0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921 \
    --manifest-dir eq_runs

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_36_mu_m0p0025_lam_0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921/syk_eq_J2_0_J4_1_beta_36_mu_m0p0025_klam_0p0025_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_25921_tol_1em12_b7ee1b03.ckpt.npz \
    --J2 0 --J4 1 --beta 36 --mu -0.0025 \
    --kernel-lambda 0.0025 --kernel-c -0.5625 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 25921 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_36_mu_m0p0025_lam_0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921 \
    --manifest-dir eq_runs

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_36_mu_m0p0025_lam_m0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921/syk_eq_J2_0_J4_1_beta_36_mu_m0p0025_klam_m0p0025_kc_m0p5625_kcut_0p75_dt_0p0064_om_36_Nw_25921_tol_1em12_a3be951f.ckpt.npz \
    --J2 0 --J4 1 --beta 36 --mu -0.0025 \
    --kernel-lambda -0.0025 --kernel-c -0.5625 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 25921 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_36_mu_m0p0025_lam_m0p0025_c_m0p5625_dt_0p0064_om_36_Nw_25921 \
    --manifest-dir eq_runs

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_48_lam_0p005_c_m0p053648_dt_0p0064_om_36_Nw_34561/syk_eq_J2_0_J4_1_beta_48_mu_0_klam_0p005_kc_m0p053648_kcut_0p75_dt_0p0064_om_36_Nw_34561_tol_1em12_7cf5bb20.ckpt.npz \
    --J2 0 --J4 1 --beta 48 --mu 0 \
    --kernel-lambda 0.005 --kernel-c -0.053648 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 34561 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_48_lam_0p005_c_m0p053648_dt_0p0064_om_36_Nw_34561 \
    --manifest-dir eq_runs

run \
    --ckpt /global/scratch/users/meabhallen/sykquench/eq_runs/eq_J4_1_b_48_lam_m0p005_c_m0p053648_dt_0p0064_om_36_Nw_34561/syk_eq_J2_0_J4_1_beta_48_mu_0_klam_m0p005_kc_m0p053648_kcut_0p75_dt_0p0064_om_36_Nw_34561_tol_1em12_b74e028d.ckpt.npz \
    --J2 0 --J4 1 --beta 48 --mu 0 \
    --kernel-lambda -0.005 --kernel-c -0.053648 --kernel-cutoff 0.75 \
    --dt 0.0064 --omega-max 36 --Nw 34561 --tol 1e-12 \
    --out-dir eq_runs/eq_J4_1_b_48_lam_m0p005_c_m0p053648_dt_0p0064_om_36_Nw_34561 \
    --manifest-dir eq_runs

echo ""
echo "Done. Check the manifest:"
echo "  tail -n 10 eq_runs/syk_eq_manifest.csv"
