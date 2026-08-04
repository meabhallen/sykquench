#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0085.lr4,n0112.lr4,n0122.lr4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/%u/syk/logs/kbe_%j.out
#SBATCH --error=/global/scratch/users/%u/syk/logs/kbe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=meabh_allen@berkeley.edu
#SBATCH --requeue

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.6-gcc-11.4.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR=/global/scratch/users/$USER/syk
cd "$WORK_DIR"

# ── Output tag ────────────────────────────────────────────────────────────────
safe() { echo "$1" | sed 's/\.$//; s/\./p/g; s/-/m/g'; }

TAG="kbe_J4i_$(safe $J4_I)_J4f_$(safe $J4_F)_b_$(safe $BETA)_dt_$(safe $DT)"
OUT_DIR="kbe_runs/$TAG"

echo "Job $SLURM_JOB_ID: J4_i=$J4_I J4_f=$J4_F beta=$BETA dt=$DT"
echo "t_pre_factor=$T_PRE_FACTOR t_post_factor=$T_POST_FACTOR"
echo "Output: $OUT_DIR"

# ── Run ───────────────────────────────────────────────────────────────────────
python3 -u "$WORK_DIR/syk_batch_tools2.py" kbe-one \
    --J4-i          "$J4_I"           \
    --J4-f          "$J4_F"           \
    --beta          "$BETA"           \
    --dt            "$DT"             \
    --t-pre-factor  "$T_PRE_FACTOR"   \
    --t-post-factor "$T_POST_FACTOR"  \
    --corr-tol      "$CORR_TOL"       \
    --n-corr        "$N_CORR"         \
    --iterations    "$ITERATIONS"     \
    --eq-dir        "$WORK_DIR/eq_runs"  \
    --out-dir       "$OUT_DIR"
