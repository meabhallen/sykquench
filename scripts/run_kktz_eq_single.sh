#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/%u/syk/logs/kktz_eq_%j.out
#SBATCH --error=/global/scratch/users/%u/syk/logs/kktz_eq_%j.err
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
WORK_DIR=${WORK_DIR:-/global/scratch/users/$USER/syk}
cd "$WORK_DIR"
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/eq_runs"

safe() { echo "$1" | sed 's/\.$//; s/\./p/g; s/-/m/g'; }

ALPHA=${ALPHA:-0.0}
SB_SEED_MU=${SB_SEED_MU:-0.0}
TAG="kktz_eq_J_$(safe $J)_alpha_$(safe $ALPHA)_mu_$(safe $MU)_sbseed_$(safe $SB_SEED_MU)_b_$(safe $BETA)_dt_$(safe $DT)_om_$(safe $OMEGA_MAX)_Nw_${NW}"
OUT_DIR="eq_runs/$TAG"

MAX_ITER=${MAX_ITER:-2000}
MIXING=${MIXING:-0.05}
ETA_RET=${ETA_RET:-1e-6}
VERBOSE_EVERY=${VERBOSE_EVERY:-25}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-25}
COMPUTE_KBE_DAB_EVERY=${COMPUTE_KBE_DAB_EVERY:-50}
DAB_TOL=${DAB_TOL:-1e-5}
REQUIRE_DAB=${REQUIRE_DAB:-0}
SEED_FROM_MU0=${SEED_FROM_MU0:-1}
PROJECT=${PROJECT:-1}

SEED_FLAG=""
if [ "$SEED_FROM_MU0" = "1" ]; then
    SEED_FLAG="--seed-from-mu0"
fi

PROJECT_FLAG=""
if [ "$PROJECT" = "0" ]; then
    PROJECT_FLAG="--no-project"
fi

TMAX_FLAG=""
if [ -n "${T_MAX:-}" ]; then
    TMAX_FLAG="--t-max $T_MAX"
fi

SEED_TOL_FLAG=""
if [ -n "${SEED_TOL:-}" ]; then
    SEED_TOL_FLAG="--seed-tol $SEED_TOL"
fi

OVERWRITE_FLAG=""
if [ "${OVERWRITE:-0}" = "1" ]; then
    OVERWRITE_FLAG="--overwrite"
fi

DAB_FLAG=""
if [ "$REQUIRE_DAB" = "1" ]; then
    DAB_FLAG="--require-dab-convergence"
fi

echo "Job $SLURM_JOB_ID: KKTZ equilibrium J=$J alpha=$ALPHA mu=$MU sb_seed_mu=$SB_SEED_MU beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL dab_tol=$DAB_TOL require_dab=$REQUIRE_DAB"
echo "Output: $OUT_DIR"

python3 -u "$WORK_DIR/kktz_batch_tools2.py" eq-one \
    --J "$J" \
    --alpha "$ALPHA" \
    --mu "$MU" \
    --beta "$BETA" \
    --dt "$DT" \
    --omega-max "$OMEGA_MAX" \
    --Nw "$NW" \
    --tol "$TOL" \
    --dab-tol "$DAB_TOL" \
    --max-iter "$MAX_ITER" \
    --mixing "$MIXING" \
    --eta-ret "$ETA_RET" \
    --verbose-every "$VERBOSE_EVERY" \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --compute-kbe-dab-every "$COMPUTE_KBE_DAB_EVERY" \
    --sb-seed-mu "$SB_SEED_MU" \
    --out-dir "$OUT_DIR" \
    $TMAX_FLAG $SEED_FLAG $SEED_TOL_FLAG $PROJECT_FLAG $DAB_FLAG $OVERWRITE_FLAG
