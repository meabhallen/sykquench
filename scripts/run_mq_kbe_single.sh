#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/%u/syk/logs/mq_kbe_%j.out
#SBATCH --error=/global/scratch/users/%u/syk/logs/mq_kbe_%j.err
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
mkdir -p "$WORK_DIR/logs" "$WORK_DIR/kbe_runs"

safe() { echo "$1" | sed 's/\.$//; s/\./p/g; s/-/m/g'; }

TAG="mq_kbe_J_$(safe $J_I)_to_$(safe ${J_F:-$J_I})_mu_$(safe $MU_I)_to_$(safe $MU_F)_b_$(safe $BETA)_dt_$(safe $DT)"
OUT_DIR="kbe_runs/$TAG"

J_F=${J_F:-$J_I}
T_PRE_FACTOR=${T_PRE_FACTOR:-2.0}
T_POST_FACTOR=${T_POST_FACTOR:-1.0}
CORR_TOL=${CORR_TOL:-1e-10}
N_CORR=${N_CORR:-6}
ITERATIONS=${ITERATIONS:-50}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-25}
EQ_DIR=${EQ_DIR:-$WORK_DIR/eq_runs}

echo "Job $SLURM_JOB_ID: MQ KBE J_i=$J_I J_f=$J_F mu_i=$MU_I mu_f=$MU_F beta=$BETA dt=$DT"
echo "t_pre_factor=$T_PRE_FACTOR t_post_factor=$T_POST_FACTOR"
echo "Output: $OUT_DIR"

CMD=(python3 -u "$WORK_DIR/mq_batch_tools2.py" kbe-one
    --J-i "$J_I"
    --J-f "$J_F"
    --mu-i "$MU_I"
    --mu-f "$MU_F"
    --beta "$BETA"
    --dt "$DT"
    --t-pre-factor "$T_PRE_FACTOR"
    --t-post-factor "$T_POST_FACTOR"
    --corr-tol "$CORR_TOL"
    --n-corr "$N_CORR"
    --iterations "$ITERATIONS"
    --checkpoint-every "$CHECKPOINT_EVERY"
    --eq-dir "$EQ_DIR"
    --out-dir "$OUT_DIR")

if [ -n "${EQ_DT:-}" ]; then CMD+=(--eq-dt "$EQ_DT"); fi
if [ -n "${EQ_OMEGA_MAX:-}" ]; then CMD+=(--eq-omega-max "$EQ_OMEGA_MAX"); fi
if [ -n "${EQ_NW:-}" ]; then CMD+=(--eq-Nw "$EQ_NW"); fi
if [ -n "${EQ_FILE:-}" ]; then CMD+=(--eq-file "$EQ_FILE"); fi
if [ "${UNCOMPRESSED:-0}" = "1" ]; then CMD+=(--uncompressed); fi
if [ "${DROP_CROSS_SELF_ENERGY:-0}" = "1" ]; then CMD+=(--drop-cross-self-energy); fi
if [ "${OVERWRITE:-0}" = "1" ]; then CMD+=(--overwrite); fi

"${CMD[@]}"
