#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0085.lr4,n0112.lr4,n0122.lr4
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=/global/scratch/users/%u/sykquench/logs/kbe_%j.out
#SBATCH --error=/global/scratch/users/%u/sykquench/logs/kbe_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=meabh_allen@berkeley.edu
#SBATCH --requeue
#SBATCH --signal=B:USR1@300
#SBATCH --open-mode=append

# ── Environment ───────────────────────────────────────────────────────────────
module load python/3.11.6-gcc-11.4.0

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export VECLIB_MAXIMUM_THREADS=${SLURM_CPUS_PER_TASK:-1}
export NUMEXPR_NUM_THREADS=${SLURM_CPUS_PER_TASK:-1}

# ── Paths ─────────────────────────────────────────────────────────────────────
WORK_DIR=/global/scratch/users/$USER/sykquench
cd "$WORK_DIR"

# ── Output tag ────────────────────────────────────────────────────────────────
safe() { echo "$1" | sed 's/\.$//; s/\./p/g; s/-/m/g'; }

KERNEL_LAMBDA=${KERNEL_LAMBDA:-0.0}
KERNEL_C=${KERNEL_C:-0.0}
EQ_KERNEL_LAMBDA=${EQ_KERNEL_LAMBDA:-0.0}
EQ_KERNEL_C=${EQ_KERNEL_C:-0.0}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-50}

TAG="kbe_J4i_$(safe $J4_I)_J4f_$(safe $J4_F)_b_$(safe $BETA)_dt_$(safe $DT)_lam_$(safe $KERNEL_LAMBDA)_c_$(safe $KERNEL_C)_eqlam_$(safe $EQ_KERNEL_LAMBDA)_eqc_$(safe $EQ_KERNEL_C)"
OUT_DIR="kbe_runs/$TAG"

KERNEL_CUTOFF_FLAG=""
if [ -n "${KERNEL_CUTOFF:-}" ]; then
    KERNEL_CUTOFF_FLAG="--kernel-cutoff $KERNEL_CUTOFF"
fi

EQ_KERNEL_CUTOFF_FLAG=""
if [ -n "${EQ_KERNEL_CUTOFF:-}" ]; then
    EQ_KERNEL_CUTOFF_FLAG="--eq-kernel-cutoff $EQ_KERNEL_CUTOFF"
fi

echo "Job $SLURM_JOB_ID: J4_i=$J4_I J4_f=$J4_F beta=$BETA dt=$DT"
echo "t_pre_factor=$T_PRE_FACTOR t_post_factor=$T_POST_FACTOR"
echo "evolution kernel: lambda=$KERNEL_LAMBDA c=$KERNEL_C cutoff=${KERNEL_CUTOFF:-auto}"
echo "eq-selection kernel: lambda=$EQ_KERNEL_LAMBDA c=$EQ_KERNEL_C cutoff=${EQ_KERNEL_CUTOFF:-auto}"
echo "Output: $OUT_DIR"

# ── Signal handling / requeue ─────────────────────────────────────────────────
# Slurm sends SIGUSR1 to this batch shell five minutes before the time limit.
# Forward it to Python, which atomically saves the latest completed time step,
# then requeue the same job so it restarts from that checkpoint.
PY_PID=""

checkpoint_and_requeue() {
    trap - USR1 TERM
    echo "Received SIGUSR1; requesting a KBE checkpoint before requeue."

    if [[ -n "$PY_PID" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        kill -USR1 "$PY_PID"
        wait "$PY_PID" || true
    fi

    echo "Requeueing Slurm job $SLURM_JOB_ID."
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
}

checkpoint_and_exit() {
    trap - USR1 TERM
    echo "Received SIGTERM; requesting a KBE checkpoint before exit."

    if [[ -n "$PY_PID" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        kill -TERM "$PY_PID"
        wait "$PY_PID" || true
    fi

    exit 143
}

trap checkpoint_and_requeue USR1
trap checkpoint_and_exit TERM

# ── Run ───────────────────────────────────────────────────────────────────────
python3 -u "$WORK_DIR/syk_batch_tools.py" kbe-one \
    --J4-i          "$J4_I"           \
    --J4-f          "$J4_F"           \
    --beta          "$BETA"           \
    --dt            "$DT"             \
    --t-pre-factor  "$T_PRE_FACTOR"   \
    --t-post-factor "$T_POST_FACTOR"  \
    --corr-tol      "$CORR_TOL"       \
    --n-corr        "$N_CORR"         \
    --checkpoint-every "$CHECKPOINT_EVERY" \
    --kernel-lambda "$KERNEL_LAMBDA"  \
    --kernel-c      "$KERNEL_C"       \
    --eq-kernel-lambda "$EQ_KERNEL_LAMBDA" \
    --eq-kernel-c   "$EQ_KERNEL_C"    \
    --eq-dir        "$WORK_DIR/eq_runs"  \
    --out-dir       "$OUT_DIR"        \
    $KERNEL_CUTOFF_FLAG $EQ_KERNEL_CUTOFF_FLAG &

PY_PID=$!
wait "$PY_PID"
STATUS=$?
exit "$STATUS"
