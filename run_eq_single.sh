#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0085.lr4,n0112.lr4,n0122.lr4
#SBATCH --mem=16G
#SBATCH --time=16:00:00
#SBATCH --output=/global/scratch/users/%u/sykquench/logs/eq_%j.out
#SBATCH --error=/global/scratch/users/%u/sykquench/logs/eq_%j.err
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

EQ_KERNEL_LAMBDA=${EQ_KERNEL_LAMBDA:-0.0}
EQ_KERNEL_C=${EQ_KERNEL_C:-0.0}
DAB_TOL=${DAB_TOL:-1e-6}
REQUIRE_DAB=${REQUIRE_DAB:-0}
VERBOSE_EVERY=${VERBOSE_EVERY:-50}

TAG="eq_J4_$(safe $J4)_b_$(safe $BETA)_lam_$(safe $EQ_KERNEL_LAMBDA)_c_$(safe $EQ_KERNEL_C)_dt_$(safe $DT)_om_$(safe $OMEGA_MAX)_Nw_${NW}"
OUT_DIR="eq_runs/$TAG"

KERNEL_CUTOFF_FLAG=""
if [ -n "${EQ_KERNEL_CUTOFF:-}" ]; then
    KERNEL_CUTOFF_FLAG="--kernel-cutoff $EQ_KERNEL_CUTOFF"
fi

DAB_FLAG=""
if [ "$REQUIRE_DAB" = "1" ]; then
    DAB_FLAG="--require-dab-convergence"
fi

echo "Job $SLURM_JOB_ID: J4=$J4 beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL"
echo "  kernel: lambda=$EQ_KERNEL_LAMBDA c=$EQ_KERNEL_C cutoff=${EQ_KERNEL_CUTOFF:-auto}"
echo "  dab_tol=$DAB_TOL require_dab=$REQUIRE_DAB"
echo "Output: $OUT_DIR"

# ── Signal handling / requeue ─────────────────────────────────────────────────
# Slurm sends SIGUSR1 to this batch shell five minutes before the time limit.
# Forward it to Python, which atomically saves the latest completed iteration,
# then requeue the same job so it restarts from that checkpoint.
PY_PID=""

checkpoint_and_requeue() {
    trap - USR1 TERM
    echo "Received SIGUSR1; requesting an equilibrium checkpoint before requeue."

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
    echo "Received SIGTERM; requesting an equilibrium checkpoint before exit."

    if [[ -n "$PY_PID" ]] && kill -0 "$PY_PID" 2>/dev/null; then
        kill -TERM "$PY_PID"
        wait "$PY_PID" || true
    fi

    exit 143
}

trap checkpoint_and_requeue USR1
trap checkpoint_and_exit TERM

# ── Run ───────────────────────────────────────────────────────────────────────
python3 -u "$WORK_DIR/syk_batch_tools.py" eq-one \
    --J4            "$J4"           \
    --beta          "$BETA"         \
    --dt            "$DT"           \
    --omega-max     "$OMEGA_MAX"    \
    --Nw            "$NW"           \
    --tol           "$TOL"          \
    --dab-tol       "$DAB_TOL"      \
    --kernel-lambda "$EQ_KERNEL_LAMBDA"\
    --kernel-c      "$EQ_KERNEL_C"  \
    --verbose-every "$VERBOSE_EVERY"\
    --out-dir       "$OUT_DIR"      \
    --manifest-dir  "eq_runs"       \
    $KERNEL_CUTOFF_FLAG $DAB_FLAG &

PY_PID=$!
wait "$PY_PID"
STATUS=$?
exit "$STATUS"
