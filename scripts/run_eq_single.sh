#!/bin/bash
#SBATCH --account=ac_onsager
#SBATCH --partition=lr4
#SBATCH --qos=lr_normal
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --exclude=n0085.lr4,n0112.lr4,n0122.lr4
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/global/scratch/users/%u/syk/logs/eq_%j.out
#SBATCH --error=/global/scratch/users/%u/syk/logs/eq_%j.err
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
WORK_DIR=/global/scratch/users/$USER/syk
cd "$WORK_DIR"

# ── Output tag ────────────────────────────────────────────────────────────────
safe() { echo "$1" | sed 's/\.$//; s/\./p/g; s/-/m/g'; }

TAG="eq_J4_$(safe $J4)_b_$(safe $BETA)_dt_$(safe $DT)_om_$(safe $OMEGA_MAX)_Nw_${NW}"
OUT_DIR="eq_runs/$TAG"

echo "Job $SLURM_JOB_ID: J4=$J4 beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL"
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
python3 -u "$WORK_DIR/syk_batch_tools2.py" eq-one \
    --J4        "$J4"        \
    --beta      "$BETA"      \
    --dt        "$DT"        \
    --omega-max "$OMEGA_MAX" \
    --Nw        "$NW"        \
    --tol       "$TOL"       \
    --out-dir   "$OUT_DIR" &

PY_PID=$!
wait "$PY_PID"
STATUS=$?
exit "$STATUS"
