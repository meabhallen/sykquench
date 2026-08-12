#!/bin/bash
# run_local_kbe.sh — run one kbe-one job directly.
# Requires a matching eq-one run already present in local_runs/eq_runs
# (see run_local_eq.sh).
#
set -e

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON=${PYTHON:-"$CONDA_PREFIX/bin/python3"}
else
    PYTHON=${PYTHON:-python3}
fi

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

J4S=(1 2 4)
BETAS=(36 48 60 72)             

# KBE evolution
KBE_DT=0.05                     # time step size
T_PRE_FACTOR=2.0                # time grid in units of beta allocated to equilibrium initial condition
T_POST_FACTOR=2.0               # time grid in units of beta taken up by pure nonequilibrium dynamics
CORR_TOL=1e-14                  # precision required at each KBE step
N_CORR=50                       # max number of solver iterations per KBE step (unless precision reached already)
PROGRESS_EVERY=1000             # checkpoint every PROGRESS_EVERY time steps

# Evolution kernel: quench dynamics run kernel-free (matches submit_kbe.sh).
KERNEL_LAMBDA=0
KERNEL_C=0
KERNEL_CUTOFF_FACTOR=1.0    # factor of J4; inactive when KERNEL_LAMBDA=0

# Initial-state equilibrium kernel: the tuned delta_star(Lambda) used to
# prepare the pre-quench state. Pick from tuned pairs:
#   (delta*, Lambda) = (c, cutoff_factor) in {(-0.043936, 0.65), (-0.053648, 0.75)}
# Both signs of lambda are run below (matches submit_kbe.sh): downstream
# real-time analysis reads off the odd-in-lambda response
# D_lambda = (G_+ - G_-)/(2*lambda), which needs a matched +/-lambda pair.
EQ_KERNEL_LAMBDA_MAG=0.005
EQ_KERNEL_C=-0.053648
EQ_KERNEL_CUTOFF_FACTOR=0.75 # factor of J4

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
#WORK_DIR="$SCRIPT_DIR/../local_runs"
WORK_DIR="$SCRIPT_DIR/../syk_sims"
mkdir -p "$WORK_DIR/../local_runs/kbe_runs"


if [ ! -f "$WORK_DIR/eq_runs/syk_eq_manifest.csv" ]; then
    echo "No equilibrium runs found in $WORK_DIR/eq_runs." >&2
    echo "Run run_local_eq.sh first (with matching J4/beta/kernel params)." >&2
    exit 1
fi

KERNEL_CUTOFF=$("$PYTHON" -c "print($KERNEL_CUTOFF_FACTOR * $J4)")
EQ_KERNEL_CUTOFF=$("$PYTHON" -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4)")

for J4 in "${J4S[@]}"; do
for BETA in "${BETAS[@]}"; do
for EQ_KERNEL_LAMBDA in "$EQ_KERNEL_LAMBDA_MAG" "-$EQ_KERNEL_LAMBDA_MAG"; do
    echo "============================================================"
    echo "KBE quench (kernel-free evolution, tuned initial state)"
    echo "  J4_i=J4_f=$J4 beta=$BETA dt=$KBE_DT"
    echo "  evolution kernel: lambda=$KERNEL_LAMBDA c=$KERNEL_C cutoff=$KERNEL_CUTOFF"
    echo "  eq-selection kernel: lambda=$EQ_KERNEL_LAMBDA c=$EQ_KERNEL_C cutoff=$EQ_KERNEL_CUTOFF"
    echo "============================================================"

    BRANCH_START=$(date +%s)
    "$PYTHON" -u syk_batch_tools.py kbe-one \
        --J4-i          "$J4"            \
        --J4-f          "$J4"            \
        --beta          "$BETA"          \
        --dt            "$KBE_DT"        \
        --t-pre-factor  "$T_PRE_FACTOR"  \
        --t-post-factor "$T_POST_FACTOR" \
        --corr-tol      "$CORR_TOL"      \
        --n-corr        "$N_CORR"        \
        --progress-every "$PROGRESS_EVERY" \
        --kernel-lambda "$KERNEL_LAMBDA" \
        --kernel-c      "$KERNEL_C"      \
        --kernel-cutoff "$KERNEL_CUTOFF" \
        --eq-kernel-lambda "$EQ_KERNEL_LAMBDA" \
        --eq-kernel-c   "$EQ_KERNEL_C"   \
        --eq-kernel-cutoff "$EQ_KERNEL_CUTOFF" \
        --eq-dir        "$WORK_DIR/eq_runs" \
        --out-dir       "$WORK_DIR/kbe_runs"

    BRANCH_ELAPSED=$(( $(date +%s) - BRANCH_START ))
    printf "  [branch elapsed: %02d:%02d:%02d]\n" $((BRANCH_ELAPSED/3600)) $((BRANCH_ELAPSED/60%60)) $((BRANCH_ELAPSED%60))
    echo ""
done
done
done

echo "Done. Output in: $WORK_DIR/kbe_runs"
printf "Total elapsed: %02d:%02d:%02d\n" $((SECONDS/3600)) $((SECONDS/60%60)) $((SECONDS%60))
