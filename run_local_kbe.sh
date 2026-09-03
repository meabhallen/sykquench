#!/bin/bash
# run_local_kbe.sh — run one kbe-one job directly.
# Requires a matching eq-one run already present in local_runs/eq_runs
# (see run_local_eq.sh).
#
set -e
set -o pipefail

if [ -n "$CONDA_PREFIX" ]; then
    PYTHON=${PYTHON:-"$CONDA_PREFIX/bin/python3"}
else
    PYTHON=${PYTHON:-python3}
fi

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────
# PARAM_MODE selects how J4/beta combinations are formed:
#   "allJ4beta" — every J4 in J4S crossed with every BETA in BETAS
#   "pairs"     —  (J4, BETA) pairs listed in J4_BETA_PAIRS below
PARAM_MODE="allJ4beta"

# when PARAM_MODE=pairs 
J4_BETA_PAIRS=(
    "1 60"
    "1 72"
    "2 6"
    "2 12"
    "2 18"
    "2 24"
    "2 30"
    "2 36"
    "3 6"
    "3 12"
    "3 18"
    "3 24"
)

# when PARAM_MODE=allJ4beta
J4S=(1) # 2 3)
BETAS=(48 60 72)
          

# KBE evolution
KBE_DTS=(0.025) # 0.025)            # time step size
T_PRE_FACTOR=2.0                # time grid in units of beta allocated to equilibrium initial condition
T_POST_FACTOR=1.0               # time grid in units of beta taken up by pure nonequilibrium dynamics
CORR_TOL_FACTOR=1e-12           # precision required at each KBE step = CORR_TOL_FACTOR/BETA
N_CORR=50                       # max number of solver iterations per KBE step (unless precision reached already)
CHECKPOINT_EVERY=500            

# Evolution kernel: quench dynamics run kernel-free.
KERNEL_LAMBDA=0
KERNEL_C=0
KERNEL_CUTOFF_FACTOR=1.0    # factor of J4; inactive when KERNEL_LAMBDA=0

# Initial-state equilibrium kernel using tuned delta_star(Lambda). Pick from tuned pairs:
# (delta*, Lambda) = (c, cutoff_factor) in {(-0.043936, 0.65), (-0.053648, 0.75)}
# Both signs of lambda are run below; real-time analysis reads off D_lambda = (G_+ - G_-)/(2*lambda)
EQ_KERNEL_LAMBDA_MAG=0.0025
EQ_KERNEL_C=-0.5625
EQ_KERNEL_CUTOFF_FACTOR=0.75 # factor of J4

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORK_DIR="$SCRIPT_DIR/../local_runs"
#WORK_DIR="$SCRIPT_DIR/../syk_sims"
mkdir -p "$WORK_DIR/kbe_runs" "$WORK_DIR/logs"


if [ ! -f "$WORK_DIR/eq_runs/syk_eq_manifest.csv" ]; then
    echo "No equilibrium runs found in $WORK_DIR/eq_runs." >&2
    echo "Run run_local_eq.sh first (with matching J4/beta/kernel params)." >&2
    exit 1
fi

# Build the flat list of "J4 BETA" pairs to run, according to PARAM_MODE.
PARAM_PAIRS=()
case "$PARAM_MODE" in
    allJ4beta)
        for J4 in "${J4S[@]}"; do
        for BETA in "${BETAS[@]}"; do
            PARAM_PAIRS+=("$J4 $BETA")
        done
        done
        ;;
    pairs)
        PARAM_PAIRS=("${J4_BETA_PAIRS[@]}")
        ;;
    *)
        echo "Unknown PARAM_MODE '$PARAM_MODE' (expected 'allJ4beta' or 'pairs')." >&2
        exit 1
        ;;
esac


for PAIR in "${PARAM_PAIRS[@]}"; do
    read -r J4 BETA <<< "$PAIR"
    KERNEL_CUTOFF=$("$PYTHON" -c "print($KERNEL_CUTOFF_FACTOR * $J4)")
    EQ_KERNEL_CUTOFF=$("$PYTHON" -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4)")


for KBE_DT in "${KBE_DTS[@]}"; do
    CORR_TOL=$("$PYTHON" -c "print($CORR_TOL_FACTOR / $BETA / $BETA * $KBE_DT / 20 * 12 * 36)")

for EQ_KERNEL_LAMBDA in "$EQ_KERNEL_LAMBDA_MAG" "-$EQ_KERNEL_LAMBDA_MAG"; do
    LOG_FILE="$WORK_DIR/logs/syk_kbe_J${J4}_beta${BETA}_dt${KBE_DT}_lam${EQ_KERNEL_LAMBDA}.log"
    {
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
        --checkpoint-every "$CHECKPOINT_EVERY" \
        --kernel-lambda "$KERNEL_LAMBDA" \
        --kernel-c      "$KERNEL_C"      \
        --kernel-cutoff "$KERNEL_CUTOFF" \
        --eq-kernel-lambda "$EQ_KERNEL_LAMBDA" \
        --eq-kernel-c   "$EQ_KERNEL_C"   \
        --eq-kernel-cutoff "$EQ_KERNEL_CUTOFF" \
        --eq-dir        "$WORK_DIR/eq_runs" \
        --out-dir       "$WORK_DIR/kbe_runs"

    BRANCH_ELAPSED=$(( $(date +%s) - BRANCH_START ))
    #printf "  [branch elapsed: %02d:%02d:%02d]\n" $((BRANCH_ELAPSED/3600)) $((BRANCH_ELAPSED/60%60)) $((BRANCH_ELAPSED%60))
    echo ""
    } 2>&1 | tee "$LOG_FILE"
done
done
done

echo "Done. Output in: $WORK_DIR/kbe_runs"
echo "Logs in:   $WORK_DIR/logs"
#printf "Total elapsed: %02d:%02d:%02d\n" $((SECONDS/3600)) $((SECONDS/60%60)) $((SECONDS%60))
