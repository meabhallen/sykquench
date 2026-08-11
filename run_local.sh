#!/bin/bash
# run_local.sh — run one eq-one + kbe-one pair directly on your own machine.
# No SLURM, no module load, no scratch paths. Usage: bash run_local.sh
#
# Reproduces a single (J4, beta, tuned-kernel) point from the submit_eq.sh /
# submit_kbe.sh sweeps using the same parameter-derivation formulas, so a
# local run here is a faithful smoke test of what the cluster jobs will do.

set -e

PYTHON=${PYTHON:-python3}

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these
# ─────────────────────────────────────────────────────────────────────────────

J4=1
BETA=5

# Equilibrium solve (mirrors submit_eq.sh's derivation formulas)
EQ_TOL=1e-3              # production uses 1e-08; loosened here for local speed
DAB_TOL=1e-3             # not enforced (REQUIRE_DAB=0 below), only printed
REQUIRE_DAB=0
EQ_DT_FACTOR=0.5         # eq dt = EQ_DT_FACTOR / J4 (production uses 0.0112; larger here for local speed)
OMEGA_MAX_FACTOR=36      # omega_max = OMEGA_MAX_FACTOR * J4
NW_RATIO=20              # Nw = max(4001, round(NW_RATIO * BETA * omega_max)), forced odd
VERBOSE_EVERY=10         # print progress every N iterations

# KBE evolution
KBE_DT=0.5               # production uses 0.025; larger here for local speed
T_PRE_FACTOR=2.0
T_POST_FACTOR=4.0
CORR_TOL=1e-3            # production uses 1e-10; loosened here for local speed
N_CORR=12
PROGRESS_EVERY=50

# Evolution kernel: quench dynamics run kernel-free (matches submit_kbe.sh).
KERNEL_LAMBDA=0
KERNEL_C=0
KERNEL_CUTOFF_FACTOR=2.0     # factor of J4; inactive since KERNEL_LAMBDA=0

# Initial-state (equilibrium) kernel: the tuned delta_star(Lambda) used to
# prepare the pre-quench state. Pick ONE of the tuned pairs from
# submit_eq.sh/submit_kbe.sh:
#   lambda = +/-0.005
#   (c, cutoff_factor) in {(-0.043936, 0.65), (-0.053648, 0.75)}
EQ_KERNEL_LAMBDA=0.005
EQ_KERNEL_C=-0.043936
EQ_KERNEL_CUTOFF_FACTOR=0.65   # factor of J4

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORK_DIR="$SCRIPT_DIR/local_runs"
mkdir -p "$WORK_DIR/eq_runs" "$WORK_DIR/kbe_runs"

if ! "$PYTHON" -c "import numpy, pandas, scipy" 2>/dev/null; then
    echo "numpy/pandas/scipy not importable via '$PYTHON'." >&2
    echo "Install them (pip install numpy pandas scipy) or set PYTHON=/path/to/python3 and re-run." >&2
    exit 1
fi

EQ_DT=$("$PYTHON" -c "print($EQ_DT_FACTOR / $J4)")
OMEGA_MAX=$("$PYTHON" -c "print($OMEGA_MAX_FACTOR * $J4)")
NW=$("$PYTHON" -c "
Nw = max(4001, int(round($NW_RATIO * $BETA * $OMEGA_MAX)))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
KERNEL_CUTOFF=$("$PYTHON" -c "print($KERNEL_CUTOFF_FACTOR * $J4)")
EQ_KERNEL_CUTOFF=$("$PYTHON" -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4)")

DAB_FLAG=""
if [ "$REQUIRE_DAB" = "1" ]; then
    DAB_FLAG="--require-dab-convergence"
fi

echo "============================================================"
echo "Step 1/2: equilibrium solve (tuned kernel)"
echo "  J4=$J4 beta=$BETA dt=$EQ_DT omega_max=$OMEGA_MAX Nw=$NW"
echo "  kernel: lambda=$EQ_KERNEL_LAMBDA c=$EQ_KERNEL_C cutoff=$EQ_KERNEL_CUTOFF"
echo "============================================================"

"$PYTHON" -u syk_batch_tools.py eq-one \
    --J4            "$J4"            \
    --beta          "$BETA"          \
    --dt            "$EQ_DT"         \
    --omega-max     "$OMEGA_MAX"     \
    --Nw            "$NW"            \
    --tol           "$EQ_TOL"        \
    --dab-tol       "$DAB_TOL"       \
    --kernel-lambda "$EQ_KERNEL_LAMBDA" \
    --kernel-c      "$EQ_KERNEL_C"   \
    --kernel-cutoff "$EQ_KERNEL_CUTOFF" \
    --verbose-every "$VERBOSE_EVERY" \
    --out-dir       "$WORK_DIR/eq_runs" \
    $DAB_FLAG

echo ""
echo "============================================================"
echo "Step 2/2: KBE quench (kernel-free evolution, tuned initial state)"
echo "  J4_i=J4_f=$J4 beta=$BETA dt=$KBE_DT"
echo "  evolution kernel: lambda=$KERNEL_LAMBDA c=$KERNEL_C cutoff=$KERNEL_CUTOFF"
echo "  eq-selection kernel: lambda=$EQ_KERNEL_LAMBDA c=$EQ_KERNEL_C cutoff=$EQ_KERNEL_CUTOFF"
echo "============================================================"

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

echo ""
echo "Done. Output in: $WORK_DIR"
