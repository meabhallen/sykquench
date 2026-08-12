#!/bin/bash
# run_local_eq.sh — run one eq-one job directly.
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

# Equilibrium solve
EQ_TOL=1e-12             
DAB_TOL=1e-6             # not enforced if REQUIRE_DAB=0 below
REQUIRE_DAB=0
EQ_DT_FACTOR=0.0064         # eq dt = EQ_DT_FACTOR / J4
OMEGA_MAX_FACTOR=36      # omega_max = OMEGA_MAX_FACTOR * J4
NW_RATIO=20              # Nw = max(4001, round(NW_RATIO * BETA * omega_max)), forced odd
VERBOSE_EVERY=200         # print progress every N iterations

# Initial-state equilibrium kernel: the tuned delta_star(Lambda) used to
# prepare the pre-quench state. Pick from tuned pairs:
#   (delta*, Lambda) = (c, cutoff_factor) in {(-0.043936, 0.65), (-0.053648, 0.75)}
# Both signs of lambda are run below (matches submit_eq.sh): downstream
# real-time analysis reads off the odd-in-lambda response
# D_lambda = (G_+ - G_-)/(2*lambda), which needs a matched +/-lambda pair.
EQ_KERNEL_LAMBDA_MAG=0.0025
EQ_KERNEL_C=-0.053648
EQ_KERNEL_CUTOFF_FACTOR=0.75   # factor of J4

# ─────────────────────────────────────────────────────────────────────────────

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
WORK_DIR="$SCRIPT_DIR/../local_runs"
mkdir -p "$WORK_DIR/eq_runs"

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
EQ_KERNEL_CUTOFF=$("$PYTHON" -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4)")

DAB_FLAG=""
if [ "$REQUIRE_DAB" = "1" ]; then
    DAB_FLAG="--require-dab-convergence"
fi

for J4 in "${J4S[@]}"; do
for BETA in "${BETAS[@]}"; do
for EQ_KERNEL_LAMBDA in "$EQ_KERNEL_LAMBDA_MAG" "-$EQ_KERNEL_LAMBDA_MAG"; do
    echo "============================================================"
    echo "Equilibrium solve (tuned kernel)"
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
done
done
done

echo "Done. Output in: $WORK_DIR/eq_runs"
