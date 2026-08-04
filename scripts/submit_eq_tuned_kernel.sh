#!/bin/bash
# Submit equilibrium +lambda/-lambda scans for tuning c_crit.

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters to scan
# ─────────────────────────────────────────────────────────────────────────────

J4S=(1)
BETAS=(6 12 18 24)
TOLS=(1e-10) # 1e-10) # 1e-08 1e-10)

# Keep |lambda| fixed during the first c scan. Repeat later with a smaller value.
LAMBDA_ABS=0.005
LAMBDAS=(-$LAMBDA_ABS $LAMBDA_ABS)

# Start coarse. Expand the range if the extracted h0 amplitude does not change sign.
CS=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16)

# These define the regulator and must stay fixed throughout the scan.
KERNEL_CUTOFF_FACTOR=2.0

WORK_DIR=/global/scratch/users/$USER/syk
SCRIPT=run_eq_single_tuned_kernel.sh

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/eq_tuned_runs"
cd "$WORK_DIR"

COUNT=0

for J4 in "${J4S[@]}"; do
for BETA in "${BETAS[@]}"; do
for TOL in "${TOLS[@]}"; do
for KERNEL_LAMBDA in "${LAMBDAS[@]}"; do
for KERNEL_C in "${CS[@]}"; do

    OMEGA_MAX=$(python3 -c "print(36 * $J4)")
    DT=$(python3 -c "print(0.0016 / $J4)")
    NW=$(python3 -c "
Nw = max(4001, int(round(20 * $BETA * $OMEGA_MAX)))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    KERNEL_CUTOFF=$(python3 -c "print($KERNEL_CUTOFF_FACTOR * $J4)")

    echo "Submitting J4=$J4 beta=$BETA lambda=$KERNEL_LAMBDA c=$KERNEL_C Lambda=$KERNEL_CUTOFF"

    sbatch \
        --job-name="syk_tune_b${BETA}_l${KERNEL_LAMBDA}_c${KERNEL_C}" \
        --export=ALL,J4=$J4,BETA=$BETA,DT=$DT,OMEGA_MAX=$OMEGA_MAX,NW=$NW,TOL=$TOL,KERNEL_LAMBDA=$KERNEL_LAMBDA,KERNEL_C=$KERNEL_C,KERNEL_CUTOFF=$KERNEL_CUTOFF \
        "$WORK_DIR/$SCRIPT"

    COUNT=$((COUNT + 1))

done
done
done
done
done

echo "Submitted $COUNT tuned-kernel equilibrium job(s)."
squeue -u "$USER"
