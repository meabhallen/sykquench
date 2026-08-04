#!/bin/bash
# submit_kbe.sh — run on login node: bash submit_kbe.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these
# ─────────────────────────────────────────────────────────────────────────────

J4_IS=(1)           # initial J4 values
J4_FS=(1) #1.01 1.02 1.04) #2.01 2.02) # 2.01 2.02)       # final J4 values (quench targets)
BETAS=(5 4.998128) #9.986470 14.958239 19.905637 24.821366 29.698372 9.985749 14.957196 19.904356 24.819786 29.633004) # 15 20) # 15 20) 
DT=0.025
T_PRE_FACTOR=2.0
T_POST_FACTOR=4.0
CORR_TOL=1e-10
N_CORR=12
ITERATIONS=50

# Tuned-kernel source for the (rare) equilibrium/stationarity-check KBE run.
# Default off (0) for normal quench production runs.
KERNEL_LAMBDAS=(0)
KERNEL_CS=(0)
KERNEL_CUTOFFS=(2.0)   # factor of J4_i; KERNEL_CUTOFF passed to the run script is FACTOR*J4_i

# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/global/scratch/users/$USER/syk
SCRIPT=run_kbe_single.sh

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/kbe_runs"
cd "$WORK_DIR"

COUNT=0

for J4_I in "${J4_IS[@]}"; do
for J4_F in "${J4_FS[@]}"; do
for BETA in "${BETAS[@]}"; do
for KERNEL_LAMBDA in "${KERNEL_LAMBDAS[@]}"; do
for KERNEL_C in "${KERNEL_CS[@]}"; do
for KERNEL_CUTOFF_FACTOR in "${KERNEL_CUTOFFS[@]}"; do

    KERNEL_CUTOFF=$(python3 -c "print($KERNEL_CUTOFF_FACTOR * $J4_I)")

    echo "Submitting: J4_i=$J4_I J4_f=$J4_F beta=$BETA kernel_lambda=$KERNEL_LAMBDA kernel_c=$KERNEL_C kernel_cutoff=$KERNEL_CUTOFF"

    sbatch \
        --job-name="kbe_Ji-${J4_I}_Jf-${J4_F}_b-${BETA}_dt-${DT}_lam-${KERNEL_LAMBDA}" \
        --export=ALL,J4_I=$J4_I,J4_F=$J4_F,BETA=$BETA,DT=$DT,T_PRE_FACTOR=$T_PRE_FACTOR,T_POST_FACTOR=$T_POST_FACTOR,CORR_TOL=$CORR_TOL,N_CORR=$N_CORR,ITERATIONS=$ITERATIONS,KERNEL_LAMBDA=$KERNEL_LAMBDA,KERNEL_C=$KERNEL_C,KERNEL_CUTOFF=$KERNEL_CUTOFF \
        "$WORK_DIR/$SCRIPT"

    COUNT=$(( COUNT + 1 ))

done
done
done
done
done
done

echo ""
echo "Submitted $COUNT job(s)."
squeue -u $USER
