#!/bin/bash
# submit_kbe.sh — run on login node: bash submit_kbe.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these
# ─────────────────────────────────────────────────────────────────────────────

J4_IS=(1)           # initial J4 values
J4_FS=(1) #1.01 1.02 1.04) #2.01 2.02) # 2.01 2.02)       # final J4 values (quench targets)
BETAS=(36 48) #9.986470 14.958239 19.905637 24.821366 29.698372 9.985749 14.957196 19.904356 24.819786 29.633004) # 15 20) # 15 20) 
DT=0.025
T_PRE_FACTOR=2.0
T_POST_FACTOR=1.0
CORR_TOL_FACTOR=1e-12           # corr_tol = CORR_TOL_FACTOR / BETA * DT / 20 * 12
N_CORR=50
CHECKPOINT_EVERY=50

# Evolution kernel: the kernel actually applied during the real-time KBE
# integration. Quench dynamics always run with this trivial (off) -- the
# engineered-kernel deformation only lives in the initial equilibrium state.
KERNEL_LAMBDAS=(0)
KERNEL_CS=(0)
KERNEL_CUTOFFS=(2.0)   # factor of J4_i; KERNEL_CUTOFF passed to the run script is FACTOR*J4_i

# Initial-state kernel: selects which tuned-kernel equilibrium file (produced
# by submit_eq.sh) to quench from. Must match those tuples exactly -- same
# delta_star(Lambda) pairs, same +/-lambda values -- or find_eq_file() in
# syk_batch_tools.py will not find a matching converged eq_runs entry.
EQ_KERNEL_LAMBDAS=(0.005 -0.005)
EQ_KERNEL_C_CUTOFF_PAIRS=(
    "-0.043936 0.65"
    "-0.053648 0.75"
)

# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/global/scratch/users/$USER/sykquench
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
for EQ_KERNEL_LAMBDA in "${EQ_KERNEL_LAMBDAS[@]}"; do
for EQ_KERNEL_C_CUTOFF in "${EQ_KERNEL_C_CUTOFF_PAIRS[@]}"; do
read -r EQ_KERNEL_C EQ_KERNEL_CUTOFF_FACTOR <<< "$EQ_KERNEL_C_CUTOFF"

    KERNEL_CUTOFF=$(python3 -c "print($KERNEL_CUTOFF_FACTOR * $J4_I)")
    EQ_KERNEL_CUTOFF=$(python3 -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4_I)")
    CORR_TOL=$(python3 -c "print($CORR_TOL_FACTOR / $BETA * $DT / 20 * 12)")

    echo "Submitting: J4_i=$J4_I J4_f=$J4_F beta=$BETA kernel_lambda=$KERNEL_LAMBDA kernel_c=$KERNEL_C kernel_cutoff=$KERNEL_CUTOFF eq_kernel_lambda=$EQ_KERNEL_LAMBDA eq_kernel_c=$EQ_KERNEL_C eq_kernel_cutoff=$EQ_KERNEL_CUTOFF corr_tol=$CORR_TOL"

    sbatch \
        --job-name="kbe_Ji-${J4_I}_Jf-${J4_F}_b-${BETA}_dt-${DT}_eqlam-${EQ_KERNEL_LAMBDA}" \
        --export=ALL,J4_I=$J4_I,J4_F=$J4_F,BETA=$BETA,DT=$DT,T_PRE_FACTOR=$T_PRE_FACTOR,T_POST_FACTOR=$T_POST_FACTOR,CORR_TOL=$CORR_TOL,N_CORR=$N_CORR,CHECKPOINT_EVERY=$CHECKPOINT_EVERY,KERNEL_LAMBDA=$KERNEL_LAMBDA,KERNEL_C=$KERNEL_C,KERNEL_CUTOFF=$KERNEL_CUTOFF,EQ_KERNEL_LAMBDA=$EQ_KERNEL_LAMBDA,EQ_KERNEL_C=$EQ_KERNEL_C,EQ_KERNEL_CUTOFF=$EQ_KERNEL_CUTOFF \
        "$WORK_DIR/$SCRIPT"

    COUNT=$(( COUNT + 1 ))

done
done
done
done
done
done
done
done

echo ""
echo "Submitted $COUNT job(s)."
squeue -u $USER
