#!/bin/bash
# submit_mq_eq.sh — run on login node: bash submit_mq_eq.sh
set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these explicitly.
# These are the initial weakly-coupled MQ equilibrium states.
# ─────────────────────────────────────────────────────────────────────────────
JS=(1.0)
MUS=(0.001 0.002 0.005 0.01)
BETAS=(4 5 6 8 12 14 16)
TOLS=(1e-15)

# Explicit grid controls, passed directly to mq_batch_tools2.py eq-one
DT=0.0025
OMEGA_MAX=36.0
MAX_ITER=2000
MIXING=0.05
ETA_RET=1e-6
CHECKPOINT_EVERY=50
COMPUTE_KBE_DAB_EVERY=50
SEED_FROM_MU0=1
PROJECT=1

# Optional: uncomment for finite real-time equilibrium window override.
# T_MAX=160

# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR=${WORK_DIR:-/global/scratch/users/$USER/syk}
SCRIPT=run_mq_eq_single.sh

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/eq_runs"
cd "$WORK_DIR"

COUNT=0
for J in "${JS[@]}"; do
for MU in "${MUS[@]}"; do
for BETA in "${BETAS[@]}"; do
for TOL in "${TOLS[@]}"; do
    NW=$(python3 -c "
Nw = max(4001, int(round(20 * float('$BETA') * float('$OMEGA_MAX'))))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    echo "Submitting MQ eq: J=$J mu=$MU beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL"
	
    sbatch \
        --job-name="mq_eq_J-${J}_mu-${MU}_b-${BETA}" \
        --export=ALL,WORK_DIR=$WORK_DIR,J=$J,MU=$MU,BETA=$BETA,DT=$DT,OMEGA_MAX=$OMEGA_MAX,NW=$NW,TOL=$TOL,MAX_ITER=$MAX_ITER,MIXING=$MIXING,ETA_RET=$ETA_RET,CHECKPOINT_EVERY=$CHECKPOINT_EVERY,COMPUTE_KBE_DAB_EVERY=$COMPUTE_KBE_DAB_EVERY,SEED_FROM_MU0=$SEED_FROM_MU0,PROJECT=$PROJECT${T_MAX:+,T_MAX=$T_MAX} \
        "$WORK_DIR/$SCRIPT"

    COUNT=$(( COUNT + 1 ))

done
done
done
done

echo ""
echo "Submitted $COUNT MQ equilibrium job(s)."
squeue -u $USER
