#!/bin/bash
# submit_mq_kbe.sh — run on login node after MQ equilibrium jobs finish: bash submit_mq_kbe.sh
set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these.
# Protocol here: weakly coupled MQ initial state, then turn off mu.
# ─────────────────────────────────────────────────────────────────────────────
J_IS=(1.0)
J_FS=(1.0)
MU_IS=(0.005)
MU_FS=(0.0)
BETAS=(10 15 20)

DT=0.05
T_PRE_FACTOR=2.0
T_POST_FACTOR=1.0
CORR_TOL=1e-9
N_CORR=25
ITERATIONS=50
CHECKPOINT_EVERY=50

# Equilibrium-file lookup filters. Use the same grid controls as submit_mq_eq.sh.
EQ_DT=0.0025
EQ_OMEGA_MAX=36.0
#EQ_NW=4097

# Full G(t1,t2) can be large. Default is compressed final .npz.
# Set UNCOMPRESSED=1 for faster save but larger files.
UNCOMPRESSED=0

# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR=${WORK_DIR:-/global/scratch/users/$USER/syk}
SCRIPT=run_mq_kbe_single.sh
EQ_DIR=$WORK_DIR/eq_runs

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/kbe_runs"
cd "$WORK_DIR"

COUNT=0
for J_I in "${J_IS[@]}"; do
for J_F in "${J_FS[@]}"; do
for MU_I in "${MU_IS[@]}"; do
for MU_F in "${MU_FS[@]}"; do
for BETA in "${BETAS[@]}"; do
    EQ_NW=$(python3 -c "
Nw = max(4001, int(round(20 * float('$BETA') * float('$EQ_OMEGA_MAX'))))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    echo "Submitting MQ KBE: J_i=$J_I J_f=$J_F mu_i=$MU_I mu_f=$MU_F beta=$BETA"

    sbatch \
        --job-name="mq_kbe_mu-${MU_I}_to-${MU_F}_b-${BETA}_dt-${DT}" \
        --export=ALL,WORK_DIR=$WORK_DIR,J_I=$J_I,J_F=$J_F,MU_I=$MU_I,MU_F=$MU_F,BETA=$BETA,DT=$DT,T_PRE_FACTOR=$T_PRE_FACTOR,T_POST_FACTOR=$T_POST_FACTOR,CORR_TOL=$CORR_TOL,N_CORR=$N_CORR,ITERATIONS=$ITERATIONS,CHECKPOINT_EVERY=$CHECKPOINT_EVERY,EQ_DIR=$EQ_DIR,EQ_DT=$EQ_DT,EQ_OMEGA_MAX=$EQ_OMEGA_MAX,EQ_NW=$EQ_NW,UNCOMPRESSED=$UNCOMPRESSED \
        "$WORK_DIR/$SCRIPT"

    COUNT=$(( COUNT + 1 ))

done
done
done
done
done

echo ""
echo "Submitted $COUNT MQ KBE job(s)."
squeue -u $USER
