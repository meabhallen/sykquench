#!/bin/bash
# submit_kktz_kbe.sh
# Run on login node after KKTZ equilibrium jobs finish:
# bash submit_kktz_kbe.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# Protocol: weakly coupled KKTZ initial state, then turn off mu.
# ─────────────────────────────────────────────────────────────────────────────
J_IS=(1.0)
J_FS=(1.0)

# Fractions are converted to decimal values below.
ALPHA_IS=(0) #1/18 1/9 1/3)

MU_IS=(0.0)
MU_FS=(0.0)
BETAS=(12) # 10 14)

DT=0.05
T_PRE_FACTOR=2.0
T_POST_FACTOR=2.0

CORR_TOL=1e-8
N_CORR=4 #25
ITERATIONS=50 #1000
CHECKPOINT_EVERY=250

# Equilibrium-file lookup filters.
# These must match the corresponding equilibrium calculations.
EQ_DT=0.005
EQ_OMEGA_MAX=36.0

UNCOMPRESSED=0

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR=${WORK_DIR:-/global/scratch/users/$USER/syk}
SCRIPT=run_kktz_kbe_single.sh
EQ_DIR=$WORK_DIR/eq_runs

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/kbe_runs"
cd "$WORK_DIR"

COUNT=0

for J_I in "${J_IS[@]}"; do
for J_F in "${J_FS[@]}"; do
for ALPHA_I_VALUE in "${ALPHA_IS[@]}"; do
for MU_I in "${MU_IS[@]}"; do
for MU_F in "${MU_FS[@]}"; do
for BETA in "${BETAS[@]}"; do

    ALPHA_I=$(python3 -c "
from fractions import Fraction
print(float(Fraction('$ALPHA_I_VALUE')))
")

    # Keep alpha fixed during the quench.
    ALPHA_F="$ALPHA_I"

    EQ_NW=$(python3 -c "
Nw = max(
    4001,
    int(round(20 * float('$BETA') * float('$EQ_OMEGA_MAX')))
)
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")

    echo "Submitting KKTZ KBE:"
    echo "  J:       $J_I -> $J_F"
    echo "  alpha:   $ALPHA_I -> $ALPHA_F"
    echo "  mu:      $MU_I -> $MU_F"
    echo "  beta:    $BETA"
    echo "  EQ_NW:   $EQ_NW"

    sbatch \
        --job-name="kktz_kbe_a-${ALPHA_I}_to-${ALPHA_F}_mu-${MU_I}_to-${MU_F}_b-${BETA}" \
        --export=ALL,WORK_DIR="$WORK_DIR",J_I="$J_I",J_F="$J_F",ALPHA_I="$ALPHA_I",ALPHA_F="$ALPHA_F",MU_I="$MU_I",MU_F="$MU_F",BETA="$BETA",DT="$DT",T_PRE_FACTOR="$T_PRE_FACTOR",T_POST_FACTOR="$T_POST_FACTOR",CORR_TOL="$CORR_TOL",N_CORR="$N_CORR",ITERATIONS="$ITERATIONS",CHECKPOINT_EVERY="$CHECKPOINT_EVERY",EQ_DIR="$EQ_DIR",EQ_DT="$EQ_DT",EQ_OMEGA_MAX="$EQ_OMEGA_MAX",EQ_NW="$EQ_NW",UNCOMPRESSED="$UNCOMPRESSED" \
        "$WORK_DIR/$SCRIPT"

    COUNT=$((COUNT + 1))

done
done
done
done
done
done

echo ""
echo "Submitted $COUNT KKTZ KBE job(s)."
squeue -u "$USER"
