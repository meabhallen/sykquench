#!/bin/bash
# submit_kktz_eq.sh — run on login node: bash submit_kktz_eq.sh
set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these explicitly.
# KKTZ quartic-coupled equilibrium states. MU is an optional tiny Q-source.
# ─────────────────────────────────────────────────────────────────────────────
JS=(1.0)
ALPHAS=(0) #.01 0.02 0.03 0.04) 
MUS=(0) # physical Q source; set 0.0 to test spontaneous symmetry breaking
SB_SEED_MU=0.01 # only applied automatically when alpha<0 and physical MU=0; sign selects branch
BETAS=(24) #6 12 18) 
TOLS=(1e-10)
DAB_TOLS=(1e-5)        # tolerance for max(d_ab**0.5)
REQUIRE_DAB=1          # set to 1 to require d_ab as well as delta for converged=True

# Explicit grid controls, passed directly to kktz_batch_tools2.py eq-one
DT=0.0025
OMEGA_MAX=36.0
MAX_ITER=5000
MIXING=0.05
ETA_RET=1e-6
CHECKPOINT_EVERY=500
COMPUTE_KBE_DAB_EVERY=50
SEED_FROM_MU0=1
PROJECT=1

# Optional: uncomment for finite real-time equilibrium window override.
# T_MAX=160

# ─────────────────────────────────────────────────────────────────────────────
WORK_DIR=${WORK_DIR:-/global/scratch/users/$USER/syk}
SCRIPT=run_kktz_eq_single.sh

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/eq_runs"
cd "$WORK_DIR"

COUNT=0
for J in "${JS[@]}"; do
for ALPHA in "${ALPHAS[@]}"; do
for MU in "${MUS[@]}"; do
for BETA in "${BETAS[@]}"; do
for TOL in "${TOLS[@]}"; do
for DAB_TOL in "${DAB_TOLS[@]}"; do
    ALPHA_VALUE=$(python3 -c "from fractions import Fraction; print(float(Fraction('$ALPHA')))")
    SB_SEED_MU_THIS=$(python3 -c "a=float('$ALPHA_VALUE'); mu=float('$MU'); print('$SB_SEED_MU' if a < 0.0 and abs(mu) < 1e-15 else '0.0')")
    NW=$(python3 -c "
Nw = max(4001, int(round(20 * float('$BETA') * float('$OMEGA_MAX'))))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    echo "Submitting KKTZ eq: J=$J alpha=$ALPHA_VALUE mu=$MU sb_seed_mu=$SB_SEED_MU_THIS beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL dab_tol=$DAB_TOL require_dab=$REQUIRE_DAB"

    sbatch \
        --job-name="kktz_eq_J-${J}_a-${ALPHA_VALUE}_mu-${MU}_b-${BETA}" \
        --export=ALL,WORK_DIR=$WORK_DIR,J=$J,ALPHA=$ALPHA_VALUE,MU=$MU,SB_SEED_MU=$SB_SEED_MU_THIS,BETA=$BETA,DT=$DT,OMEGA_MAX=$OMEGA_MAX,NW=$NW,TOL=$TOL,DAB_TOL=$DAB_TOL,REQUIRE_DAB=$REQUIRE_DAB,MAX_ITER=$MAX_ITER,MIXING=$MIXING,ETA_RET=$ETA_RET,CHECKPOINT_EVERY=$CHECKPOINT_EVERY,COMPUTE_KBE_DAB_EVERY=$COMPUTE_KBE_DAB_EVERY,SEED_FROM_MU0=$SEED_FROM_MU0,PROJECT=$PROJECT${T_MAX:+,T_MAX=$T_MAX} \
        "$WORK_DIR/$SCRIPT"

    COUNT=$(( COUNT + 1 ))

done
done
done
done
done
done

echo ""
echo "Submitted $COUNT KKTZ equilibrium job(s)."
squeue -u $USER
