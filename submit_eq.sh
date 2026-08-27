#!/bin/bash
# submit_eq.sh — run on login node: bash submit_eq.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

J4S=(1)
BETAS=(36 48)

# Tuned-kernel source. lambda is independent (submitted crossed with every
# (kernel_c, kernel_cutoff) pair below) -- both signs are needed since the
# real-time analysis reads off the odd-in-lambda response D_lambda =
# (G_+ - G_-)/(2*lambda), which requires a matched +/-lambda pair per tuned
# (delta_star, Lambda). kernel_c/kernel_cutoff are NOT independent: they are
# delta_star(Lambda) pairs tuned in syk_matsubara_kernel.ipynb (large-betaJ
# converged values, betaJ in [240,300,400]) so they must stay paired, not
# looped independently -- an earlier version of this script (and, separately,
# an earlier version of the analysis notebook) hit exactly this bug via
# independent arrays silently forming the wrong (c, Lambda) combinations.
EQ_KERNEL_LAMBDAS=(0.005 -0.005)
EQ_KERNEL_C_CUTOFF_PAIRS=(
    #"-0.043936 0.65"
    "-0.053648 0.75"
)

TOLS=(1e-012) 		    # tol for delta_F
DAB_TOLS=(1e-06)        # tolerance for max(d_ab**0.5)
REQUIRE_DAB=1          	# set to 1 to require d_ab as well as delta_F for converged=True
EQ_DT_FACTORS=(0.0064)      # eq dt = EQ_DT_FACTOR / J4
OMEGA_MAX_FACTORS=(36)      # omega_max = OMEGA_MAX_FACTOR * J4
NW_RATIOS=(20)              # Nw = max(4001, round(NW_RATIO * BETA * omega_max)), forced odd
VERBOSE_EVERY=200          # print progress every N iterations

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/global/scratch/users/$USER/sykquench
SCRIPT=run_eq_single.sh   

mkdir -p "$WORK_DIR/logs" "$WORK_DIR/eq_runs"
cd "$WORK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Loop and submit
# ─────────────────────────────────────────────────────────────────────────────

COUNT=0

for J4 in "${J4S[@]}"; do
for BETA in "${BETAS[@]}"; do
for TOL in "${TOLS[@]}"; do
for EQ_KERNEL_LAMBDA in "${EQ_KERNEL_LAMBDAS[@]}"; do
for EQ_KERNEL_C_CUTOFF in "${EQ_KERNEL_C_CUTOFF_PAIRS[@]}"; do
read -r EQ_KERNEL_C EQ_KERNEL_CUTOFF_FACTOR <<< "$EQ_KERNEL_C_CUTOFF"
for DAB_TOL in "${DAB_TOLS[@]}"; do
for EQ_DT_FACTOR in "${EQ_DT_FACTORS[@]}"; do
for OMEGA_MAX_FACTOR in "${OMEGA_MAX_FACTORS[@]}"; do
for NW_RATIO in "${NW_RATIOS[@]}"; do

    OMEGA_MAX=$(python3 -c "print($OMEGA_MAX_FACTOR * $J4)")
    DT=$(python3 -c "print($EQ_DT_FACTOR / $J4)")
    NW=$(python3 -c "
Nw = max(4001, int(round($NW_RATIO * $BETA * $OMEGA_MAX)))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    EQ_KERNEL_CUTOFF=$(python3 -c "print($EQ_KERNEL_CUTOFF_FACTOR * $J4)")

    echo "Submitting: J4=$J4 beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL kernel_lambda=$EQ_KERNEL_LAMBDA kernel_c=$EQ_KERNEL_C kernel_cutoff=$EQ_KERNEL_CUTOFF dab_tol=$DAB_TOL"

    sbatch \
    	--job-name="syk_eq_J-${J4}_beta-${BETA}_lam-${EQ_KERNEL_LAMBDA}_c-${EQ_KERNEL_C}_Lam-${EQ_KERNEL_CUTOFF}" \
    	--export=ALL,J4=$J4,BETA=$BETA,DT=$DT,OMEGA_MAX=$OMEGA_MAX,NW=$NW,TOL=$TOL,EQ_KERNEL_LAMBDA=$EQ_KERNEL_LAMBDA,EQ_KERNEL_C=$EQ_KERNEL_C,EQ_KERNEL_CUTOFF=$EQ_KERNEL_CUTOFF,DAB_TOL=$DAB_TOL,REQUIRE_DAB=$REQUIRE_DAB,VERBOSE_EVERY=$VERBOSE_EVERY \
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
done

echo ""
echo "Submitted $COUNT job(s)."
squeue -u $USER
