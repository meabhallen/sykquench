#!/bin/bash
# submit_eq.sh — run on login node: bash submit_eq.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters
# ─────────────────────────────────────────────────────────────────────────────

J4S=(1)
BETAS=(5)

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
KERNEL_LAMBDAS=(0.005 -0.005)
KERNEL_C_CUTOFF_PAIRS=(
    "-0.043936 0.65"
    "-0.053648 0.75"
)

TOLS=(1e-08) 		    # tol for delta_F
DAB_TOLS=(1e-04)        # tolerance for max(d_ab**0.5)
REQUIRE_DAB=1          	# set to 1 to require d_ab as well as delta_F for converged=True
DTS=(0.0112)
OMEGA_MAXS=(36)
NWS=(20)

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
for KERNEL_LAMBDA in "${KERNEL_LAMBDAS[@]}"; do
for KERNEL_C_CUTOFF in "${KERNEL_C_CUTOFF_PAIRS[@]}"; do
read -r KERNEL_C KERNEL_CUTOFF_FACTOR <<< "$KERNEL_C_CUTOFF"
for DAB_TOL in "${DAB_TOLS[@]}"; do
for T in "${DTS[@]}"; do
for OMEGA_M in "${OMEGA_MAXS[@]}"; do
for NWRATIO in "${NWS[@]}"; do

    OMEGA_MAX=$(python3 -c "print($OMEGA_M * $J4)")
    DT=$(python3 -c "print($T / $J4)")
    NW=$(python3 -c "
Nw = max(4001, int(round($NWRATIO * $BETA * $OMEGA_MAX)))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    KERNEL_CUTOFF=$(python3 -c "print($KERNEL_CUTOFF_FACTOR * $J4)")

    echo "Submitting: J4=$J4 beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL kernel_lambda=$KERNEL_LAMBDA kernel_c=$KERNEL_C kernel_cutoff=$KERNEL_CUTOFF dab_tol=$DAB_TOL"

    sbatch \
    	--job-name="syk_eq_J-${J4}_beta-${BETA}_lam-${KERNEL_LAMBDA}_c-${KERNEL_C}_Lam-${KERNEL_CUTOFF}" \
    	--export=ALL,J4=$J4,BETA=$BETA,DT=$DT,OMEGA_MAX=$OMEGA_MAX,NW=$NW,TOL=$TOL,KERNEL_LAMBDA=$KERNEL_LAMBDA,KERNEL_C=$KERNEL_C,KERNEL_CUTOFF=$KERNEL_CUTOFF,DAB_TOL=$DAB_TOL,REQUIRE_DAB=$REQUIRE_DAB \
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
