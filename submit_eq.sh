#!/bin/bash
# submit_eq.sh — run on login node: bash submit_eq.sh

set -e

module load python/3.11.6-gcc-11.4.0

# ─────────────────────────────────────────────────────────────────────────────
# Parameters — edit these
# ─────────────────────────────────────────────────────────────────────────────

J4S=(1) # 1.02 1.04)
BETAS=(5 4.998128 9.986470 14.958239 19.905637 24.821366 29.698372 9.985749 14.957196 19.904356 24.819786 29.633004) #(35 40 45 50)
TOLS=(1e-12)

# Tuned-kernel source. Defaults reproduce plain production eq runs
# (kernel off). Override for a kernel-tuning scan, e.g.
#   KERNEL_LAMBDAS=(-0.005 0.005); KERNEL_CS=(0.02 0.04 0.06 0.08 0.1 0.12 0.14 0.16)
# This replaces the old submit_eq_tuned_kernel.sh.
KERNEL_LAMBDAS=(0)
KERNEL_CS=(0)
KERNEL_CUTOFFS=(2.0)   # factor of J4; KERNEL_CUTOFF passed to the run script is FACTOR*J4

DAB_TOLS=(1e-6)        # tolerance for max(d_ab**0.5)
REQUIRE_DAB=0          # set to 1 to require d_ab as well as delta_F for converged=True

# ─────────────────────────────────────────────────────────────────────────────
# Fixed settings
# ─────────────────────────────────────────────────────────────────────────────

WORK_DIR=/global/scratch/users/$USER/syk
SCRIPT=run_eq_single.sh    # the slurm script (see below)

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
for KERNEL_C in "${KERNEL_CS[@]}"; do
for KERNEL_CUTOFF_FACTOR in "${KERNEL_CUTOFFS[@]}"; do
for DAB_TOL in "${DAB_TOLS[@]}"; do

    # Derived parameters (mirrors make_eq_jobs.py logic)
    OMEGA_MAX=$(python3 -c "print(36 * $J4)")
    DT=$(python3 -c "print(0.0008 / $J4)")
    NW=$(python3 -c "
Nw = max(4001, int(round(20 * $BETA * $OMEGA_MAX)))
if Nw % 2 == 0:
    Nw += 1
print(Nw)
")
    KERNEL_CUTOFF=$(python3 -c "print($KERNEL_CUTOFF_FACTOR * $J4)")

    echo "Submitting: J4=$J4 beta=$BETA dt=$DT omega_max=$OMEGA_MAX Nw=$NW tol=$TOL kernel_lambda=$KERNEL_LAMBDA kernel_c=$KERNEL_C kernel_cutoff=$KERNEL_CUTOFF dab_tol=$DAB_TOL"

    sbatch \
    	--job-name="syk_eq_J-${J4}_beta-${BETA}_lam-${KERNEL_LAMBDA}" \
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

echo ""
echo "Submitted $COUNT job(s)."
squeue -u $USER
