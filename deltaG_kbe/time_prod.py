"""Seconds per step of the production evolve_syk4_kbe at the test size (beta J = 32, dt = 0.053, t_pre = 2 beta, t_post = 2: the first
38 post-quench steps, Nt = 1246), for comparison with run_test.py 2 on the same grid."""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import syk_batch_tools as sbt
from common import PROD, BETA, DT, T_PRE

with np.load(PROD[1]) as z:
    omega, A = z["omega_real"], z["A"]
t0 = time.time()
t, G, diag = sbt.evolve_syk4_kbe(omega, A, BETA, J2_i=0.0, J2_f=0.0, J4_i=1.0, J4_f=1.0, t_pre=T_PRE, t_post=2.0, dt=DT,
                                 n_corr=50, corr_tol=1e-16, progress_every=0, return_diagnostics=True)
n0 = int(diag["n0"]); steps = len(t) - 1 - n0
print(f"production evolve_syk4_kbe: {time.time() - t0:.1f} s total for {steps} steps (incl. setup); iters median "
      f"{np.median(diag['corr_iters_used'][n0 + 1:]):.0f}")
t1 = time.time()
sbt.evolve_syk4_kbe(omega, A, BETA, J2_i=0.0, J2_f=0.0, J4_i=1.0, J4_f=1.0, t_pre=T_PRE, t_post=0.0, dt=DT,
                    n_corr=50, corr_tol=1e-16, progress_every=0)
setup = time.time() - t1
print(f"setup only (t_post=0): {setup:.1f} s  ->  {(time.time() - t0 - 2 * setup) / steps:.3f} s/step")
