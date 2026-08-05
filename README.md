# sykquench
These scripts simulate finite-temperature quenches in various SYK models. Schwinger-Dyson numerics prepare the Green's function and self-energy of the finite temperature equilibrium ensemble, and simulate real-time dynamics using Kadanoff-Baym equations. This can be done for quenches in the single SYK, and the KKTZ model which includes the Maldacena-Qi model for $\alpha=0$. 


The bash scripts are used for submitting single and coupled KKTZ jobs to the cluster, and the respective tools.py scripts contain equilibrium SDE and non-equilibrium KBE solvers.


Additionally, for the single SYK quench with an engineered kernel acting as a source of bosonic modes: the syk_matsubara_f00.ipynb script determines $f_{00}(\tau/\beta)$ numerically using the coefficients computed to very high precision in *Precise low-temperature expansions for the Sachdev-Ye-Kitaev model*, Phys. Rev. B **108**, 035103 (2023), and syk_matsubara_kernel.ipynb then uses numerical $f_{00}(\tau/\beta)$ and analytical $f_{0}(\tau/\beta), f_{1}(\tau/\beta)$ to tune the kernel couplings such that there is virtually no overlap with the soft mode and its pure insertions ($\alpha_0\approx0$). These tuned couplings can be fed into real time equilibrium preparation using the above scripts. 
