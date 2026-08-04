"""
MQ / coupled-SYK q=4 real-time equilibrium + KBE patch v2.

Conventions:
    G^>_{ab}(t1,t2) = -i < psi_a(t1) psi_b(t2) >
    G^<_{ab}(t1,t2) = - G^>_{ba}(t2,t1)
    Sigma^>_{ab} = - J(t1) J(t2) (G^>_{ab})^3

The important v2 change is the equilibrium FDT for the off-diagonal component:
    G^>_aa(omega) = 2i Im G^R_aa(omega) / (exp(-beta omega)+1)
    G^>_LR(omega) = Re G^R_LR(omega) / (exp(-beta omega)+1)
not the diagonal spectral formula applied componentwise.

The KBE initializer uses the returned Ggt_eq directly; it never reconstructs LR from A_LR.
"""

import numpy as np
from scipy.interpolate import interp1d

''' formerly 
def _trapz_weights(x):
    x = np.asarray(x)
    dx = x[1] - x[0]
    w = np.ones_like(x, dtype=float) * dx
    w[0] *= 0.5
    w[-1] *= 0.5
    return w'''

def _trap_weights_uniform_from_dx(n, dx):
    w = np.ones(n, dtype=float) * dx
    if n:
        w[0] *= 0.5
        w[-1] *= 0.5
    return w

def _trap_weights_grid(x):
    x = np.asarray(x)
    dx = x[1] - x[0]
    return _trap_weights_uniform_from_dx(len(x), dx)

''' Fourier transforms, same as single SYK'''

def _time_to_omega_mat(X_t, t, omega, chunk=384):
    t = np.asarray(t)
    omega = np.asarray(omega)
    wt = _trap_weights_grid(t)
    Xt = np.asarray(X_t) * wt[:, None, None]
    out = np.empty((len(omega), 2, 2), dtype=complex)
    for a in range(0, len(omega), chunk):
        phase = np.exp(1j * np.outer(omega[a:a+chunk], t))
        out[a:a+chunk] = np.einsum("ot,tab->oab", phase, Xt, optimize=True)
    return out


def _omega_to_time_mat(X_w, omega, t, chunk=384):
    omega = np.asarray(omega)
    t = np.asarray(t)
    ww = _trap_weights_grid(omega) / (2.0 * np.pi)
    Xw = np.asarray(X_w) * ww[:, None, None]
    out = np.empty((len(t), 2, 2), dtype=complex)
    for a in range(0, len(t), chunk):
        phase = np.exp(-1j * np.outer(t[a:a+chunk], omega))
        out[a:a+chunk] = np.einsum("to,oab->tab", phase, Xw, optimize=True)
    return out


def h_mu_matrix(mu):
    # h in i d_t G = h G + ... ; h = i * mu_matrix, real antisymmetric matrix appearing in d psi / dt = mu_matrix psi
    return 1j * np.array([[0.0, mu], [-mu, 0.0]], dtype=complex)

def get_G_Sigma_mq_matsubara(
    J, mu, beta,
    q=4,
    N_half=4*512,
    max_iter=2000,
    tol=1e-10,
    mixing=0.03,iterations=50,
):
    """
    Imaginary-time SD self-consistency for the Maldacena-Qi model.

    Self-energy (diagonal only):
        Sigma_LL(tau) = J^2 * [2 G_LL(tau)]^{q-1}
        Sigma_LR = 0

    Dyson (2x2 matrix inversion with mu contact term):
        D(iw) = (iw - Sigma_LL)^2 - mu^2
        G_LL(iw) = (iw - Sigma_LL) / D
        G_LR(iw) = -i*mu / D

    The self-consistency loop runs on G_LL only.
    G_LR is computed from G_LL at convergence (no separate loop).

    Fourier convention: G(tau) = (1/beta) sum_n G(iw_n) exp(+i w_n tau)
    => free G_LL(0+) = +1/2  (matches MQ paper section 5.1)

    Returns
    -------
    tau_grid    : imaginary times in [0, beta)
    G_LL_tau    : real array
    G_LR_tau    : complex array (purely imaginary at convergence)
    omega_n     : Matsubara frequencies
    Sigma_LL_iw : self-energy in Matsubara space
    """
    N        = 2 * N_half
    n_arr    = np.arange(-N_half, N_half)
    omega_n  = (2 * n_arr + 1) * np.pi / beta
    tau_grid = np.linspace(0, beta, N, endpoint=False)
    dtau     = beta / N

    # Fourier matrices with the +i w tau convention
    M_iw2tau = np.exp(+1j * np.outer(omega_n, tau_grid)).T / beta  # (N_tau, N_freq) #. T means transpose, np.outer takes every combo 
    M_tau2iw = np.exp(-1j * np.outer(tau_grid, omega_n)).T * dtau  # (N_freq, N_tau)

    G_LL_iw = 1.0 / (1j * omega_n)   # free propagator seed

    print(f"\nMQ Matsubara SD: J={J}, mu={mu}, beta={beta}, q={q}")
    print(f"{'Iter':>5}  {'||delta G_LL||':>15}  {'G_LL(b/2)':>12}  {'G_LL(0+)':>10}")
    print("-" * 50)

    for it in range(max_iter):
        G_LL_tau  = np.real(M_iw2tau @ G_LL_iw)

        # Diagonal self-energy: SYK4 on each dot independently
        Sigma_LL_tau = J**2 * ( G_LL_tau)**(q - 1)
        Sigma_LL_iw  = M_tau2iw @ Sigma_LL_tau

        # 2x2 Dyson: mu enters as off-diagonal mass, not self-energy
        A_iw     = 1j * omega_n - Sigma_LL_iw
        D_iw     = A_iw**2 - mu**2
        G_LL_new = A_iw / D_iw

        delta    = np.mean(np.abs(G_LL_new - G_LL_iw))
        G_LL_iw  = (1.0 - mixing) * G_LL_iw + mixing * G_LL_new

        if it % iterations == 0 or delta < tol:
            G_now = np.real(M_iw2tau @ G_LL_iw)
            print(f"{it:5d}  {delta:15.3e}  {G_now[N_half]:12.6f}  {G_now[1]:10.5f}")

        if delta < tol:
            print(f"\nConverged at iteration {it}.")
            break
    else:
        print(f"\nDid not converge within {max_iter} iterations.")

    # Final G_LL and Sigma_LL
    G_LL_tau     = np.real(M_iw2tau @ G_LL_iw)
    Sigma_LL_tau = J**2 * ( G_LL_tau)**(q - 1)
    Sigma_LL_iw  = M_tau2iw @ Sigma_LL_tau

    # G_LR from Dyson: no loop needed, fully determined by G_LL
    A_iw     = 1j * omega_n - Sigma_LL_iw
    D_iw     = A_iw**2 - mu**2
    G_LR_iw  = (-1j * mu) / D_iw
    G_LR_tau = M_iw2tau @ G_LR_iw   # purely imaginary at convergence

    print(f"\nFinal checks:")
    print(f"  G_LL(0+)      = {G_LL_tau[1]:.6f}  (should be ~+0.5)")
    print(f"  G_LL(beta/2)  = {G_LL_tau[N_half]:.6f}")
    print(f"  max|Re G_LR|  = {np.max(np.abs(np.real(G_LR_tau))):.2e}  (should be ~0)")
    print(f"  iG_LR(0+)     = {np.imag(G_LR_tau[1]):.6f}")

    return tau_grid, G_LL_tau, G_LR_tau, omega_n, Sigma_LL_iw


def _fd_factor(beta, omega):
    # 1/(exp(-beta omega)+1) = 1 - nF, stable
    x = np.clip(-beta * omega, -500, 500)
    return 1.0 / (np.exp(x) + 1.0)


def wightman_from_GR_mq(omega, GR, beta):
    """Frequency-space G^>(omega) from the MQ/Milekhin component FDT."""
    fac = _fd_factor(beta, omega)
    Ggt_w = np.zeros_like(GR, dtype=complex)

    # diagonal FDT, eq. (102)
    Ggt_w[:, 0, 0] = fac * (2j * np.imag(GR[:, 0, 0]))
    Ggt_w[:, 1, 1] = fac * (2j * np.imag(GR[:, 1, 1]))

    # off-diagonal FDT: eq. (103)
    Ggt_w[:, 0, 1] = fac * (1 * np.real(GR[:, 0, 1]))
    Ggt_w[:, 1, 0] = fac * (1 * np.real(GR[:, 0, 1]))
    return Ggt_w

def retarded_self_energy_from_greater(Sgt, t):
    i0 = int(np.argmin(np.abs(t)))
    theta = np.zeros_like(t, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5
    Sless = -np.transpose(Sgt[::-1], (0, 2, 1))
    return  theta[:, None, None] * (Sgt - Sless)  # note -i factor, -1j *theta[:, None, None] * (Sgt - Sless)
def _interp_complex_to_grid(x_old, y_old, x_new, kind="linear"):
    """
    Interpolate a complex-valued function y_old(x_old) onto x_new.
    """
    x_old = np.asarray(x_old)
    y_old = np.asarray(y_old)

    f_re = interp1d(
        x_old, np.real(y_old),
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )
    f_im = interp1d(
        x_old, np.imag(y_old),
        kind=kind,
        bounds_error=False,
        fill_value="extrapolate",
    )

    return f_re(x_new) + 1j * f_im(x_new)


def _build_GR0_mq_seed(
    omega,
    eta_ret,
    mu,
    *,
    init_mode="free",
    init_omega=None,
    init_GR_LL=None,
    init_GR_RR=None,
    init_offdiag="zero",
):
    """
    Build initial 2x2 retarded Green function GR0[k,a,b].

    init_mode options:
        "free":
            GR0 = [(w+i eta0) I]^{-1}

        "uncoupled_syk":
            Use supplied single-SYK GR for LL/RR.

    init_offdiag options for init_mode="uncoupled_syk":
        "zero":
            GR_LR = GR_RL = 0

        "linear_mu":
            GR_LR ~ g_L h_LR g_R, linear in mu

        "dyson_mu":
            GR0 = [diag(g_L^{-1}, g_R^{-1}) - h]^{-1},
            i.e. resum the bilinear mu in the seed.
    """
    omega = np.asarray(omega)
    Nw = len(omega)

    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)
    eta0 = max(eta_ret, 0.05)

    GR0 = np.zeros((Nw, 2, 2), dtype=complex)

    if init_mode == "free":
        for k, w in enumerate(omega):
            GR0[k] = np.linalg.inv((w + 1j * eta0) * eye)
        return GR0

    if init_mode != "uncoupled_syk":
        raise ValueError(
            "init_mode must be 'free' or 'uncoupled_syk', "
            f"got {init_mode!r}"
        )

    if init_omega is None or init_GR_LL is None:
        raise ValueError(
            "For init_mode='uncoupled_syk', pass init_omega and init_GR_LL."
        )

    if init_GR_RR is None:
        init_GR_RR = init_GR_LL

    gL = _interp_complex_to_grid(init_omega, init_GR_LL, omega)
    gR = _interp_complex_to_grid(init_omega, init_GR_RR, omega)

    if init_offdiag == "zero":
        GR0[:, 0, 0] = gL
        GR0[:, 1, 1] = gR
        GR0[:, 0, 1] = 0.0
        GR0[:, 1, 0] = 0.0

    elif init_offdiag == "linear_mu":
        GR0[:, 0, 0] = gL
        GR0[:, 1, 1] = gR

        # Since Dyson is D = D0 - h, inverse to first order is
        # G ≈ G0 + G0 h G0.
        GR0[:, 0, 1] = gL * h[0, 1] * gR
        GR0[:, 1, 0] = gR * h[1, 0] * gL

    elif init_offdiag == "dyson_mu":
        for k in range(Nw):
            D = np.array(
                [
                    [1.0 / gL[k], 0.0],
                    [0.0, 1.0 / gR[k]],
                ],
                dtype=complex,
            ) - h
            GR0[k] = np.linalg.inv(D)

    else:
        raise ValueError(
            "init_offdiag must be 'zero', 'linear_mu', or 'dyson_mu', "
            f"got {init_offdiag!r}"
        )

    return GR0

def solve_equilibrium_mq_real_time_matrix_v2(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-8,
    mixing=0.02,
    eta_ret=1e-4,
    verbose_every=25,
    Ggt_init=None,
    t_init=None,
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ using FDT.

    Returns:
        omega, t, Ggt_t, GR_w, Ggt_w
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    # Use an odd number of time points centered on zero.
    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]

    omega = np.linspace(-omega_max, omega_max, Nw)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)

    # Initial guess: noninteracting coupled Majoranas, broadened by eta_ret.
    if Ggt_init is not None and t_init is not None:
        Ggt = np.empty((len(t), 2, 2), dtype=complex)
        for a in range(2):
            for b in range(2):
                f = interp1d(t_init, Ggt_init[:, a, b], kind="cubic",
                             bounds_error=False, fill_value="extrapolate")
                Ggt[:, a, b] = f(t)
    else:
        GR0 = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            #GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye - h)
            GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye +h)
        Ggt_w0 = wightman_from_GR_mq(omega, GR0, beta)
        Ggt = _omega_to_time_mat(Ggt_w0, omega, t)
    print("Initial Ggt_w_LR at k=Nw//2+100:", Ggt_w0[Nw//2+100, 0, 1])
    print("Initial Ggt_LR t=0:", Ggt[len(t)//2, 0, 1])
    print("Initial Ggt_LR t=dt:", Ggt[len(t)//2+1, 0, 1])
    print("Initial GR0_LR at k=Nw//2+100:", GR0[Nw//2+100, 0, 1])

    GR = None
    Ggt_w = None
    for it in range(max_iter):
        Sgt = -J**2 * Ggt**3
        #Sgt = -J**2 * Ggt**3 + h
        Sigma_R_t = retarded_self_energy_from_greater(Sgt, t)
        Sigma_R_w = _time_to_omega_mat(Sigma_R_t, t, omega)

        GR_new = np.empty((Nw, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_new[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - Sigma_R_w[k])

        Ggt_w_new = wightman_from_GR_mq(omega, GR_new, beta)
        Ggt_new = _omega_to_time_mat(Ggt_w_new, omega, t)

        # enforce exact equal-time Majorana normalization on the diagonal only;
        # do not touch LR/RL.
        Ggt_new[i0, 0, 0] = -0.5j
        Ggt_new[i0, 1, 1] = -0.5j

        delta = np.max(np.abs(Ggt_new - Ggt))
        Ggt = (1.0 - mixing) * Ggt + mixing * Ggt_new
        GR = GR_new
        Ggt_w = Ggt_w_new

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            B = -np.real(Ggt[i0, 0, 1])
            print(f"{it:5d}  max|dG|={delta:.3e}  iGLL0={1j*Ggt[i0,0,0]:.8f}  B=Re GLR0={-B:.8f}")
        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    return omega, t, Ggt, GR, Ggt_w

def _ds_residual_mq(F_t, iSgt, t, omega, eta_ret, h, eye, one_minus_nF, i0):
    """
    Compute the DS/convergence residual d_ab as in Eq. (104) of the paper.

    The paper defines (Eq. 104, translated to real-time / F = iG^> language):

        d_ab = (1/Npoints) * || d_tau F_ab(tau) - (iSigma^R * F)_ab(tau) - (F * iSigma^A)_ab(tau)
                                - i h_ac F_cb(tau) + delta_ab delta(tau) ||_2

    In real time with F = iG^>:
        i d_t G^>_ab = h_ac G^>_cb + (Sigma^R * G^>)_ab + (Sigma^> * G^A)_ab     [Eq. 53, left branch]

    Multiplying through by i:
        -d_t F_ab = i h_ac F_cb + (iSigma^R * F)_ab + (iSigma^> * iG^A)_ab

    In equilibrium, the FDT ties together iG^> = F and G^A, so we check
    self-consistency in frequency space where the DS equation is algebraic:

        [(omega - h - Sigma^R) G^R]_ab = delta_ab   (Dyson equation)

    The most natural residual matching Eq. (104) is therefore the frequency-domain
    DS discrepancy for the diagonal spectral function A_aa = -2 Im G^R_aa
    (which should integrate to 1/(2pi) by the sum rule) and the time-domain
    discrepancy of the DS equation of motion for F, measured per-point as:

        d_aa  = (1/Nt) * || d_t F_aa - [iSR * F]_aa ||_2   (diagonal)
        d_LR  = (1/Nt) * || d_t F_LR - i*mu*F_RL - [iSR * F]_LR ||_2   (off-diagonal)

    Here we evaluate a simpler but faithful proxy: the RMS of
        residual_ab(t) = (d/dt - i*h) F_ab(t)  -  (iSigma^R * F)_ab(t)
    computed in frequency space where convolution is multiplication:
        residual_ab(omega) = (-i*omega - h) F_ab(omega) - iSigma^R(omega) F_ab(omega)
                           = [(-i*omega)*eye - h - iSR_w] F_w  +  F_w  (from the Dyson identity = eye)
    This reduces to checking how well (omega*eye - h - SR_w) G_R = eye, which is
    exact by construction. Instead we directly measure the DS equation violation
    in time: for each ab component we compute

        d_ab = (1/Nt) * sum_t |d_t F_ab(t) - i*h_ac F_cb(t) - (iSR * F)_ab(t)|

    which we approximate via finite differences for d_t and trapezoidal convolution
    for (iSR * F).  To keep cost low we evaluate it only at the *current* iterate.
    """
    Nt = len(t)
    dt = t[1] - t[0]

    # Finite-difference time derivative of F_t: shape (Nt, 2, 2)
    # Use centred differences in the interior, one-sided at boundaries.
    dF = np.empty_like(F_t)
    dF[1:-1] = (F_t[2:] - F_t[:-2]) / (2 * dt)
    dF[0]    = (F_t[1]  - F_t[0])   / dt
    dF[-1]   = (F_t[-1] - F_t[-2])  / dt

    # i h F term: (i h)_ac F_cb  -- h is already i*mu_matrix so i*h = i*(i*mu_mat) = -mu_mat
    # actually h = h_mu_matrix(mu) = i * [[0,mu],[-mu,0]], so
    # the contribution to d_t F from the mu term is  i*h @ F = i*(i*mu_mat) @ F = -mu_mat @ F.
    # But the full EOM is  d_t F = i*h @ F + (iSR * F),
    # so residual = d_t F - i*h @ F - (iSR * F).
    ih = 1j * h  # shape (2,2) -- this is -mu_matrix, but let's keep it general

    # (iSR * F) convolution in time: (iSR * F)_ab(t1) = int dt' iSR_ac(t1-t') F_cb(t')
    # We only need the diagonal strip so approximate by a simple running trapezoid sum.
    # For the residual we use only the causal (t1>=t') part since iSR is retarded.
    w = _trap_weights_uniform_from_dx(Nt, dt)
    iSR_t = iSgt.copy()  # shape (Nt, 2, 2) -- we reuse the already computed iSgt here
    # Build iSigma^R in time (retarded part only)
    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5
    iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))
    iSR_t_full = theta[:, None, None] * (iSgt + iSgt_flipped)

    conv = np.zeros((Nt, 2, 2), dtype=complex)
    for n in range(Nt):
        # causal sum: t' from t[0] to t[n] where t[n] >= t'
        # iSR(t[n]-t') is nonzero for t[n]-t' >= 0, i.e. t' <= t[n]
        n_causal = n + 1
        for m in range(n_causal):
            dt_rel = t[n] - t[m]
            # find index in iSR_t_full closest to dt_rel
            idx = int(round((dt_rel - t[0]) / dt))
            idx = max(0, min(Nt - 1, idx))
            conv[n] += w[m] * (iSR_t_full[idx] @ F_t[m])

    residual = dF - (ih[None, :, :] @ F_t) - conv
    Npoints = Nt

    # d_ab per component (Eq. 104 spirit: RMS over time points)
    d = {}
    for a, aname in enumerate(['L', 'R']):
        for b, bname in enumerate(['L', 'R']):
            d[f'd_{aname}{bname}'] = np.sqrt(np.mean(np.abs(residual[:, a, b])**2))

    return d


def _enforce_spectral(A, omega):
    """
    Enforce the three exact symmetry constraints on a diagonal SYK spectral function:
      1. Even:        A(omega) = A(-omega)
      2. Non-negative: A(omega) >= 0
      3. Sum rule:    int A(omega) dw/(2pi) = 1

    These are exact properties of A_aa = -2 Im G^R_aa for any single SYK or MQ diagonal
    component.  Enforcing them at each iteration mirrors what solve_equilibrium_greater_real_time
    does in the single-SYK notebook and prevents the spectral function drifting negative
    or off the sum rule, which would destabilise the iteration.
    """
    # 1. Symmetrise (A should be exactly even; any asymmetry is numerical noise)
    A = 0.5 * (A + A[::-1])
    # 2. Clip (causality / positivity of the spectral function)
    A = np.clip(A, 0.0, None)
    # 3. Normalise (sum rule: fermion anti-commutator = 1 in units where {psi,psi}=1)
    dw = omega[1] - omega[0]
    sum_A = (np.sum(A) - 0.5 * A[0] - 0.5 * A[-1]) * dw / (2.0 * np.pi)  # trapezoid
    if sum_A > 0:
        A = A / sum_A
    return A


def _enforce_odd(B):
    """
    Enforce exact odd symmetry on the off-diagonal "spectral" function
    B_LR(omega) = Re G^R_LR(omega).

    This follows from G^R_LR(omega)^* = -G^R_LR(-omega) (consequence of the Z4
    symmetry psi_L -> -psi_R, psi_R -> psi_L combined with the retarded GF reality
    condition), which forces Re G^R_LR to be odd and Im G^R_LR to be even.
    Any even contamination in Re G^R_LR is numerical noise and should be projected out.
    """
    return 0.5 * (B - B[::-1])

def _residual_even(B):
    """
    Checking even contamination in Re G^R_LR.
    """
    return 0.5 * (B + B[::-1])

def solve_equilibrium_mq_matrix_F(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-9,
    mixing=0.05,
    eta_ret=1e-6,
    verbose_every=25,
    compute_kbe_dab_every=50,
    kbe_dab_t_cut=None,
    kbe_dab_edge_skip=4,
    kbe_dab_power=0.5,
    return_dab_history=False,    
    init_mode="free",
    init_omega=None,
    init_GR_LL=None,
    init_GR_RR=None,
    init_offdiag="zero",
    project_A=True,
    clip_A=True,
    normalize_A=True,
    project_B=True,
    pin_equal_time=True,    
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ q=4.

    Same as before, but now also computes the actual real-time KBE residual
    using calc_kbe_d_ab_mq_equilibrium every `compute_kbe_dab_every` iterations.

    The printed quantity is d_ab**kbe_dab_power. Default kbe_dab_power=0.5
    prints the RMS residual, since calc_kbe_d_ab_mq_equilibrium returns mean |res|^2.

    Returns:
        omega, t, Ggt, GR_w, F_t

    If return_dab_history=True:
        omega, t, Ggt, GR_w, F_t, dab_history
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]
    Nt = len(t)
    Nw_ = Nw

    omega = np.linspace(-omega_max, omega_max, Nw_)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)

    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    one_minus_nF = 1.0 - nF

    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5

    # ---- initial guess ----
    GR0 = _build_GR0_mq_seed(
    omega=omega,
    eta_ret=eta_ret,
    mu=mu,
    init_mode=init_mode,
    init_omega=init_omega,
    init_GR_LL=init_GR_LL,
    init_GR_RR=init_GR_RR,
    init_offdiag=init_offdiag,)

    #A_LL_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 0, 0]), omega)
    #A_RR_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 1, 1]), omega)
    #B_LR_init = _enforce_odd(np.real(GR0[:, 0, 1]))
    #B_RL_init = _enforce_odd(np.real(GR0[:, 1, 0]))
    A_LL_init = -2.0 * np.imag(GR0[:, 0, 0])
    A_RR_init = -2.0 * np.imag(GR0[:, 1, 1])
    B_LR_init = 2*np.real(GR0[:, 0, 1])
    B_RL_init = 2*np.real(GR0[:, 1, 0])
    

    F_w = np.zeros((Nw_, 2, 2), dtype=complex)
    F_w[:, 0, 0] = one_minus_nF * A_LL_init
    F_w[:, 1, 1] = one_minus_nF * A_RR_init
    F_w[:, 0, 1] = 1j * one_minus_nF * B_LR_init
    F_w[:, 1, 0] = 1j * one_minus_nF * B_RL_init

    F_t = _omega_to_time_mat(F_w, omega, t)

    GR_w = None
    delta = np.inf
    dab_history = []

    for it in range(max_iter):

        # i Sigma^>_ab(t) = J^2 F_ab(t)^3
        iSgt = J**2 * F_t**3

        # i Sigma^R_ab(t) = theta(t)[i Sigma^>_ab(t) + i Sigma^>_ba(-t)]
        iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))
        iSR_t = theta[:, None, None] * (iSgt + iSgt_flipped)

        iSR_w = _time_to_omega_mat(iSR_t, t, omega)
        SR_w = -1j * iSR_w

        # Dyson equation
        GR_w = np.empty((Nw_, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

        # FDT
        #A_LL_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 0, 0]), omega)
        #A_RR_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 1, 1]), omega)
        #B_LR_new = _enforce_odd(np.real(GR_w[:, 0, 1]))
        #B_LR_res = _residual_even(np.real(GR_w[:, 0, 1]))
        #B_RL_new = _enforce_odd(np.real(GR_w[:, 1, 0]))
        A_LL_new = -2.0 * np.imag(GR_w[:, 0, 0])
        A_RR_new = -2.0 * np.imag(GR_w[:, 1, 1])
        B_LR_new = 2*np.real(GR_w[:, 0, 1])
        B_RL_new = 2*np.real(GR_w[:, 1, 0])

        if project_A:
            A_LL_new = 0.5 * (A_LL_new + A_LL_new[::-1])
            A_RR_new = 0.5 * (A_RR_new + A_RR_new[::-1])

        if clip_A:
            A_LL_new = np.clip(A_LL_new, 0.0, None)
            A_RR_new = np.clip(A_RR_new, 0.0, None)

        if normalize_A:
            sum_LL = np.trapz(A_LL_new, omega) / (2*np.pi)
            sum_RR = np.trapz(A_RR_new, omega) / (2*np.pi)
            if sum_LL > 0:
                A_LL_new = A_LL_new / sum_LL
            if sum_RR > 0:
                A_RR_new = A_RR_new / sum_RR

        if project_B:
            B_LR_new = 0.5 * (B_LR_new - B_LR_new[::-1])
            B_RL_new = 0.5 * (B_RL_new - B_RL_new[::-1])

        F_w_new = np.zeros((Nw_, 2, 2), dtype=complex)
        F_w_new[:, 0, 0] = one_minus_nF * A_LL_new
        F_w_new[:, 1, 1] = one_minus_nF * A_RR_new
        F_w_new[:, 0, 1] = 1j * one_minus_nF * B_LR_new
        F_w_new[:, 1, 0] = 1j * one_minus_nF * B_RL_new

        F_t_new = _omega_to_time_mat(F_w_new, omega, t)

        # Equal-time pins
        #F_t_new[i0, 0, 0] = 0.5
        #F_t_new[i0, 1, 1] = 0.5
        #F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
        #F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])
        if pin_equal_time:
            F_t_new[i0, 0, 0] = 0.5
            F_t_new[i0, 1, 1] = 0.5
            F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
            F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])

        delta = np.max(np.abs(F_t_new - F_t))

        # Mix in time domain
        F_t = (1.0 - mixing) * F_t + mixing * F_t_new

        # Convert current mixed iterate to G^>
        Ggt_current = -1j * F_t

        # ---- actual real-time KBE residual d_ab ----
        do_kbe_dab = (
            compute_kbe_dab_every is not None
            and compute_kbe_dab_every > 0
            and (it % compute_kbe_dab_every == 0 or delta < tol)
        )

        dab_info = None
        if do_kbe_dab:
            dab_info = calc_kbe_d_ab_mq_equilibrium(
                t=t,
                Ggt=Ggt_current,
                J=J,
                mu=mu,
                t_cut=kbe_dab_t_cut,
                edge_skip=kbe_dab_edge_skip,
                keep_cross_self_energy=True,
                return_details=True,
            )

            d_ab = dab_info["d_ab"]
            d_t1_ab = dab_info["d_t1_ab"]
            d_t2_ab = dab_info["d_t2_ab"]

            d_print = d_ab**kbe_dab_power
            d_t1_print = d_t1_ab**kbe_dab_power
            d_t2_print = d_t2_ab**kbe_dab_power

            dab_history.append({
                "it": it,
                "delta": delta,
                "d_ab": d_ab.copy(),
                "d_t1_ab": d_t1_ab.copy(),
                "d_t2_ab": d_t2_ab.copy(),
                "d_print": d_print.copy(),
                "d_t1_print": d_t1_print.copy(),
                "d_t2_print": d_t2_print.copy(),
            })

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            B = np.imag(F_t[i0, 0, 1])
            msg = (
                f"{it:5d}  max|dF|={delta:.3e}  "
                f"F_LL(0)={F_t[i0,0,0].real:.8f}  "
                f"B=ImF_LR(0)={B:.8f}"
            )

            if dab_info is not None:
                d_print = dab_info["d_ab"]**kbe_dab_power
                msg += (
                    f"\n       KBE d_ab^{kbe_dab_power:g} = "
                    f"[[{d_print[0,0]:.3e}, {d_print[0,1]:.3e}], "
                    f"[{d_print[1,0]:.3e}, {d_print[1,1]:.3e}]]"
                )

            print(msg)

        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    Ggt = -1j * F_t

    print(f"\nFinal: G^>_LL(0) = {Ggt[i0,0,0]:.6f}  (should be -0.5i)")
    print(f"Final: G^>_LR(0) = {Ggt[i0,0,1]:.6f}  (should be purely real)")
    B_final = np.imag(F_t[i0, 0, 1])
    print(f"Final: B = Im F_LR(0) = -Re G^>_LR(0) = {B_final:.8f}")

    A_LL_final = -2.0 * np.imag(GR_w[:, 0, 0])
    A_RR_final = -2.0 * np.imag(GR_w[:, 1, 1])
    dw = omega[1] - omega[0]

    def _trap_sum(A):
        return (np.sum(A) - 0.5*A[0] - 0.5*A[-1]) * dw / (2.0*np.pi)

    print(f"Final: int A_LL dw/(2pi) = {_trap_sum(A_LL_final):.6f}  (should be 1)")
    print(f"Final: int A_RR dw/(2pi) = {_trap_sum(A_RR_final):.6f}  (should be 1)")

    B_LR_final = np.real(GR_w[:, 0, 1])
    print(
        "Final: B_LR odd-symmetry max|B(w)+B(-w)| = "
        f"{np.max(np.abs(B_LR_final + B_LR_final[::-1])):.2e}  (should be ~0)"
    )
    print(f"Final: mean|dF| = {delta:.3e}  (converged when < tol={tol:.0e})")

    if dab_history:
        last = dab_history[-1]
        print(f"Final: KBE d_ab^{kbe_dab_power:g} =")
        print(last["d_print"])

    if return_dab_history:
        return omega, t, Ggt, GR_w, F_t, dab_history

    return omega, t, Ggt, GR_w, F_t

def solve_equilibrium_mq_matrix_Fv1(
    J,
    mu,
    beta,
    omega_max=8.0,
    Nw=4097,
    t_max=None,
    dt=0.1,
    max_iter=2000,
    tol=1e-9,
    mixing=0.05,
    eta_ret=1e-6,
    verbose_every=25,
):
    """
    Equilibrium real-time DS iteration for coupled SYK/MQ q=4.

    Works in terms of F_ab = i * G^>_ab.

    Conventions:
        F_aa(t=0)  = +0.5       [real; from G^>_aa(0) = -i/2]
        F_LR(t)    = purely imaginary at all t
                     [since G^>_LR(t) is real: G^>_LR(t)* = G^>_LR(-t) by Z4 + Majorana]
        F_LR(t=0)  = 0          [real part is zero; imaginary part = B = <psi_L psi_R>]

        i*Sigma^>_ab  = J^2 * F_ab^3
        i*Sigma^R_ab(t) = theta(t)[i*Sigma^>_ab(t) + i*Sigma^>_ba(-t)]
                          (off-diagonal: ba-transpose on the time-reversed term)

        FDT diagonal (eq. 102):
            G^>_aa(omega) = 2i Im G^R_aa(omega) * (1-nF)
            F_aa(omega)   = (1-nF) * A_aa(omega)
            A_aa          = -2 Im G^R_aa  [real, EVEN, NON-NEGATIVE, int dw/(2pi)=1]
            Stabilising:  enforce even + clip>=0 + normalise (as in single-SYK notebook)

        FDT off-diagonal (eq. 103):
            G^>_LR(omega) = Re G^R_LR(omega) * (1-nF)     [no factor of 2]
            F_LR(omega)   = 1j * (1-nF) * B_LR(omega)     [purely imaginary in omega]
            B_LR          = Re G^R_LR                     [real, ODD]
            Stabilising:  enforce odd (anti-symmetrise)

    Returns:
        omega, t, Ggt, GR_w, F_t
    """
    if t_max is None:
        t_max = max(120.0, 6.0 * beta)

    t = np.arange(-t_max, t_max + 0.5 * dt, dt)
    i0 = int(np.argmin(np.abs(t)))
    t = t - t[i0]
    Nt = len(t)
    Nw_ = Nw

    omega = np.linspace(-omega_max, omega_max, Nw_)
    eye = np.eye(2, dtype=complex)
    h = h_mu_matrix(mu)  # h = i * [[0,mu],[-mu,0]]

    # Fermi factor: nF(omega) = 1/(exp(beta*omega)+1)
    nF = 1.0 / (np.exp(np.clip(beta * omega, -500, 500)) + 1.0)
    one_minus_nF = 1.0 - nF  # shape (Nw,)

    # theta function for retarded
    theta = np.zeros(Nt, dtype=float)
    theta[t > 0.0] = 1.0
    theta[i0] = 0.5

    # ---- initial guess: noninteracting F from GR0 ----
    # We start from the free (non-interacting but coupled) retarded Green function,
    # which is the 2x2 matrix inverse of (omega + i*eta - h).
    # From GR0 we extract:
    #   A_aa(omega) = -2 Im GR0_aa(omega)  [diagonal spectral function]
    #   B_LR(omega) =  2 Re GR0_LR(omega)  [off-diagonal "spectral" function for FDT eq.(103)]
    # and then set F_w via FDT:
    #   F_aa(omega) = (1-nF) * A_aa(omega)
    #   F_LR(omega) = (1-nF) * B_LR(omega)
    #
    # This is analogous to what solve_equilibrium_greater_real_time does for the
    # single SYK, where the initial A(omega) is a Lorentzian (the free spectral
    # function).  The MQ version must use the 2x2 matrix to correctly seed the
    # off-diagonal components.
    GR0 = np.empty((Nw_, 2, 2), dtype=complex)
    for k, w in enumerate(omega):
        GR0[k] = np.linalg.inv((w + 1j * max(eta_ret, 0.05)) * eye - 0*h)

    # F_ab(omega) initial guess from FDT
    #   Diagonal:    F_aa(omega) = (1-nF) * A_aa(omega),  A_aa = -2 Im G^R_aa  [real, even, >=0]
    #   Off-diagonal: F_LR(omega) = 1j * (1-nF) * B_LR(omega),  B_LR = Re G^R_LR [real, odd]
    #     The factor of 1j makes F_LR(t) purely imaginary in time, consistent with
    #     G^>_LR(t) being real (paper eq 11 + Majorana symmetry).  No factor of 2 for off-diag.
    A_LL_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 0, 0]), omega)
    A_RR_init = _enforce_spectral(-2.0 * np.imag(GR0[:, 1, 1]), omega)
    B_LR_init = _enforce_odd(np.real(GR0[:, 0, 1]))
    B_RL_init = _enforce_odd(np.real(GR0[:, 1, 0]))

    F_w = np.zeros((Nw_, 2, 2), dtype=complex)
    F_w[:, 0, 0] = one_minus_nF * A_LL_init
    F_w[:, 1, 1] = one_minus_nF * A_RR_init
    F_w[:, 0, 1] = 1j * one_minus_nF * B_LR_init
    F_w[:, 1, 0] = 1j * one_minus_nF * B_RL_init

    # F_t = i G^> in time domain
    F_t = _omega_to_time_mat(F_w, omega, t)

    GR_w = None

    for it in range(max_iter):

        # i Sigma^>_ab(t) = J^2 * F_ab(t)^3
        # Derivation: G^> = -iF  and Sigma^> = -J^2 (G^>)^3
        iSgt = J**2 * F_t**3 #- np.array([[0.0, mu], [-mu, 0.0]], dtype=complex)

        # ------------------------------------------------------------------ #
        # Build i Sigma^R(t) = theta(t) * [i Sigma^>(t) - i Sigma^<(t)]
        #
        # The lesser self-energy satisfies (from the Majorana/Keldysh identity):
        #     i Sigma^<_ab(t) = -i Sigma^>_ba(-t)
        #
        # So:
        #     i Sigma^>(t) - i Sigma^<(t) = i Sigma^>_ab(t) + i Sigma^>_ba(-t)
        #
        # For DIAGONAL components (a=b): Sigma^>_aa(-t) is just iSgt[::-1] evaluated at -t.
        # For OFF-DIAGONAL components: we need Sigma^>_ba(-t), i.e. the TRANSPOSED
        # component evaluated at -t.  This is why we need iSgt_flipped, not iSgt[::-1].
        #
        # iSgt_flipped[n, a, b] = iSgt[Nt-1-n, b, a]  = i Sigma^>_ba(-t[n])
        # ------------------------------------------------------------------ #
        iSgt_flipped = np.transpose(iSgt[::-1], (0, 2, 1))   # shape (Nt, 2, 2)
        iSR_t = theta[:, None, None] * (iSgt + iSgt_flipped)

        # Fourier to frequency: i Sigma^R(omega)
        iSR_w = _time_to_omega_mat(iSR_t, t, omega)

        # Sigma^R(omega) = iSR_w / i = -i * iSR_w
        SR_w = -1j * iSR_w

        # Dyson equation: G^R = (omega + i*eta - h - Sigma^R)^{-1}
        GR_w = np.empty((Nw_, 2, 2), dtype=complex)
        for k, w in enumerate(omega):
            GR_w[k] = np.linalg.inv((w + 1j * eta_ret) * eye - h - SR_w[k])

        # FDT to get new F_w  (Eqs. 102-103 of paper)
        #
        #   Diagonal (eq. 102):
        #     G^>_aa(omega) = 2i Im G^R_aa(omega) * (1-nF)
        #     F_aa(omega) = i*G^>_aa = (1-nF)*(-2 Im G^R_aa) = (1-nF)*A_aa
        #     A_aa = -2 Im G^R_aa  [real, even, non-negative, int dw/(2pi) = 1]
        #
        #   Off-diagonal (eq. 103):
        #     G^>_LR(omega) = Re G^R_LR(omega) * (1-nF)     [NO factor of 2]
        #     F_LR(omega) = i*G^>_LR = i*(1-nF)*Re G^R_LR   [purely imaginary in omega]
        #     => F_LR(t) = IFT(i*real_odd) = purely imaginary in time  ✓
        #     B_LR = Re G^R_LR  [real, odd]
        #
        # After extracting A_aa and B_LR, apply the same stabilising projections
        # that solve_equilibrium_greater_real_time uses for the single SYK:
        #   A_aa: symmetrise (even), clip (>=0), normalise (sum rule)
        #   B_LR: anti-symmetrise (odd)

        A_LL_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 0, 0]), omega)
        A_RR_new = _enforce_spectral(-2.0 * np.imag(GR_w[:, 1, 1]), omega)
        B_LR_new = _enforce_odd(np.real(GR_w[:, 0, 1]))
        B_LR_res = _residual_even(np.real(GR_w[:, 0, 1]))
        B_RL_new = _enforce_odd(np.real(GR_w[:, 1, 0]))

        F_w_new = np.zeros((Nw_, 2, 2), dtype=complex)
        F_w_new[:, 0, 0] = one_minus_nF * A_LL_new
        F_w_new[:, 1, 1] = one_minus_nF * A_RR_new
        F_w_new[:, 0, 1] = 1j * one_minus_nF * B_LR_new
        F_w_new[:, 1, 0] = 1j * one_minus_nF * B_RL_new

        F_t_new = _omega_to_time_mat(F_w_new, omega, t)

        # Equal-time pins:
        #   Diagonal:    F_aa(t=0) = +0.5  [from G^>_aa(0) = -i/2]
        #   Off-diagonal: F_LR(t=0) = 0    [F_LR(t) is purely imaginary, so Re F_LR(0) = 0;
        #                                    the imaginary part carries the physical B value]
        F_t_new[i0, 0, 0] = 0.5
        F_t_new[i0, 1, 1] = 0.5
        # Off-diagonal: zero the real part at t=0 (it should be exactly zero by symmetry)
        F_t_new[i0, 0, 1] = 1j * np.imag(F_t_new[i0, 0, 1])
        F_t_new[i0, 1, 0] = 1j * np.imag(F_t_new[i0, 1, 0])

        # ---- convergence metric (mean absolute change, matching single-SYK convention) ----
        delta = np.mean(np.abs(F_t_new - F_t))

        # mix in time domain
        F_t = (1.0 - mixing) * F_t + mixing * F_t_new

        # ---- DS self-consistency residual (Eq. 104 analogue) ----

        if verbose_every and (it % verbose_every == 0 or delta < tol):
            # B = -Re G^>_LR(0) = Im F_LR(0)  [since F_LR(t) purely imaginary]
            B = np.imag(F_t[i0, 0, 1])
            #dab_str = f"  sc_LL={d_LL:.2e}  sc_LR={d_LR:.2e}" if np.isfinite(d_LL) else ""
            #print(f"{it:5d}  mean|dF|={delta:.3e}  F_LL(0)={F_t[i0,0,0].real:.8f}  B=ImF_LR(0)={B:.8f}{dab_str}")
            print(f"{it:5d}  mean|dF|={delta:.3e}  F_LL(0)={F_t[i0,0,0].real:.8f} B=ImF_LR(0)={B:.8f}" f"residual_even_B={np.max(np.abs(B_LR_res)):.3e}")

        if delta < tol:
            break
    else:
        print("WARNING: equilibrium iteration did not converge")

    # convert back to G^> = -i F
    Ggt = -1j * F_t

    # Final diagnostics
    print(f"\nFinal: G^>_LL(0) = {Ggt[i0,0,0]:.6f}  (should be -0.5i)")
    print(f"Final: G^>_LR(0) = {Ggt[i0,0,1]:.6f}  (should be purely real)")
    B_final = np.imag(F_t[i0, 0, 1])
    print(f"Final: B = Im F_LR(0) = -Re G^>_LR(0) = {B_final:.8f}")

    # Spectral sum-rule and symmetry checks
    A_LL_final = -2.0 * np.imag(GR_w[:, 0, 0])
    A_RR_final = -2.0 * np.imag(GR_w[:, 1, 1])
    dw = omega[1] - omega[0]
    def _trap_sum(A):
        return (np.sum(A) - 0.5*A[0] - 0.5*A[-1]) * dw / (2.0*np.pi)
    print(f"Final: int A_LL dw/(2pi) = {_trap_sum(A_LL_final):.6f}  (should be 1)")
    print(f"Final: int A_RR dw/(2pi) = {_trap_sum(A_RR_final):.6f}  (should be 1)")
    B_LR_final = np.real(GR_w[:, 0, 1])
    print(f"Final: B_LR odd-symmetry max|B(w)+B(-w)| = {np.max(np.abs(B_LR_final + B_LR_final[::-1])):.2e}  (should be ~0)")
    print(f"Final: mean|dF| = {delta:.3e}  (converged when < tol={tol:.0e})")

    return omega, t, Ggt, GR_w, F_t


# ---------- KBE evolution ----------

def lesser_mat(G, i, j):
    return -G[j, i].T


def retarded_mat(G, i, k):
    return G[i, k] - lesser_mat(G, i, k)


def advanced_mat(G, k, j):
    return lesser_mat(G, k, j) - G[k, j]


def lesser_sigma_mat(S, i, j):
    return -S[j, i].T


def retarded_sigma_mat(S, i, k):
    return S[i, k] - lesser_sigma_mat(S, i, k)


def advanced_sigma_mat(S, k, j):
    return lesser_sigma_mat(S, k, j) - S[k, j]


def sigma_greater_mq(G, J_of_t, keep_cross_self_energy=True):
    JJ = J_of_t[:, None] * J_of_t[None, :]
    S = -JJ[:, :, None, None] * G**3
    if not keep_cross_self_energy:
        S[:, :, 0, 1] = 0.0
        S[:, :, 1, 0] = 0.0
    return S


def _trap_weights_segment(m, dt):
    return _trap_weights_uniform_from_dx(m, dt)


def rhs_t1_matrix(G, S, h_list, i, j, dt):
    conv = np.zeros((2, 2), dtype=complex)
    ks = np.arange(i + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (retarded_sigma_mat(S, i, k) @ G[k, j])

    ks = np.arange(j + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (S[i, k] @ advanced_mat(G, k, j))

    return -1j * (h_list[i] @ G[i, j] + conv)


def rhs_t2_matrix(G, S, h_list, i, j, dt):
    conv = np.zeros((2, 2), dtype=complex)
    ks = np.arange(i + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (retarded_mat(G, i, k) @ S[k, j])

    ks = np.arange(j + 1)
    ws = _trap_weights_segment(len(ks), dt)
    for w, k in zip(ws, ks):
        conv += w * (G[i, k] @ advanced_sigma_mat(S, k, j))

    return +1j * (G[i, j] @ h_list[j] + conv)


def enforce_equal_time_majorana(G, n):
    # Only exact equal-time constraints. Do not re-project off-diagonal time slices.
    G[n, n, 0, 0] = -0.5j
    G[n, n, 1, 1] = -0.5j
    # ensure anti-Hermitian equal-time matrix
    off = 0.5 * (G[n, n, 0, 1] - np.conj(G[n, n, 1, 0]))
    G[n, n, 0, 1] = off
    G[n, n, 1, 0] = -np.conj(off)


def fill_equilibrium_block(t, n0, t_eq, Ggt_eq):
    G = np.zeros((len(t), len(t), 2, 2), dtype=complex)
    interps = [[interp1d(t_eq, Ggt_eq[:, a, b], kind="cubic",
                         bounds_error=False, fill_value="extrapolate")
                for b in range(2)] for a in range(2)]
    for i in range(n0 + 1):
        for j in range(n0 + 1):
            tau = t[i] - t[j]
            for a in range(2):
                for b in range(2):
                    G[i, j, a, b] = interps[a][b](tau)
    for n in range(n0 + 1):
        enforce_equal_time_majorana(G, n)
    return G


def evolve_mq_kbe_matrix_v2(
    t_eq,
    Ggt_eq,
    beta=None,  # unused; kept for call compatibility
    J_i=1.0,
    J_f=1.0,
    mu_i=0.075,
    mu_f=0.0,
    t_pre=80.0,
    t_post=40.0,
    dt=0.1,
    n_corr=6,
    corr_tol=1e-10,
    print_every=50,
    keep_cross_self_energy=True,
):
    """Full 2x2 KBE evolution initialized directly from equilibrium Ggt_eq."""
    t = np.arange(-t_pre, t_post + 0.5 * dt, dt)
    n0 = int(np.argmin(np.abs(t)))
    t = t - t[n0]
    Nt = len(t)

    if np.max(np.abs(t[:n0+1, None] - t[None, :n0+1])) > np.max(np.abs(t_eq)):
        print("WARNING: t_eq does not cover the needed equilibrium tau range; interpolation will extrapolate.")

    G = fill_equilibrium_block(t, n0, t_eq, Ggt_eq)

    J_of_t = np.where(t < 0.0, J_i, J_f)
    mu_of_t = np.where(t < 0.0, mu_i, mu_f)
    h_list = np.array([h_mu_matrix(m) for m in mu_of_t])

    for n in range(n0 + 1, Nt):
        S = sigma_greater_mq(G, J_of_t, keep_cross_self_energy=keep_cross_self_energy)

        d1_old = np.array([rhs_t1_matrix(G, S, h_list, n - 1, j, dt) for j in range(n)])
        d2_old = np.array([rhs_t2_matrix(G, S, h_list, i, n - 1, dt) for i in range(n)])
        ddiag_old = rhs_t1_matrix(G, S, h_list, n - 1, n - 1, dt) + rhs_t2_matrix(G, S, h_list, n - 1, n - 1, dt)

        G[n, :n] = G[n - 1, :n] + dt * d1_old
        G[:n, n] = G[:n, n - 1] + dt * d2_old
        G[n, n] = G[n - 1, n - 1] + dt * ddiag_old
        enforce_equal_time_majorana(G, n)

        for _ in range(n_corr):
            row_old = G[n, :n].copy()
            col_old = G[:n, n].copy()
            diag_old = G[n, n].copy()

            S = sigma_greater_mq(G, J_of_t, keep_cross_self_energy=keep_cross_self_energy)
            d1_new = np.array([rhs_t1_matrix(G, S, h_list, n, j, dt) for j in range(n)])
            d2_new = np.array([rhs_t2_matrix(G, S, h_list, i, n, dt) for i in range(n)])
            ddiag_new = rhs_t1_matrix(G, S, h_list, n, n, dt) + rhs_t2_matrix(G, S, h_list, n, n, dt)

            G[n, :n] = G[n - 1, :n] + 0.5 * dt * (d1_old + d1_new)
            G[:n, n] = G[:n, n - 1] + 0.5 * dt * (d2_old + d2_new)
            G[n, n] = G[n - 1, n - 1] + 0.5 * dt * (ddiag_old + ddiag_new)
            enforce_equal_time_majorana(G, n)

            err = max(
                np.max(np.abs(G[n, :n] - row_old)),
                np.max(np.abs(G[:n, n] - col_old)),
                np.max(np.abs(G[n, n] - diag_old)),
            )
            if err < corr_tol:
                break

        if print_every and (n - n0) % print_every == 0:
            B = -np.real(G[n, n, 0, 1])
            print(f"evolved to t={t[n]:.3f}; B={B:.8f}")

    return t, G


def _central_derivative_1d(X, t):
    """
    Central derivative along axis 0.
    X shape: (Nt,) or (Nt,2,2)
    """
    t = np.asarray(t)
    dt = t[1] - t[0]

    dX = np.empty_like(X, dtype=complex)
    dX[1:-1] = (X[2:] - X[:-2]) / (2.0 * dt)
    dX[0] = (X[1] - X[0]) / dt
    dX[-1] = (X[-1] - X[-2]) / dt
    return dX


def _theta_plus_minus(t):
    t = np.asarray(t)
    i0 = int(np.argmin(np.abs(t)))

    theta_p = np.zeros_like(t, dtype=float)
    theta_m = np.zeros_like(t, dtype=float)

    theta_p[t > 0.0] = 1.0
    theta_m[t < 0.0] = 1.0

    theta_p[i0] = 0.5
    theta_m[i0] = 0.5

    return theta_p, theta_m, i0


def _conv_1d_same(A, B, t):
    """
    Discrete convolution

        C(t) = int du A(t-u) B(u)

    for scalar arrays on a symmetric uniform grid.
    """
    w = _trap_weights_grid(t)
    return np.convolve(B * w, A, mode="same")


def _conv_mat_same(A, B, t):
    """
    Matrix convolution

        C_ab(t) = int du A_ac(t-u) B_cb(u)

    A, B shape: (Nt,2,2)
    Returns shape: (Nt,2,2)
    """
    A = np.asarray(A)
    B = np.asarray(B)
    w = _trap_weights_grid(t)

    Nt = len(t)
    C = np.zeros((Nt, 2, 2), dtype=complex)

    for a in range(2):
        for b in range(2):
            tmp = np.zeros(Nt, dtype=complex)
            for c in range(2):
                tmp += np.convolve(B[:, c, b] * w, A[:, a, c], mode="same")
            C[:, a, b] = tmp

    return C

def calc_kbe_d_ab_mq_equilibrium(
    t,
    Ggt,
    J,
    mu,
    *,
    t_cut=None,
    edge_skip=4,
    keep_cross_self_energy=True,
    return_details=False,
):
    """
    Real-time KBE residual d_ab for equilibrium coupled SYK / MQ.

    Input:
        t    : relative-time grid
        Ggt  : G^>_ab(t), shape (Nt,2,2)
        J    : SYK coupling
        mu   : MQ bilinear coupling

    Checks both equilibrium KBE equations:

        i d_t G^>(t) =
            h G^>(t)
            + Sigma^R * G^>
            + Sigma^> * G^A

    and

        i d_t G^>(t) =
            G^>(t) h
            + G^R * Sigma^>
            + G^> * Sigma^A

    where:
        h = h_mu_matrix(mu) = i [[0, mu], [-mu, 0]]

    and all products/convolutions are matrix products in L/R space.

    Returns:
        d_ab = 0.5 * (d_t1_ab + d_t2_ab)

    with:
        d_t1_ab = mean_t |res_t1_ab(t)|^2
        d_t2_ab = mean_t |res_t2_ab(t)|^2
    """
    t = np.asarray(t)
    Ggt = np.asarray(Ggt, dtype=complex)

    if Ggt.shape != (len(t), 2, 2):
        raise ValueError(f"Ggt must have shape {(len(t), 2, 2)}, got {Ggt.shape}")

    theta_p, theta_m, i0 = _theta_plus_minus(t)

    h = h_mu_matrix(mu)

    # Lesser Green's function:
    # G^<_ab(t) = -G^>_ba(-t)
    Glt = -np.transpose(Ggt[::-1], (0, 2, 1))

    GR = theta_p[:, None, None] * (Ggt - Glt)
    GA = theta_m[:, None, None] * (Glt - Ggt)

    # Greater self-energy:
    # Sigma^>_ab(t) = -J^2 G^>_ab(t)^3
    Sgt = -J**2 * Ggt**3

    if not keep_cross_self_energy:
        Sgt[:, 0, 1] = 0.0
        Sgt[:, 1, 0] = 0.0

    # Lesser self-energy:
    # Sigma^<_ab(t) = -Sigma^>_ba(-t)
    Slt = -np.transpose(Sgt[::-1], (0, 2, 1))

    SR = theta_p[:, None, None] * (Sgt - Slt)
    SA = theta_m[:, None, None] * (Slt - Sgt)

    dG = _central_derivative_1d(Ggt, t)

    conv_t1 = _conv_mat_same(SR, Ggt, t) + _conv_mat_same(Sgt, GA, t)
    conv_t2 = _conv_mat_same(GR, Sgt, t) + _conv_mat_same(Ggt, SA, t)

    hG = np.einsum("ac,tcb->tab", h, Ggt)
    Gh = np.einsum("tac,cb->tab", Ggt, h)

    res_t1 = 1j * dG - hG - conv_t1
    res_t2 = 1j * dG - Gh - conv_t2

    mask = np.ones(len(t), dtype=bool)

    if edge_skip is not None and edge_skip > 0:
        mask[:edge_skip] = False
        mask[-edge_skip:] = False

    if t_cut is not None:
        mask &= np.abs(t) <= t_cut

    d_t1_ab = np.mean(np.abs(res_t1[mask])**2, axis=0)
    d_t2_ab = np.mean(np.abs(res_t2[mask])**2, axis=0)
    d_ab = 0.5 * (d_t1_ab + d_t2_ab)

    if not return_details:
        return d_ab

    return {
        "d_ab": d_ab,
        "d_t1_ab": d_t1_ab,
        "d_t2_ab": d_t2_ab,
        "res_t1": res_t1,
        "res_t2": res_t2,
        "Glt": Glt,
        "GR": GR,
        "GA": GA,
        "Sgt": Sgt,
        "Slt": Slt,
        "SR": SR,
        "SA": SA,
        "conv_t1": conv_t1,
        "conv_t2": conv_t2,
        "mask": mask,
    }