import numpy as np
import glob

# --- load all runs ---
files = sorted(glob.glob('lbm_run_N*.npz'))
runs = [np.load(f, allow_pickle=True) for f in files]

def band_energy(K, kL, E_dim, x_max):
    m = kL <= x_max
    frac = np.trapezoid(E_dim[m], kL[m])          # ∫ E_dim d(kL)
    return K * frac                           # band-limited energy

def band_enstrophy(K, kL, E_dim, L2D, x_max):
    m = kL <= x_max
    integral = np.trapezoid((kL[m]**2) * E_dim[m], kL[m])  # ∫ (kL)^2 E_dim d(kL)
    return (K / (L2D**2)) * integral                   # Ω = (K/L^2) ∫ (kL)^2 E_dim d(kL)

# choose reference = highest N
Ns = [int(r['Nx']) for r in runs]
ref_idx = int(np.argmax(Ns))
ref = runs[ref_idx]
N_ref = int(ref['Nx'])
print("Reference resolution:", N_ref)

#interpolate ref spectrum onto coarse kL grid (only within overlap)
def spectrum_L2_rel_err(kL_c, E_c, kL_ref, E_ref):
    kmin = max(kL_c.min(), kL_ref.min())
    kmax = min(kL_c.max(), kL_ref.max())
    mask = (kL_c >= kmin) & (kL_c <= kmax)
    if not np.any(mask):
        return np.nan
    E_ref_i = np.interp(kL_c[mask], kL_ref, E_ref)
    num = np.sum((E_c[mask] - E_ref_i)**2)
    den = np.sum(E_ref_i**2)
    return float(np.sqrt(num/den))

# gather common sample times (intersection)
common_times = set(ref['spec_t'])
for r in runs:
    common_times &= set(r['spec_t'])
common_times = sorted(int(t) for t in common_times)
print("Common sample times:", common_times)

# --- compute errors for each run vs reference ---
summary = []
for r in runs:
    N = int(r['Nx'])
    if N == N_ref:
        continue  # skip reference

    # map times -> index for spectra
    t2idx_r   = {int(t): i for i, t in enumerate(r['spec_t'])}
    t2idx_ref = {int(t): i for i, t in enumerate(ref['spec_t'])}

    # spectrum errors (average across selected times)
    spec_errs = []
    for t in common_times:
        ir   = t2idx_r[t]
        iref = t2idx_ref[t]
        kL_c, E_c     = r['spec_kL'][ir],   r['spec_E'][ir]
        kL_ref, E_ref = ref['spec_kL'][iref], ref['spec_E'][iref]
        e = spectrum_L2_rel_err(kL_c, E_c, kL_ref, E_ref)
        if not np.isnan(e):
            spec_errs.append(e)
    spec_err_mean = float(np.mean(spec_errs)) if spec_errs else np.nan

    # --- band-limited energy & enstrophy errors (use only up to coarsest Nyquist) ---
    t_r,  K_r  = np.array(r['K_t']),   np.array(r['K'])
    t_ref, K_R = np.array(ref['K_t']), np.array(ref['K'])
    L2D_run = float(r['L2D'])
    L2D_ref = float(ref['L2D'])

    K_errs, Om_errs = [], []
    for t in common_times:
        # indices of this sample in each run (already built above)
        ir   = t2idx_r[t]
        iref = t2idx_ref[t]

        # dimensionless spectra at this time
        kL_c, E_c   = r['spec_kL'][ir],    r['spec_E'][ir]
        kL_ref, E_r = ref['spec_kL'][iref], ref['spec_E'][iref]

        # overlap cutoff in kL (coarsest Nyquist)
        x_max = min(kL_c.max(), kL_ref.max())

        # energies at this time (interpolate each run's K(t) to the sample time)
        Kc  = float(np.interp(t, t_r,  K_r))
        Kr  = float(np.interp(t, t_ref, K_R))

        # band-limited energy (uses ∫ E_dim d(kL) fraction)
        m_c   = kL_c  <= x_max
        m_ref = kL_ref <= x_max
        frac_c   = float(np.trapezoid(E_c[m_c], kL_c[m_c]))
        frac_ref = float(np.trapezoid(E_r[m_ref], kL_ref[m_ref]))
        Kc_band  = Kc * frac_c
        Kr_band  = Kr * frac_ref

        # band-limited enstrophy: Ω = (K/L^2) ∫ (kL)^2 E_dim d(kL)
        int_c   = float(np.trapezoid((kL_c[m_c]**2)  * E_c[m_c],   kL_c[m_c]))
        int_ref = float(np.trapezoid((kL_ref[m_ref]**2) * E_r[m_ref], kL_ref[m_ref]))
        Oc_band = (Kc / (L2D_run**2)) * int_c
        Or_band = (Kr / (L2D_ref**2)) * int_ref

        # relative errors
        K_errs.append(abs(Kc_band - Kr_band) / Kr_band)
        Om_errs.append(abs(Oc_band - Or_band) / Or_band)

    K_err_mean  = float(np.mean(K_errs))  if K_errs  else np.nan
    Om_err_mean = float(np.mean(Om_errs)) if Om_errs else np.nan


    summary.append((N, spec_err_mean, K_err_mean, Om_err_mean))

# pretty print
summary.sort()
print("\nError vs resolution (reference N={}):".format(N_ref))
for N, eS, eK, eO in summary:
    print(f"N={N:4d}:  spectrum L2 rel err ≈ {eS:.3e},   |ΔK|/K ≈ {eK:.3e},   |ΔΩ|/Ω ≈ {eO:.3e}")
