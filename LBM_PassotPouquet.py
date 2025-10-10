import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# D2Q9 LBM + Passot–Pouquet ICs
# ------------------------------

def d2q9_setup():
    # D2Q9 discrete velocities and weights
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    w   = np.array([4/9,
                    1/9, 1/9, 1/9, 1/9,
                    1/36,1/36,1/36,1/36])
    return cxs, cys, w

#D2Q9 equilibrium distribution function
def feq_D2Q9(rho, ux, uy, cxs, cys, w):    
    cs2 = 1.0/3.0 # cs^2 = 1/3 in lattice units for D2Q9
    cu  = np.zeros((rho.shape[0], rho.shape[1], 9))
    u2  = ux**2 + uy**2
    Feq = np.zeros_like(cu)
    for i, (cx, cy, wi) in enumerate(zip(cxs, cys, w)):
        cu[:, :, i] = cx*ux + cy*uy  # dot product ci . u
        Feq[:, :, i] = wi * rho * (
            1 + (cu[:, :, i]/cs2)
              + 0.5*(cu[:, :, i]**2)/(cs2**2)
              - 0.5*(u2)/cs2
        )
    return Feq

    """
    Build a periodic 2D velocity field with PassotPouquet spectrum:
      E(k) ∝ (k/kp)^4 * exp[-2*(k/kp)^2]
    Steps:
      1) Create random complex field in Fourier space with amplitude ∝ sqrt(E(k))
      2) Project to solenoidal (divergence-free): P = I - kk^T/k^2
      3) IFFT to physical space, then scale to desired u_rms
    """
def passot_pouquet_velocity(Nx, Ny, k_peak, urms_target, dx=1.0, dy=1.0, seed=1):
    rng = np.random.default_rng(seed) #creating a reproducible random number generator
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx) #Nx=Ny=256 or 512
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0,0] = 1.0  # avoid divide-by-zero for shaping

    # -------- Passot Pouquet spectral shape ---------
    E = (K / k_peak)**4 * np.exp(-2.0*(K / k_peak)**2)

    # random complex coefficients for two components
    phase1 = rng.normal(size=(Ny, Nx)) + 1j*rng.normal(size=(Ny, Nx)) #complex gaussian random field
    phase2 = rng.normal(size=(Ny, Nx)) + 1j*rng.normal(size=(Ny, Nx))

    # amplitude ∝ sqrt(E). =>normalize to urms_target
    amp = np.sqrt(E)
    ux_hat = amp * phase1
    uy_hat = amp * phase2

    # ------ Solenoidal projection: (I - k k^T / k^2) to make it divergence-free ------
    K2 = (KX**2 + KY**2)
    mask = K2 > 0 # avoid k=0
    Px11 = np.ones_like(K2) #P=2x2 projection tensor
    Px22 = np.ones_like(K2)
    Px12 = np.zeros_like(K2)
    Px21 = np.zeros_like(K2)
    # compute projection tensor entries only where k != 0 and assign back
    Px11[mask] = 1.0 - (KX[mask]*KX[mask])/K2[mask]
    Px12[mask] = - (KX[mask]*KY[mask])/K2[mask]
    Px21[mask] = - (KY[mask]*KX[mask])/K2[mask]
    Px22[mask] = 1.0 - (KY[mask]*KY[mask])/K2[mask]
    # apply projection
    ux_hat_s = Px11*ux_hat + Px12*uy_hat
    uy_hat_s = Px21*ux_hat + Px22*uy_hat
    # Zero the mean (k=0) explicitly
    ux_hat_s[0,0] = 0.0
    uy_hat_s[0,0] = 0.0

    # ------- Back to physical space -------
    ux = np.fft.ifft2(ux_hat_s).real
    uy = np.fft.ifft2(uy_hat_s).real
    # Scale to desired u_rms to ensure Ma=urms/cs<<1
    urms = np.sqrt((np.mean(ux**2 + uy**2)))  # == sqrt(mean(u^2))
    if urms > 0:
        s = urms_target / urms #rescaling factor to get urms=0.05
        ux *= s
        uy *= s
    return ux, uy

    """
    Compute 1D isotropic energy spectrum E(k) by shell averaging:
      E(k) = 1/2 * sum_{shell} |u_hat(k)|^2  (with proper normalization)
    Returns k_shell_centers, E_shell
    """
def shell_avg_spectrum(ux, uy, dx=1.0, dy=1.0, nbins=None):
    Ny, Nx = ux.shape
    uxh = np.fft.fft2(ux)
    uyh = np.fft.fft2(uy)

    # kinetic energy density in Fourier space
    Ek2d = 0.5*(np.abs(uxh)**2 + np.abs(uyh)**2)/(Nx*Ny)**2  # Parseval normalization

    # Build the 2D wavenumber grid
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)

    kmax = np.max(K)
    if nbins is None: #to get an isotropic 1D spectrum
        nbins = min(Nx, Ny)//2
    bins = np.linspace(0.0, kmax, nbins+1)
    E_shell = np.zeros(nbins)
    k_shell = np.zeros(nbins)

    # bin by K magnitude
    inds = np.digitize(K.ravel(), bins) - 1 # assigns each Fourier mode (pixel) to a bin index depending on its radius.
    for b in range(nbins):
        mask = (inds == b)
        if np.any(mask):
            E_shell[b] = np.sum(Ek2d.ravel()[mask]) # total energy in that shell.
            k_shell[b] = np.mean(K.ravel()[mask]) # the average radius of each ring
    return k_shell, E_shell

""""
-------------------------------------------
                    Main 
-------------------------------------------
"""
def main():
    # --------------------
    # Simulation parameters
    # --------------------
    Nx, Ny = 128, 128         # grid size, number of lattice nodes in each direction, best 512
    L_box = 2*np.pi           # choose a fixed physical box (common choice)
    dx = L_box / Nx
    dy = L_box / Ny
    m = 8                     # best 8
    k_peak = m * 2*np.pi / L_box 
    Nt = 4000                 # number of time steps
    tau = 0.53                # relaxation time (viscosity via nu = cs^2*(tau-1/2)), best 0.53
    rho0 = 1.0
    urms0 = 0.05              # initial rms velocity (keep low Mach: u << cs ~ 1/sqrt(3))
    plot_every = 200
    SAMPLE_TIMES = [1000, 2000, 3000]  # To compare across runs for error
    K_YMIN_FIXED = 0.0        #y-axis range for kinetic energy spectrum
    K_YMAX_FIXED = 0.0013

    # normalization length so x-axis is kL
    L2D = 2*np.pi / k_peak  
    k_nyq = np.pi / dx              # physical Nyquist wavenumber
    kL_nyq = k_nyq * L2D            # dimensionless Nyquist for the plot

    #convert to dimensionless units
    def to_dimensionless(k_shell, E_shell, ux, uy, L2D): 
        urms = np.sqrt(np.mean(ux**2 + uy**2))
        u_prime = urms / np.sqrt(2.0)
        kL = k_shell * L2D
        E_dim = E_shell / (u_prime**2 * L2D)
        return kL, E_dim

    cxs, cys, w = d2q9_setup() 

    # --------------------
    # Passot–Pouquet initial velocity field (periodic)
    # --------------------
    ux0, uy0 = passot_pouquet_velocity(Nx, Ny, k_peak=k_peak, urms_target=urms0, dx=dx, dy=dy, seed=4)

    # Fixed vorticity color scale from t=0 (used for all frames)
    vorticity0 = ((np.roll(uy0, -1, axis=1) - np.roll(uy0,  1, axis=1)) / (2.0*dx)
                - (np.roll(ux0, -1, axis=0) - np.roll(ux0,  1, axis=0)) / (2.0*dy))
    omega_rms0 = float(np.sqrt(np.mean(vorticity0**2)))
    omega_clim = 3.0 * omega_rms0          # 3σ symmetric range
    omega_vmin, omega_vmax = -omega_clim, +omega_clim # (alternative: omega_clim = 1.05 * np.max(np.abs(vorticity0)))

    # initialize distributions with Feq(rho0, u0)
    rho = rho0 * np.ones((Ny, Nx))
    F = feq_D2Q9(rho, ux0, uy0, cxs, cys, w)

    # prepare plotting fig1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.tight_layout()
    cbar = None  # colorbar for vorticity

     # Energy history arrays
    K_hist, t_hist = [], []
    # save initial energy
    urms_init = np.sqrt(np.mean(ux0**2 + uy0**2))
    K0 = 0.5 * urms_init**2

    # for time-averaged spectrum
    E_accum = None 
    count = 0
    start_average = 1000 

    # Spectrum figure
    fig_spec, ax_spec = plt.subplots(figsize=(6,4))
    ax_spec.set_xscale('log')
    ax_spec.set_yscale('log')
    ax_spec.set_xlabel(r'$kL$')
    ax_spec.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
    ax_spec.set_title('Energy spectra at different t')
    ax_spec.grid(True, which='both', alpha=0.3)
    xlim_set = False

    # Zoomed spectra figure
    fig_zoom, ax_zoom = plt.subplots(figsize=(6,4))
    ax_zoom.set_xscale('log'); ax_zoom.set_yscale('log')
    ax_zoom.set_xlabel(r'$kL$'); ax_zoom.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
    ax_zoom.set_title('Energy spectra (zoomed)')
    ax_zoom.grid(True, which='both', alpha=0.3)
    zoom_line_added = False

    # inertial-range zoom bounds
    kL_peak = k_peak * L2D                  # = 2*pi
    zoom_x_lo = max(0.9 * kL_peak, 1.0)     # start past injection
    zoom_x_hi = min(0.1 * kL_nyq, 1000.0)   # stop before dissipation/Nyquist
    ZOOM_YLIMS = (1e-10, 1e2)              # fixed y-limits
    ax_zoom.set_ylim(*ZOOM_YLIMS)           

    # For storage for error
    results = {
    'Nx': Nx, 'Ny': Ny, 'tau': tau, 'k_peak': k_peak, 'L2D': L2D,
    'spec_t': [],        # times when we store spectra
    'spec_kL': [],       # list of arrays
    'spec_E': [],        # list of arrays (E/(u'^2 L))
    'K_t': [],           # full time history
    'K': [],             
    'Omega_t': [],       # enstrophy times (match SAMPLE_TIMES)
    'Omega': [],         # enstrophy values at those times
}

    #------------------------- time loop -------------------------------
    for it in range(Nt):

        # --- Streaming (periodic via roll) ---
        for i, (cx, cy) in enumerate(zip(cxs, cys)):
            if cx != 0:
                F[:, :, i] = np.roll(F[:, :, i], cx, axis=1) # if the velocity for this direction is (1,0), then all the particles moving east are rolled one cell to the right and so on
            if cy != 0:
                F[:, :, i] = np.roll(F[:, :, i], cy, axis=0) # “Roll around” means that when populations stream beyond one edge of the grid, they re-enter from the opposite edge = periodic BC

        # --- Macros ---
        rho = np.sum(F, axis=2)
        ux  = np.sum(F * cxs[np.newaxis, np.newaxis, :], axis=2) / rho
        uy  = np.sum(F * cys[np.newaxis, np.newaxis, :], axis=2) / rho

        # --- Collision (BGK) ---
        Feq = feq_D2Q9(rho, ux, uy, cxs, cys, w)
        F += -(1.0/tau) * (F - Feq)

        #--- Time averaging of spectrum ---
        if it % plot_every == 0 and it >= start_average:
            k_shell, E_shell = shell_avg_spectrum(ux, uy, dx=dx, dy=dy ,nbins=80)
            if E_accum is None:
                E_accum = np.zeros_like(E_shell)
            E_accum += E_shell
            count += 1

        # --- Kinetic energy ---
        urms = np.sqrt(np.mean(ux**2 + uy**2))
        K = 0.5 * urms**2
        K_hist.append(K)
        t_hist.append(it)

        # mirror the history into the results bundle
        results['K_t'] = t_hist[:]  
        results['K']   = K_hist[:]

        # --- Diagnostics/plots ---
        if it % (5*plot_every) == 0 or it == Nt-1:
            k_shell, E_shell = shell_avg_spectrum(ux, uy,dx=dx, dy=dy, nbins=80)
            K_from_spectrum = np.sum(E_shell)
            K_from_field = K
            print(f"t={it:5d} | K_spectrum={K_from_spectrum:.6e} | K_field={K_from_field:.6e} | rel.err={(K_from_spectrum-K_from_field)/K_from_field:.2e}")

            # --- Vorticity field ---
            ax1.clear()
            vorticity = (1/(2*dx)*(np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)) # 2-point centered difference
                         - (1/(2*dy)*(np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))))
            im = ax1.imshow(vorticity, cmap='bwr', origin='lower', vmin=omega_vmin, vmax=omega_vmax)
            ax1.set_title(f'Vorticity, t={it}')
            ax1.set_xticks([]); ax1.set_yticks([])
            # add/update colorbar
            if cbar is None:
                cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.08, location='left')
                cbar.set_label(r'Vorticity')
            else:
                cbar.update_normal(im)

           # --- Energy spectra evolution ---
            msk = (k_shell > 0) & (E_shell > 0)
            if np.any(msk):
                kL, E_dim = to_dimensionless(k_shell[msk], E_shell[msk], ux, uy, L2D)

            # --- store spectra & enstrophy at selected times for convergence study ---
            if it in SAMPLE_TIMES and np.any(msk):
                # store spectrum
                results['spec_t'].append(it)
                results['spec_kL'].append(kL.copy())
                results['spec_E'].append(E_dim.copy())
                # enstrophy Ω = 1/2 <ω^2> 
                omega = vorticity  
                Omega = 0.5 * float(np.mean(omega**2))
                results['Omega_t'].append(it)
                results['Omega'].append(Omega)

            # right panel of fig1, instantaneous spectrum
            ax2.clear()
            ax2.loglog(kL, E_dim, linestyle='-', lw=1.5)
            ax2.set_xlabel(r'$kL$')
            ax2.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
            ax2.set_title(f'Energy spectrum, t={it}')
            ax2.axvline(k_peak*L2D, color='r', linestyle='--', label=r'$k_pL$')
            ax2.axvline(kL_nyq,    ls=':', color='black', alpha=0.5, label='Nyquist')
            ax2.legend(loc="best")
            ax2.set_xlim(1, min(1000, float(kL.max())*1.05))

            # persistent multi-time spectra figure2 (dimensionless)
            ax_spec.plot(kL, E_dim, label=f't={it}', lw=1)

            # set limits and add k^-3
            if not xlim_set:
                ax_spec.set_xlim(1, min(1000, float(kL.max())*1.05))
                k_ref = 10.0
                if not (kL.min() < k_ref < kL.max()):
                    k_ref = float(np.exp(0.5*(np.log(kL.min()) + np.log(kL.max()))))
                E_ref = float(np.interp(k_ref, kL, E_dim))
                k_line = np.array([k_ref/4, k_ref*4])
                k_line[0] = max(k_line[0], ax_spec.get_xlim()[0])
                k_line[1] = min(k_line[1], ax_spec.get_xlim()[1])
                E_line = E_ref * (k_line / k_ref)**(-3.0)
                ax_spec.plot(k_line, E_line, 'k--', lw=2, label=r'$k^{-3}$')
                xlim_set = True

            ax_spec.legend(loc='lower left', fontsize=8)

            # zoomed figure update
            ax_zoom.plot(kL, E_dim, lw=1, label=f't={it}')
            ax_zoom.set_xlim(zoom_x_lo, min(zoom_x_hi, float(kL.max())))
            ax_zoom.set_ylim(*ZOOM_YLIMS)   

            # add k^-3 line in the zoomed figure 
            if not zoom_line_added:
                # anchor slope at geometric mid of the zoomed x-window
                x0, x1 = ax_zoom.get_xlim()
                k_ref_z = np.sqrt(x0 * x1)
                if not (kL.min() < k_ref_z < kL.max()):
                    k_ref_z = float(np.exp(0.5*(np.log(kL.min()) + np.log(kL.max()))))
                E_ref_z = float(np.interp(k_ref_z, kL, E_dim))
                k_line_z = np.array([x0, x1])
                E_line_z = E_ref_z * (k_line_z / k_ref_z)**(-3.0)
                ax_zoom.plot(k_line_z, E_line_z, 'k--', lw=2, label=r'$k^{-3}$')
                zoom_line_added = True

            ax_zoom.legend(loc='lower left', fontsize=8)
        plt.pause(0.001)

    plt.show(block=False)

    # ----- Plot kinetic energy decay -----
    plt.figure(figsize=(6,4))
    plt.plot(t_hist, K_hist, lw=2)
    plt.xlabel("Time step")
    plt.ylabel("Turbulent kinetic energy K(t)")
    plt.title("Turbulent energy decay")
    plt.ylim(K_YMIN_FIXED, K_YMAX_FIXED)  
    plt.grid(True, alpha=0.3)
    plt.show()

    # ----- Saving for error -----
    np.savez_compressed(f'lbm_run_N{Nx}.npz', **results)
    print(f"Saved convergence data to lbm_run_N{Nx}.npz")


if __name__ == "__main__":
    main()

