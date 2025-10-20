import numpy as np
import matplotlib.pyplot as plt

# Global font sizes 
plt.rcParams.update({
    "axes.titlesize": 18,   # subplot titles
    "axes.labelsize": 16,   # x/y labels
    "legend.fontsize": 14,  # legend text
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 20  # suptitle
})

# ------------------------------
# D2Q9 LBM + Passot–Pouquet ICs (MRT collision)
# ------------------------------

def d2q9_setup():
    # D2Q9 discrete velocities and weights
    cxs = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
    cys = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
    w   = np.array([4/9,
                    1/9, 1/9, 1/9, 1/9,
                    1/36,1/36,1/36,1/36])
    return cxs, cys, w

# D2Q9 equilibrium distribution function
def feq_D2Q9(rho, ux, uy, cxs, cys, w):
    cs2 = 1.0/3.0  # cs^2 = 1/3
    cu  = np.zeros((rho.shape[0], rho.shape[1], 9))
    u2  = ux**2 + uy**2
    Feq = np.zeros_like(cu)
    for i, (cx, cy, wi) in enumerate(zip(cxs, cys, w)):
        cu[:, :, i] = cx*ux + cy*uy
        Feq[:, :, i] = wi * rho * (
            1 + (cu[:, :, i]/cs2)
              + 0.5*(cu[:, :, i]**2)/(cs2**2)
              - 0.5*(u2)/cs2
        )
    return Feq

# ------------------- MRT ----------------------
def d2q9_M_Minv():
    # d'Humières / Lallemand–Luo basis (rows are moments)
    M = np.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 1],           # 0: rho
        [-4,-1,-1,-1,-1, 2, 2, 2, 2],          # 1: e
        [ 4,-2,-2,-2,-2, 1, 1, 1, 1],          # 2: eps
        [ 0, 1, 0,-1, 0, 1,-1,-1, 1],          # 3: jx
        [ 0,-2, 0, 2, 0, 1,-1,-1, 1],          # 4: qx
        [ 0, 0, 1, 0,-1, 1, 1,-1,-1],          # 5: jy
        [ 0, 0,-2, 0, 2, 1, 1,-1,-1],          # 6: qy
        [ 0, 1,-1, 1,-1, 0, 0, 0, 0],          # 7: pxx
        [ 0, 0, 0, 0, 0, 1,-1, 1,-1]           # 8: pxy
    ], dtype=float)
    Minv = np.linalg.inv(M)
    return M, Minv

"""
Map BGK tau (shear) to MRT rates so viscosity matches
Keep conserved moments at 0-relaxation.
"""
def mrt_relaxation_from_tau(tau, s_e=1.70, s_eps=1.60, s_q=1.92): #update s_i when change Re
    s_nu = 1.0 / tau
    # diag S in order [rho, e, eps, jx, qx, jy, qy, pxx, pxy]
    S_diag = np.array([0.0, s_e, s_eps, 0.0, s_q, 0.0, s_q, s_nu, s_nu], dtype=float)
    return S_diag

"""
One-site MRT collision:
  m      = M f
  m_eq   = M feq
  m_post = m - S (m - m_eq)
  f_post = Minv m_post
Vectorized over the (y,x) grid.
"""
def collide_MRT(F, rho, ux, uy, cxs, cys, w, M, Minv, S_diag):
    Feq = feq_D2Q9(rho, ux, uy, cxs, cys, w)
    # transform to moment space: m = M f  (einsum over velocity index)
    m    = np.einsum('...i,ji->...j', F,   M)
    m_eq = np.einsum('...i,ji->...j', Feq, M)
    # diagonal relaxation per moment
    m_post = m - (m - m_eq) * S_diag[None, None, :]
    # back to populations: f_post = Minv m_post
    F_post = np.einsum('...j,ij->...i', m_post, Minv)
    return F_post

"""
Build ICs on an NxN grid by cropping the centered low-k block from the
master spectra, with correct FFT normalization so amplitudes are preserved.
Assumes even sizes and Nx,Ny <= master size.
"""
def ic_from_master(Nx, Ny, Ux_hat_master, Uy_hat_master, urms_target=None):
    Ny_m, Nx_m = Ux_hat_master.shape
    assert Nx % 2 == 0 and Ny % 2 == 0, "Nx, Ny must be even"
    assert Nx <= Nx_m and Ny <= Ny_m, "Target grid must not exceed master grid"

    # 1) shift to center DC at [Ny_m/2, Nx_m/2]
    Ux_c = np.fft.fftshift(Ux_hat_master)
    Uy_c = np.fft.fftshift(Uy_hat_master)

    # 2) crop exact center window of size Ny×Nx
    cy, cx = Ny_m // 2, Nx_m // 2
    hy, hx = Ny // 2, Nx // 2
    sy = slice(cy - hy, cy + hy)
    sx = slice(cx - hx, cx + hx)
    Ux_small_c = Ux_c[sy, sx]
    Uy_small_c = Uy_c[sy, sx]

    # 3) unshift back to FFT storage layout
    Ux_small = np.fft.ifftshift(Ux_small_c)
    Uy_small = np.fft.ifftshift(Uy_small_c)

    # 4) normalization so ifft2 amplitude matches the master’s band
    alpha = (Nx * Ny) / (Nx_m * Ny_m)
    Ux_small *= alpha
    Uy_small *= alpha

    # 5) back to physical space
    ux = np.fft.ifft2(Ux_small).real
    uy = np.fft.ifft2(Uy_small).real

    if urms_target is not None:
        urms = float(np.sqrt(np.mean(ux**2 + uy**2)))
        if urms > 0:
            s = urms_target / urms
            ux *= s
            uy *= s
    return ux, uy

def shell_avg_spectrum(ux, uy, dx=1.0, dy=1.0, nbins=None):
    Ny, Nx = ux.shape
    uxh = np.fft.fft2(ux)
    uyh = np.fft.fft2(uy)
    Ek2d = 0.5*(np.abs(uxh)**2 + np.abs(uyh)**2)/(Nx*Ny)**2
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    kmax = np.max(K)
    if nbins is None:
        nbins = min(Nx, Ny)//2
    bins = np.linspace(0.0, kmax, nbins+1)
    E_shell = np.zeros(nbins)
    k_shell = np.zeros(nbins)
    inds = np.digitize(K.ravel(), bins) - 1
    for b in range(nbins):
        mask = (inds == b)
        if np.any(mask):
            E_shell[b] = np.sum(Ek2d.ravel()[mask])
            k_shell[b] = np.mean(K.ravel()[mask])
    return k_shell, E_shell

def build_nearest_index_maps(Nx_src, Ny_src, Nx_ref=512, Ny_ref=512):
    ix_ref = np.arange(Nx_ref)
    iy_ref = np.arange(Ny_ref)
    ix_src_for_ref = np.rint(ix_ref * (Nx_src / Nx_ref)).astype(int)
    iy_src_for_ref = np.rint(iy_ref * (Ny_src / Ny_ref)).astype(int)
    ix_src_for_ref = np.mod(ix_src_for_ref, Nx_src)
    iy_src_for_ref = np.mod(iy_src_for_ref, Ny_src)
    return ix_src_for_ref, iy_src_for_ref

""""
-------------------------------------------
                    Main 
-------------------------------------------
"""
def main():
    # --------------------
    # Simulation parameters
    # --------------------
    Nx_ref, Ny_ref = 512, 512
    Nx, Ny = 128, 128
    L_box = 2*np.pi
    dx = L_box / Nx
    dy = L_box / Ny
    m = 8
    k_peak = m * 2*np.pi / L_box 
    Nt = 500
    rho0 = 1.0
    urms0 = 0.05
    tau = 0.503          # shear relaxation time (τ_ν)
    plot_every = 50
    SAMPLE_TIMES = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]
    K_YMIN_FIXED = 0.0
    K_YMAX_FIXED = 0.0013
    ix_map, iy_map = build_nearest_index_maps(Nx, Ny, Nx_ref, Ny_ref)

    # normalization length so x-axis is kL
    L2D = 2*np.pi / k_peak  
    k_nyq = np.pi / dx
    kL_nyq = k_nyq * L2D

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
    master = np.load(f"pp_master_512_seed4_L{int(L_box)}.npz")
    ux0, uy0 = ic_from_master(Nx, Ny, master["Ux_hat"], master["Uy_hat"], urms_target=urms0)
    print("t=0 urms:", np.sqrt(np.mean(ux0**2 + uy0**2)))
    k_shell0, E_shell0 = shell_avg_spectrum(ux0, uy0, dx=dx, dy=dy, nbins=80)
    print("E_shell0 sum:", np.sum(E_shell0))

    vorticity0 = ((np.roll(uy0, -1, axis=1) - np.roll(uy0,  1, axis=1)) / (2.0*dx)
                - (np.roll(ux0, -1, axis=0) - np.roll(ux0,  1, axis=0)) / (2.0*dy))
    omega_rms0 = float(np.sqrt(np.mean(vorticity0**2)))
    omega_clim = 3.0 * omega_rms0
    omega_vmin, omega_vmax = -omega_clim, +omega_clim 

    # initialize distributions with Feq(rho0, u0)
    rho = rho0 * np.ones((Ny, Nx))
    F = feq_D2Q9(rho, ux0, uy0, cxs, cys, w)

    # --- MRT precompute ---
    M, Minv = d2q9_M_Minv()
    S_diag  = mrt_relaxation_from_tau(tau, s_e=1.64, s_eps=1.54, s_q=1.90)

    # prepare plotting fig1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.tight_layout()
    cbar = None  # colorbar for vorticity

    # Energy history
    K_hist, t_hist = [], []
    urms_init = np.sqrt(np.mean(ux0**2 + uy0**2))
    K0 = 0.5 * urms_init**2

    # time-averaged spectrum
    E_accum = None 
    count = 0
    start_average = 1000 

    # Spectrum figure
    fig_spec, ax_spec = plt.subplots(figsize=(6,4))
    ax_spec.set_xscale('log'); ax_spec.set_yscale('log')
    ax_spec.set_xlabel(r'$kL$'); ax_spec.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
    ax_spec.set_title('Energy spectra at different time steps')
    ax_spec.grid(True, which='both', alpha=0.3)
    xlim_set = False

    # Zoomed spectra figure
    fig_zoom, ax_zoom = plt.subplots(figsize=(6,4))
    ax_zoom.set_xscale('log'); ax_zoom.set_yscale('log')
    ax_zoom.set_xlabel(r'$kL$'); ax_zoom.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
    ax_zoom.set_title('Energy spectra (zoomed)')
    ax_zoom.grid(True, which='both', alpha=0.3)
    zoom_line_added = False

    kL_peak = k_peak * L2D
    zoom_x_lo = max(0.9 * kL_peak, 1.0)
    zoom_x_hi = min(0.1 * kL_nyq, 1000.0)
    ZOOM_YLIMS = (1e-10, 1e2)
    ax_zoom.set_ylim(*ZOOM_YLIMS)

    cs2 = 1.0/3.0
    nu   = cs2 * (tau - 0.5)             # matches BGK via s_nu=1/tau
    Re0  = urms0 * L_box / nu

    results = {
        'Nx': Nx, 'Ny': Ny,
        'tau': tau,
        'nu': nu,
        'Re0': Re0,
        'L_box': L_box,
        'm': m,
        'k_peak': k_peak,
        'L2D': L2D,
        'spec_t': [],
        'spec_kL': [],
        'spec_E': [],
        'K_t': [],
        'K': [],
        'Omega_t': [],
        'Omega': [],
        'u_times': [],
        'u_snaps': [],
        'v_snaps': [],
    }

    #------------------------- time loop -------------------------------
    for it in range(Nt):

        # --- Streaming (periodic via roll) ---
        for i, (cx, cy) in enumerate(zip(cxs, cys)):
            if cx != 0:
                F[:, :, i] = np.roll(F[:, :, i], cx, axis=1)
            if cy != 0:
                F[:, :, i] = np.roll(F[:, :, i], cy, axis=0)

        # --- Macros ---
        rho = np.sum(F, axis=2)
        ux  = np.sum(F * cxs[np.newaxis, np.newaxis, :], axis=2) / rho
        uy  = np.sum(F * cys[np.newaxis, np.newaxis, :], axis=2) / rho

        # --- Collision (MRT) ---
        F = collide_MRT(F, rho, ux, uy, cxs, cys, w, M, Minv, S_diag)

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
        results['K_t'] = t_hist[:]
        results['K']   = K_hist[:]

        # --- Diagnostics/plots ---
        if it % (plot_every) == 0 or it == Nt-1:
            k_shell, E_shell = shell_avg_spectrum(ux, uy, dx=dx, dy=dy, nbins=80)
            msk = (k_shell > 0) & (E_shell > 0)
            K_from_spectrum = np.sum(E_shell)
            K_from_field = K
            print(f"t={it:5d} | K_spectrum={K_from_spectrum:.6e} | K_field={K_from_field:.6e} | rel.err={(K_from_spectrum-K_from_field)/K_from_field:.2e}")

            # --- Vorticity field ---
            ax1.clear()
            vorticity = (1/(2*dx)*(np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1))
                         - (1/(2*dy)*(np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))))
            im = ax1.imshow(vorticity, cmap='bwr', origin='lower', vmin=omega_vmin, vmax=omega_vmax)
            ax1.set_title(f'Vorticity, n={it}')
            ax1.set_xticks([]); ax1.set_yticks([])
            if cbar is None:
                cbar = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.08, location='left')
                cbar.set_label(r'Vorticity', fontsize=14)
            else:
                cbar.update_normal(im)

            # --- Energy spectra evolution ---
            if np.any(msk):
                kL, E_dim = to_dimensionless(k_shell[msk], E_shell[msk], ux, uy, L2D)

                if it in SAMPLE_TIMES:
                    results['spec_t'].append(it)
                    results['spec_kL'].append(kL.copy())
                    results['spec_E'].append(E_dim.copy())
                    omega = vorticity
                    Omega = 0.5 * float(np.mean(omega**2))
                    results['Omega_t'].append(it)
                    results['Omega'].append(Omega)
                    results['u_times'].append(int(it))
                    results['u_snaps'].append(ux.copy())
                    results['v_snaps'].append(uy.copy())

                ax2.clear()
                ax2.loglog(kL, E_dim, linestyle='-', lw=1.5)
                ax2.set_xlabel(r'$kL$'); ax2.set_ylabel(r'$E(k)/(u^{\prime 2}L)$')
                ax2.set_title(f'Energy spectrum, t={it}')
                ax2.axvline(k_peak*L2D, linestyle='--', label=r'$k_pL$')
                ax2.axvline(kL_nyq,    ls=':', alpha=0.5, label='Nyquist')
                ax2.legend(loc="best", fontsize=14)
                ax2.set_xlim(1, min(1000, float(kL.max())*1.05))

                ax_spec.plot(kL, E_dim, label=f't={it}', lw=1)
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
                ax_spec.legend(loc='lower left', fontsize=14)

                ax_zoom.plot(kL, E_dim, lw=1, label=f't={it}')
                ax_zoom.set_xlim(zoom_x_lo, min(zoom_x_hi, float(kL.max())))
                ax_zoom.set_ylim(*ZOOM_YLIMS)   
                if not zoom_line_added:
                    x0, x1 = ax_zoom.get_xlim()
                    k_ref_z = np.sqrt(x0 * x1)
                    if not (kL.min() < k_ref_z < kL.max()):
                        k_ref_z = float(np.exp(0.5*(np.log(kL.min()) + np.log(kL.max()))))
                    E_ref_z = float(np.interp(k_ref_z, kL, E_dim))
                    k_line_z = np.array([x0, x1])
                    E_line_z = E_ref_z * (k_line_z / k_ref_z)**(-3.0)
                    ax_zoom.plot(k_line_z, E_line_z, 'k--', lw=2, label=r'$k^{-3}$')
                    zoom_line_added = True
                ax_zoom.legend(loc='lower left', fontsize=14)
        plt.pause(0.001)

    plt.show(block=False)

    # ----- Plot kinetic energy decay -----
    plt.figure(figsize=(6,4))
    plt.plot(t_hist, K_hist, lw=2)
    plt.xlabel("Time step"); plt.ylabel("Turbulent kinetic energy K(t)")
    plt.title("Turbulent energy decay")
    plt.ylim(K_YMIN_FIXED, K_YMAX_FIXED)
    plt.grid(True, alpha=0.3)
    plt.show()

    # ----- Saving for error -----
    fname = f"lbm_runMRT_N{Nx}_{tau}.npz"
    np.savez_compressed(fname, **results)
    print(f"Saved convergence data to {fname}  (Re≈{Re0:.0f})")

if __name__ == "__main__":
    main()
