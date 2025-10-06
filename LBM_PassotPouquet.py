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

def feq_D2Q9(rho, ux, uy, cxs, cys, w):
    # cs^2 = 1/3 in lattice units for D2Q9
    cs2 = 1.0/3.0
    cu  = np.zeros((rho.shape[0], rho.shape[1], 9))
    u2  = ux**2 + uy**2
    Feq = np.zeros_like(cu)
    for i, (cx, cy, wi) in enumerate(zip(cxs, cys, w)):
        cu[:, :, i] = cx*ux + cy*uy # dot product ci . u
        Feq[:, :, i] = wi * rho * (
            1 + (cu[:, :, i]/cs2)
              + 0.5*(cu[:, :, i]**2)/(cs2**2)
              - 0.5*(u2)/cs2
        )
    return Feq

def passot_pouquet_velocity(Nx, Ny, k_peak, urms_target, seed=1):
    """
    Build a periodic 2D velocity field with PassotPouquet spectrum:
      E(k) ∝ (k/kp)^4 * exp[-2*(k/kp)^2]
    Steps:
      1) Create random complex field in Fourier space with amplitude ∝ sqrt(E(k))
      2) Project to solenoidal (divergence-free): P = I - kk^T/k^2
      3) IFFT to physical space, then scale to desired u_rms
    """
    rng = np.random.default_rng(seed) #creating a reproducible random number generator
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=1.0) #Nx=Ny=256
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=1.0)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    K[0,0] = 1.0  # avoid divide-by-zero for shaping; we'll zero it later

    # PP spectral shape (unnormalized)
    E = (K / k_peak)**4 * np.exp(-2.0*(K / k_peak)**2)

    # random complex coefficients for two components
    phase1 = rng.normal(size=(Ny, Nx)) + 1j*rng.normal(size=(Ny, Nx)) #complex gaussian random field
    phase2 = rng.normal(size=(Ny, Nx)) + 1j*rng.normal(size=(Ny, Nx))

    # amplitude ∝ sqrt(E). (Factor choices only affect scaling; we'll normalize to urms_target)
    amp = np.sqrt(E)
    ux_hat = amp * phase1
    uy_hat = amp * phase2

    # Solenoidal projection: (I - k k^T / k^2) to make it divergence-free
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

    # Back to physical space
    ux = np.fft.ifft2(ux_hat_s).real
    uy = np.fft.ifft2(uy_hat_s).real

    # Scale to desired u_rms to ensure Ma=urms/cs<<1
    urms = np.sqrt((np.mean(ux**2 + uy**2)))  # == sqrt(mean(u^2))
    if urms > 0:
        s = urms_target / urms #rescaling factor to get urms=0.05
        ux *= s
        uy *= s
    return ux, uy

def shell_avg_spectrum(ux, uy, dx=1.0, dy=1.0, nbins=None):
    """
    Compute 1D isotropic energy spectrum E(k) by shell averaging:
      E(k) = 1/2 * sum_{shell} |u_hat(k)|^2  (with proper normalization)
    Returns k_shell_centers, E_shell
    """
    Ny, Nx = ux.shape
    uxh = np.fft.fft2(ux)
    uyh = np.fft.fft2(uy)

    # kinetic energy density in Fourier space
    Ek2d = 0.5*(np.abs(uxh)**2 + np.abs(uyh)**2)/(Nx*Ny)  # Parseval normalization

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

def main():
    # --------------------
    # Simulation parameters
    # --------------------
    Nx, Ny = 256, 256
    Nt = 4000                 # number of time steps
    tau = 0.7                 # relaxation time (viscosity via nu = cs^2*(tau-1/2))
    rho0 = 1.0
    urms0 = 0.05              # initial rms velocity (keep low Mach: u << cs ~ 1/sqrt(3))
    k_peak = 8.0 * 2*np.pi/Nx # peak wavenumber ~ 8 grid modes (adjust as you like)
    plot_every = 200

    cxs, cys, w = d2q9_setup()

    # --------------------
    # Passot–Pouquet initial velocity field (periodic)
    # --------------------
    ux0, uy0 = passot_pouquet_velocity(Nx, Ny, k_peak=k_peak, urms_target=urms0, seed=4)

    # initialize distributions with Feq(rho0, u0)
    rho = rho0 * np.ones((Ny, Nx))
    F = feq_D2Q9(rho, ux0, uy0, cxs, cys, w)

    # prepare plotting
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,4))
    plt.tight_layout()

    # time loop
    for it in range(Nt):

        # --- Streaming (periodic via roll) ---
        # note: this is "pull" or "push" equivalent; here we "push" by rolling each direction
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

        # --- Diagnostics/plots ---
        if it % plot_every == 0 or it == Nt-1:
            # energy spectrum
            k_shell, E_shell = shell_avg_spectrum(ux, uy, nbins=80)

            ax1.clear()
            vorticity = (np.roll(uy, -1, axis=1) - np.roll(uy, 1, axis=1)  # dUy/dx
                        - (np.roll(ux, -1, axis=0) - np.roll(ux, 1, axis=0))) # dUx/dy
            im = ax1.imshow(vorticity, cmap='bwr', origin='lower')
            ax1.set_title(f'Vorticity, t={it}')
            ax1.set_xticks([]); ax1.set_yticks([])

            ax2.clear()
            # Avoid zero bin; plot where k>0 and E>0
            m = (k_shell > 0) & (E_shell > 0)
            ax2.loglog(k_shell[m], E_shell[m], linestyle='-')
            ax2.set_xlabel('k')
            ax2.set_ylabel('E(k)')
            ax2.set_title('1D energy spectrum')

            # --- Add vertical line for k_peak ---
            ax2.axvline(k_peak, color='r', linestyle='--', label=r'$k_p$')
            ax2.legend()
            ax2.axvline(np.pi, ls=':', color='black' ,alpha=0.5, label='Nyquist')  # if dx=1

            ax2.legend(loc="best")

            plt.pause(0.001)

    plt.show()

if __name__ == "__main__":
    main()
