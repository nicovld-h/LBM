# save_master_pp.py
import numpy as np

L_box = 2*np.pi

def make_master_pp_hat(Nm=512, L_box=L_box, m=8, urms_target=0.05, seed=4):
    dx = L_box / Nm
    kx = 2*np.pi*np.fft.fftfreq(Nm, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Nm, d=dx)
    KX, KY = np.meshgrid(kx, ky)
    K  = np.sqrt(KX**2 + KY**2); K[0,0]=1.0
    k_peak = (2*np.pi/L_box)*m
    E = (K/k_peak)**4 * np.exp(-2*(K/k_peak)**2)

    rng = np.random.default_rng(seed)
    phase1 = rng.normal(size=(Nm,Nm)) + 1j*rng.normal(size=(Nm,Nm))
    phase2 = rng.normal(size=(Nm,Nm)) + 1j*rng.normal(size=(Nm,Nm))
    amp = np.sqrt(E)
    ux_hat = amp * phase1
    uy_hat = amp * phase2

    K2 = KX**2+KY**2; mask = K2>0
    Px11 = np.ones_like(K2); Px22 = np.ones_like(K2)
    Px12 = np.zeros_like(K2); Px21 = np.zeros_like(K2)
    Px11[mask] = 1 - (KX[mask]*KX[mask])/K2[mask]
    Px12[mask] = -(KX[mask]*KY[mask])/K2[mask]
    Px21[mask] = -(KY[mask]*KX[mask])/K2[mask]
    Px22[mask] = 1 - (KY[mask]*KY[mask])/K2[mask]
    ux_hat_s = Px11*ux_hat + Px12*uy_hat
    uy_hat_s = Px21*ux_hat + Px22*uy_hat
    ux_hat_s[0,0]=0; uy_hat_s[0,0]=0

    # scale to target urms
    ux = np.fft.ifft2(ux_hat_s).real
    uy = np.fft.ifft2(uy_hat_s).real
    urms = np.sqrt(np.mean(ux**2+uy**2))
    s = (urms_target/urms) if urms>0 else 1.0
    return ux_hat_s*s, uy_hat_s*s, k_peak

def ic_from_master(Nx, Ny, master_ux_hat, master_uy_hat):
    Nm = master_ux_hat.shape[0]  # 512
    # take centered low-k block (crop/pad handles even N)
    hx, hy = Nx//2, Ny//2
    cx, cy = Nm//2, Nm//2
    sx = slice(cx - hx, cx + hx)
    sy = slice(cy - hy, cy + hy)
    ux_hat_small = np.zeros((Ny, Nx), dtype=complex)
    uy_hat_small = np.zeros((Ny, Nx), dtype=complex)
    ux_hat_small[:, :] = master_ux_hat[sy, sx]
    uy_hat_small[:, :] = master_uy_hat[sy, sx]
    ux = np.fft.ifft2(ux_hat_small).real
    uy = np.fft.ifft2(uy_hat_small).real
    return ux, uy


m = 8; urms0 = 0.05
Ux_hat, Uy_hat, k_peak = make_master_pp_hat(Nm=512, L_box=L_box, m=m, urms_target=urms0, seed=4)

ux_test = np.fft.ifft2(Ux_hat).real
uy_test = np.fft.ifft2(Uy_hat).real
print("master urms =", (np.mean(ux_test**2 + uy_test**2))**0.5)

np.savez_compressed(f"pp_master_512_seed4_L{int(L_box)}.npz",
                    Ux_hat=Ux_hat, Uy_hat=Uy_hat, L_box=L_box, m=m, k_peak=k_peak)
print("Saved pp_master_512_seed4_L", L_box/np.pi, "pi.npz")
