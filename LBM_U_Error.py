import numpy as np
import matplotlib.pyplot as plt
import glob, os

L_BOX_DEFAULT = 2*np.pi

def _as_list(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        return [xi for xi in x]
    if isinstance(x, np.ndarray):
        return [xi for xi in x]
    return list(x)

def build_nearest_index_maps(Nx_src, Ny_src, Nx_ref, Ny_ref):
    ix_ref = np.arange(Nx_ref)
    iy_ref = np.arange(Ny_ref)
    ix_src = np.rint(ix_ref * (Nx_src / Nx_ref)).astype(int) % Nx_src
    iy_src = np.rint(iy_ref * (Ny_src / Ny_ref)).astype(int) % Ny_src
    return ix_src, iy_src

def resample_to_ref(ux, uy, ix_map, iy_map):
    return ux[np.ix_(iy_map, ix_map)], uy[np.ix_(iy_map, ix_map)]

def fft_lowpass(ux, uy, k_cut, L_box):
    Ny, Nx = ux.shape
    dx = L_box / Nx; dy = L_box / Ny
    kx = 2*np.pi*np.fft.fftfreq(Nx, d=dx)
    ky = 2*np.pi*np.fft.fftfreq(Ny, d=dy)
    KX, KY = np.meshgrid(kx, ky)
    K = np.sqrt(KX**2 + KY**2)
    mask = (K <= k_cut)
    Ux = np.fft.fft2(ux); Uy = np.fft.fft2(uy)
    Ux *= mask; Uy *= mask
    return np.fft.ifft2(Ux).real, np.fft.ifft2(Uy).real

def spectral_resample_to_ref(ux, uy, Nx_src, Ny_src, Nx_ref, Ny_ref):
    """
    Band-limited resampling from Ny_src×Nx_src -> Ny_ref×Nx_ref using FFT.
    - Crops (if ref < src) or zero-pads (if ref > src) the centered spectrum.
    - Applies the correct normalization so physical amplitudes are preserved:
        beta = (Nx_ref*Ny_ref)/(Nx_src*Ny_src)
    Returns (ux_ref, uy_ref) on the reference grid.
    """
    # FFT on the source grid
    Ux = np.fft.fft2(ux); Uy = np.fft.fft2(uy)

    # Shift DC to center for easy crop/pad
    Ux_c = np.fft.fftshift(Ux)
    Uy_c = np.fft.fftshift(Uy)

    # Prepare destination (centered)
    Gx_c = np.zeros((Ny_ref, Nx_ref), dtype=complex)
    Gy_c = np.zeros((Ny_ref, Nx_ref), dtype=complex)

    # Overlap extents (centered crop/pad)
    hx = min(Nx_src, Nx_ref) // 2
    hy = min(Ny_src, Ny_ref) // 2

    # centers
    cx_src, cy_src = Nx_src // 2, Ny_src // 2
    cx_ref, cy_ref = Nx_ref // 2, Ny_ref // 2

    # source slices
    sx_src = slice(cx_src - hx, cx_src + hx)
    sy_src = slice(cy_src - hy, cy_src + hy)
    # destination slices
    sx_ref = slice(cx_ref - hx, cx_ref + hx)
    sy_ref = slice(cy_ref - hy, cy_ref + hy)

    # copy low-k block
    Gx_c[sy_ref, sx_ref] = Ux_c[sy_src, sx_src]
    Gy_c[sy_ref, sx_ref] = Uy_c[sy_src, sx_src]

    # unshift back
    Gx = np.fft.ifftshift(Gx_c)
    Gy = np.fft.ifftshift(Gy_c)

    # *** normalization so ifft amplitude stays consistent ***
    beta = (Nx_ref * Ny_ref) / (Nx_src * Ny_src)
    Gx *= beta
    Gy *= beta

    # back to physical grid
    ux_ref = np.fft.ifft2(Gx).real
    uy_ref = np.fft.ifft2(Gy).real
    return ux_ref, uy_ref

def field_L2(ux, uy):
    return np.sqrt(np.sum(ux*ux + uy*uy))

def load_run_native(path):
    d = np.load(path, allow_pickle=True)
    Nx = int(d["Nx"]); Ny = int(d["Ny"])

    # Reconstruct L_box if not saved explicitly
    if "L_box" in d:
        L_box = float(d["L_box"])
    elif "L2D" in d and "k_peak" in d:
        L_box = float(d["L2D"]) * (2*np.pi / float(d["k_peak"]))
    else:
        L_box = L_BOX_DEFAULT

    times = np.array(d["u_times"], dtype=int) if "u_times" in d else np.array([], dtype=int)

    # Prefer ref snapshots ONLY if they are NON-EMPTY
    has_ref_keys = ("u_snaps_ref" in d and "v_snaps_ref" in d)
    ref_nonempty = has_ref_keys and (d["u_snaps_ref"].size > 0) and (d["v_snaps_ref"].size > 0)

    if ref_nonempty:
        ux_list = _as_list(d["u_snaps_ref"])
        uy_list = _as_list(d["v_snaps_ref"])
        Ny_ref, Nx_ref = ux_list[0].shape if ux_list else (None, None)
        ref = {"has_ref": True, "Nx_ref": Nx_ref, "Ny_ref": Ny_ref}
    else:
        # Use native and we’ll resample later
        ux_list = _as_list(d["u_snaps"]) if "u_snaps" in d else []
        uy_list = _as_list(d["v_snaps"]) if "v_snaps" in d else []
        ref = {"has_ref": False}

    # Align lengths
    n = min(len(times), len(ux_list), len(uy_list))
    times = times[:n]; ux_list = ux_list[:n]; uy_list = uy_list[:n]

    return {
        "path": path, "Nx": Nx, "Ny": Ny, "L_box": L_box,
        "times": times,
        "ux_list_native": ux_list, "uy_list_native": uy_list,
        "ref_info": ref
    }

def main(pattern="lbm_run_N*.npz"):
    MAX_T = 2800   # compare up to this time step (t=0..MAX_T)

    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files match {pattern}")

    runs = [load_run_native(f) for f in files]
    runs = [r for r in runs if len(r["times"]) > 0 and len(r["ux_list_native"]) > 0]
    if not runs:
        raise RuntimeError("No runs with usable snapshots found (empty ref arrays detected & no native fallback).")

    # DNS = largest Nx (reference grid)
    dns = max(runs, key=lambda r: r["Nx"])
    L_box = dns["L_box"]
    Nx_ref, Ny_ref = dns["Nx"], dns["Ny"]

    # Build ref snapshots for every run (spectral resample to DNS grid)
    for r in runs:
        r["ux_list_ref"], r["uy_list_ref"] = [], []
        for ux, uy in zip(r["ux_list_native"], r["uy_list_native"]):
            ux_r, uy_r = spectral_resample_to_ref(
                ux, uy, Nx_src=r["Nx"], Ny_src=r["Ny"], Nx_ref=Nx_ref, Ny_ref=Ny_ref
            )
            r["ux_list_ref"].append(ux_r)
            r["uy_list_ref"].append(uy_r)


    # Smallest Nyquist (min N)
    N_min = min(r["Nx"] for r in runs)
    k_cut = np.pi * N_min / L_box
    print(f"Using smallest Nyquist cut-off: N_min={N_min}, k_cut={k_cut:.6f} rad/unit")
    dns_lp_cache = {}
    for i, t in enumerate(dns["times"]):
        ux_a, uy_a = dns["ux_list_ref"][i], dns["uy_list_ref"][i]
        dns_lp_cache[int(t)] = fft_lowpass(ux_a, uy_a, k_cut, L_box)

    # Intersect times with DNS
    times_dns = set(dns["times"].tolist())
    pts = []  # (Nx, rel_L2_RMS, filename)

    for r in sorted(runs, key=lambda x: x["Nx"]):
        common = np.array(sorted(list(times_dns.intersection(set(r["times"].tolist())))))
        common = common[common <= MAX_T]
        if common.size == 0:
            print(f"Skipping {os.path.basename(r['path'])}: no common times ≤ {MAX_T} with DNS.")
            continue

        idx_r = {int(t): i for i, t in enumerate(r["times"])}
        idx_a = {int(t): i for i, t in enumerate(dns["times"])}

        errs = []
        for t in common:
            ir = idx_r[int(t)]
            ia = idx_a[int(t)]

            ux_c, uy_c = r["ux_list_ref"][ir], r["uy_list_ref"][ir]
            ux_a, uy_a = dns["ux_list_ref"][ia], dns["uy_list_ref"][ia]

            # low-pass to smallest Nyquist
            ux_c, uy_c = fft_lowpass(ux_c, uy_c, k_cut, L_box)
            ux_a, uy_a = fft_lowpass(ux_a, uy_a, k_cut, L_box)

            # discrete L2 norms over the grid
            diff_L2 = np.sqrt(np.sum((ux_c - ux_a)**2 + (uy_c - uy_a)**2))
            ref_L2  = np.sqrt(np.sum(ux_a**2 + uy_a**2))
            if ref_L2 > 0:
                errs.append(diff_L2 / ref_L2)

            ux_c, uy_c = fft_lowpass(r["ux_list_ref"][ir], r["uy_list_ref"][ir], k_cut, L_box)
            ux_a, uy_a = dns_lp_cache[int(t)]

            if t in (0, 1000, 2000, 3000):
                print(f"N={r['Nx']}, t={t}: rel_L2 = {np.sqrt(np.sum((ux_c-ux_a)**2 + (uy_c-uy_a)**2)) / np.sqrt(np.sum(ux_a**2 + uy_a**2)):.3e}")

        if len(errs) == 0:
            print(f"Skipping {os.path.basename(r['path'])}: empty/NaN errors.")
            continue

        # RMS over sample times
        rel_L2_RMS = float(np.sqrt(np.mean(np.square(errs))))
        pts.append((r["Nx"], rel_L2_RMS, os.path.basename(r["path"])))
        
        if not pts:
            raise RuntimeError("No comparable runs (no overlapping times with DNS).")
        
    # Print and plot
    print("\nGrid size vs relative error (k ≤ k_nyq(Nmin)):")
    for N, e, name in pts:
        tag = " (DNS)" if N==Nx_ref else ""
        print(f"  N={N:4d}  RelErr={e:.6e}   {name}{tag}")

    Ns  = [p[0] for p in pts]
    Err = [p[1] for p in pts]
    plt.figure(figsize=(6,4))
    plt.plot(Ns, Err, marker="o", lw=2)
    plt.xlabel("Grid size N")
    plt.ylabel("Relative error")
    plt.title(f"Convergence vs resolution (k ≤ k_nyq(Nmin={N_min}))")
    plt.grid(True, alpha=0.3)
    # optional:
    # plt.xscale('log', base=2); plt.yscale('log')
    plt.show()

if __name__ == "__main__":
    main()
