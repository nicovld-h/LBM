import numpy as np
import matplotlib.pyplot as plt
import glob, os

L_BOX_DEFAULT = 2*np.pi
URMS0_DEFAULT = 0.05   # initial urms used in the solver

plt.rcParams.update({
    "axes.titlesize": 18,   # subplot titles
    "axes.labelsize": 16,   # x/y labels
    "legend.fontsize": 14,  # legend text
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 20  # suptitle
})

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

"""
Band-limited resampling from Ny_srcxNx_src -> Ny_refxNx_ref using FFT.
    - Crops (if ref < src) or zero-pads (if ref > src) the centered spectrum.
    - Applies the correct normalization so physical amplitudes are preserved:
        beta = (Nx_ref*Ny_ref)/(Nx_src*Ny_src)
Returns (ux_ref, uy_ref) on the reference grid.
"""
def spectral_resample_to_ref(ux, uy, Nx_src, Ny_src, Nx_ref, Ny_ref):
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

    # L_box as before
    if "L_box" in d:
        L_box = float(d["L_box"])
    elif "L2D" in d and "k_peak" in d:
        L_box = float(d["L2D"]) * (2*np.pi / float(d["k_peak"]))
    else:
        L_box = L_BOX_DEFAULT

    # tau / Re0 (if present), else compute Re0 from URMS0_DEFAULT and L2D
    tau  = float(d["tau"]) if "tau" in d else np.nan
    Re0  = float(d["Re0"]) if "Re0" in d else None
    if Re0 is None:
        L2D = float(d["L2D"]) if "L2D" in d else (L_box / 8.0)  # m=8 default
        if not np.isnan(tau):
            nu = (1.0/3.0)*(tau-0.5)
            Re0 = URMS0_DEFAULT * L2D / nu
        else:
            Re0 = np.nan

    times = np.array(d["u_times"], dtype=int) if "u_times" in d else np.array([], dtype=int)

    # Prefer non-empty ref snapshots (same as you had)
    has_ref_keys = ("u_snaps_ref" in d and "v_snaps_ref" in d)
    ref_nonempty = has_ref_keys and (d["u_snaps_ref"].size > 0) and (d["v_snaps_ref"].size > 0)

    if ref_nonempty:
        ux_list = _as_list(d["u_snaps_ref"])
        uy_list = _as_list(d["v_snaps_ref"])
    else:
        ux_list = _as_list(d["u_snaps"]) if "u_snaps" in d else []
        uy_list = _as_list(d["v_snaps"]) if "v_snaps" in d else []

    n = min(len(times), len(ux_list), len(uy_list))
    times = times[:n]; ux_list = ux_list[:n]; uy_list = uy_list[:n]

    return {
        "path": path, "Nx": Nx, "Ny": Ny, "L_box": L_box,
        "tau": tau, "Re0": Re0,
        "times": times,
        "ux_list_native": ux_list, "uy_list_native": uy_list,
    }

"""
Per-τ convergence analysis. Returns the maximum error seen (for y-axis scaling).
    - y_limits: tuple (ymin, ymax) to enforce same scale across groups (or None).
    - do_plot: False to only compute/print and return ymax without plotting.
"""
def analyze_group(runs_group, group_name, y_limits=None, do_plot=True):
    if not runs_group:
        print(f"[{group_name}] No runs with usable snapshots; skipping.")
        return 0.0

    # ---- bucket by tau ----
    buckets = {}
    for r in runs_group:
        tau = float(r["tau"]) if np.isfinite(r["tau"]) else None
        if tau is None:
            print(f"[{group_name}] Skipping {os.path.basename(r['path'])}: missing tau")
            continue
        buckets.setdefault(tau, []).append(r)

    ymax_all = 0.0
    if do_plot:
        plt.figure(figsize=(6,4))

    for tau, bucket in sorted(buckets.items()):
        # DNS for this τ = largest N in this τ-bucket
        dns = max(bucket, key=lambda rr: rr["Nx"])
        L_box = dns["L_box"]; Nx_ref, Ny_ref = dns["Nx"], dns["Ny"]

        # spectral-resample every run’s snapshots to this τ-DNS grid
        for rr in bucket:
            rr["ux_list_ref"], rr["uy_list_ref"] = [], []
            for ux, uy in zip(rr["ux_list_native"], rr["uy_list_native"]):
                ux_r, uy_r = spectral_resample_to_ref(
                    ux, uy, Nx_src=rr["Nx"], Ny_src=rr["Ny"], Nx_ref=Nx_ref, Ny_ref=Ny_ref
                )
                rr["ux_list_ref"].append(ux_r); rr["uy_list_ref"].append(uy_r)

        # bandlimit at smallest Nyquist within this τ-bucket
        N_min = min(rr["Nx"] for rr in bucket)
        k_cut = np.pi * N_min / L_box
        print(f"[{group_name}] τ={tau:.4f}: N_min={N_min}, k_cut={k_cut:.6f}")

        # cache low-passed DNS snapshots for this τ
        dns_lp_cache = {}
        for i, t in enumerate(dns["times"]):
            ux_a, uy_a = dns["ux_list_ref"][i], dns["uy_list_ref"][i]
            dns_lp_cache[int(t)] = fft_lowpass(ux_a, uy_a, k_cut, L_box)

        # compute errors vs this τ-DNS
        items = []
        for rr in sorted(bucket, key=lambda x: x["Nx"]):
            common = np.intersect1d(rr["times"], dns["times"])
            if common.size == 0:
                print(f"[{group_name}] τ={tau:.4f}: no common times with {os.path.basename(dns['path'])} for {os.path.basename(rr['path'])}")
                continue
            idx_r = {int(tt): i for i, tt in enumerate(rr["times"])}
            errs = []
            for tt in common:
                ir = idx_r[int(tt)]
                ux_c, uy_c = fft_lowpass(rr["ux_list_ref"][ir], rr["uy_list_ref"][ir], k_cut, L_box)
                ux_a, uy_a = dns_lp_cache[int(tt)]
                diff_L2 = np.sqrt(np.sum((ux_c - ux_a)**2 + (uy_c - uy_a)**2))
                ref_L2  = np.sqrt(np.sum(ux_a**2 + uy_a**2))
                if ref_L2 > 0:
                    errs.append(diff_L2 / ref_L2)
            if not errs:
                continue
            rel_L2_RMS = float(np.sqrt(np.mean(np.square(errs))))
            items.append((rr["Nx"], rel_L2_RMS, rr["Re0"], os.path.basename(rr["path"]), rr is dns))

        if not items:
            print(f"[{group_name}] τ={tau:.4f}: nothing to plot"); 
            continue

        Ns   = [it[0] for it in items]
        Errs = [it[1] for it in items]
        Re0s = [it[2] for it in items]
        ymax_all = max(ymax_all, max(Errs) if Errs else 0.0)

        # plot τ curve
        if do_plot:
            plt.plot(Ns, Errs, marker="o", lw=2, label=f"τ={tau:.4f} (Re≈{Re0s[-1]:.0f})")

        # console print with correct DNS tag by file identity
        print(f"\n[{group_name}] τ={tau:.4f}")
        for (N, e, Re0, name, is_dns) in items:
            tag = " (DNS)" if is_dns else ""
            print(f"  N={N:4d}  RelErr={e:.6e}  {name}{tag}")

    # finalize plot
    if do_plot:
        plt.xlabel("Grid size N")
        plt.ylabel("Relative error in velocity U")
        plt.title(f"{group_name}: Grid convergence of the cascade LBM for the U-velocity component")
        plt.grid(True, alpha=0.3)
        if y_limits is not None:
            plt.ylim(*y_limits)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return ymax_all


def main(pattern="lbm_run_N*.npz"):
    MAX_T = 400   # compare up to this time step (t=0..MAX_T)

    files = sorted(glob.glob(pattern))
    files += sorted(glob.glob("lbm_runMRT_N*.npz"))  # add MRT files
    files = sorted(set(files))  # dedupe

    if not files:
        raise FileNotFoundError(f"No files match {pattern}")

    runs_all = [load_run_native(f) for f in files]
    runs_all = [r for r in runs_all if len(r["times"]) > 0 and len(r["ux_list_native"]) > 0]
    if not runs_all:
        raise RuntimeError("No runs with usable snapshots found.")

    # Split by filename into two groups
    runs_bgk = [r for r in runs_all if "MRT" not in r["path"]]
    runs_mrt = [r for r in runs_all if "MRT"     in r["path"]]

    # First pass: compute maxima without plotting
    ymax_bgk = analyze_group(runs_bgk, "BGK", do_plot=False)
    ymax_mrt = analyze_group(runs_mrt, "MRT", do_plot=False)

    # Common y-limits
    common_ylim = (-0.1, 1.05 * max(ymax_bgk, ymax_mrt, 1e-12))

    # Second pass: plot both with identical y-scale
    analyze_group(runs_bgk, "BGK", y_limits=common_ylim, do_plot=True)
    analyze_group(runs_mrt, "MRT", y_limits=common_ylim, do_plot=True)


if __name__ == "__main__":
    main()
