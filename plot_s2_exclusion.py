"""
S2 upper limit scan using saved toymc_s2 data.

Reads electron_data.npz and solar_nu_background.npz from each
(U2, MH) parameter directory, computes Asimov likelihood ratio,
and produces the expected 90% CL exclusion contour.

Usage:
    python plot_s2_exclusion.py <simulation_base_dir>
    python plot_s2_exclusion.py plots_grid_scan_s2_new
"""

import os
import sys
import glob
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm

from core.stats import chi2_poisson_likelihood_ratio


def plot_electron_counts_2d(param_dir, outdir=None):
    """Plot 2D scattered electron counts (E_e, cosθ) from saved NPZ data.

    Parameters
    ----------
    param_dir : str
        Path to parameter directory containing electron_data.npz and
        solar_nu_background.npz.
    outdir : str or None
        Output directory for plots (default: param_dir).
    """
    import matplotlib.pyplot as plt

    sig_path = os.path.join(param_dir, "electron_data.npz")
    bg_path = os.path.join(param_dir, "solar_nu_background.npz")
    if not os.path.exists(sig_path) or not os.path.exists(bg_path):
        print(f"  Error: missing NPZ files in {param_dir}")
        return

    sig = np.load(sig_path)
    bg = np.load(bg_path)

    e_w = np.diff(sig["e_bins"])
    ct_w = np.diff(sig["costheta_lab_bins"])
    sig_counts = sig["counts_2d"] * e_w[:, None] * ct_w[None, :]
    bg_counts = bg["bg_counts"]

    e_ctrs = 0.5 * (sig["e_bins"][:-1] + sig["e_bins"][1:])
    ct_ctrs = 0.5 * (sig["costheta_lab_bins"][:-1] + sig["costheta_lab_bins"][1:])

    if outdir is None:
        outdir = param_dir
    os.makedirs(outdir, exist_ok=True)

    dirname = os.path.basename(param_dir)
    tag = dirname.replace(".", "p")

    # ── Signal 2D counts (matplotlib) ──
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for ax, data, title, cmap, norm in [
        (axes[0], sig_counts, "Signal", "inferno", "log"),
        (axes[1], bg_counts, "Solar nu background", "viridis", "log"),
    ]:
        im = ax.pcolormesh(sig["e_bins"], sig["costheta_lab_bins"],
                           data.T, shading="auto", cmap=cmap, norm=norm)
        ax.set_xlabel(r"$E_e$ [MeV]")
        ax.set_ylabel(r"$\cos\theta$")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, label="Counts / bin")

    plt.suptitle(f"{dirname}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"electron_counts_2d_{tag}.pdf"),
                dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  Saved: electron_counts_2d_{tag}.pdf")

    # ── Signal 2D counts (ROOT) ──
    try:
        import ROOT as rt
        from pytools.rt_ploter import rt_plot_2d_heatmap

        h2d = rt.TH2D("h2d", ";Energy (MeV);Solar Angle Cosine",
                       len(e_ctrs), sig["e_bins"],
                       len(ct_ctrs), sig["costheta_lab_bins"])
        for ix in range(len(e_ctrs)):
            for iy in range(len(ct_ctrs)):
                h2d.SetBinContent(ix + 1, iy + 1, sig_counts[ix, iy])
        h2d.GetZaxis().SetTitle("Counts 500t 1yr")
        rt_plot_2d_heatmap(h2d, f"/electron_counts_2d_{tag}_rt",
                           n_levels=10, dir=outdir, type="pdf")
        print(f"  Saved: electron_counts_2d_{tag}_rt.pdf")
    except Exception as e:
        print(f"  ROOT plot skipped: {e}")


def compute_chi2_from_saved(param_dir):
    """Compute chi2 for a single saved parameter point.
    
    Returns
    -------
    dict or None
        {'MH', 'U2', 'chi2', 'signal_total'} if successful, else None
    """
    # Parse MH and U2 from directory name
    dirname = os.path.basename(param_dir)
    parts = dirname.split("_")
    try:
        u2_idx = parts.index("U2")
        mh_idx = parts.index("MH")
        U2 = float(parts[u2_idx + 1])
        MH = float(parts[mh_idx + 1])
    except (ValueError, IndexError):
        return None

    # Load signal counts (2D)
    sig_path = os.path.join(param_dir, "electron_data.npz")
    if not os.path.exists(sig_path):
        return None
    sig_data = np.load(sig_path)
    # counts_2d stored is counts MeV⁻¹ (Δcosθ)⁻¹
    e_bins = sig_data["e_bins"]
    ct_bins = sig_data["costheta_lab_bins"]
    e_widths = np.diff(e_bins)
    ct_widths = np.diff(ct_bins)
    sig_density = sig_data["counts_2d"]
    sig_per_bin = sig_density * e_widths[:, None] * ct_widths[None, :]  # → counts/bin

    # Load background counts (already counts per bin)
    bg_path = os.path.join(param_dir, "solar_nu_background.npz")
    if not os.path.exists(bg_path):
        return None
    bg_data = np.load(bg_path)
    bg_counts = bg_data["bg_counts"]

    # Ensure same shape (resample if needed)
    nE_sig, nA_sig = sig_per_bin.shape
    nE_bg, nA_bg = bg_counts.shape
    if nE_sig != nE_bg or nA_sig != nA_bg:
        from scipy.interpolate import interp2d
        bg_e_ctrs = 0.5 * (bg_data["e_bins"][:-1] + bg_data["e_bins"][1:])
        bg_ct_ctrs = 0.5 * (bg_data["ct_bins"][:-1] + bg_data["ct_bins"][1:])
        sig_e_ctrs = 0.5 * (e_bins[:-1] + e_bins[1:])
        sig_ct_ctrs = 0.5 * (ct_bins[:-1] + ct_bins[1:])
        f = interp2d(bg_ct_ctrs, bg_e_ctrs, bg_counts, kind="linear", fill_value=0)
        bg_counts = np.maximum(f(sig_ct_ctrs, sig_e_ctrs), 0.0)

    # Apply energy fit window (4.8–12.8 MeV) (> 2 MeV)
    # e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
    # fit_mask = (e_centers >= 2)
    # sig_per_bin = sig_per_bin[fit_mask, :]
    # bg_counts = bg_counts[fit_mask, :]

    chi2 = chi2_poisson_likelihood_ratio(sig_per_bin.ravel(), bg_counts.ravel())
    return {
        "MH": MH,
        "U2": U2,
        "chi2": chi2,
        "signal_total": np.sum(sig_per_bin),
        "background_total": np.sum(bg_counts),
    }


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    base_dir = sys.argv[1]
    start_time = time.time()

    # ======================================================================
    # CONFIGURATION
    # ======================================================================
    chi2_crit = 2.71  # 90% CL, 1 DOF, one-sided
    max_workers = min(8, os.cpu_count() or 4)

    outdir = os.path.join(base_dir, "s2_upper_limit")
    os.makedirs(outdir, exist_ok=True)

    # ======================================================================
    # FIND ALL PARAMETER DIRECTORIES
    # ======================================================================
    param_dirs = sorted(glob.glob(os.path.join(base_dir, "U2_*_MH_*")))
    if not param_dirs:
        print(f"No parameter directories found under {base_dir}")
        sys.exit(1)
    print(f"Found {len(param_dirs)} parameter sets")

    # ======================================================================
    # COMPUTE CHI2 FOR EACH POINT (PARALLEL)
    # ======================================================================
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(compute_chi2_from_saved, pd): pd
            for pd in param_dirs
        }
        for future in tqdm(futures, desc="Computing chi2", unit="pt"):
            try:
                res = future.result()
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"  Error: {e}")

    if not results:
        print("No valid results!")
        sys.exit(1)

    # Build grid
    all_mh = sorted(set(r["MH"] for r in results))
    all_u2 = sorted(set(r["U2"] for r in results))
    nU2 = len(all_u2)
    nMh = len(all_mh)
    mh_map = {mh: i for i, mh in enumerate(all_mh)}
    u2_map = {u2: i for i, u2 in enumerate(all_u2)}

    chi2_grid = np.full((nU2, nMh), np.nan)
    sig_grid = np.full((nU2, nMh), np.nan)
    bg_grid = np.full((nU2, nMh), np.nan)

    for r in results:
        iu2 = u2_map[r["U2"]]
        imh = mh_map[r["MH"]]
        chi2_grid[iu2, imh] = r["chi2"]
        sig_grid[iu2, imh] = r["signal_total"]
        bg_grid[iu2, imh] = r["background_total"]

    mh_vals = np.array(list(all_mh))
    u2_vals = np.array(list(all_u2))

    # ======================================================================
    # SAVE
    # ======================================================================
    np.savez(
        os.path.join(outdir, "s2_upper_limit_grid.npz"),
        MH_values=mh_vals,
        U2_values=u2_vals,
        chi2_grid=chi2_grid,
        signal_grid=sig_grid,
        background_grid=bg_grid,
        chi2_crit=chi2_crit,
    )

    # Exclusion contour
    mh_excl, u2_excl = [], []
    for imh in range(nMh):
        chi2_row = chi2_grid[:, imh]
        above = chi2_row >= chi2_crit
        if np.sum(~np.isnan(chi2_row)) < 2:
            continue
        if np.any(above):
            i_first = np.where(above)[0][0]
            if i_first == 0:
                u2_cross = u2_vals[0]
            else:
                log_u = np.interp(
                    chi2_crit,
                    [chi2_row[i_first - 1], chi2_row[i_first]],
                    [np.log10(u2_vals[i_first - 1]), np.log10(u2_vals[i_first])],
                )
                u2_cross = 10.0 ** log_u
            mh_excl.append(mh_vals[imh])
            u2_excl.append(u2_cross)

    excl_csv = os.path.join(outdir, "s2_expected_exclusion.csv")
    np.savetxt(
        excl_csv,
        np.column_stack([mh_excl, u2_excl]),
        delimiter=",", header="mH,u2_90CL", comments="",
    )
    print(f"\n>>> Saved: {excl_csv}")

    # ======================================================================
    # PLOTS
    # ======================================================================
    import matplotlib.pyplot as plt

    # Signal counts contour (Morandi purple)
    from matplotlib.colors import LinearSegmentedColormap
    morandi_purple = LinearSegmentedColormap.from_list(
        "morandi_purple",
        ["#E8E0ED", "#C4B5D0", "#9B86A8", "#7B628E", "#5C4270", "#3D2852"],
        N=256
    )
    plt.figure(figsize=(10, 7))
    plot_sig = np.where(sig_grid > 0, sig_grid, np.nan)
    log_max = np.nanmax(np.log10(plot_sig))
    if np.isfinite(log_max) and log_max > 0:
        levels = np.logspace(0, log_max, 8)
        cs = plt.contourf(mh_vals, u2_vals, plot_sig,
                      levels=levels, cmap=morandi_purple, alpha=0.85)
        cs2 = plt.contour(mh_vals, u2_vals, plot_sig,
                       levels=levels, colors="#4A3560", linewidths=0.8)
        plt.clabel(cs2, inline=True, fontsize=8, fmt="%.0e", colors="#4A3560")
    plt.xlabel(r"$M_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^{2}$")
    plt.yscale("log")
    plt.title("S2: total scattered electron counts (500t 1yr)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "s2_signal_counts_contour.pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {outdir}/s2_signal_counts_contour.pdf")

    # Chi2 grid (Morandi purple)
    plt.figure(figsize=(10, 7))
    plot_grid = np.where(chi2_grid > 0, chi2_grid, np.nan)
    log_max_c = np.nanmax(np.log10(np.where(plot_grid > 0, plot_grid, np.nan)))
    if np.isfinite(log_max_c) and log_max_c > 0:
        levels_c = np.logspace(0, log_max_c, 10)
        plt.contourf(mh_vals, u2_vals, plot_grid,
                      levels=levels_c, cmap=morandi_purple, alpha=0.85)
        cs2 = plt.contour(mh_vals, u2_vals, plot_grid,
                       levels=levels_c, colors="#4A3560", linewidths=0.6)
        plt.clabel(cs2, inline=True, fontsize=7, fmt="%.1f", colors="#4A3560")
    plt.xlabel(r"$M_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^{2}$")
    plt.yscale("log")
    plt.title("S2 Expected Exclusion (Asimov)")

    if mh_excl:
        plt.plot(mh_excl, u2_excl, "-", color="#D45050", linewidth=2.5, label="90% CL")
        plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "s2_chi2_grid.pdf"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved: {outdir}/s2_chi2_grid.pdf")

    # Exclusion contour (matplotlib, skip ROOT due to segfault issues)
    if mh_excl:
        plt.figure(figsize=(10, 7))
        plt.plot(mh_excl, u2_excl, "-", color="#D45050", linewidth=2.5)
        plt.yscale("log")
        plt.xlim(2.0, 15.0)
        plt.ylim(1e-6, 1e0)
        plt.xlabel(r"$m_H$ [MeV]")
        plt.ylabel(r"$|U_{eH}|^{2}$")
        plt.title("S2 Expected Exclusion (90% CL, Asimov)")
        plt.grid(True, which="both", ls=":", alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "s2_expected_exclusion.pdf"), dpi=150)
        plt.close()
        print(f"    Saved: {outdir}s2_expected_exclusion.pdf")

    # ======================================================================
    # TIMING
    # ======================================================================
    elapsed = time.time() - start_time
    m, s = divmod(elapsed, 60)
    print(f"\nTotal runtime: {int(m):d}m {s:.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--plot":
        if len(sys.argv) < 3:
            print("Usage: python plot_s2_exclusion.py --plot <param_dir>")
            sys.exit(1)
        plot_electron_counts_2d(sys.argv[2])
    else:
        main()
