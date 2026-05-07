"""
Plot saved S2 simulation results.

Usage:
    python plot_s2_results.py <param_dir>
    python plot_s2_results.py plots_grid_scan_s2/U2_1.00e-01_MH_4.0

Or to plot all subdirectories under a given directory:
    python plot_s2_results.py plots_grid_scan_s2 --all
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

from ploter import (
    plot_El_costheta_map,
    plot_1d_energy_distribution,
    plot_1d_angle_distribution,
)


def plot_one_param_set(param_dir):
    """Generate all plots from saved data in param_dir."""
    print(f"\n=== Plotting: {param_dir} ===")

    # ── Load neutrino data (saved by get_and_save_nuL_El_costheta_decay_in_flight) ──
    # Infer MH and U2 from directory name
    dirname = os.path.basename(param_dir)
    parts = dirname.split("_")
    # Expected format: U2_{value}_MH_{value}
    try:
        u2_idx = parts.index("U2")
        mh_idx = parts.index("MH")
        U2 = float(parts[u2_idx + 1])
        MH = float(parts[mh_idx + 1])
    except (ValueError, IndexError):
        print(f"  Warning: Could not parse U2/MH from directory name '{dirname}', trying files...")
        U2, MH = None, None

    # Load neutrino CSV files
    nu_2d_csv = glob.glob(os.path.join(param_dir, "diff_El_costheta_M*.csv"))
    nu_el_csv = glob.glob(os.path.join(param_dir, "diff_El_M*.csv"))
    nu_ct_csv = glob.glob(os.path.join(param_dir, "diff_costheta_M*.csv"))

    if nu_2d_csv:
        nu_2d = np.loadtxt(nu_2d_csv[0], delimiter=",", skiprows=1)
        # Reshape from flat to (nE, nTh, 3)
        nE_nu = len(np.unique(nu_2d[:, 0]))
        nTh_nu = len(np.unique(nu_2d[:, 1]))
        diff_El_costheta_nu = nu_2d.reshape(nE_nu, nTh_nu, 3)
        print(f"  Loaded neutrino 2D: {nE_nu}×{nTh_nu}")
    else:
        print(f"  Warning: No neutrino 2D CSV found")
        diff_El_costheta_nu = None

    if nu_el_csv:
        diff_El_nu = np.loadtxt(nu_el_csv[0], delimiter=",", skiprows=1)
        print(f"  Loaded neutrino 1D energy: {len(diff_El_nu)} points")
    else:
        diff_El_nu = None

    if nu_ct_csv:
        diff_costheta_nu = np.loadtxt(nu_ct_csv[0], delimiter=",", skiprows=1)
        print(f"  Loaded neutrino 1D angle: {len(diff_costheta_nu)} points")
    else:
        diff_costheta_nu = None

    # ── Load electron data (saved by process2_single_parameter_set) ──
    elec_npz = os.path.join(param_dir, "electron_data.npz")
    if os.path.exists(elec_npz):
        data = np.load(elec_npz)
        e_centers = data["e_centers"]
        costheta_centers = data["costheta_centers"]
        e_bins = data["e_bins"]
        costheta_lab_bins = data["costheta_lab_bins"]

        nE_e = len(e_centers)
        nA_e = len(costheta_centers)

        # Reconstruct 3D arrays for plot functions
        counts_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
        counts_El_costheta_electron[:, :, 0] = e_centers[:, None]
        counts_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
        counts_El_costheta_electron[:, :, 2] = data["counts_2d"]

        # Convert 2D counts density from Counts/(MeV·Δcosθ) to Counts/MeV for plotting
        costheta_widths = np.diff(costheta_lab_bins)
        counts_El_costheta_electron_perMeV = counts_El_costheta_electron.copy()
        counts_El_costheta_electron_perMeV[:, :, 2] = (
            counts_El_costheta_electron[:, :, 2] * costheta_widths[None, :]
        )

        diff_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
        diff_El_costheta_electron[:, :, 0] = e_centers[:, None]
        diff_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
        diff_El_costheta_electron[:, :, 2] = data["rate_2d"]

        diff_El_electron = np.zeros((nE_e, 2))
        diff_El_electron[:, 0] = e_centers
        diff_El_electron[:, 1] = data["energy_1d_rate"]

        counts_El_electron = np.zeros((nE_e, 2))
        counts_El_electron[:, 0] = e_centers
        counts_El_electron[:, 1] = data["energy_1d_counts"]

        diff_costheta_electron = np.zeros((nA_e, 2))
        diff_costheta_electron[:, 0] = costheta_centers
        diff_costheta_electron[:, 1] = data["angle_1d"]

        # Extract metadata
        U2 = float(data["U2"]) if U2 is None else U2
        MH = float(data["MH"]) if MH is None else MH

        print(f"  Loaded electron data: {nE_e}×{nA_e}")
    else:
        print(f"  Error: No electron_data.npz found in {param_dir}")
        return

    # ── Load solar neutrino background data ──
    bg_npz = os.path.join(param_dir, "solar_nu_background.npz")
    has_bg = os.path.exists(bg_npz)
    if has_bg:
        bg = np.load(bg_npz)
        print(f"  Loaded solar neutrino background data")

    title_tag = f"U²={U2:.2e}, M={MH:.1f} MeV"

    # ── Plot 1: Neutrino 2D ──
    if diff_El_costheta_nu is not None:
        print("\nPlotting neutrino 2D distribution...")
        plot_El_costheta_map(
            diff_El_costheta_nu,
            param_dir,
            filename=f"neutrino_2d_U2_{U2:.2e}_MH_{MH:.1f}.png",
            title_prefix=f"Neutrino: {title_tag}",
        )

    # ── Plot 2: Neutrino 1D energy ──
    if diff_El_nu is not None:
        print("Plotting neutrino 1D energy distribution...")
        plot_1d_energy_distribution(
            diff_El_nu,
            param_dir,
            filename=f"neutrino_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Neutrino Energy: {title_tag}",
            ylabel="Flux (MeV$^{-1}$ cm$^{-2}$ s$^{-1}$)",
        )

    # ── Plot 3: Neutrino 1D angle ──
    if diff_costheta_nu is not None:
        print("Plotting neutrino 1D angular distribution...")
        plot_1d_angle_distribution(
            diff_costheta_nu,
            param_dir,
            filename=f"neutrino_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Neutrino Angular: {title_tag}",
            ylabel="Flux (cm$^{-2}$ s$^{-1}$)",
        )

    # ── Plot 4: Electron 2D counts (matplotlib) ──
    print("Plotting electron 2D counts distribution...")
    plot_El_costheta_map(
        counts_El_costheta_electron_perMeV,
        param_dir,
        filename=f"electron_counts_2d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Electron: {title_tag}",
    )

    # ── Plot 4b: Electron 2D counts (ROOT) ──
    print("Plotting electron 2D counts distribution (ROOT)...")
    import ROOT as rt
    from pytools.rt_ploter import rt_plot_2d_heatmap
    h2d = rt.TH2D(
        "h2d", ";Energy (MeV);Solar Angle Cosine",
        nE_e, e_bins, nA_e, costheta_lab_bins,
    )
    for ix in range(nE_e):
        for iy in range(nA_e):
            h2d.SetBinContent(ix + 1, iy + 1, counts_El_costheta_electron_perMeV[ix, iy, 2])
    h2d.GetZaxis().SetTitle("Counts / MeV 500t 1yr")
    rt_plot_2d_heatmap(
        h2d,
        f"/electron_counts_2d_U2_{U2:.2e}_MH_{MH:.1f}_rt",
        n_levels=10,
        dir=param_dir,
        type="pdf",
    )

    # ── Plot 5: Electron 1D energy ──
    print("Plotting electron 1D energy distribution...")
    plot_1d_energy_distribution(
        diff_El_electron,
        param_dir,
        filename=f"electron_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Electron Energy: {title_tag}",
        ylabel="Event rate [s$^{-1}$ MeV$^{-1}$]",
    )

    # ── Plot 6: Electron 1D angle ──
    print("Plotting electron 1D angular distribution...")
    plot_1d_angle_distribution(
        diff_costheta_electron,
        param_dir,
        filename=f"electron_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Electron Angular: {title_tag}",
        ylabel="Event rate [s$^{-1}$]",
    )

    # ── Plot 7: Solar neutrino background 2D counts (ROOT) ──
    if has_bg:
        print("Plotting solar neutrino background 2D counts (ROOT)...")
        bg_e_bins = bg["e_bins"]
        bg_ct_bins = bg["ct_bins"]
        bg_counts = bg["bg_counts"]
        nE_bg = len(bg_e_bins) - 1
        nA_bg = len(bg_ct_bins) - 1
        h2d_bg = rt.TH2D(
            "h2d_bg", ";Energy (MeV);Solar Angle Cosine",
            nE_bg, bg_e_bins, nA_bg, bg_ct_bins,
        )
        for ix in range(nE_bg):
            for iy in range(nA_bg):
                h2d_bg.SetBinContent(ix + 1, iy + 1, bg_counts[ix, iy])
        h2d_bg.GetZaxis().SetTitle("Counts 500t 1yr")
        # Keep palette direction consistent with signal 2D ROOT plot (4b)
        rt.TColor.InvertPalette()
        rt_plot_2d_heatmap(
            h2d_bg,
            f"/solar_nu_bg_counts_2d_rt",
            n_levels=10,
            dir=param_dir,
            type="pdf",
        )

    # ── Plot 8: RHN signal vs solar neutrino background ──
    if has_bg:
        print("Plotting RHN signal vs solar neutrino background comparison...")
        ct_signal = bg["costheta_signal"]
        angle_signal = bg["angle_1d_signal"]
        ct_bg = bg["costheta_bg"]
        angle_bg = bg["angle_1d_bg"]

        _, ax = plt.subplots(1, 1, figsize=(10, 6))
        ax.plot(ct_signal, angle_signal, "b-", lw=2, label="RHN signal (ES from decay nu)")
        ax.plot(ct_bg, angle_bg, "r--", lw=2, label="Solar nu ES background")
        ax.set_xlabel(r"cos$\theta$ (lab frame)")
        ax.set_ylabel("Event rate [s$^{-1}$]")
        ax.set_title(f"ES Electron Angular Distribution: {title_tag}")
        ax.grid(True, alpha=0.3)
        ax.legend()

        plt.savefig(
            os.path.join(param_dir, f"comparison_signal_vs_bg_U2_{U2:.2e}_MH_{MH:.1f}.png"),
            dpi=150, bbox_inches="tight",
        )
        plt.close()

        print("  Saved signal vs background comparison plot")

    # ── Plot 9: Signal / Background 2D ratio (ROOT) ──
    if has_bg:
        print("Plotting signal / background 2D ratio (ROOT)...")
        sig_counts = counts_El_costheta_electron[:, :, 2]  # counts density
        bg_counts = bg["bg_counts"]  # counts per bin

        # Resample to common binning if needed (assumes same scatter function output)
        # Both use internal scatter bins: energy range 0-16 MeV, cosθ range -1 to 1
        # Check if binning matches; if not, skip
        e_match = len(e_bins) == len(bg["e_bins"]) and np.allclose(e_bins, bg["e_bins"])
        ct_match = len(costheta_lab_bins) == len(bg["ct_bins"]) and np.allclose(costheta_lab_bins, bg["ct_bins"])

        if e_match and ct_match:
            nE_r = nE_e
            nA_r = nA_e

            # Build ratio: signal / background (handle zeros)
            ratio = np.where(bg_counts > 0, sig_counts / bg_counts, 0.0)
            # Mask bins with < 0.01 bg counts as not statistically meaningful
            ratio = np.where(bg_counts > 0.01, ratio, np.nan)

            h2d_ratio = rt.TH2D(
                "h2d_ratio", ";Energy (MeV);Solar Angle Cosine",
                nE_r, e_bins, nA_r, costheta_lab_bins,
            )
            for ix in range(nE_r):
                for iy in range(nA_r):
                    if not np.isnan(ratio[ix, iy]):
                        h2d_ratio.SetBinContent(ix + 1, iy + 1, ratio[ix, iy])

            h2d_ratio.GetZaxis().SetTitle("Signal / Background")
            rt_plot_2d_heatmap(
                h2d_ratio,
                f"/signal_bg_ratio_2d_U2_{U2:.2e}_MH_{MH:.1f}_rt",
                n_levels=20,
                dir=param_dir,
                type="pdf",
            )
            print("  Saved signal/bg 2D ratio plot")
        else:
            print("  Warning: Signal and background binning mismatch, skipping ratio plot")

    print(f"  Done: {param_dir}")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    target = sys.argv[1]
    plot_all = "--all" in sys.argv

    if plot_all:
        # Plot all subdirectories
        subdirs = sorted(glob.glob(os.path.join(target, "U2_*_MH_*")))
        if not subdirs:
            print(f"No parameter subdirectories found under {target}")
            sys.exit(1)
        for sd in subdirs:
            if os.path.isdir(sd):
                plot_one_param_set(sd)
    else:
        # Single directory
        if not os.path.isdir(target):
            print(f"Error: {target} is not a directory")
            sys.exit(1)
        plot_one_param_set(target)


if __name__ == "__main__":
    main()
