"""
Complete analysis workflows for Solar RHN parameter scans.

This module provides high-level workflow functions for:
- Listing and discovering neutrino distribution CSV files
- Batch processing of electron scattering computations
- End-to-end parameter set processing with full visualization pipeline

Key Functions
-------------
- list_neutrino_csv_files: Find all neutrino CSV files in a directory
- batch_compute_electrons_from_csv: Process all CSV files in batch
- process_single_parameter_set: Complete workflow for single (U², M_H) point
  including neutrino decay, electron scattering, and plotting
"""

from math import ceil, floor
import os
import glob
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import ticker
from array import array

from core import *


def plot_ER(
    ES: np.ndarray, MH: np.ndarray, U2: float, savepath: str = "./plots/ER/"
) -> None:
    """Plot energy distribution of RH neutrinos mixed from solar neutrinos (U2=1.0).

    Parameters
    ----------
    ES : ndarray (N, 2)
        Solar neutrino energy distribution
    MH : ndarray
        RH neutrino mass
    U2 : float
        Mixing parameter squared
    savepath : str
        ER plot save path
    log : bool
        Whether to use logarithmic scale for y-axis
    """
    from core.rhn_physics import getRHNSpectrum

    os.makedirs(savepath, exist_ok=True)

    cmap = plt.get_cmap("viridis")
    RH_colors = cmap(np.linspace(0, 1, len(MH)))

    # plt.figure()
    # plt.xlim(0, 16)
    # plt.plot(ES[:, 0], ES[:, 1], label='SN Spectrum', color='black')

    # for m, color in zip(MH, RH_colors):
    #     rhn_spectrum = getRHNSpectrum(ES, m, U2)

    #     plt.plot(rhn_spectrum[:, 0], rhn_spectrum[:, 1],
    #              label=f'RHN Spectrum (M={m:.1f} MeV)', color=color)
    #     plt.xlabel(r'\text{Energy} ($\unit{MeV}$)', fontsize=12)
    #     plt.ylabel(r'\text{Flux} ($\unit{MeV^{-1}cm^{-2}s^{-1}}$)', fontsize=12)
    #     plt.grid(True, alpha=0.3)
    #     plt.legend()
    #     plt.tight_layout()

    # plot_file = os.path.join(savepath, f'RHN_spectrum_U{U2:.1f}_lin.pdf')
    # plt.savefig(plot_file)
    # plt.close()

    # print(f"Saved ER lin plots to {savepath}.")

    eps = 1e1
    plt.figure()
    plt.yscale("log")
    plt.xlim(0, 16)
    plt.ylim(eps + 1, 1e7)
    plt.plot(ES[:, 0], np.clip(ES[:, 1], eps, None), label="SN Spectrum", color="black")
    for m, color in zip(MH, RH_colors):
        rhn_spectrum = getRHNSpectrum(ES, m, U2)
        plt.plot(
            rhn_spectrum[:, 0],
            np.clip(rhn_spectrum[:, 1], eps, None),
            label=f"RHN Spectrum ($m_{{\nu_H}}={m:.1f}$ MeV)",
            color=color,
        )

    plt.xlabel("Energy (MeV)", fontsize="small")
    plt.ylabel("Flux (MeV$^{-1}$ cm$^{-2}$ s$^{-1}$)", fontsize="small")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(savepath, f"RHN_spectrum_U{U2:.1f}_log.pdf")
    plt.legend(fontsize="small")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved ER log plots to {savepath}.")


def plot_lifetimes(savepath: str = "plots/LT/") -> None:
    """Plot RHN lifetimes for various U2 values.

    Parameters
    ----------
    savepath : str
        Lifetime plot save path.
    """
    from core.rhn_physics import RHN_TauCM

    os.makedirs(savepath, exist_ok=True)

    MH = np.linspace(2.0, 16.0, 151)
    U2 = np.logspace(-6, 0, 71)

    lifetime = np.zeros((MH.size, U2.size))

    for i, u2 in enumerate(U2):
        lifetime[:, i] = RHN_TauCM(MH, u2)

    log_norm = colors.LogNorm(vmin=lifetime.min(), vmax=lifetime.max())
    levels = np.logspace(
        floor(np.log10(lifetime.min())), ceil(np.log10(lifetime.max())), 12
    )

    # Heatmap
    fig, ax = plt.subplots()

    c = ax.pcolormesh(MH, U2, lifetime.T, shading="auto", cmap="viridis", norm=log_norm)
    ax.set_yscale("log")
    ax.set_xlim(2, 16)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$m_{\nu_H}$ (MeV)")
    ax.set_ylabel(r"$|U_{eH}|^2$")
    cbar = fig.colorbar(c, ax=ax, label=r"$\tau_{c.m.}$ (s)")
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())

    # sns.heatmap(lifetime, cbar=True, ax=ax)
    # n_cols, n_rows = lifetime.shape
    # x_ticks = np.arange(0, n_cols, 10)
    # y_ticks = np.arange(0, n_rows, 10)
    # ax.set_xticks(x_ticks)
    # ax.set_yticks(y_ticks)
    # ax.set_xticklabels([f"{MH[i]:.1f}" for i in x_ticks], rotation=45)
    # ax.set_yticklabels([f"{U2[i]:.1e}" for i in y_ticks], rotation=45)

    plt.tight_layout()
    plt.savefig(
        os.path.join(savepath, "RHN_lifetime_heatmap.pdf"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        f"Saved RHN lifetime heatmap to {os.path.join(savepath, 'RHN_lifetime_heatmap.pdf')}."
    )

    # Contour
    fig, ax = plt.subplots()
    CS = ax.contour(MH, U2, lifetime.T, levels=20, cmap="viridis", norm=log_norm)
    ax.set_yscale("log")
    ax.set_xlim(2, 16)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$m_{\nu_H}$ (MeV)")
    ax.set_ylabel(r"$|U_{eH}|^2$")
    cbar = fig.colorbar(CS, ax=ax, label=r"$\tau_{c.m.}$ (s)")
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    plt.tight_layout()
    plt.savefig(
        os.path.join(savepath, "RHN_lifetime_contour.pdf"), dpi=300, bbox_inches="tight"
    )
    plt.close()
    print(
        f"Saved RHN lifetime contour to {os.path.join(savepath, 'RHN_lifetime_contour.pdf')}."
    )

    # Contourf
    fig, ax = plt.subplots()
    CS = ax.contourf(MH, U2, lifetime.T, levels=levels, cmap="viridis", norm=log_norm)
    ax.set_yscale("log")
    ax.set_xlim(2, 16)
    ax.set_ylim(1e-6, 1)
    ax.set_xlabel(r"$m_{\nu_H}$ (MeV)")
    ax.set_ylabel(r"$|U_{eH}|^2$")
    ax.set_xticks(np.arange(2, 17, 2))
    cbar = fig.colorbar(CS, ax=ax, label=r"$\tau_{c.m.}$ (s)")
    cbar.ax.yaxis.set_major_formatter(ticker.LogFormatterMathtext())
    plt.tight_layout()
    plt.savefig(
        os.path.join(savepath, "RHN_lifetime_contourf.pdf"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(
        f"Saved RHN lifetime contourf to {os.path.join(savepath, 'RHN_lifetime_contourf.pdf')}."
    )


def plot_branch_ratios(savepath: str = "plots/BR/") -> None:
    """Plot vll/(vll + vvv).

    Parameters
    ----------
    savepath : str
        Branch ratio save path.
    """
    from core.rhn_physics import RHN_BR_vll

    os.makedirs(savepath, exist_ok=True)

    MH = np.linspace(2.0, 16.0, 151)
    U2 = 1.0

    br_vll = RHN_BR_vll(MH, U2)

    plt.figure()
    plt.plot(MH, br_vll, "-", linewidth=2)
    plt.xlim(2, 16)
    plt.xlabel(r"$m_{\nu_H}$ (MeV)")
    plt.ylabel(r"$\nu ll/(\nu ll + \nu\nu\nu)$")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plot_file = os.path.join(savepath, f"RHN_branch_ratio_vll.pdf")
    plt.savefig(plot_file, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved RHN branching ratio plot to {plot_file}.")


# @timer
def getNuleeInDetector(
    spectrum_RHN: np.ndarray,
    MH: float,
    U2: float,
    cosphi_bins: np.ndarray = np.linspace(-1, 1, 201),
) -> tuple:
    """Calculate energy and angular distribution of neutrino and electron-pair from RHN decay inside detector.

    Parameters
    ----------
    spectrum_RHN : ndarray (N, 2)
        RHN spectrum
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    cosphi_bins : ndarray (P,)
        Cosine of angle bins

    Returns
    -------
    tuple of ndarray
        (diff_El_decayed, diff_cosphi_decayed, diff_El_cosphi_decayed, diff_Eee_decayed, diff_cosphi_ee_decayed)
        - diff_El_decayed : ndarray (N, 2)
            Energy distribution of decayed neutrinos
        - diff_cosphi_decayed : ndarray (P, 2)
            Angular distribution of decayed neutrinos
        - diff_El_cosphi_decayed : ndarray (N, P, 3)
            2D distribution of decayed neutrinos in (E, cosphi)
        - diff_Eee_decayed : ndarray (N, 2)
            Energy distribution of decayed electron-pairs
        - diff_cosphi_ee_decayed : ndarray (P, 2)
            Angular distribution of decayed electron-pairs
        - diff_Eee_cosphi_decayed : ndarray (N, P, 3)
            2D distribution of decayed electron-pairs in (Eee, cosphi)
    """
    # print(
    # f"Calculating decayed neutrino distributions for MH={MH:.1f} MeV, U2={U2:.1e}...")
    # print("Step 1: Calculating decayed LHN energy distribution inside detector...")
    spectrum_decayed = getDecayedRHNSpectrum_vll(
        spectrum_RHN, MH, U2, distance_SE, attenuation_length
    )  # vll branch only for my simulation
    # if boolBorexino:
    #     spectrum_decayed = getDecayedRHNSpectrum(
    #         spectrum_RHN, MH, U2, distance_SE, attenuation_length
    #     )  # vll + vvv branch for Borexino simulation
    # else:
    #     spectrum_decayed = getDecayedRHNSpectrum_vll(
    #         spectrum_RHN, MH, U2, distance_SE, attenuation_length
    #     )  # vll branch only for my simulation
    # print("Sum(decayed RHN flux) = ", np.sum(spectrum_decayed[:, 1]))
    energy = spectrum_decayed[:, 0]
    estep = energy[1] - energy[0]
    flux_RHN_decayed = spectrum_decayed[:, 1]
    # sum_flux_decayed = np.sum(flux_RHN_decayed)

    npoints_cosphi = len(cosphi_bins)
    cosstep = cosphi_bins[1] - cosphi_bins[0]

    diff_El_decayed = np.zeros((len(energy), 2))
    diff_cosphi_decayed = np.zeros((npoints_cosphi, 2))
    diff_El_cosphi_decayed = np.zeros((len(energy), npoints_cosphi, 3))

    diff_El_decayed[:, 0] = energy
    diff_cosphi_decayed[:, 0] = cosphi_bins
    diff_El_cosphi_decayed[:, :, 0] = energy[:, np.newaxis]
    diff_El_cosphi_decayed[:, :, 1] = cosphi_bins[np.newaxis, :]

    diff_Eee_decayed = np.zeros((len(energy), 2))
    diff_cosphi_ee_decayed = np.zeros((npoints_cosphi, 2))
    diff_Eee_cosphi_decayed = np.zeros((len(energy), npoints_cosphi, 3))

    diff_Eee_decayed[:, 0] = energy
    diff_cosphi_ee_decayed[:, 0] = cosphi_bins
    diff_Eee_cosphi_decayed[:, :, 0] = energy[:, np.newaxis]
    diff_Eee_cosphi_decayed[:, :, 1] = cosphi_bins[np.newaxis, :]

    for ie, EH in enumerate(energy):
        if EH <= MH:
            continue
        flux_EH = flux_RHN_decayed[ie]  # RHN flux at EH
        PH = np.sqrt(EH**2 - MH**2)  # RHN momentum
        # Loop for differential distributions
        for iel in range(len(energy)):
            El = energy[iel]  # LHN diff energy
            if El >= EH or El <= 0.0:
                continue
            Eee = EH - El
            iee = int(np.round(Eee / estep))
            if iee < 0 or iee >= len(energy):
                continue
            diff_Eee_decayed[iee, 1] += flux_EH * diff_Eee(Eee, MH, EH)
            for icosphi in range(npoints_cosphi):
                cosphi = cosphi_bins[icosphi]  # LHN diff angle
                diff_El_costheta = diff_El_costheta_lab(El, cosphi, MH, EH)
                diff_El_cosphi_decayed[iel, icosphi, 2] += flux_EH * diff_El_costheta
                diff_cosphi_decayed[icosphi, 1] += flux_EH * diff_El_costheta
                diff_El_decayed[iel, 1] += flux_EH * diff_El_costheta

                # Electron-pair angle distribution
                Pee_sq = PH**2 + El**2 - 2 * PH * El * cosphi
                if Pee_sq <= 0:
                    Pee = 0.0
                    cos_ee = 1.0
                else:
                    Pee = np.sqrt(Pee_sq)
                    cos_ee = (PH - El * cosphi) / Pee

                cos_ee = np.clip(cos_ee, -1.0, 1.0)
                icos_ee = int(np.round((cos_ee - cosphi_bins[0]) / cosstep))
                if icos_ee < 0 or icos_ee >= npoints_cosphi:
                    continue
                diff_cosphi_ee_decayed[icos_ee, 1] += flux_EH * diff_El_costheta
                diff_Eee_cosphi_decayed[iee, icos_ee, 2] += flux_EH * diff_El_costheta

    # Normalize distributions to total decayed flux with numerical guards.
    target_flux = float(np.nansum(flux_RHN_decayed))
    if not np.isfinite(target_flux) or target_flux <= 0.0:
        target_flux = 0.0

    def _safe_rescale(arr, target):
        denom = float(np.nansum(arr))
        if target > 0.0 and np.isfinite(denom) and denom > 0.0:
            arr *= target / denom
        else:
            arr[:] = 0.0
        np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    _safe_rescale(diff_El_decayed[:, 1], target_flux)
    _safe_rescale(diff_cosphi_decayed[:, 1], target_flux)
    _safe_rescale(diff_El_cosphi_decayed[:, :, 2], target_flux)
    _safe_rescale(diff_Eee_decayed[:, 1], target_flux)
    _safe_rescale(diff_cosphi_ee_decayed[:, 1], target_flux)
    _safe_rescale(diff_Eee_cosphi_decayed[:, :, 2], target_flux)

    return (
        diff_El_decayed,
        diff_cosphi_decayed,
        diff_El_cosphi_decayed,
        diff_Eee_decayed,
        diff_cosphi_ee_decayed,
        diff_Eee_cosphi_decayed,
    )


def list_neutrino_csv_files(directory="./output/"):
    """List all neutrino 2D distribution CSV files in a directory.

    Parameters
    ----------
    directory : str
        Directory to search for CSV files

    Returns
    -------
    list of dict
        List of dictionaries with file info: {'path', 'MH', 'U2', 'filename'}
    """
    pattern = os.path.join(directory, "diff_El_costheta_M*.csv")
    files = glob.glob(pattern)

    results = []
    for fpath in sorted(files):
        basename = os.path.basename(fpath)

        # Try to extract M and U values
        m_match = re.search(r"M([\d.]+)", basename)
        u_match = re.search(r"U([\d.eE+-]+)", basename)

        info = {
            "path": fpath,
            "filename": basename,
            "MH": float(m_match.group(1)) if m_match else None,
            "U2": float(u_match.group(1)) if u_match else None,
        }
        results.append(info)

    return results


def batch_compute_electrons_from_csv(
    directory="./output/", N_int_local=100000, plot=True
):
    """Batch process all neutrino CSV files in a directory to compute electron spectra.

    Parameters
    ----------
    directory : str
        Directory containing neutrino CSV files
    N_int_local : int
        Number of Monte Carlo samples for each scattering computation
    plot : bool
        Whether to generate plots for each file

    Returns
    -------
    list of dict
        Results for each file processed
    """
    from core.decay_and_scattering import (
        get_and_save_nuL_scatter_electron_El_costheta_from_csv,
    )

    files = list_neutrino_csv_files(directory)

    if not files:
        print(f"No neutrino CSV files found in {directory}")
        return []

    print(f"\n{'=' * 70}")
    print(f"Found {len(files)} neutrino distribution files")
    print(f"{'=' * 70}")

    for i, finfo in enumerate(files):
        print(f"{i + 1}. {finfo['filename']}")
        if finfo["MH"] is not None and finfo["U2"] is not None:
            print(f"   MH={finfo['MH']:.1f} MeV, U²={finfo['U2']:.2e}")

    print(f"\n{'=' * 70}")
    print("Starting batch processing...")
    print(f"{'=' * 70}\n")

    results = []
    for i, finfo in enumerate(files):
        print(f"\n[{i + 1}/{len(files)}] Processing: {finfo['filename']}")

        try:
            electron_2d, e_bins, costheta_bins, _ = (
                get_and_save_nuL_scatter_electron_El_costheta_from_csv(
                    csv_path=finfo["path"],
                    savepath=None,  # Use same directory as input
                    N_int_local=N_int_local,
                    plot=plot,
                )
            )

            result = {
                "success": True,
                "file": finfo["filename"],
                "MH": finfo["MH"],
                "U2": finfo["U2"],
                "electron_total": integrateSpectrum2D(
                    electron_2d,
                    0.5 * (e_bins[:-1] + e_bins[1:]),
                    0.5 * (costheta_bins[:-1] + costheta_bins[1:]),
                ),
            }
            results.append(result)
            print(f"✓ Success! Total electron events: {result['electron_total']:.6e}")

        except Exception as e:
            result = {
                "success": False,
                "file": finfo["filename"],
                "MH": finfo["MH"],
                "U2": finfo["U2"],
                "error": str(e),
            }
            results.append(result)
            print(f"✗ Failed: {e}")

    # Print summary
    print(f"\n{'=' * 70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'=' * 70}")

    n_success = sum(1 for r in results if r["success"])
    n_failed = len(results) - n_success

    print(f"Total files: {len(results)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")

    if n_failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r["success"]:
                print(f"  - {r['file']}: {r['error']}")

    print(f"{'=' * 70}\n")

    return results


def process1_single_parameter_set(spectrum_nuL_orig, U2, MH):
    """Process a single (U2, MH) parameter set and generate all plots in scenario 1

    Complete workflow:
    1. Compute LHN distributions from RHN decay inside detector

    Parameters
    ----------
    spectrum_nuL_orig : ndarray
        Original SN spectrum
    U2 : float
        Mixing parameter squared
    MH : float
        RHN mass
    """
    from core.rhn_physics import getRHNSpectrum
    from core.decay_and_scattering import getNulEAndAngleFromRHNDecay

    # Create output directory
    output_base = "plots_grid_scan_s1"
    os.makedirs(output_base, exist_ok=True)

    print("Processing U2={:.2e}, MH={:.1f} MeV...".format(U2, MH))
    # print("Step 1: Calculating decayed RHN spectrum in detector...")
    # decayed_spectrum = getDecayedRHNSpectrum_vll(
    #     spectrum_nuL_orig, MH, U2,
    #     distance_SE,
    #     detector_size  # Detector length in meters
    # )

    print("Step 1: Calculate decayed LHN energy distribution inside detector...")
    spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
    (
        diff_El_decayed,
        diff_costheta_decayed,
        diff_cosphi_decayed,
        _,
        _
    ) = getNulEAndAngleFromRHNDecay(
        spectrum_RHN,
        MH,
        U2,
        distance_SE,
        attenuation_length,
        costheta_bins=np.linspace(-1, 1, 201),
    )

    print("Step 2: Plotting decayed neutrino energy distribution...")
    savedir = f"plots_grid_scan_s1/U{U2:.1e}M{MH:.1f}/"
    os.makedirs(savedir, exist_ok=True)

    plt.figure()
    plt.plot(diff_El_decayed[:, 0], diff_El_decayed[:, 1], "-", linewidth=2)
    plt.xlabel("Energy (MeV)")
    plt.ylabel("Flux (MeV$^{-1}$ cm$^{-2}$ s$^{-1}$)")
    plt.grid(True)
    plt.savefig(
        f"{savedir}decayed_nuL_spectrum_U{U2:.1e}_MH{MH:.1f}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved decayed neutrino spectrum plot for U2={U2:.1e}, MH={MH:.1f} MeV.")

    print("Step 3: Plotting decayed neutrino angular distribution...")
    plt.figure()
    plt.plot(diff_costheta_decayed[:, 0], diff_costheta_decayed[:, 1], "-", linewidth=2)
    plt.xlabel(r"$\cos\theta$")
    plt.ylabel("Flux /sr")
    plt.grid(True)
    plt.savefig(
        f"{savedir}decayed_nuL_theta_U{U2:.1e}_MH{MH:.1f}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved decayed neutrino theta plot for U2={U2:.1e}, MH={MH:.1f} MeV.")
    plt.figure()
    plt.plot(diff_cosphi_decayed[:, 0], diff_cosphi_decayed[:, 1], "-", linewidth=2)
    plt.xlabel(r"$\cos\phi$")
    plt.ylabel("Flux")
    plt.grid(True)
    plt.savefig(
        f"{savedir}decayed_nuL_phi_U{U2:.1e}_MH{MH:.1f}.pdf",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Saved decayed neutrino phi plot for U2={U2:.1e}, MH={MH:.1f} MeV.")


def process2_single_parameter_set(args):
    """Process a single (U2, MH) parameter set and generate all plots in scenario 2

    Complete workflow:
    1. Compute neutrino distributions from RHN decay
    2. Compute scattered electron distributions


    Parameters
    ----------
    args : tuple
        (spectrum_nuL_orig, U2, MH, output_dir)

    Returns
    -------
    dict
        Summary statistics for this parameter set
    """
    from core.decay_and_scattering import (
        get_and_save_nuL_El_costheta_decay_in_flight,
        get_and_save_nuL_scatter_electron_El_costheta,
    )
    from core.spectrum_utils import integrateSpectrum, integrateSpectrum2D

    spectrum_nuL_orig, U2, MH, output_dir = args

    # print(f"\n{'=' * 60}")
    # print(f"Processing U2={U2:.2e}, MH={MH:.1f} MeV")
    # print(f"{'=' * 60}")

    # Create subdirectory for this parameter set
    param_dir = os.path.join(output_dir, f"U2_{U2:.2e}_MH_{MH:.1f}")
    os.makedirs(param_dir, exist_ok=True)

    # Step 1: Get neutrino distributions from RHN decay
    # print("Step 1: Computing neutrino distributions from RHN decay...")
    (
        diff_El_nu,  # MeV^-1 cm^-2 s^-1
        _,
        _,
        diff_El_costheta_nu,  # MeV^-1 cm^-2 s^-1 cosine^-1
    ) = get_and_save_nuL_El_costheta_decay_in_flight(
        spectrum_nuL_orig, U2, MH, savepath=param_dir
    )

    # Step 2: Compute scattered electron distributions
    # print("\nStep 5: Computing scattered electron distributions...")
    electron_2d, e_bins, costheta_lab_bins, _ = (
        get_and_save_nuL_scatter_electron_El_costheta(
            diff_El_costheta_nu, savepath=param_dir, N_int_local=100000
        )
    )

    # Convert electron_2d to same format as neutrino distribution
    e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
    costheta_centers = 0.5 * (costheta_lab_bins[:-1] + costheta_lab_bins[1:])

    # Apply detector resolution smearing
    from core.stats import apply_energy_resolution_convolution, apply_angle_resolution_convolution

    # Energy smearing: 5% fractional resolution, applied per cosθ bin
    for ia in range(len(costheta_centers)):
        electron_2d[:, ia] = apply_energy_resolution_convolution(
            electron_2d[:, ia], e_centers, frac_resolution=0.05,
        )
    # Angle smearing: 25° resolution, applied per energy bin
    for ie in range(len(e_centers)):
        electron_2d[ie, :] = apply_angle_resolution_convolution(
            electron_2d[ie, :], costheta_centers, sigma_deg=25.0,
        )

    # Build diff_El_costheta format for electron
    nE_e = len(e_centers)
    nA_e = len(costheta_centers)
    e_widths = np.diff(e_bins)
    costheta_widths = np.diff(costheta_lab_bins)
    # print(f"bin width: energy {e_widths[0]} MeV; costheta {costheta_widths[0]}")
    diff_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
    diff_El_costheta_electron[:, :, 0] = e_centers[:, None]
    diff_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
    diff_El_costheta_electron[:, :, 2] = electron_2d  # s^-1
    counts_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
    counts_El_costheta_electron[:, :, 0] = e_centers[:, None]
    counts_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
    counts_El_costheta_electron[:, :, 2] = (
        electron_2d / (e_widths[:, None] * costheta_widths[None, :]) * exposure_time
    )  # counts, S included in simulation before

    # Step 3: Compute derived electron quantities and save all data
    # print("\nStep 6: Computing derived electron quantities...")
    diff_El_electron = np.zeros((nE_e, 2))
    diff_El_electron[:, 0] = e_centers
    counts_El_electron = np.zeros((nE_e, 2))
    counts_El_electron[:, 0] = e_centers
    for ie in range(nE_e):
        diff_El_electron[ie, 1] = np.sum(electron_2d[ie, :]) / e_widths[ie]
        counts_El_electron[ie, 1] = np.sum(electron_2d[ie, :])

    diff_costheta_electron = np.zeros((nA_e, 2))
    diff_costheta_electron[:, 0] = costheta_centers
    for ia in range(nA_e):
        diff_costheta_electron[ia, 1] = np.sum(electron_2d[:, ia]) / costheta_widths[ia]

    # Save all derived data
    # print("Saving derived electron data...")
    np.savez(
        os.path.join(param_dir, "electron_data.npz"),
        counts_2d=counts_El_costheta_electron[:, :, 2],
        rate_2d=diff_El_costheta_electron[:, :, 2],
        energy_1d_rate=diff_El_electron[:, 1],
        energy_1d_counts=counts_El_electron[:, 1],
        angle_1d=diff_costheta_electron[:, 1],
        e_bins=e_bins,
        costheta_lab_bins=costheta_lab_bins,
        e_centers=e_centers,
        costheta_centers=costheta_centers,
        U2=U2,
        MH=MH,
    )
    # print(f"  Saved: electron_data.npz")

    # Step 4: Compute solar neutrino ES background for comparison
    # print("\nStep 4: Computing solar neutrino ES background...")
    try:
        nA_bg = diff_El_costheta_nu.shape[1]
        costheta_bg = diff_El_costheta_nu[0, :, 1]
        forward_bin = np.argmax(costheta_bg)
        dcos_bg = costheta_bg[1] - costheta_bg[0]

        nE_sun = len(spectrum_nuL_orig)
        energy_sun = spectrum_nuL_orig[:, 0]

        # Subtract the part that oscillated into RHNs
        spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
        flux_bg = spectrum_nuL_orig[:, 1] - spectrum_RHN[:, 1]
        # Below MH threshold, spectrum_RHN is zero, so flux_bg = original flux
        flux_bg = np.maximum(flux_bg, 0.0)  # guard against numerical negatives

        diff_El_costheta_sun = np.zeros((nE_sun, nA_bg, 3))
        diff_El_costheta_sun[:, :, 0] = energy_sun[:, None]
        diff_El_costheta_sun[:, :, 1] = costheta_bg[None, :]
        for ie in range(nE_sun):
            diff_El_costheta_sun[ie, forward_bin, 2] = (
                flux_bg[ie] / dcos_bg
            )

        electron_2d_bg, e_bins_bg, ct_lab_bins_bg, _ = (
            get_and_save_nuL_scatter_electron_El_costheta(
                diff_El_costheta_sun,
                savepath=param_dir,
                N_int_local=100000,
            )
        )

        ct_bg = 0.5 * (ct_lab_bins_bg[:-1] + ct_lab_bins_bg[1:])
        nA_bg_out = len(ct_bg)
        nE_bg_out = len(e_bins_bg) - 1
        e_bg_ctrs = 0.5 * (e_bins_bg[:-1] + e_bins_bg[1:])

        # Apply detector resolution smearing (same as signal)
        for ia in range(nA_bg_out):
            electron_2d_bg[:, ia] = apply_energy_resolution_convolution(
                electron_2d_bg[:, ia], e_bg_ctrs, frac_resolution=0.05,
            )
        for ie in range(nE_bg_out):
            electron_2d_bg[ie, :] = apply_angle_resolution_convolution(
                electron_2d_bg[ie, :], ct_bg, sigma_deg=25.0,
            )

        # Convert to counts (per bin)
        bg_counts = electron_2d_bg * exposure_time  # counts per (E, cosθ) bin

        # Save 2D counts as CSV
        rows = []
        for ie in range(nE_bg_out):
            for ia in range(nA_bg_out):
                rows.append([e_bg_ctrs[ie], ct_bg[ia], bg_counts[ie, ia]])
        bg_csv = os.path.join(param_dir, "solar_nu_bg_counts_2d.csv")
        np.savetxt(
            bg_csv, np.array(rows), delimiter=",",
            header="energy,costheta,counts", fmt="%0.6e", comments="",
        )
        # print(f"  Saved: {bg_csv}")

        # 1D projections (counts)
        diff_ct_bg = np.zeros((nA_bg_out, 2))
        diff_ct_bg[:, 0] = ct_bg
        diff_El_bg = np.zeros((nE_bg_out, 2))
        diff_El_bg[:, 0] = e_bg_ctrs
        for ia in range(nA_bg_out):
            diff_ct_bg[ia, 1] = np.sum(bg_counts[:, ia])
        for ie in range(nE_bg_out):
            diff_El_bg[ie, 1] = np.sum(bg_counts[ie, :])

        np.savez(
            os.path.join(param_dir, "solar_nu_background.npz"),
            bg_counts=bg_counts,
            e_bins=e_bins_bg,
            ct_bins=ct_lab_bins_bg,
            e_centers=e_bg_ctrs,
            costheta_bg=ct_bg,
            angle_1d_bg=diff_ct_bg[:, 1],
            energy_1d_bg=diff_El_bg[:, 1],
            angle_1d_signal=diff_costheta_electron[:, 1],
            costheta_signal=costheta_centers,
        )
        # print(f"  Saved: solar_nu_background.npz")

    except Exception as e:
        import traceback

        print(f"  Warning: Solar neutrino background failed: {e}")
        traceback.print_exc()

    # Collect summary statistics
    summary = {
        "U2": U2,
        "MH": MH,
        "neutrino_total_flux": integrateSpectrum(diff_El_nu),
        "neutrino_2d_integral": integrateSpectrum2D(diff_El_costheta_nu),
        "electron_total_events": integrateSpectrum2D(diff_El_costheta_electron),
        "output_dir": param_dir,
    }

    print(f"\nCompleted U2={U2:.2e}, MH={MH:.1f} MeV")
    # print(f"  Neutrino total flux: {summary['neutrino_total_flux']:.6e}")
    # print(f"  Output directory: {param_dir}")

    return summary
