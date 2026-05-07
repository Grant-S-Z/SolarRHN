import os
import time
import matplotlib.pyplot as plt
import numpy as np
import uproot as ur
from typing import Any
from tqdm import tqdm

# Import from core package
from core import *
from workflows import *
from pytools.rt_ploter import rt_plot_exclusion_region

# Plot directory
ER_plots_path = './plots/ER/'

if __name__ == "__main__":
    start_time = time.time()
    
    # Plot settings
    # plt.rcParams['font.sans-serif'] = ''
    plt.rcParams.update({'text.usetex': True,
                         'font.family': 'sans-serif',
                         'font.serif': ["Computer Modern Roman"],
                         'font.sans-serif': ["Helvetica"],
                         'font.size': 14
                         })
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
    plt.rcParams['figure.dpi'] = 300
    # plt.rcParams['legend.fontsize'] = 'small'
    # plt.rcParams['axes.labelsize'] = 'small'

    # Define grid
    cosstep = 0.01
    estep = 0.2
    energy = np.arange(0.0, 16.0, step=estep)

    print("=== Extract 8B spectrum from csv file ===")
    spectrum_nuL_orig = interpolateSpectrum(
        "data/8BSpectrum.csv", energy)  # unit: 0.2MeV^-1 cm^-2 s^-1

    print("B8 neutrino flux (integrated): ",
          integrateSpectrum(spectrum_nuL_orig), "cm^-2 s^-1")
    
    # Original background calculation (kept for reference):
    # print("Calculating background ES spectrum...")
    # _, _, diff_Ebkg, _, _, bg_bins_E, _, _ = scatter_electron_spectrum(
    #     energy,
    #     spectrum_nuL_orig,
    #     target_energy_centers=energy+0.5*estep,
    #     use_numba=True,
    # )
    # bg_centers = 0.5 * (bg_bins_E[:-1] + bg_bins_E[1:])
    # B_bin = diff_Ebkg * exposure_time

    print("Loading background ES spectrum from data/Solar.root (he_es)...")
    f_bg: Any = ur.open("data/Solar.root")
    h_bg: Any = f_bg["he_es"]
    bg_values = np.asarray(h_bg.values())
    bg_edges = np.asarray(h_bg.axis().edges())
    f_bg.close()

    bg_bin_width_src = bg_edges[1] - bg_edges[0]
    bg_centers_src = 0.5 * (bg_edges[:-1] + bg_edges[1:])
    bg_per_mev_src = bg_values / bg_bin_width_src

    # Align background to the same bin centers used by signal for direct chi2.
    bg_centers = energy + 0.5 * estep
    bg_per_mev = np.interp(bg_centers, bg_centers_src, bg_per_mev_src, left=0.0, right=0.0)
    B_bin = bg_per_mev * estep

    # Define parameter grids
    U2_values = np.logspace(-7, -1, 40)
    MH_values = np.linspace(2.0, 14.0, 40)

    print("\n" + "="*60)
    print("PARAMETER SCAN CONFIGURATION")
    print("="*60)
    print(f"U² values: {U2_values}")
    print(f"MH values (MeV): {MH_values}")
    print(f"Total parameter sets: {len(U2_values) * len(MH_values)}")
    print("="*60 + "\n")

    chi2_grid = np.full((len(U2_values), len(MH_values)), np.nan)

    # Method 1 process
    print("Starting Method 1 processing...")
    # print("Plot ER spectra for U2=1.0...")
    # plot_ER(spectrum_nuL_orig, MH_values, 1.0, ER_plots_path)

    # print("Plot RHN lifetime in cms...")
    # plot_lifetimes('plots/LT/')

    # print("Plot RHN decay branching ratios...")
    # plot_branch_ratios('plots/BR/')

    print("Plot for different MH and fixed U2...")
    
    print("Plot decayed electron energy distribution...")
    
    for iu2, U2 in enumerate(tqdm(U2_values, desc="Scan U2", unit="U2")):
        chi2_results = []
        plt.figure()
        savedir = f'plots/Eee/'
        # savedir = f'plots/Cee/'
        os.makedirs(savedir, exist_ok=True)
        for imh, MH in enumerate(tqdm(MH_values, desc=f"Scan MH @ U2={U2:.1e}", unit="MH", leave=False)):
            # print(f"Processing parameter set: U2={U2:.1e}, MH={MH:.1f} MeV")

            spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
            diff_El_decayed, diff_cosphi_decayed, diff_El_cosphi_decayed, diff_Eee_decayed, diff_cosphi_ee_decayed, _ = getNuleeInDetector(
                spectrum_RHN, MH, U2)
            S_bin = np.nan_to_num(diff_Eee_decayed[:, 1] * exposure, nan=0.0, posinf=0.0, neginf=0.0)
            S_bin = np.clip(S_bin, 0.0, None)

            # Use bin centers consistently for direct bin-by-bin chi2.
            signal_centers = diff_Eee_decayed[:, 0] + 0.5 * estep
            if signal_centers.shape != bg_centers.shape or not np.allclose(signal_centers, bg_centers, atol=1e-12):
                raise ValueError("Signal and background energy bins are not aligned for direct chi2")

            chi2_val = chi2_poisson_likelihood_ratio(S_bin, B_bin)
            chi2_grid[iu2, imh] = chi2_val
            chi2_results.append((MH, chi2_val))

            print("Sum(diff_Eee_decayed flux) = ", np.sum(S_bin))
            print(f"chi2(MH={MH:.1f} MeV) = {chi2_val:.6e}")
            plt.plot(signal_centers,
                     S_bin / estep, '-', linewidth=2, label=f'MH={MH:.1f} MeV')
            # plt.plot(diff_cosphi_ee_decayed[:, 0],
            #          diff_cosphi_ee_decayed[:, 1] * exposure / estep, '-', linewidth=2, label=f'MH={MH:.1f} MeV')


        # Background counts in detector: bg_counts_per_mev (rate) * exposure_time
        plt.plot(bg_centers, B_bin / estep, 'k--', linewidth=2, label=r'$^8$B ES Background')
        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"Counts / MeV / 1 yr / 500 t")
        plt.grid(True)        
        plt.legend()
        plt.xlim(0.0, 16.0)
        plt.savefig(
            f'{savedir}decayed_ee_count_U{U2:.1e}.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(
            f"{savedir}decayed_ee_count_U{U2:.1e} picture saved.")

        print("\n" + "-" * 60)
        print(f"chi2 summary for U2={U2:.1e}")
        for mh_val, chi2_val in chi2_results:
            print(f"MH={mh_val:.1f} MeV: chi2={chi2_val:.6e}")
        print("-" * 60)

    # Save full chi2 parameter scan results for downstream fitting/plotting.
    os.makedirs('output', exist_ok=True)
    np.savez(
        'output/chi2_grid_s1.npz',
        U2_values=U2_values,
        MH_values=MH_values,
        chi2_grid=chi2_grid,
    )

    chi2_table = np.column_stack((U2_values, chi2_grid))
    header = 'U2 ' + ' '.join([f'MH_{mh:.1f}' for mh in MH_values])
    np.savetxt('output/chi2_grid_s1.txt', chi2_table, header=header)
    print("Saved chi2 grid to output/chi2_grid_s1.npz and output/chi2_grid_s1.txt")

    # Draw 90% CL exclusion region in (U2, MH) plane using ROOT.
    chi2_crit, excl_path = rt_plot_exclusion_region(
        U2_values,
        MH_values,
        chi2_grid,
        file_name='s1_exclusion_90CL',
        dir='plots/exclusion/',
        cl=0.90,
        ndof=2,
        xlog=False,
        ylog=True,
        type='png',
    )
    print(f"Saved 90% CL exclusion plot to: {excl_path}")
    print(f"90% CL threshold (chi2_crit, dof=2): {chi2_crit:.6f}")
        
    # print("Plot decayed electron angular distribution...")
    # for U2 in U2_values:
    #     plt.figure()
    #     savedir = f'plots/Cee/'
    #     os.makedirs(savedir, exist_ok=True)
    #     for MH in MH_values:
    #         print(f"Processing parameter set: U2={U2:.1e}, MH={MH:.1f} MeV")

    #         spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
    #         diff_El_decayed, diff_cosphi_decayed, diff_El_cosphi_decayed, diff_Eee_decayed, diff_cosphi_ee_decayed, diff_Eee_cosphi_decayed = getNuleeInDetector(
    #             spectrum_RHN, MH, U2)

    #         plt.plot(0.5 * (diff_cosphi_ee_decayed[:-1, 0] + diff_cosphi_ee_decayed[1:, 0]),
    #                  0.5 * (diff_cosphi_ee_decayed[:-1, 1] + diff_cosphi_ee_decayed[1:, 1]) * exposure, '-', linewidth=2, label=f'MH={MH:.1f} MeV')
    #         plt.xlabel(r"$e^{+}e^{-}$ cosine")
    #         plt.ylabel(r"Counts / MeV 1 yr 500 t")
    #         plt.grid(True)

    #     plt.xlim(-1.0, 1.0)
    #     plt.legend()
    #     plt.savefig(
    #         f'{savedir}decayed_cosphi_ee_count_U{U2:.1e}.pdf', dpi=300, bbox_inches='tight')
    #     plt.close()
    #     print(
    #         f"{savedir}decayed_cosphi_ee_count_U{U2:.1e}.pdf saved.")
        
    # print("Parameter grid scan and plotting...")
    # for U2 in U2_values:
    #     for MH in MH_values:

    #         print(f"Processing parameter set: U2={U2:.1e}, MH={MH:.1f} MeV")
    #         savedir = f'plots_grid_scan_s1/U{U2:.1e}M{MH:.1f}/'
    #         os.makedirs(savedir, exist_ok=True)
    #         spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
    #         diff_El_decayed, diff_cosphi_decayed, diff_El_cosphi_decayed, diff_Eee_decayed = getNuleeInDetector(spectrum_RHN, MH, U2)

    #         # print("Plotting decayed neutrino energy spectrum...")
    #         # plt.figure()
    #         # plt.plot(diff_El_decayed[:, 0],
    #         #         diff_El_decayed[:, 1], '-', linewidth=2)
    #         # plt.xlabel("Energy (MeV)")
    #         # plt.ylabel(r"Flux (\unit{MeV^{-1}cm^{-2}s^{-1}})")
    #         # plt.grid(True)
    #         # plt.savefig(
    #         #     f'{savedir}decayed_nuL_flux_U{U2:.1e}_MH{MH:.1f}.pdf', dpi=300, bbox_inches='tight')
    #         # plt.close()
    #         # print(
    #         #     f"Saved decayed neutrino spectrum plot for U2={U2:.1e}, MH={MH:.1f} MeV.")

    #         # plt.figure()
    #         # plt.plot(diff_El_decayed[:, 0], diff_El_decayed[:, 1] * exposure, '-', linewidth=2)
    #         # plt.xlabel("Energy (MeV)")
    #         # plt.ylabel(r"Counts / MeV 1 yr 500 t")
    #         # plt.grid(True)
    #         # plt.savefig(
    #         #     f'{savedir}decayed_nuL_count_U{U2:.1e}_MH{MH:.1f}.pdf', dpi=300, bbox_inches='tight')
    #         # plt.close()
    #         # print(
    #         #     f"Saved decayed neutrino count plot for U2={U2:.1e}, MH={MH:.1f} MeV.")

    #         # print("Plotting decayed neutrino angular distribution...")
    #         # plt.figure()
    #         # plt.plot(diff_cosphi_decayed[:, 0],
    #         #         diff_cosphi_decayed[:, 1], '-', linewidth=2)
    #         # plt.xlabel(r"$\cos\phi$")
    #         # plt.ylabel(r"Flux (\unit{sr^{-1}cm^{-2}s^{-1}})")
    #         # plt.xlim(-1.0, 1.0)
    #         # plt.grid(True)
    #         # plt.savefig(
    #         #     f'{savedir}decayed_nuL_phi_U{U2:.1e}_MH{MH:.1f}.pdf', dpi=300, bbox_inches='tight')
    #         # plt.close()
    #         # print(
    #         #     f"Saved decayed neutrino phi plot for U2={U2:.1e}, MH={MH:.1f} MeV.")

    #         print("Plotting decayed electron energy distribution...")
    #         plt.figure()
    #         plt.plot(diff_Eee_decayed[:, 0],
    #                 diff_Eee_decayed[:, 1] * exposure / estep , '-', linewidth=2)
    #         plt.xlabel("Energy (MeV)")
    #         plt.ylabel(r"Counts / MeV 1 yr 500 t")
    #         plt.grid(True)
    #         plt.savefig(
    #             f'{savedir}decayed_ee_count_U{U2:.1e}_MH{MH:.1f}.pdf', dpi=300, bbox_inches='tight')
    #         plt.close()
    #         print(
    #             f"Saved decayed electron count plot for U2={U2:.1e}, MH={MH:.1f} MeV.")

    # Print total elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'='*60}")
    print(f"Total runtime: {int(hours):d}h {int(minutes):d}m {seconds:.1f}s ({elapsed:.1f}s)")
    print(f"{'='*60}")