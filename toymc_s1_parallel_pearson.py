import os
import time
import matplotlib.pyplot as plt
import uproot as ur
import numpy as np
from typing import Any
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

# Import from core package
from core import *
from workflows import *
from pytools.rt_ploter import rt_plot_exclusion_region

# Plot directory
ER_plots_path = './plots/ER/'

def compute_mh_point_pearson(args):
    """Compute Pearson chi2 and signal for a single (U2, MH) point.
    
    Separated as a module-level function for ProcessPoolExecutor compatibility.
    Returns: (imh, MH, chi2_val, S_bin, signal_centers, sum_S_bin)
    """
    imh, MH, U2, spectrum_nuL_orig, _, B_bin, fit_mask, estep, energy_resolution = args
    
    spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(
        spectrum_RHN, MH, U2)
    S_bin = np.nan_to_num(diff_Eee_decayed[:, 1] * exposure, nan=0.0, posinf=0.0, neginf=0.0)
    S_bin = np.clip(S_bin, 0.0, None)

    signal_centers = diff_Eee_decayed[:, 0] + 0.5 * estep
    S_bin = apply_energy_resolution_convolution(
        S_bin,
        signal_centers,
        frac_resolution=energy_resolution,
    )

    S_bin_fit = S_bin[fit_mask]
    B_bin_fit = B_bin[fit_mask]
    chi2_val = chi2_pearson(S_bin_fit, B_bin_fit)
    sum_S_bin = np.sum(S_bin)
    
    return imh, MH, chi2_val, S_bin, signal_centers, sum_S_bin


if __name__ == "__main__":
    start_time = time.time()
    
    # ============================================================================
    # CONFIGURATION
    # ============================================================================
    
    # Parallel execution settings
    max_workers = 10

    # Energy binning parameters
    cosstep = 0.01
    estep = 0.2
    energy = np.arange(0.0, 16.0, step=estep)

    # Detector and physics parameters
    # energy_resolution = 0.03
    energy_resolution = 0.05  # 5% fractional energy resolution (Borexino)
    angle_resolution_deg = 25.0  # Angular resolution in degrees
    chi2_fit_min = 4.8
    chi2_fit_max = 12.8

    # Parameter grids for scan
    U2_values = np.logspace(-7, -1, 7)
    MH_values = np.linspace(2.0, 14.0, 7)

    # ============================================================================
    # DATA LOADING
    # ============================================================================

    print("="*70)
    print("SOLAR RHN PARAMETER SCAN (PARALLEL) WITH PEARSON χ²")
    print("="*70)
    print()

    print(">>> Loading 8B neutrino spectrum from csv file...")
    spectrum_nuL_orig = interpolateSpectrum(
        "data/8BSpectrum.csv", energy)  # unit: 0.2MeV^-1 cm^-2 s^-1

    print("B8 neutrino flux (integrated): ",
          integrateSpectrum(spectrum_nuL_orig), "cm^-2 s^-1")
    print()

    print(">>> Loading background ES spectrum from data/Solar.root...")
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
    print(f">>> Applying {100.0*energy_resolution:.1f}% energy resolution to background...")
    B_bin = apply_energy_resolution_convolution(
        B_bin,
        bg_centers,
        frac_resolution=energy_resolution,
    )
    chi2_fit_mask = (bg_centers >= chi2_fit_min) & (bg_centers <= chi2_fit_max)
    print()

    # ============================================================================
    # SCAN CONFIGURATION AND EXECUTION
    # ============================================================================

    print("\n" + "="*60)
    print("PARAMETER SCAN CONFIGURATION (PARALLEL WITH PEARSON χ²)")
    print("="*60)
    print(f"U² values: {U2_values}")
    print(f"MH values (MeV): {MH_values}")
    print(f"Total parameter sets: {len(U2_values) * len(MH_values)}")
    print(f"Energy resolution: {100.0 * energy_resolution:.1f}% (Gaussian convolution)")
    print(f"Angular resolution: {angle_resolution_deg:.1f}° (for future implementations)")
    print(f"χ² method: Pearson (S²/B)")
    print(f"Max workers: {max_workers}")
    print("="*60 + "\n")

    grid_suffix = f"{len(U2_values)}x{len(MH_values)}_pearson"

    chi2_grid = np.full((len(U2_values), len(MH_values)), np.nan)

    print("Starting parallel computation with Pearson χ²...")
    print()
    
    # Prepare per-U2 computation
    for iu2, U2 in enumerate(tqdm(U2_values, desc="Scan U2", unit="U2", position=0)):
        chi2_results = []
        plt.figure()
        savedir = f'plots/Eee_pearson/'
        os.makedirs(savedir, exist_ok=True)
        
        # Build task list for all MH values at this U2
        compute_tasks = [
            (imh, MH, U2, spectrum_nuL_orig, bg_centers, B_bin, chi2_fit_mask, estep, energy_resolution)
            for imh, MH in enumerate(MH_values)
        ]
        
        # Parallel computation of all MH points
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            results = list(tqdm(
                executor.map(compute_mh_point_pearson, compute_tasks),
                total=len(MH_values),
                desc=f"Compute @ U2={U2:.1e}",
                unit="MH",
                leave=False,
                position=1
            ))
        
        # Process results and populate chi2_grid
        for imh, MH, chi2_val, S_bin, signal_centers, sum_S_bin in results:
            if signal_centers.shape != bg_centers.shape or not np.allclose(signal_centers, bg_centers, atol=1e-12):
                raise ValueError("Signal and background energy bins are not aligned for direct chi2")
            
            chi2_grid[iu2, imh] = chi2_val
            chi2_results.append((MH, chi2_val))
            
            plt.plot(signal_centers,
                     S_bin, '-', linewidth=2, label=f'MH={MH:.1f} MeV')
        
        # Background counts in detector: bg_counts_per_mev (rate) * exposure_time
        plt.plot(bg_centers, B_bin / estep, 'k--', linewidth=2, label=r'$^8$B ES Background')
        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"Counts / MeV / 1 yr / 500 t")
        plt.grid(True)        
        plt.legend()
        plt.xlim(0.0, 16.0)
        plt.savefig(
            f'{savedir}decayed_ee_count_U{U2:.1e}_pearson.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    # ============================================================================
    # RESULTS AND OUTPUT
    # ============================================================================

    # Save full chi2 parameter scan results for downstream fitting/plotting.
    os.makedirs('output', exist_ok=True)
    chi2_npz_path = f'output/chi2_grid_s1_{grid_suffix}.npz'
    chi2_txt_path = f'output/chi2_grid_s1_{grid_suffix}.txt'
    np.savez(
        chi2_npz_path,
        U2_values=U2_values,
        MH_values=MH_values,
        chi2_grid=chi2_grid,
    )

    chi2_table = np.column_stack((U2_values, chi2_grid))
    header = 'U2 ' + ' '.join([f'MH_{mh:.1f}' for mh in MH_values])
    np.savetxt(chi2_txt_path, chi2_table, header=header)
    print()
    print(f"Saved Pearson χ² grid to {chi2_npz_path} and {chi2_txt_path}")

    # Draw 90% CL exclusion region in (U2, MH) plane using ROOT.
    chi2_crit, excl_path = rt_plot_exclusion_region(
        U2_values,
        MH_values,
        chi2_grid,
        file_name=f's1_exclusion_90CL_{grid_suffix}',
        dir='plots/exclusion/',
        cl=0.90,
        ndof=2,
        xlog=False,
        ylog=True,
        type='pdf',
    )
    print(f"Saved 90% CL exclusion plot (Pearson χ²) to: {excl_path}")
    print(f"90% CL threshold (chi2_crit, dof=2): {chi2_crit:.6f}")
    
    # Print total elapsed time
    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n{'='*60}")
    print(f"Total runtime: {int(hours):d}h {int(minutes):d}m {seconds:.1f}s ({elapsed:.1f}s)")
    print(f"{'='*60}")
