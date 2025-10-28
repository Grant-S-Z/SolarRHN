"""
Toy Monte Carlo simulation for Solar RHN parameter scan.

This script performs a grid scan over (U², M_H) parameter space:
- Computes neutrino distributions from RHN decay in flight
- Computes electron spectra from neutrino-electron scattering
- Generates comprehensive plots for each parameter point
- Supports both sequential and parallel processing
"""

import os
import numpy as np
import multiprocessing as mp

# Import from core package
from core import interpolateSpectrum

# Import workflow functions
from workflows import process_single_parameter_set

# Output directory
savepath = './data/simulation/'

if __name__ == "__main__":
    os.makedirs("plots", exist_ok=True)
    os.makedirs(savepath, exist_ok=True)

    # Define energy grid
    energy = np.arange(0.0, 16.0, step=0.2)
    
    print("=== Extract 8B spectrum from csv file ===")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)  # unit: MeV^-1 cm^-2 s^-1
    
    # Define parameter grids
    # Example: scan over multiple U2 and MH values
    U2_values = [0.1, 0.01]  # mixing parameter squared
    MH_values = [2.0, 4.0, 6.0, 8.0]  # RHN mass in MeV
    
    print("\n" + "="*60)
    print("PARAMETER SCAN CONFIGURATION")
    print("="*60)
    print(f"U² values: {U2_values}")
    print(f"MH values (MeV): {MH_values}")
    print(f"Total parameter sets: {len(U2_values) * len(MH_values)}")
    print("="*60 + "\n")
    
    # Create output directory
    output_base = "plots_grid_scan"
    os.makedirs(output_base, exist_ok=True)
    
    # Prepare arguments for parallel processing
    args_list = []
    for U2 in U2_values:
        for MH in MH_values:
            args_list.append((spectrum_nuL_orig, U2, MH, output_base))
    
    # Option 1: Sequential processing (easier for debugging)
    print("Starting sequential processing...")
    results = []
    for args in args_list:
        result = process_single_parameter_set(args)
        results.append(result)
    
    # Option 2: Parallel processing (uncomment to use)
    # print("Starting parallel processing...")
    # with mp.Pool(processes=4) as pool:
    #     results = pool.map(process_single_parameter_set, args_list)
    
    # Save summary results
    print("\n" + "="*60)
    print("SUMMARY OF ALL RUNS")
    print("="*60)
    
    summary_file = os.path.join(output_base, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("U2\tMH\tNeutrino_Flux\tNeutrino_2D_Integral\tElectron_Success\tOutput_Dir\n")
        for res in results:
            line = f"{res['U2']:.2e}\t{res['MH']:.1f}\t{res['neutrino_total_flux']:.6e}\t"
            line += f"{res['neutrino_2d_integral']:.6e}\t{res['electron_success']}\t{res['output_dir']}\n"
            f.write(line)
            print(line.strip())
    
    print(f"\nSummary saved to: {summary_file}")
    print(f"All plots saved to: {output_base}/")
    print("\n" + "="*60)
    print("PROCESSING COMPLETE")
    print("="*60)
