import os
import matplotlib.pyplot as plt
import numpy as np

# Import from core package
from core import *
from core.sampling import getNuLEAndAngleBySampling
from core.rhn_physics import getDecayedRHNSpectrum_vll

if __name__ == "__main__":
    # Plot settings
    plt.rcParams.update({
        'text.usetex': True,
        'font.family': 'serif',
        'font.serif': ["Computer Modern Roman"],
        'font.size': 14
    })
    plt.rcParams['text.latex.preamble'] = r'\usepackage{siunitx}'
    plt.rcParams['figure.dpi'] = 300

    # Define energy grid
    estep = 0.2
    energy = np.arange(0.0, 16.0, step=estep)

    print("=== Extract 8B spectrum from csv file ===")
    spectrum_nuL_orig = interpolateSpectrum(
        "data/8BSpectrum.csv", energy)  # unit: MeV^-1 cm^-2 s^-1

    # Define parameter grids
    U2_values = [1.0e-5]
    MH_values = np.linspace(2.0, 12.0, 6)
    
    # Sampling settings
    num_samples = 100000  # Number of MC samples per parameter set
    
    # Angular bins for sampling (needed by the function signature)
    costheta_bins = np.linspace(-1, 1, 201)

    print("\n" + "="*60)
    print("SAMPLING METHOD CONFIGURATION")
    print("="*60)
    print(f"U² values: {U2_values}")
    print(f"MH values (MeV): {MH_values}")
    print(f"Samples per point: {num_samples}")
    print("="*60 + "\n")

    savedir = 'plots_grid_scan_s1_sampling/'
    os.makedirs(savedir, exist_ok=True)

    for U2 in U2_values:
        plt.figure()    
        for MH in MH_values:
            print(f"Sampling parameter set: U2={U2:.1e}, MH={MH:.1f} MeV")

            # 1. Calculate Production Spectrum
            spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2)
            
            # 2. Calculate Decayed Spectrum (Flux of RHNs decaying in detector)
            # Note: Ensure getDecayedRHNSpectrum includes the Branching Ratio if you modified it previously.
            # If not, we might need to multiply by RHN_BR_vll(MH, U2) manually.
            # Here we assume getDecayedRHNSpectrum returns the flux of the specific channel or total decays.
            # If it returns total decays, we multiply by BR here to be safe for nu-e-e channel.
            
            spectrum_decayed = getDecayedRHNSpectrum_vll(
                spectrum_RHN, MH, U2, distance_SE, detector_size
            )

            spectrum_decayed[:, 1] *= exposure
            # valid_mask = (spectrum_decayed[:, 0] > MH) & (spectrum_decayed[:, 1] > 0)
            # spectrum_decayed = spectrum_decayed[valid_mask]
            
            # Check if we need to apply BR manually (depends on your rhn_physics.py modification)
            # Assuming standard implementation returns total decays, we apply BR for nu-e-e events:
            # br = RHN_BR_vll(MH, U2)
            # spectrum_decayed[:, 1] *= br 
            # (Uncomment above lines if your getDecayedRHNSpectrum does NOT include BR)

            # 3. Generate Samples
            # diff_Eee_sample returns shape (N, 2) -> [Energy, Count/Flux]
            # The function normalizes the output histogram to match the total flux of spectrum_decayed.
            diff_El_sample, diff_costheta_sample, diff_Eee_sample = getNuLEAndAngleBySampling(
                spectrum_decayed, MH, num_samples, costheta_bins
            )

            # 4. Plotting
            # diff_Eee_sample[:, 1] is the flux density (or count density depending on normalization)
            # We multiply by exposure / estep to match the units "Counts / MeV"
            # Note: getNuLEAndAngleBySampling normalizes such that sum(hist) * width ~ sum(input_flux)
            # So diff_Eee_sample[:, 1] is effectively dN/dE * total_flux.
            
            plt.plot(diff_Eee_sample[:, 0],
                     diff_Eee_sample[:, 1] / estep, 
                     '--', linewidth=2, label=f'MH={MH:.1f} MeV')

        plt.xlabel("Energy (MeV)")
        plt.ylabel(r"Counts / MeV 1 yr 500 t")
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        outfile = f'{savedir}decayed_ee_count_sampling_U{U2:.1e}.pdf'
        plt.savefig(outfile, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved sampling plot to {outfile}")