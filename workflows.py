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

import os
import glob
import re
import numpy as np
from core.spectrum_utils import integrateSpectrum2D


def list_neutrino_csv_files(directory='./output/'):
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
    pattern = os.path.join(directory, 'diff_El_costheta_M*.csv')
    files = glob.glob(pattern)
    
    results = []
    for fpath in sorted(files):
        basename = os.path.basename(fpath)
        
        # Try to extract M and U values
        m_match = re.search(r'M([\d.]+)', basename)
        u_match = re.search(r'U([\d.eE+-]+)', basename)
        
        info = {
            'path': fpath,
            'filename': basename,
            'MH': float(m_match.group(1)) if m_match else None,
            'U2': float(u_match.group(1)) if u_match else None,
        }
        results.append(info)
    
    return results


def batch_compute_electrons_from_csv(directory='./output/', N_int_local=100000, plot=True):
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
    from core.decay_and_scattering import get_and_save_nuL_scatter_electron_El_costheta_from_csv
    
    files = list_neutrino_csv_files(directory)
    
    if not files:
        print(f"No neutrino CSV files found in {directory}")
        return []
    
    print(f"\n{'='*70}")
    print(f"Found {len(files)} neutrino distribution files")
    print(f"{'='*70}")
    
    for i, finfo in enumerate(files):
        print(f"{i+1}. {finfo['filename']}")
        if finfo['MH'] is not None and finfo['U2'] is not None:
            print(f"   MH={finfo['MH']:.1f} MeV, U²={finfo['U2']:.2e}")
    
    print(f"\n{'='*70}")
    print("Starting batch processing...")
    print(f"{'='*70}\n")
    
    results = []
    for i, finfo in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing: {finfo['filename']}")
        
        try:
            electron_2d, e_bins, costheta_bins, _ = get_and_save_nuL_scatter_electron_El_costheta_from_csv(
                csv_path=finfo['path'],
                savepath=None,  # Use same directory as input
                N_int_local=N_int_local,
                plot=plot
            )
            
            result = {
                'success': True,
                'file': finfo['filename'],
                'MH': finfo['MH'],
                'U2': finfo['U2'],
                'electron_total': integrateSpectrum2D(electron_2d, 
                                                      0.5*(e_bins[:-1]+e_bins[1:]),
                                                      0.5*(costheta_bins[:-1]+costheta_bins[1:]))
            }
            results.append(result)
            print(f"✓ Success! Total electron events: {result['electron_total']:.6e}")
            
        except Exception as e:
            result = {
                'success': False,
                'file': finfo['filename'],
                'MH': finfo['MH'],
                'U2': finfo['U2'],
                'error': str(e)
            }
            results.append(result)
            print(f"✗ Failed: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    
    if n_failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}: {r['error']}")
    
    print(f"{'='*70}\n")
    
    return results


def process_single_parameter_set(args):
    """Process a single (U2, MH) parameter set and generate all plots.
    
    Complete workflow:
    1. Compute neutrino distributions from RHN decay
    2. Plot 2D and 1D neutrino distributions
    3. Compute scattered electron distributions
    4. Plot 2D and 1D electron distributions
    
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
        get_and_save_nuL_scatter_electron_El_costheta
    )
    from core.spectrum_utils import integrateSpectrum, integrateSpectrum2D
    from ploter import (
        plot_El_costheta_map,
        plot_1d_energy_distribution,
        plot_1d_angle_distribution
    )
    
    spectrum_nuL_orig, U2, MH, output_dir = args
    
    print(f"\n{'='*60}")
    print(f"Processing U2={U2:.2e}, MH={MH:.1f} MeV")
    print(f"{'='*60}")
    
    # Create subdirectory for this parameter set
    param_dir = os.path.join(output_dir, f"U2_{U2:.2e}_MH_{MH:.1f}")
    os.makedirs(param_dir, exist_ok=True)
    
    # Step 1: Get neutrino distributions from RHN decay
    print("Step 1: Computing neutrino distributions from RHN decay...")
    (
        diff_El_nu,
        diff_costheta_nu,
        diff_cosphi_nu,
        diff_El_costheta_nu,
    ) = get_and_save_nuL_El_costheta_decay_in_flight(
        spectrum_nuL_orig, U2, MH, savepath=param_dir
    )
    
    # Step 2: Plot neutrino 2D distribution
    print("\nStep 2: Plotting neutrino 2D distribution...")
    plot_El_costheta_map(
        diff_El_costheta_nu, 
        param_dir, 
        filename=f"neutrino_2d_U2_{U2:.2e}_MH_{MH:.1f}.png",
        title_prefix=f"Neutrino: U²={U2:.2e}, M={MH:.1f} MeV"
    )
    
    # Step 3: Plot neutrino 1D energy distribution
    print("\nStep 3: Plotting neutrino 1D energy distribution...")
    plot_1d_energy_distribution(
        diff_El_nu,
        param_dir,
        filename=f"neutrino_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.png",
        title_prefix=f"Neutrino Energy: U²={U2:.2e}, M={MH:.1f} MeV",
        ylabel="Flux (MeV⁻¹ cm⁻² s⁻¹)"
    )
    
    # Step 4: Plot neutrino 1D angular distribution
    print("\nStep 4: Plotting neutrino 1D angular distribution...")
    plot_1d_angle_distribution(
        diff_costheta_nu,
        param_dir,
        filename=f"neutrino_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.png",
        title_prefix=f"Neutrino Angular: U²={U2:.2e}, M={MH:.1f} MeV",
        ylabel="Flux (sr⁻¹ cm⁻² s⁻¹)"
    )
    
    # Step 5: Compute scattered electron distributions
    print("\nStep 5: Computing scattered electron distributions...")
    try:
        electron_2d, e_bins, costheta_lab_bins, _ = get_and_save_nuL_scatter_electron_El_costheta(
            diff_El_costheta_nu, 
            savepath=param_dir,
            N_int_local=100000
        )
        
        # Convert electron_2d to same format as neutrino distribution
        e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
        costheta_centers = 0.5 * (costheta_lab_bins[:-1] + costheta_lab_bins[1:])
        
        # Build diff_El_costheta format for electron
        nE_e = len(e_centers)
        nA_e = len(costheta_centers)
        diff_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
        diff_El_costheta_electron[:, :, 0] = e_centers[:, None]
        diff_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
        diff_El_costheta_electron[:, :, 2] = electron_2d
        
        # Step 6: Plot electron 2D distribution
        print("\nStep 6: Plotting electron 2D distribution...")
        plot_El_costheta_map(
            diff_El_costheta_electron,
            param_dir,
            filename=f"electron_2d_U2_{U2:.2e}_MH_{MH:.1f}.png",
            title_prefix=f"Electron: U²={U2:.2e}, M={MH:.1f} MeV"
        )
        
        # Step 7: Plot electron 1D energy distribution (integrate over angle)
        print("\nStep 7: Plotting electron 1D energy distribution...")
        diff_El_electron = np.zeros((nE_e, 2))
        diff_El_electron[:, 0] = e_centers
        # Integrate over angle bins (approximate with simple sum × bin width)
        costheta_widths = np.diff(costheta_lab_bins)
        for ie in range(nE_e):
            diff_El_electron[ie, 1] = np.sum(electron_2d[ie, :] * costheta_widths)
        
        plot_1d_energy_distribution(
            diff_El_electron,
            param_dir,
            filename=f"electron_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Electron Energy: U²={U2:.2e}, M={MH:.1f} MeV",
            ylabel="Event rate (MeV⁻¹)"
        )
        
        # Step 8: Plot electron 1D angular distribution (integrate over energy)
        print("\nStep 8: Plotting electron 1D angular distribution...")
        diff_costheta_electron = np.zeros((nA_e, 2))
        diff_costheta_electron[:, 0] = costheta_centers
        # Integrate over energy bins
        energy_widths = np.diff(e_bins)
        for ia in range(nA_e):
            diff_costheta_electron[ia, 1] = np.sum(electron_2d[:, ia] * energy_widths)
        
        plot_1d_angle_distribution(
            diff_costheta_electron,
            param_dir,
            filename=f"electron_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.png",
            title_prefix=f"Electron Angular: U²={U2:.2e}, M={MH:.1f} MeV",
            ylabel="Event rate (sr⁻¹)"
        )
        
        # Step 9: Compute collimated beam comparison
        print("\nStep 9: Computing collimated beam comparison...")
        try:
            from core.decay_and_scattering import get_and_save_nuL_scatter_electron_El_costheta
            from core.spectrum_utils import integrateSpectrum
            import matplotlib.pyplot as plt
            
            # Get total neutrino flux for normalization
            total_nu_flux = integrateSpectrum(diff_El_nu)
            
            # Create collimated beam neutrino distribution
            # Same energy bins as RHN decay, but all flux in forward direction (cosθ=1.0)
            nE = diff_El_costheta_nu.shape[0]
            nA = diff_El_costheta_nu.shape[1]
            
            # Get energy and angle arrays from original distribution
            energy_centers = diff_El_costheta_nu[:, 0, 0]
            costheta_centers_orig = diff_El_costheta_nu[0, :, 1]
            
            # Create new distribution with all flux in forward bin
            diff_El_costheta_collimated = np.zeros((nE, nA, 3))
            diff_El_costheta_collimated[:, :, 0] = energy_centers[:, None]
            diff_El_costheta_collimated[:, :, 1] = costheta_centers_orig[None, :]
            
            # Put all flux in the most forward angle bin (highest cosθ)
            forward_bin_idx = np.argmax(costheta_centers_orig)
            
            # Simply sum all angular contributions at each energy and put in forward bin
            # diff_El_costheta_nu[:, :, 2] has shape (nE, nA) with dN/(dE·dΩ)
            # Sum over angle axis to get total at each energy, then put in forward bin
            for ie in range(nE):
                # Sum all angular bins at this energy to get total flux
                total_at_energy = np.sum(diff_El_costheta_nu[ie, :, 2])
                # Put it all in the forward angular bin
                diff_El_costheta_collimated[ie, forward_bin_idx, 2] = total_at_energy
            
            # Verify total flux matches (for debugging)
            collimated_total = integrateSpectrum2D(diff_El_costheta_collimated)
            print(f"  Total neutrino flux (RHN): {total_nu_flux:.6e}")
            print(f"  Total neutrino flux (collimated): {collimated_total:.6e}")
            print(f"  Ratio: {collimated_total/total_nu_flux:.6f}")
            
            # Scatter collimated beam
            electron_2d_beam, e_bins_beam, costheta_lab_bins_beam, _ = get_and_save_nuL_scatter_electron_El_costheta(
                diff_El_costheta_collimated, 
                savepath=param_dir,  # Save to same directory
                N_int_local=100000
            )
            
            # Build diff_costheta format for beam electrons
            costheta_beam_centers = 0.5 * (costheta_lab_bins_beam[:-1] + costheta_lab_bins_beam[1:])
            nA_beam = len(costheta_beam_centers)
            diff_costheta_beam = np.zeros((nA_beam, 2))
            diff_costheta_beam[:, 0] = costheta_beam_centers
            
            # Integrate over energy
            energy_widths_beam = np.diff(e_bins_beam)
            for ia in range(nA_beam):
                diff_costheta_beam[ia, 1] = np.sum(electron_2d_beam[:, ia] * energy_widths_beam)
            
            # Plot comparison side-by-side
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            
            # Left: RHN decay electron angular distribution
            ax = axes[0]
            ax.plot(costheta_centers, diff_costheta_electron[:, 1], 'b-', linewidth=2, label='RHN decay')
            ax.set_xlabel('cos(θ) (lab frame)', fontsize=12)
            ax.set_ylabel('Event rate (sr⁻¹)', fontsize=12)
            ax.set_title(f'RHN Decay: U²={U2:.2e}, M={MH:.1f} MeV', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            # Right: Collimated beam electron angular distribution
            ax = axes[1]
            ax.plot(costheta_beam_centers, diff_costheta_beam[:, 1], 'r-', linewidth=2, label='Collimated beam')
            ax.set_xlabel('cos(θ) (lab frame)', fontsize=12)
            ax.set_ylabel('Event rate (sr⁻¹)', fontsize=12)
            ax.set_title(f'Collimated Beam: Same Total Flux', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            plt.tight_layout()
            comparison_file = os.path.join(param_dir, f"comparison_rhn_vs_collimated_U2_{U2:.2e}_MH_{MH:.1f}.png")
            plt.savefig(comparison_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved comparison plot: {comparison_file}")
            
            # Also plot both on same axis for direct comparison
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            ax.plot(costheta_centers, diff_costheta_electron[:, 1], 'b-', linewidth=2, label='RHN decay (extended source)')
            ax.plot(costheta_beam_centers, diff_costheta_beam[:, 1], 'r--', linewidth=2, label='Collimated beam (forward)')
            ax.set_xlabel('cos(θ) (lab frame)', fontsize=12)
            ax.set_ylabel('Event rate (sr⁻¹)', fontsize=12)
            ax.set_title(f'Scattered Electron Angular Distribution: U²={U2:.2e}, M={MH:.1f} MeV', fontsize=13)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=11)
            
            overlay_file = os.path.join(param_dir, f"comparison_overlay_U2_{U2:.2e}_MH_{MH:.1f}.png")
            plt.savefig(overlay_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"  Saved overlay comparison: {overlay_file}")
            
        except Exception as e:
            import traceback
            print(f"  Warning: Collimated beam comparison failed: {e}")
            traceback.print_exc()
        
        electron_success = True
        
    except Exception as e:
        print(f"Warning: Electron scattering failed for U2={U2}, MH={MH}: {e}")
        electron_success = False
    
    # Collect summary statistics
    summary = {
        'U2': U2,
        'MH': MH,
        'neutrino_total_flux': integrateSpectrum(diff_El_nu),
        'neutrino_2d_integral': integrateSpectrum2D(diff_El_costheta_nu),
        'electron_success': electron_success,
        'output_dir': param_dir
    }
    
    print(f"\nCompleted U2={U2:.2e}, MH={MH:.1f} MeV")
    print(f"  Neutrino total flux: {summary['neutrino_total_flux']:.6e}")
    print(f"  Output directory: {param_dir}")
    
    return summary


__all__ = [
    'list_neutrino_csv_files',
    'batch_compute_electrons_from_csv',
    'process_single_parameter_set',
]
