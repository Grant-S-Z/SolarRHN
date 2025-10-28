#!/usr/bin/env python
"""
Compute scattered electron spectrum from saved neutrino CSV file.

This script demonstrates how to use get_and_save_nuL_scatter_electron_El_costheta_from_csv
to compute electron scattering from a previously saved neutrino distribution.

Usage:
    python compute_electron_from_csv.py <path_to_neutrino_csv> [--no-plot] [--samples N]

Example:
    python compute_electron_from_csv.py ./data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv
    python compute_electron_from_csv.py ./data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv --samples 50000
"""

import argparse
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.electron_scattering import get_and_save_nuL_scatter_electron_El_costheta_from_csv


def main():
    parser = argparse.ArgumentParser(
        description='Compute scattered electron spectrum from neutrino CSV file'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='Path to neutrino 2D distribution CSV file (energy, costheta, value)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory for results (default: same as input CSV)'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=100000,
        help='Number of Monte Carlo samples for scattering (default: 100000)'
    )
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Disable automatic plotting'
    )
    
    args = parser.parse_args()
    
    # Check if input file exists
    if not os.path.exists(args.csv_path):
        print(f"Error: Input file not found: {args.csv_path}")
        sys.exit(1)
    
    # Compute electron spectrum
    print(f"\n{'='*70}")
    print(f"  ELECTRON SCATTERING COMPUTATION FROM CSV")
    print(f"{'='*70}")
    print(f"Input:  {args.csv_path}")
    print(f"Output: {args.output_dir or 'Same as input'}")
    print(f"MC samples: {args.samples:,}")
    print(f"Plot: {not args.no_plot}")
    print(f"{'='*70}\n")
    
    try:
        electron_2d, e_bins, costheta_bins, _ = get_and_save_nuL_scatter_electron_El_costheta_from_csv(
            csv_path=args.csv_path,
            savepath=args.output_dir,
            N_int_local=args.samples,
            plot=not args.no_plot
        )
        
        print("\n" + "="*70)
        print("  SUCCESS!")
        print("="*70)
        print(f"Electron spectrum shape: {electron_2d.shape}")
        print(f"Energy bins: {len(e_bins)-1} ({e_bins[0]:.2f} - {e_bins[-1]:.2f} MeV)")
        print(f"Angle bins: {len(costheta_bins)-1} ({costheta_bins[0]:.3f} - {costheta_bins[-1]:.3f})")
        
        if not args.no_plot:
            output_dir = args.output_dir or os.path.dirname(args.csv_path)
            print(f"\nGenerated plots:")
            print(f"  - {os.path.join(output_dir, 'electron_2d_from_csv.pdf')}")
            print(f"  - {os.path.join(output_dir, 'electron_energy_1d_from_csv.pdf')}")
            print(f"  - {os.path.join(output_dir, 'electron_angle_1d_from_csv.pdf')}")
        
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  ERROR!")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
