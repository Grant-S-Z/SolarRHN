#!/usr/bin/env python
"""
Quick demo: compute electron spectrum from a saved neutrino CSV file.

This is a minimal example showing how to use the CSV-based electron
computation functionality.
"""

import os
import sys

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils import (
    get_and_save_nuL_scatter_electron_El_costheta_from_csv,
    list_neutrino_csv_files
)


def demo_single_file():
    """Demo: Process a single neutrino CSV file"""
    
    print("\n" + "="*70)
    print("DEMO 1: Process single neutrino CSV file")
    print("="*70 + "\n")
    
    # Example CSV path (adjust to your actual file)
    csv_path = './data/simulation/diff_El_costheta_M4.0_U1.0e-01.csv'
    
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        print("Please update csv_path to point to your neutrino CSV file.")
        return
    
    # Compute electron spectrum
    electron_2d, e_bins, costheta_bins, _ = get_and_save_nuL_scatter_electron_El_costheta_from_csv(
        csv_path=csv_path,
        savepath=None,  # Use same directory as input
        N_int_local=50000,  # Use fewer samples for quick demo
        plot=True
    )
    
    print("\nDone! Check the output directory for:")
    print("  - scattered_electrons_2d_lab.csv")
    print("  - electron_2d_from_csv.pdf")
    print("  - electron_energy_1d_from_csv.pdf")
    print("  - electron_angle_1d_from_csv.pdf")


def demo_list_files():
    """Demo: List all available neutrino CSV files"""
    
    print("\n" + "="*70)
    print("DEMO 2: List all neutrino CSV files")
    print("="*70 + "\n")
    
    directory = './data/simulation/'
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    files = list_neutrino_csv_files(directory)
    
    if not files:
        print(f"No neutrino CSV files found in {directory}")
        return
    
    print(f"Found {len(files)} neutrino distribution files:\n")
    
    for i, finfo in enumerate(files, 1):
        print(f"{i}. {finfo['filename']}")
        if finfo['MH'] is not None and finfo['U2'] is not None:
            print(f"   MH = {finfo['MH']:.1f} MeV")
            print(f"   U² = {finfo['U2']:.2e}")
        print(f"   Path: {finfo['path']}\n")


def demo_batch():
    """Demo: Batch process multiple files"""
    
    print("\n" + "="*70)
    print("DEMO 3: Batch process neutrino CSV files")
    print("="*70 + "\n")
    
    from utils import batch_compute_electrons_from_csv
    
    directory = './data/simulation/'
    
    if not os.path.exists(directory):
        print(f"Directory not found: {directory}")
        return
    
    # Process all files (with reduced samples for speed)
    results = batch_compute_electrons_from_csv(
        directory=directory,
        N_int_local=50000,  # Fewer samples for demo
        plot=True
    )
    
    print(f"\nProcessed {len(results)} files")
    print("Check electron_batch_summary.csv for details")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Demo electron computation from CSV')
    parser.add_argument('--demo', type=int, choices=[1, 2, 3], default=1,
                       help='Which demo to run (1=single, 2=list, 3=batch)')
    
    args = parser.parse_args()
    
    if args.demo == 1:
        demo_single_file()
    elif args.demo == 2:
        demo_list_files()
    elif args.demo == 3:
        demo_batch()
    
    print("\n" + "="*70)
    print("Demo complete!")
    print("="*70 + "\n")
