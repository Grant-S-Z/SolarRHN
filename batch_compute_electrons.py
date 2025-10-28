#!/usr/bin/env python
"""
Batch process all neutrino CSV files to compute electron spectra.

This script finds all neutrino 2D distribution CSV files in a directory
and computes the corresponding electron scattering spectra.

Usage:
    python batch_compute_electrons.py [directory] [--samples N] [--no-plot]

Example:
    python batch_compute_electrons.py ./data/simulation/
    python batch_compute_electrons.py ./plots_grid_scan/ --samples 50000
"""

import argparse
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backup.utils import batch_compute_electrons_from_csv, list_neutrino_csv_files


def main():
    parser = argparse.ArgumentParser(
        description='Batch compute electron spectra from neutrino CSV files'
    )
    parser.add_argument(
        'directory',
        type=str,
        nargs='?',
        default='./data/simulation/',
        help='Directory containing neutrino CSV files (default: ./data/simulation/)'
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
    parser.add_argument(
        '--list-only',
        action='store_true',
        help='Only list files without processing'
    )
    
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.exists(args.directory):
        print(f"Error: Directory not found: {args.directory}")
        sys.exit(1)
    
    # List files
    if args.list_only:
        files = list_neutrino_csv_files(args.directory)
        if not files:
            print(f"No neutrino CSV files found in {args.directory}")
            sys.exit(0)
        
        print(f"\nFound {len(files)} neutrino distribution files in {args.directory}:")
        print("-" * 70)
        for i, finfo in enumerate(files):
            print(f"{i+1}. {finfo['filename']}")
            if finfo['MH'] is not None and finfo['U2'] is not None:
                print(f"   MH={finfo['MH']:.1f} MeV, U²={finfo['U2']:.2e}")
            print(f"   Path: {finfo['path']}")
        print("-" * 70 + "\n")
        sys.exit(0)
    
    # Process files
    print(f"\n{'='*70}")
    print(f"  BATCH ELECTRON SCATTERING COMPUTATION")
    print(f"{'='*70}")
    print(f"Directory: {args.directory}")
    print(f"MC samples: {args.samples:,}")
    print(f"Plot: {not args.no_plot}")
    print(f"{'='*70}\n")
    
    try:
        results = batch_compute_electrons_from_csv(
            directory=args.directory,
            N_int_local=args.samples,
            plot=not args.no_plot
        )
        
        if not results:
            print("No files were processed.")
            sys.exit(1)
        
        # Save summary to file
        import pandas as pd
        summary_file = os.path.join(args.directory, 'electron_batch_summary.csv')
        
        df_data = []
        for r in results:
            row = {
                'filename': r['file'],
                'MH': r.get('MH', ''),
                'U2': r.get('U2', ''),
                'success': r['success'],
            }
            if r['success']:
                row['electron_total'] = r['electron_total']
            else:
                row['error'] = r.get('error', '')
            df_data.append(row)
        
        df = pd.DataFrame(df_data)
        df.to_csv(summary_file, index=False)
        print(f"\nSummary saved to: {summary_file}")
        
    except Exception as e:
        print(f"\n{'='*70}")
        print(f"  ERROR!")
        print(f"{'='*70}")
        print(f"{type(e).__name__}: {e}")
        print(f"{'='*70}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
