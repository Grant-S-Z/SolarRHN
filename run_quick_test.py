#!/usr/bin/env python
"""
Quick test script for RHN parameter scan
Tests with a small parameter set for validation
"""

import os
import sys
import numpy as np

# Import the necessary functions from the new core package
from core.spectrum_utils import interpolateSpectrum
from workflows import process_single_parameter_set

if __name__ == "__main__":
    print("="*60)
    print("QUICK TEST: RHN Parameter Scan")
    print("="*60)
    
    # Create output directory
    output_dir = "test_output"
    os.makedirs(output_dir, exist_ok=True)
    
    # Define energy grid (coarser for faster testing)
    energy = np.arange(0.0, 16.0, step=0.5)  # Coarser grid for speed
    
    print(f"Energy bins: {len(energy)}")
    print(f"Energy range: {energy[0]:.1f} - {energy[-1]:.1f} MeV")
    
    # Load spectrum
    print("\nLoading 8B solar neutrino spectrum...")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    
    # Test with single parameter set
    U2_test = 0.1
    MH_test = 4.0
    
    print(f"\nTest parameters:")
    print(f"  U² = {U2_test}")
    print(f"  MH = {MH_test} MeV")
    
    # Run processing
    args = (spectrum_nuL_orig, U2_test, MH_test, output_dir)
    
    print("\nStarting processing...")
    result = process_single_parameter_set(args)
    
    # Display results
    print("\n" + "="*60)
    print("TEST COMPLETE")
    print("="*60)
    print(f"Neutrino total flux: {result['neutrino_total_flux']:.6e}")
    print(f"Neutrino 2D integral: {result['neutrino_2d_integral']:.6e}")
    print(f"Electron scattering success: {result['electron_success']}")
    print(f"Output directory: {result['output_dir']}")
    print("\nCheck the output directory for plots!")
    print("="*60)
