#!/usr/bin/env python
"""
Test script to compare Pearson chi2 vs Asimov chi2 for RHN analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from core.stats import chi2_poisson_likelihood_ratio, chi2_pearson

def generate_test_data():
    """Generate test signal and background arrays."""
    # Simple test: signal peak at 5 MeV, background flat
    nbins = 50
    energy_centers = np.linspace(0, 15, nbins)
    
    # Background: flat with some energy dependence
    B = 10.0 * np.exp(-energy_centers / 10.0)
    
    # Signal: Gaussian peak around 5 MeV
    signal_mean = 5.0
    signal_sigma = 1.0
    S = 5.0 * np.exp(-0.5 * ((energy_centers - signal_mean) / signal_sigma)**2)
    
    return energy_centers, S, B

def main():
    print("="*60)
    print("Comparison of Pearson χ² vs Likelihood ratio χ²")
    print("="*60)
    
    # Generate test data
    energy, S, B = generate_test_data()
    
    # Calculate both test statistics
    chi2_likelihood = chi2_poisson_likelihood_ratio(S, B)
    chi2_pearson = chi2_pearson(S, B)
    
    print(f"\nTest Results:")
    print(f"  Total signal: {np.sum(S):.2f}")
    print(f"  Total background: {np.sum(B):.2f}")
    print(f"  Signal/Background ratio: {np.sum(S)/np.sum(B):.4f}")
    print(f"\n  Likelihood ratio χ²: {chi2_likelihood:.4f}")
    print(f"  Pearson χ²: {chi2_pearson:.4f}")
    print(f"  Ratio (Pearson/Likelihood): {chi2_pearson/chi2_likelihood:.4f}")
    
    # Plot the distributions
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot signal and background
    ax1.plot(energy, B, 'b-', linewidth=2, label='Background')
    ax1.plot(energy, S, 'r-', linewidth=2, label='Signal')
    ax1.set_xlabel('Energy (MeV)')
    ax1.set_ylabel('Counts')
    ax1.set_title('Signal and Background Distributions')
    ax1.legend()
    ax1.grid(True)
    
    # Plot χ² contributions per bin
    # For Likelihood ratio: 2 * [S + B * log(B/(S+B))]
    likelihood_ratio_per_bin = np.zeros_like(S)
    mask = B > 0
    likelihood_ratio_per_bin[mask] = 2.0 * (S[mask] + B[mask] * np.log(B[mask] / (S[mask] + B[mask])))
    
    # For Pearson: S² / B
    pearson_per_bin = np.zeros_like(S)
    pearson_per_bin[mask] = S[mask]**2 / B[mask]
    
    ax2.plot(energy[mask], likelihood_ratio_per_bin[mask], 'b-', linewidth=2, label='Likelihood ratio χ² contribution')
    ax2.plot(energy[mask], pearson_per_bin[mask], 'r-', linewidth=2, label='Pearson χ² contribution')
    ax2.set_xlabel('Energy (MeV)')
    ax2.set_ylabel('χ² contribution per bin')
    ax2.set_title('χ² Contributions per Energy Bin')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('chi2_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: chi2_comparison.png")
    
    # Test with different signal strengths
    print("\n" + "="*60)
    print("Testing different signal strengths:")
    print("="*60)
    
    signal_scales = [0.1, 0.5, 1.0, 2.0, 5.0]
    print("\nSignal scale | Likelihood χ² | Pearson χ² | Ratio")
    print("-" * 50)
    
    for scale in signal_scales:
        S_scaled = S * scale
        chi2_likelihood = chi2_poisson_likelihood_ratio(S_scaled, B)
        chi2_p = chi2_pearson(S_scaled, B)
        ratio = chi2_p / chi2_likelihood if chi2_likelihood > 0 else np.nan
        print(f"  {scale:5.1f}      | {chi2_likelihood:9.2f} | {chi2_p:9.2f} | {ratio:5.2f}")

if __name__ == "__main__":
    main()
