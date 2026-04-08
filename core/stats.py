"""
Statistical functions for Solar RHN simulation.
"""
import numpy as np


def apply_energy_resolution_convolution(counts_bin, energy_centers, frac_resolution=0.03):
    """Apply Gaussian energy response to binned counts.

    Parameters
    ----------
    counts_bin : ndarray
        Counts per true-energy bin.
    energy_centers : ndarray
        Energy bin centers (MeV).
    frac_resolution : float, optional
        Fractional energy resolution used in sigma(E)=frac_resolution*E.

    Returns
    -------
    ndarray
        Counts per reconstructed-energy bin.
    """
    counts_bin = np.asarray(counts_bin, dtype=float)
    energy_centers = np.asarray(energy_centers, dtype=float)

    if counts_bin.shape != energy_centers.shape:
        raise ValueError("counts_bin and energy_centers must have the same shape")
    if frac_resolution < 0.0:
        raise ValueError("frac_resolution must be non-negative")

    smeared = np.zeros_like(counts_bin)
    for j, (e_true, n_true) in enumerate(zip(energy_centers, counts_bin)):
        if n_true <= 0.0:
            continue

        sigma = frac_resolution * e_true
        if sigma <= 0.0:
            smeared[j] += n_true
            continue

        x = (energy_centers - e_true) / sigma
        weights = np.exp(-0.5 * x * x)
        weight_sum = np.sum(weights)
        if weight_sum > 0.0:
            smeared += n_true * (weights / weight_sum)
        else:
            smeared[j] += n_true

    return smeared


def apply_angle_resolution_convolution(counts_bin, cos_centers, sigma_deg=25.0):
    """Apply Gaussian angular response to binned counts in cos(theta).

    Parameters
    ----------
    counts_bin : ndarray
        Counts per true-angle bin.
    cos_centers : ndarray
        cos(theta) bin centers, typically in [-1, 1].
    sigma_deg : float, optional
        Angular resolution in degrees, used as Gaussian sigma in theta space.

    Returns
    -------
    ndarray
        Counts per reconstructed-angle bin.
    """
    counts_bin = np.asarray(counts_bin, dtype=float)
    cos_centers = np.asarray(cos_centers, dtype=float)

    if counts_bin.shape != cos_centers.shape:
        raise ValueError("counts_bin and cos_centers must have the same shape")
    if sigma_deg < 0.0:
        raise ValueError("sigma_deg must be non-negative")

    sigma_rad = np.deg2rad(sigma_deg)
    theta_centers = np.arccos(np.clip(cos_centers, -1.0, 1.0))

    smeared = np.zeros_like(counts_bin)
    for j, n_true in enumerate(counts_bin):
        if n_true <= 0.0:
            continue

        if sigma_rad <= 0.0:
            smeared[j] += n_true
            continue

        x = (theta_centers - theta_centers[j]) / sigma_rad
        weights = np.exp(-0.5 * x * x)
        weight_sum = np.sum(weights)
        if weight_sum > 0.0:
            smeared += n_true * (weights / weight_sum)
        else:
            smeared[j] += n_true

    return smeared

def chi2_poisson_likelihood_ratio(S, B):
    """Likelihood ratio chi2 for Poisson distribution (Wilks' theorem).
    
    Also known as profile likelihood chi2. Computes -2ΔlnL for signal test.
    
    For signal vs background hypothesis test: 
    L(μ) = Π Pois(n_i | B_i + μS_i), test μ=1 vs μ=0
    Under Wilks' theorem: -2ln[L(μ=0)/L(μ=1)] ~ χ²
    
    Parameters
    ----------
    S : ndarray
        Signal counts per bin under alternative hypothesis (μ=1)
    B : ndarray
        Background counts per bin under null hypothesis (μ=0)

    Returns
    -------
    float
        Likelihood ratio chi2 statistic (-2ΔlnL)
    """    
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)

    if np.any(S < 0):
        raise ValueError("S must be non-negative")
    if np.any(B < 0):
        raise ValueError("B must be non-negative")

    # For B=0 bins, use the limiting value B*log(B/(S+B)) -> 0.
    log_lambda_term = np.array(S, copy=True)
    mask_b_pos = B > 0
    if np.any(mask_b_pos):
        sb = S[mask_b_pos] + B[mask_b_pos]
        if np.any(sb <= 0):
            raise ValueError("S + B must be positive where B > 0")
        log_lambda_term[mask_b_pos] = S[mask_b_pos] + B[mask_b_pos] * np.log(B[mask_b_pos] / sb)

    return 2.0 * np.sum(log_lambda_term)


def chi2_pearson(S, B, min_background=1e-10):
    """Pearson chi2 test statistic.
    
    Pearson's chi-squared: χ² = Σ (O - E)² / E
    For signal+background test: O = S + B, E = B (null hypothesis)
    So χ² = Σ S² / B
    
    Parameters
    ----------
    S : ndarray
        Signal counts per bin
    B : ndarray
        Background counts per bin
    min_background : float, optional
        Minimum background value to avoid division by zero
        
    Returns
    -------
    float
        Pearson chi2 statistic
    """
    S = np.asarray(S, dtype=float)
    B = np.asarray(B, dtype=float)
    
    if np.any(S < 0):
        raise ValueError("S must be non-negative")
    if np.any(B < 0):
        raise ValueError("B must be non-negative")
    
    # Avoid division by zero by using min_background where B is zero or very small
    B_safe = np.where(B > min_background, B, min_background)
    
    # Pearson chi2: Σ (S² / B)
    chi2 = np.sum(S**2 / B_safe)
    
    return chi2


__all__ = [
    'apply_energy_resolution_convolution',
    'apply_angle_resolution_convolution',
    'chi2_poisson_likelihood_ratio',
    'chi2_pearson',
]
