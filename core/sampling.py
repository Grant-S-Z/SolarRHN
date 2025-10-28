"""
Monte Carlo sampling algorithms for particle distributions.

This module provides rejection sampling and histogram-based sampling
for generating particle kinematics from theoretical distributions.

Key Functions
-------------
- rejection_sampling_2Dfunc: Rejection sampling for 2D distributions
- rejection_sampling_1Dfunc: Rejection sampling for 1D distributions  
- generate_samples_from_spectrum: Sample from discrete histogram
- getNuLEAndAngleBySampling: Generate neutrino samples from RHN decay
"""

import numpy as np


def rejection_sampling_2Dfunc(f, num_samples, x_bounds, y_bounds, M, *args):
    """Generate samples from 2D distribution using rejection sampling.
    
    The rejection sampling algorithm:
    1. Generate uniform random point (x, y) in bounds
    2. Generate uniform random u in [0, M]
    3. Accept if u < f(x, y), otherwise reject and repeat
    
    Parameters
    ----------
    f : callable
        Target distribution function f(x, y, *args)
    num_samples : int
        Number of samples to generate
    x_bounds : tuple
        (x_min, x_max) bounds for x
    y_bounds : tuple
        (y_min, y_max) bounds for y
    M : float
        Upper bound on f over the sampling region (must satisfy f(x,y) ≤ M)
    *args : optional
        Additional arguments passed to f
    
    Returns
    -------
    ndarray
        Array of shape (num_samples, 2) with sampled (x, y) pairs
        
    Notes
    -----
    Efficiency depends on M being close to max(f). If M is too large,
    many samples will be rejected. If M < max(f), some regions will
    be undersampled.
    """
    samples = []
    
    while len(samples) < num_samples:
        # Generate candidate sample uniformly within bounds
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        y = np.random.uniform(y_bounds[0], y_bounds[1])
        
        # Generate random number for acceptance check
        u = np.random.uniform(0, M)
        
        # Accept or reject the candidate sample
        if u < f(x, y, *args):
            samples.append([x, y])
    
    return np.array(samples)


def rejection_sampling_1Dfunc(f, num_samples, x_bounds, M, *args):
    """Generate samples from 1D distribution using rejection sampling.
    
    Parameters
    ----------
    f : callable
        Target distribution function f(x, *args)
    num_samples : int
        Number of samples to generate
    x_bounds : tuple
        (x_min, x_max) bounds for x
    M : float
        Upper bound on f over the sampling region
    *args : optional
        Additional arguments passed to f
    
    Returns
    -------
    ndarray
        Array of sampled x values
    """
    samples = []
    
    while len(samples) < num_samples:
        # Generate candidate sample uniformly within bounds
        x = np.random.uniform(x_bounds[0], x_bounds[1])
        
        # Generate random number for acceptance check
        u = np.random.uniform(0, M)
        
        # Accept or reject the candidate sample
        if u < f(x, *args):
            samples.append(x)
    
    return np.array(samples)


def generate_samples_from_spectrum(spectrum, num_samples):
    """Generate samples from a discrete spectrum using histogram sampling.
    
    This function treats the input spectrum as a discrete probability
    distribution and samples from it using NumPy's choice function.
    
    Parameters
    ----------
    spectrum : ndarray
        2D array with shape (N, 2) where:
        - spectrum[:, 0] = x values (bin centers)
        - spectrum[:, 1] = weights/probabilities
    num_samples : int
        Number of samples to generate
    
    Returns
    -------
    samples : ndarray
        Array of sampled x values
    spectrum_generated : ndarray
        Histogram of generated samples (same shape as input spectrum)
        
    Notes
    -----
    If the total weight is negligible (< 1e-6), returns zero arrays.
    The output spectrum_generated shows the distribution of generated
    samples for validation purposes.
    """
    x_value = spectrum[:, 0]
    y_value = spectrum[:, 1]

    # Initialize output arrays
    samples = np.zeros(num_samples)
    spectrum_generated = np.zeros((len(x_value), 2))
    for i in range(len(x_value)):
        spectrum_generated[i][0] = x_value[i]

    # Check if spectrum is essentially zero
    if np.sum(y_value) < 1e-6:
        return samples, spectrum_generated
    
    # Normalize to probability distribution
    y_value = y_value / np.sum(y_value)
    
    # Sample from discrete distribution
    indices = np.arange(len(x_value))
    samples_index = np.random.choice(indices, size=num_samples, p=y_value)

    # Fill output arrays
    for i in range(num_samples):
        samples[i] = x_value[samples_index[i]]
        spectrum_generated[samples_index[i]][1] += 1.0
        
    return samples, spectrum_generated


def getMaximumValue2D(f, x_bounds, y_bounds, *args):
    """Find maximum value of 2D function over grid.
    
    Evaluates f on a 100×100 grid and returns the maximum value.
    Used to determine the envelope M for rejection sampling.
    
    Parameters
    ----------
    f : callable
        Function to maximize f(x, y, *args)
    x_bounds : tuple
        (x_min, x_max)
    y_bounds : tuple
        (y_min, y_max)
    *args : optional
        Additional arguments passed to f
    
    Returns
    -------
    float
        Maximum value found on grid
        
    Notes
    -----
    This is a simple grid search and may miss the true maximum if
    the function is highly peaked between grid points. Increase
    num_points for better accuracy at cost of computation time.
    """
    values = []
    num_points = 100
    
    for ix in range(num_points):
        x_this = x_bounds[0] + ix * (x_bounds[1] - x_bounds[0]) / num_points
        for iy in range(num_points):
            y_this = y_bounds[0] + iy * (y_bounds[1] - y_bounds[0]) / num_points
            value = f(x_this, y_this, *args)
            values.append(value)
            
    return max(values)


def getNuLEAndAngleBySampling(spectrum_R, MH, num_samples, costheta_bins):
    """Generate neutrino energy and angle samples from RHN decay spectrum.
    
    This function uses a two-stage sampling approach:
    1. Sample RHN energies from spectrum_R
    2. For each RHN, sample neutrino (E, cosθ) from decay kinematics
    
    The decay kinematics are sampled from diff_El_costheta_cms using
    rejection sampling, then boosted to lab frame.
    
    Parameters
    ----------
    spectrum_R : ndarray
        RHN spectrum with shape (N, 2): [energy, flux]
    MH : float
        RHN mass (MeV)
    num_samples : int
        Number of samples to generate
    costheta_bins : array-like
        Bin centers for angular distribution
    
    Returns
    -------
    diff_El : ndarray
        Neutrino energy distribution
    diff_costheta : ndarray
        Neutrino angular distribution (lab frame)
    diff_Eee : ndarray
        e⁺e⁻ pair energy distribution
        
    Notes
    -----
    This sampling-based approach is an alternative to the analytical
    integration used in getNulEAndAngleFromRHNDecay. It can be useful
    for validation or when dealing with complex distributions.
    
    The function requires utils functions:
    - diff_El_costheta_cms: Decay distribution in CMS
    - cms_to_lab: Lorentz transformation
    """
    from .decay_distributions import diff_El_costheta_cms
    from .transformations import cms_to_lab
    
    energy_bins = spectrum_R[:, 0]
    flux_R = spectrum_R[:, 1]

    # Construct bin edges from centers
    energy_bin_edges = np.zeros(len(energy_bins) + 1)
    step_i = 0.0
    for i in range(len(energy_bins)):
        if i > 0:
            step_i = energy_bins[i] - energy_bins[i - 1]
        else:
            step_i = energy_bins[i + 1] - energy_bins[i]
        energy_bin_edges[i] = energy_bins[i] - 0.999 * step_i
    energy_bin_edges[-1] = energy_bins[-1] + 0.001 * step_i

    costheta_bin_edges = np.zeros(len(costheta_bins) + 1)
    step_i = 0.0
    for i in range(len(costheta_bins)):
        if i > 0:
            step_i = costheta_bins[i] - costheta_bins[i - 1]
        else:
            step_i = costheta_bins[i + 1] - costheta_bins[i]
        costheta_bin_edges[i] = costheta_bins[i] - 0.999 * step_i
    costheta_bin_edges[-1] = costheta_bins[-1] + 0.001 * step_i
    
    # Find maximum of decay distribution for rejection sampling
    maxDiff = 0.0
    for EH in energy_bins:
        if EH > MH:
            maxDiff_this = getMaximumValue2D(
                diff_El_costheta_cms,
                [energy_bins[0], energy_bins[-1]],
                [costheta_bins[-1], costheta_bins[0]],
                MH, EH
            )
            if maxDiff_this > maxDiff:
                maxDiff = maxDiff_this

    # Sample RHN energies from spectrum
    samples_generated, spectrum_generated = generate_samples_from_spectrum(spectrum_R, num_samples)
    
    # For each RHN, sample decay kinematics
    Els = []
    Eees = []
    costhetas = []

    for EH in samples_generated:
        # Sample (El, costheta) in CMS frame
        sample_this = rejection_sampling_2Dfunc(
            diff_El_costheta_cms,
            1,
            [energy_bins[0], energy_bins[-1]],
            [costheta_bins[0], costheta_bins[-1]],
            maxDiff * 2.0,  # Safety factor for rejection bound
            MH, EH
        )
        El_this = sample_this[0][0]
        costheta_this = sample_this[0][1]
        
        # Transform to lab frame
        El_this_lab, costheta_this_lab = cms_to_lab(El_this, costheta_this, MH, EH)
        
        Els.append(El_this_lab)
        Eees.append(EH - El_this_lab)  # e+e- pair energy
        costhetas.append(costheta_this_lab)

    # Histogram the samples
    diff_El = np.zeros((len(energy_bins), 2))
    diff_Eee = np.zeros((len(energy_bins), 2))
    diff_costheta = np.zeros((len(costheta_bins), 2))

    Els_count, Els_edges = np.histogram(Els, bins=energy_bin_edges)
    Eees_count, Eees_edges = np.histogram(Eees, bins=energy_bin_edges)
    costhetas_count, costhetas_edges = np.histogram(costhetas, bins=costheta_bin_edges)
    
    # Normalize to match input flux
    sum_flux_R = np.sum(flux_R)
    sum_Els_count = np.sum(Els_count)
    sum_Eees_count = np.sum(Eees_count)
    sum_costhetas_count = np.sum(costhetas_count)

    for ieL in range(len(Els_count)):
        diff_El[ieL][0] = energy_bins[ieL]
        diff_El[ieL][1] = Els_count[ieL] * 1.0 * sum_flux_R / sum_Els_count if sum_Els_count > 0 else 0

        diff_Eee[ieL][0] = energy_bins[ieL]
        diff_Eee[ieL][1] = Eees_count[ieL] * 1.0 * sum_flux_R / sum_Eees_count if sum_Eees_count > 0 else 0
        
    for iTheta in range(len(costhetas_count)):
        diff_costheta[iTheta][0] = costheta_bins[iTheta]
        diff_costheta[iTheta][1] = costhetas_count[iTheta] * 1.0 * sum_flux_R / sum_costhetas_count if sum_costhetas_count > 0 else 0
        
    return diff_El, diff_costheta, diff_Eee


__all__ = [
    'rejection_sampling_2Dfunc',
    'rejection_sampling_1Dfunc',
    'generate_samples_from_spectrum',
    'getMaximumValue2D',
    'getNuLEAndAngleBySampling',
]
