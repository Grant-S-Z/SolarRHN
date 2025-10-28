"""
Neutrino-electron scattering cross section and spectrum calculations.

This module provides the core physics for neutrino-electron elastic scattering,
including:
- Differential cross sections
- MSW flavor oscillation probabilities
- 2D spectrum generation (energy × angle)
- Integration over recoil electron kinematics

All constants are imported from the constants module.
"""

import numpy as np
import matplotlib.pyplot as plt
from .constants import (
    pi, sw2, gL_e, gR_e, gL_mu, gR_mu, me,
    Ne, sig0_es, Q0, RunTime,
    N_int, Ntime, osci_mode, bin_width, max_energy,
    n_bins, bin_array, bin_mid_array
)


def cal_Tmax(q):
    """Calculate maximum recoil electron energy.
    
    For neutrino energy q, the maximum kinetic energy of the
    recoil electron in elastic scattering is:
    T_max = 2q² / (m_e + 2q)
    
    Parameters
    ----------
    q : float or array
        Neutrino energy (MeV)
    
    Returns
    -------
    float or array
        Maximum electron recoil energy (MeV)
    """
    return 2.0 * q * q / (0.511 + 2.0 * q)


def cal_costheta(T, q):
    """Calculate scattering angle from recoil energy.
    
    Given electron recoil energy T and neutrino energy q,
    compute cos(θ) where θ is the electron scattering angle.
    
    Parameters
    ----------
    T : float or array
        Electron recoil energy (MeV)
    q : float or array
        Neutrino energy (MeV)
    
    Returns
    -------
    float or array
        cos(θ_scatter)
    """
    return np.sqrt((T * (0.511 + q) * (0.511 + q)) / ((T + 2 * 0.511) * q * q))


def mswlma(energy):
    """MSW-LMA solar neutrino oscillation probability.
    
    Calculate the survival probability P(νₑ → νₑ) for solar neutrinos
    after propagation through the Sun and to Earth, including MSW
    matter effects.
    
    Parameters
    ----------
    energy : float or array
        Neutrino energy (MeV)
    
    Returns
    -------
    float or array
        Survival probability P_ee
    """
    A = 1.0
    sin12 = 0.31  # sin²θ₁₂
    sin13 = 0.02241  # sin²θ₁₃
    ps = 92.5  # Production region parameter
    ksi = 0.203 * A * (1 - sin13) * energy * ps / 100
    c = ((1 - 2 * sin12) - ksi) / np.sqrt(1 - 2 * ksi * (1 - 2 * sin12) + ksi * ksi)
    return sin13 * sin13 + (1 - sin13) * (1 - sin13) * (0.5 + 0.5 * c * (1 - 2 * sin12))


def scatter_electron_spectrum(energy, flux, *, energy_centers=None, bin_width_local=None, N_int_local=None):
    """Calculate electron spectrum from neutrino-electron scattering.

    This function computes the 2D distribution of recoil electrons
    (energy × scattering angle) from neutrino-electron elastic scattering,
    properly accounting for:
    - Weak coupling constants (gL, gR)
    - Differential cross section angular dependence
    - MSW flavor oscillations (if enabled)
    - Integration over recoil kinematics
    
    The function is backwards compatible: if only (energy, flux) are provided,
    it treats `energy` as the input energy grid. To support arbitrary binning,
    pass `energy_centers` explicitly.

    Parameters
    ----------
    energy : array-like
        The source energy grid (kept for backwards compatibility)
    flux : array-like
        Corresponding flux values for each energy bin
    energy_centers : array-like, optional
        Explicit energy bin centers for the scattering loop.
        If None, uses `energy` as centers.
    bin_width_local : float, optional
        Energy bin width for converting densities to per-bin values.
        If None, uses module-level `bin_width`.
    N_int_local : int, optional
        Number of integration steps per energy bin.
        If None, uses module-level `N_int`.

    Returns
    -------
    spectrum_2d : ndarray
        2D histogram (n_energy_bins × n_angle_bins) of electron events
    spectrum_energy : ndarray
        1D energy projection (sum over angles)
    spectrum_angle : ndarray
        1D angular projection (sum over energies)
    energy_bins : ndarray
        Energy bin edges
    angle_bins : ndarray
        Angular bin edges (radians)
        
    Notes
    -----
    The 2D spectrum is computed by:
    1. Loop over incoming neutrino energies
    2. For each energy, integrate over recoil electron energies T ∈ [0, T_max]
    3. Compute differential cross section dσ/dT
    4. Convert to scattering angle and bin in (E, θ) space
    
    The total event rate includes:
    - Cross section × number of target electrons × flux × bin width × runtime
    """
    # Determine parameters with fallbacks
    if energy_centers is None:
        centers = np.asarray(energy)
    else:
        centers = np.asarray(energy_centers)

    bw = bin_width_local if bin_width_local is not None else bin_width
    Nint = int(N_int_local) if N_int_local is not None else int(N_int)

    total_flux = np.sum(flux) * bw
    print("Total incoming neutrino flux: ", total_flux)

    # Define output energy and angle bins
    n_energy_bins = 100
    energy_bins = np.linspace(0, max_energy, n_energy_bins + 1)

    n_angle_bins = 50
    angle_bins = np.linspace(0, pi, n_angle_bins + 1)

    spectrum_2d = np.zeros((n_energy_bins + 1, n_angle_bins + 1))
    spectrum_energy = np.zeros(n_energy_bins + 1)
    spectrum_angle = np.zeros(n_angle_bins + 1)

    # Iterate over available centers and flux entries
    n_loop = min(len(centers), len(flux))
    for i in range(n_loop):
        ienergy = centers[i]
        
        # Apply MSW oscillation probability
        pee = mswlma(ienergy)
        if osci_mode == 0:
            pee = 1

        Tmax = cal_Tmax(ienergy)
        iflux = flux[i]
        iflux_e = pee * iflux  # Effective electron neutrino flux

        # Integrate over recoil electron energies
        for j in range(Nint):
            T = Tmax / Nint * (j + 0.5)  # Midpoint of integration bin
            
            # Differential cross section
            sig = (
                sig0_es
                / me
                * (
                    gL_e * gL_e
                    + gR_e * gR_e * (1 - T / ienergy) * (1 - T / ienergy)
                    - gL_e * gR_e * T * me / (ienergy * ienergy)
                )
            )
            
            # Event rate in this integration bin
            event = sig * Ne * iflux_e
            event_in_bin = event * Tmax / Nint * bw
            
            # Convert to scattering angle
            cosine = cal_costheta(T, ienergy)
            rad = pi - np.arccos(cosine)  # Scattering angle in radians

            # Bin in 2D (energy, angle) space
            energy_idx = np.digitize(T, energy_bins) - 1
            rad_idx = np.digitize(rad, angle_bins) - 1

            if 0 <= energy_idx < n_energy_bins and 0 <= rad_idx < n_angle_bins:
                spectrum_2d[energy_idx, rad_idx] += event_in_bin
                spectrum_energy[energy_idx] += event_in_bin
                spectrum_angle[rad_idx] += event_in_bin

    return spectrum_2d, spectrum_energy, spectrum_angle, energy_bins, angle_bins


def read_flux_data(filename):
    """Read neutrino flux data from text file.
    
    Parameters
    ----------
    filename : str
        Path to data file (2 columns: energy, flux)
    
    Returns
    -------
    energy : ndarray
        Neutrino energies
    flux : ndarray
        Flux values
    """
    data = np.loadtxt(filename)
    return data[:, 0], data[:, 1]


def plot_2d_distribution_contour(spectrum_2d, energy_bins, angle_bins, title="2D Distribution"):
    """Plot 2D electron spectrum as contour map.
    
    Parameters
    ----------
    spectrum_2d : ndarray
        2D histogram of events
    energy_bins : ndarray
        Energy bin edges
    angle_bins : ndarray
        Angular bin edges
    title : str
        Plot title
    
    Returns
    -------
    fig, ax : matplotlib figure and axes
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create mesh grid
    E, A = np.meshgrid(energy_bins, angle_bins)

    # Filled contour plot
    contour = ax.contourf(E, A, spectrum_2d.T, levels=20, cmap='viridis')

    # Contour lines
    ax.contour(E, A, spectrum_2d.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)

    # Color bar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Event Rate')

    # Labels and title
    ax.set_xlabel('Energy (MeV)')
    ax.set_ylabel('Angle (rad)')
    ax.set_title(title)

    return fig, ax


__all__ = [
    # Functions
    'cal_Tmax', 'cal_costheta', 'mswlma',
    'scatter_electron_spectrum',
    'read_flux_data',
    'plot_2d_distribution_contour',
]


if __name__ == "__main__":
    # Example usage
    energy, flux = read_flux_data('./DecayData/DecayDetectorCount_U0.01_M2.0.txt')
    spectrum_2d, spectrum_energy, spectrum_angle, energy_bins, angle_bins = scatter_electron_spectrum(energy, flux)

    plt.figure()
    plt.plot(energy_bins, spectrum_energy)
    plt.savefig('solar_energy.pdf')
    
    plt.figure()
    plt.plot(angle_bins, spectrum_angle)
    plt.savefig('solar_angle.pdf')

    print(spectrum_2d.shape[0])
    print(spectrum_2d.shape[1])
    energy_bins = np.linspace(0, 16, spectrum_2d.shape[0])
    angle_bins = np.linspace(0, pi, spectrum_2d.shape[1])
    _, _ = plot_2d_distribution_contour(spectrum_2d, energy_bins, angle_bins)
    plt.savefig('solar.pdf')
