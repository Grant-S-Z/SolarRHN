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

# Try to import numba for JIT compilation
try:
    from numba import jit
    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False
    # Dummy decorator if numba not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


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


@jit(nopython=True, cache=True)
def _scatter_core_numba(centers, flux, bw, Nint, n_energy_bins, n_angle_bins, 
                        energy_bins, angle_bins, osci_on):
    """JIT-compiled core scattering calculation with bilinear binning.
    
    This function uses bilinear interpolation to distribute each scattering event
    across neighboring bins, eliminating binning artifacts and producing smooth
    angular distributions.
    """
    spectrum_2d = np.zeros((n_energy_bins, n_angle_bins))
    spectrum_energy = np.zeros(n_energy_bins)
    spectrum_angle = np.zeros(n_angle_bins)
    
    # Constants (copy to avoid accessing module-level in numba)
    _pi = 3.141592653589793
    _me = 0.511
    _gL_e = 0.5 + 0.23
    _gR_e = 0.23
    _sig0_es = 88.083e-46
    _Ne = 1.673e32
    
    # MSW parameters
    A = 1.0
    sin12 = 0.31
    sin13 = 0.02241
    ps = 92.5
    
    # Pre-compute bin widths for interpolation
    energy_bin_widths = energy_bins[1:] - energy_bins[:-1]
    angle_bin_widths = angle_bins[1:] - angle_bins[:-1]
    
    n_loop = min(len(centers), len(flux))
    
    for i in range(n_loop):
        ienergy = centers[i]
        
        # Apply MSW oscillation probability (inlined)
        if osci_on:
            ksi = 0.203 * A * (1 - sin13) * ienergy * ps / 100
            c = ((1 - 2 * sin12) - ksi) / np.sqrt(1 - 2 * ksi * (1 - 2 * sin12) + ksi * ksi)
            pee = sin13 * sin13 + (1 - sin13) * (1 - sin13) * (0.5 + 0.5 * c * (1 - 2 * sin12))
        else:
            pee = 1.0
        
        # Calculate Tmax
        Tmax = 2.0 * ienergy * ienergy / (_me + 2.0 * ienergy)
        iflux = flux[i]
        iflux_e = pee * iflux
        
        # Integrate over recoil electron energies
        for j in range(Nint):
            T = Tmax / Nint * (j + 0.5)
            
            # Differential cross section
            sig = (
                _sig0_es / _me * (
                    _gL_e * _gL_e
                    + _gR_e * _gR_e * (1 - T / ienergy) * (1 - T / ienergy)
                    - _gL_e * _gR_e * T * _me / (ienergy * ienergy)
                )
            )
            
            # Event rate
            event = sig * _Ne * iflux_e
            event_in_bin = event * Tmax / Nint * bw
            
            # Calculate scattering angle
            cosine = np.sqrt((T * (_me + ienergy) * (_me + ienergy)) / ((T + 2 * _me) * ienergy * ienergy))
            rad = _pi - np.arccos(cosine)
            
            # === BILINEAR BINNING: Distribute event to 4 neighboring bins ===
            # Find bin indices
            energy_idx = np.searchsorted(energy_bins, T) - 1
            rad_idx = np.searchsorted(angle_bins, rad) - 1
            
            # Clamp to valid range
            energy_idx = max(0, min(energy_idx, n_energy_bins - 1))
            rad_idx = max(0, min(rad_idx, n_angle_bins - 1))
            
            # Calculate fractional positions within bins
            # For energy
            if energy_idx < n_energy_bins - 1:
                e_low = energy_bins[energy_idx]
                e_width = energy_bin_widths[energy_idx]
                e_frac = (T - e_low) / e_width if e_width > 0 else 0.5
                e_frac = max(0.0, min(1.0, e_frac))
            else:
                e_frac = 1.0
            
            # For angle
            if rad_idx < n_angle_bins - 1:
                a_low = angle_bins[rad_idx]
                a_width = angle_bin_widths[rad_idx]
                a_frac = (rad - a_low) / a_width if a_width > 0 else 0.5
                a_frac = max(0.0, min(1.0, a_frac))
            else:
                a_frac = 1.0
            
            # Distribute to neighboring bins using bilinear weights
            # Bottom-left bin
            w00 = (1.0 - e_frac) * (1.0 - a_frac)
            spectrum_2d[energy_idx, rad_idx] += w00 * event_in_bin
            
            # Bottom-right bin (if not at right edge)
            if rad_idx < n_angle_bins - 1:
                w01 = (1.0 - e_frac) * a_frac
                spectrum_2d[energy_idx, rad_idx + 1] += w01 * event_in_bin
            
            # Top-left bin (if not at top edge)
            if energy_idx < n_energy_bins - 1:
                w10 = e_frac * (1.0 - a_frac)
                spectrum_2d[energy_idx + 1, rad_idx] += w10 * event_in_bin
                
                # Top-right bin (if not at any edge)
                if rad_idx < n_angle_bins - 1:
                    w11 = e_frac * a_frac
                    spectrum_2d[energy_idx + 1, rad_idx + 1] += w11 * event_in_bin
            
            # For 1D projections, use simple binning (already integrated)
            spectrum_energy[energy_idx] += event_in_bin
            spectrum_angle[rad_idx] += event_in_bin
    
    return spectrum_2d, spectrum_energy, spectrum_angle


def _scatter_core_numpy(centers, flux, bw, Nint, n_energy_bins, n_angle_bins,
                        energy_bins, angle_bins, osci_on):
    """NumPy-based core scattering calculation with bilinear binning (fallback without numba)."""
    spectrum_2d = np.zeros((n_energy_bins, n_angle_bins))
    spectrum_energy = np.zeros(n_energy_bins)
    spectrum_angle = np.zeros(n_angle_bins)
    
    # Pre-compute bin widths
    energy_bin_widths = energy_bins[1:] - energy_bins[:-1]
    angle_bin_widths = angle_bins[1:] - angle_bins[:-1]
    
    n_loop = min(len(centers), len(flux))
    
    for i in range(n_loop):
        ienergy = centers[i]
        
        # Apply MSW oscillation probability
        if osci_on:
            pee = mswlma(ienergy)
        else:
            pee = 1.0
        
        Tmax = cal_Tmax(ienergy)
        iflux = flux[i]
        iflux_e = pee * iflux
        
        # Integrate over recoil electron energies
        for j in range(Nint):
            T = Tmax / Nint * (j + 0.5)
            
            # Differential cross section
            sig = (
                sig0_es / me * (
                    gL_e * gL_e
                    + gR_e * gR_e * (1 - T / ienergy) * (1 - T / ienergy)
                    - gL_e * gR_e * T * me / (ienergy * ienergy)
                )
            )
            
            event = sig * Ne * iflux_e
            event_in_bin = event * Tmax / Nint * bw
            
            # Convert to scattering angle
            cosine = cal_costheta(T, ienergy)
            rad = pi - np.arccos(cosine)
            
            # === BILINEAR BINNING ===
            energy_idx = np.digitize(T, energy_bins) - 1
            rad_idx = np.digitize(rad, angle_bins) - 1
            
            # Clamp to valid range
            energy_idx = max(0, min(energy_idx, n_energy_bins - 1))
            rad_idx = max(0, min(rad_idx, n_angle_bins - 1))
            
            # Calculate fractional positions
            if energy_idx < n_energy_bins - 1:
                e_low = energy_bins[energy_idx]
                e_width = energy_bin_widths[energy_idx]
                e_frac = (T - e_low) / e_width if e_width > 0 else 0.5
                e_frac = max(0.0, min(1.0, e_frac))
            else:
                e_frac = 1.0
            
            if rad_idx < n_angle_bins - 1:
                a_low = angle_bins[rad_idx]
                a_width = angle_bin_widths[rad_idx]
                a_frac = (rad - a_low) / a_width if a_width > 0 else 0.5
                a_frac = max(0.0, min(1.0, a_frac))
            else:
                a_frac = 1.0
            
            # Distribute to 4 neighboring bins
            w00 = (1.0 - e_frac) * (1.0 - a_frac)
            spectrum_2d[energy_idx, rad_idx] += w00 * event_in_bin
            
            if rad_idx < n_angle_bins - 1:
                w01 = (1.0 - e_frac) * a_frac
                spectrum_2d[energy_idx, rad_idx + 1] += w01 * event_in_bin
            
            if energy_idx < n_energy_bins - 1:
                w10 = e_frac * (1.0 - a_frac)
                spectrum_2d[energy_idx + 1, rad_idx] += w10 * event_in_bin
                
                if rad_idx < n_angle_bins - 1:
                    w11 = e_frac * a_frac
                    spectrum_2d[energy_idx + 1, rad_idx + 1] += w11 * event_in_bin
            
            # 1D projections
            spectrum_energy[energy_idx] += event_in_bin
            spectrum_angle[rad_idx] += event_in_bin
    
    return spectrum_2d, spectrum_energy, spectrum_angle


def scatter_electron_spectrum(energy, flux, *, energy_centers=None, bin_width_local=None, N_int_local=None, use_numba=True):
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
    use_numba : bool, optional
        Whether to use Numba JIT compilation for speedup.
        If None, auto-detect (use if available).

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
    
    Performance:
    - With Numba JIT: ~5-10x faster
    - Without Numba: Standard NumPy implementation
    """
    # Determine parameters with fallbacks
    if energy_centers is None:
        centers = np.asarray(energy)
    else:
        centers = np.asarray(energy_centers)

    bw = bin_width_local if bin_width_local is not None else bin_width
    Nint = int(N_int_local) if N_int_local is not None else int(N_int)
    
    # Auto-detect numba usage
    if use_numba is None:
        use_numba = HAS_NUMBA
    elif use_numba and not HAS_NUMBA:
        print("Warning: Numba requested but not available. Install with: pip install numba")
        use_numba = False

    total_flux = np.sum(flux) * bw
    if use_numba and HAS_NUMBA:
        print(f"Total incoming neutrino flux: {total_flux:.6e} (using Numba JIT)")
    else:
        print(f"Total incoming neutrino flux: {total_flux:.6e}")

    # Define output energy and angle bins
    n_energy_bins = 100
    energy_bins = np.linspace(0, max_energy, n_energy_bins + 1)

    n_angle_bins = 50
    angle_bins = np.linspace(0, pi, n_angle_bins + 1)

    # Choose implementation
    osci_on = (osci_mode == 1)
    
    if use_numba and HAS_NUMBA:
        spectrum_2d, spectrum_energy, spectrum_angle = _scatter_core_numba(
            centers, flux, bw, Nint, n_energy_bins, n_angle_bins,
            energy_bins, angle_bins, osci_on
        )
    else:
        spectrum_2d, spectrum_energy, spectrum_angle = _scatter_core_numpy(
            centers, flux, bw, Nint, n_energy_bins, n_angle_bins,
            energy_bins, angle_bins, osci_on
        )

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

    # Filter out energy bins that are zero or very close to zero
    energy_centers = 0.5 * (energy_bins[:-1] + energy_bins[1:])
    energy_mask = energy_centers > 1e-6
    if np.any(energy_mask):
        # Filter energy bins and spectrum
        energy_bins_filtered = np.concatenate([[energy_bins[0]], energy_bins[1:][energy_mask]])
        spectrum_2d_filtered = spectrum_2d[energy_mask, :]
    else:
        # Keep all bins if none can be filtered
        energy_bins_filtered = energy_bins
        spectrum_2d_filtered = spectrum_2d

    # Create mesh grid
    E, A = np.meshgrid(energy_bins_filtered, angle_bins)

    # Filled contour plot
    contour = ax.contourf(E, A, spectrum_2d_filtered.T, levels=20, cmap='viridis')

    # Contour lines
    ax.contour(E, A, spectrum_2d_filtered.T, levels=10, colors='black', linewidths=0.5, alpha=0.5)

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
