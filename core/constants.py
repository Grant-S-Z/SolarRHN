"""
Physical constants used in Solar RHN calculations.
"""
import numpy as np

# Mathematical constant
pi = 3.141592653
na = 6.02e23

# Fundamental constants
GFermi = 1.1663787e-11  # Fermi constant, MeV^-2
m_electron = 0.5109989507  # Electron mass, MeV
me = 0.511  # Electron mass (MeV) - used in neutrino-electron scattering
hbar = 6.582119569e-22  # Reduced Planck constant, MeV·second

# Electroweak parameters
s2w = 0.22305  # Weak mixing angle (sin²θ_W)
sw2 = 0.23  # sin²θ_W for neutrino-electron scattering

# Weak coupling constants for neutrino-electron scattering
gL_e = 0.5 + sw2  # Left-handed coupling (e)
gR_e = sw2  # Right-handed coupling (e)
gL_mu = -0.5 + sw2  # Left-handed coupling (μ)
gR_mu = sw2  # Right-handed coupling (μ)


# Astronomical constants
distance_SE = 1.4960e11  # Sun-Earth distance, meters
speed_of_light = 299792458.0  # Speed of light, m/s


# Detector parameters
mass = 5e8 # 500t, gram
# mass = 1e8 # 100t (Borexino)
Ne = na * mass / 18 * 10 # number of electrons in detector
sig0_es = 88.083e-46  # baseline cross section
Q0 = 0.862  # Reference Q value
# volume = 500.0 # Detector volume in cubic meters

exposure_time = 365.0 * 24.0 * 3600.0  # 365d, second
# exposure_time = 1000.0 * 24.0 * 3600.0 # Borexino 1000d

# detector_size = pow(volume, 1.0/3.0) # meters
# S = pow(detector_size * 100.0, 2)  # cm^2

detector_radius = 5.0  # meters
# detector_radius = 3.02 # Borexino FV radius cut

S = pi * pow(detector_radius * 100.0, 2)  # cm²
exposure = S * exposure_time  # cm²·s
attenuation_length = 4 / 3 * detector_radius


# Integration and binning parameters for neutrino-electron scattering
N_int = 100000
Ntime = 1e4  # Number of time steps
osci_mode = 1  # Oscillation mode (1=on, 0=off)
bin_width = 0.2  # Energy bin width (MeV)
max_energy = 15.8  # Maximum energy (MeV)
n_bins = int(max_energy / bin_width)  # Number of energy bins
bin_array = np.linspace(0.2, 15.8, n_bins)  # Energy bin edges
bin_mid_array = np.linspace(0.3, 15.7, n_bins - 1)  # Energy bin centers

__all__ = [
    # Basic constants
    'pi', 'GFermi', 'm_electron', 'me', 'hbar',
    # Electroweak parameters
    's2w', 'sw2', 'gL_e', 'gR_e', 'gL_mu', 'gR_mu',
    # Astronomical constants
    'distance_SE', 'speed_of_light',
    # Detector parameters
    'Ne', 'sig0_es', 'Q0',
    # Integration and binning
    'N_int', 'Ntime', 'osci_mode', 'bin_width', 'max_energy',
    'n_bins', 'bin_array', 'bin_mid_array',
    # Detector parameters
    'exposure_time', 'detector_radius', 'S', 'exposure', 'attenuation_length',
]
