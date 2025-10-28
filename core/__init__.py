"""
Core physics modules for Solar RHN simulation.

This package contains modular components for:
- Physical constants
- RHN properties and decay rates
- Coordinate transformations
- Decay distributions
- Spectrum processing
- Neutrino-electron scattering
- Electron scattering and angle mapping
- Monte Carlo sampling algorithms
- Utility tools (timing, debugging)
"""

from .constants import *
from .rhn_physics import *
from .transformations import *
from .decay_distributions import *
from .spectrum_utils import *
from .neutrino_electron_scattering import *
from .electron_scattering import *
from .sampling import *
from .tools import *

__all__ = [
    # Constants
    'pi', 'GFermi', 'm_electron', 'hbar', 's2w', 'distance_SE', 'speed_of_light',
    
    # RHN physics
    'RHN_Gamma_vvv', 'RHN_Gamma_vll', 'RHN_TauCM', 'RHN_BR_vll', 'RHN_TauF',
    'getRHNSpectrum', 'getRHNSpectrums', 'getDecayedRHNSpectrum',
    'findRatioForDistance', 'findDistanceForRatio', 'findRatioForDistanceSpectrum',
    
    # Transformations
    'cms_to_lab', 'lab_to_cms', 'transform_phi_to_theta', 'transform_theta_to_phi',
    
    # Decay distributions
    'diff_lambda', 'diff_El_costheta_cms', 'diff_El_costheta_lab',
    'diff_El_costheta_lab_wrong', 'diff_costheta', 'diff_El', 'diff_Eee',
    'getNulEAndAngleFromRHNDecay', 'get_and_save_nuL_El_costheta_decay_in_flight',
    
    # Spectrum utilities
    'interpolateSpectrum', 'integrateSpectrum', 'integrateSpectrum2D',
    'saveSpectrums',
    
    # Neutrino-electron scattering (from solar.py)
    'sw2', 'gL_e', 'gR_e', 'gL_mu', 'gR_mu', 'me',
    'Ne', 'sig0_es', 'Q0', 'RunTime',
    'N_int', 'Ntime', 'osci_mode', 'bin_width', 'max_energy',
    'n_bins', 'bin_array', 'bin_mid_array',
    'cal_Tmax', 'cal_costheta', 'mswlma',
    'scatter_electron_spectrum',
    'read_flux_data', 'plot_2d_distribution_contour',
    
    # Electron scattering (RHN decay products)
    'get_and_save_nuL_scatter_electron_El_costheta',
    'get_and_save_nuL_scatter_electron_El_costheta_from_csv',
    
    # Sampling algorithms
    'rejection_sampling_2Dfunc', 'rejection_sampling_1Dfunc',
    'generate_samples_from_spectrum', 'getMaximumValue2D',
    'getNuLEAndAngleBySampling',
    
    # Utility tools
    'timer',
]
