"""
RHN (Right-Handed Neutrino) physics properties and decay calculations.

This module contains functions for calculating RHN decay widths, lifetimes,
branching ratios, and spectrum transformations.
"""

import math
import numpy as np
from .constants import pi, GFermi, m_electron, hbar, s2w, distance_SE, speed_of_light


def RHN_Gamma_vvv(MH, U2):
    """Calculate RHN decay width to three neutrinos.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    
    Returns
    -------
    float
        Decay width Γ(N → ννν) in MeV
    """
    return pow(GFermi, 2) * pow(MH, 5) * U2 * 1.0 / (192.0 * pow(pi, 3))


def RHN_Gamma_vll(MH, U2):
    """Calculate RHN decay width to neutrino and lepton pair.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    
    Returns
    -------
    float
        Decay width Γ(N → νe⁺e⁻) in MeV
    """
    C1 = 0.25 * (1 - 4 * s2w + 8 * s2w**2)
    C2 = 0.5 * s2w * (2 * s2w - 1)
    C3 = 0.25 * (1 + 4 * s2w + 8 * s2w**2)
    C4 = 0.5 * s2w * (2 * s2w + 1)

    xl = m_electron / MH

    L = math.log(
        (1.0 - 3 * xl**2 - (1 - xl**2) * math.sqrt(1.0 - 4.0 * xl**2))
        / (xl**2 * (1.0 + math.sqrt(1.0 - 4.0 * xl**2)))
    )

    hfactor = (C1 * (1.0 - 1.0) + C3 * 1.0) * (
        (1.0 - 14 * xl**2 - 2.0 * pow(xl, 4) - 12.0 * pow(xl, 6))
        * math.sqrt(1.0 - 4.0 * xl**2)
        + 12.0 * pow(xl, 4) * (pow(xl, 4) - 1) * L
    ) + 4 * (C2 * (1 - 1) + C4 * 1) * (
        xl**2 * (2 + 10 * xl**2 - 12 * pow(xl, 4)) *
        math.sqrt(1.0 - 4.0 * xl**2)
        + 6 * pow(xl, 4) * (1 - 2 * xl**2 + 2 * pow(xl, 4)) * L
    )

    gamma = pow(GFermi, 2) * pow(MH, 5) * U2 * \
        hfactor / (192.0 * pow(pi, 3))
    return gamma


def RHN_TauCM(MH, U2):
    """Calculate RHN lifetime in center-of-mass frame.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    
    Returns
    -------
    float
        Lifetime τ_CM in seconds
    """
    return hbar / (RHN_Gamma_vll(MH, U2) + RHN_Gamma_vvv(MH, U2))


def RHN_BR_vll(MH, U2):
    """Calculate branching ratio for RHN → νe⁺e⁻.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    
    Returns
    -------
    float
        Branching ratio BR(N → νe⁺e⁻)
    """
    gamma_vll = RHN_Gamma_vll(MH, U2)
    gamma_vvv = RHN_Gamma_vvv(MH, U2)
    return gamma_vll / (gamma_vll + gamma_vvv)


def RHN_TauF(MH, EH):
    """Calculate time of flight for RHN to travel from Sun to Earth in CMS.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    EH : float
        RHN energy in MeV
    
    Returns
    -------
    float
        Time of flight in seconds
    """
    if MH >= EH:
        return 0.0

    beta = math.sqrt(EH * EH - MH * MH) / EH
    return MH * distance_SE / (EH * beta * speed_of_light)


def getRHNSpectrum(spectrum_L, MH, U2):
    """Calculate RHN spectrum from left-handed neutrino spectrum.
    
    Parameters
    ----------
    spectrum_L : ndarray
        Left-handed neutrino spectrum, shape (N, 2): [energy, flux]
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    
    Returns
    -------
    ndarray
        RHN spectrum, shape (N, 2): [energy, flux]
    """
    energy = spectrum_L[:, 0]
    flux_L = spectrum_L[:, 1]
    spectrum_R = []
    for ie in range(len(energy)):
        if energy[ie] <= MH:
            spectrum_R.append([energy[ie], 0.0])
        else:
            spectrum_R.append(
                [energy[ie], flux_L[ie] * U2 *
                    math.sqrt(1.0 - (MH / energy[ie]) ** 2)]
            )
    return np.array(spectrum_R)


def getRHNSpectrums(spectrum_L, grid_U2M):
    """Calculate RHN spectra for a grid of (U2, MH) parameters.
    
    Parameters
    ----------
    spectrum_L : ndarray
        Left-handed neutrino spectrum
    grid_U2M : list
        Grid of (U2, MH) parameter pairs
    
    Returns
    -------
    list
        List of RHN spectra for each parameter combination
    """
    nU2 = len(grid_U2M)
    nM = len(grid_U2M[0])

    spectrums_R = []

    for i in range(nU2):
        spectrums_i = []
        for j in range(nM):
            U2_this = grid_U2M[i][j][0]
            M_this = grid_U2M[i][j][1]
            spectrums_ij = getRHNSpectrum(spectrum_L, M_this, U2_this)
            spectrums_i.append(spectrums_ij)
        spectrums_R.append(spectrums_i)

    return spectrums_R


def getDecayedRHNSpectrum(spectrum_orig, MH, U2, distance, length):
    """Calculate decayed RHN spectrum at given distance.
    
    Parameters
    ----------
    spectrum_orig : ndarray
        Original RHN spectrum
    MH : float
        RHN mass in MeV
    U2 : float
        Mixing parameter squared
    distance : float
        Distance from sun to decay point in meters
    length : float
        Length of detector in meters
    
    Returns
    -------
    ndarray
        Decayed RHN spectrum
    """
    energy = spectrum_orig[:, 0]
    flux_orig = spectrum_orig[:, 1]

    spectrum_decayed = []
    for ie in range(len(energy)):
        EH = energy[ie]
        if EH <= MH:
            spectrum_decayed.append([EH, 0.0])
        else:
            tau_cm = RHN_TauCM(MH, U2)
            PH = math.sqrt(EH * EH - MH * MH)
            beta = PH / EH
            tau_f = MH * distance / (EH * beta * speed_of_light)
            delta_tau = MH * length / (EH * beta * speed_of_light)
            spectrum_decayed.append(
                [
                    EH,
                    flux_orig[ie]
                    * math.exp(-1.0 * tau_f / tau_cm)
                    * (1.0 - math.exp(-1.0 * delta_tau / tau_cm)),
                ]
            )
    return np.array(spectrum_decayed)


def findRatioForDistance(MH, EH, U2, distance):
    """Find decay ratio at given distance.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    EH : float
        RHN energy in MeV
    U2 : float
        Mixing parameter squared
    distance : float
        Distance in meters
    
    Returns
    -------
    float
        Fraction of RHNs that have decayed
    """
    if EH < MH + 0.001:
        return 0.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH * EH - MH * MH)
    beta = PH / EH
    tau_f = MH * distance / (EH * beta * speed_of_light)
    return 1.0 - math.exp(-1.0 * tau_f / tau_cm)


def findDistanceForRatio(MH, EH, U2, ratio):
    """Find distance corresponding to given decay ratio.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    EH : float
        RHN energy in MeV
    U2 : float
        Mixing parameter squared
    ratio : float
        Target decay fraction (0 to 1)
    
    Returns
    -------
    float
        Distance in meters, or -999.0 if ratio is invalid
    """
    if ratio < 0.0 or ratio >= 1.0:
        return -999.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH * EH - MH * MH)
    beta = PH / EH
    tau_f = -1.0 * tau_cm * math.log(1.0 - ratio)
    return tau_f * EH * beta * speed_of_light / MH


def findRatioForDistanceSpectrum(MH, spectrum, U2, distance):
    """Calculate average decay ratio for entire spectrum at given distance.
    
    Parameters
    ----------
    MH : float
        RHN mass in MeV
    spectrum : ndarray
        RHN spectrum
    U2 : float
        Mixing parameter squared
    distance : float
        Distance in meters
    
    Returns
    -------
    float
        Weighted average decay fraction
    """
    energy = spectrum[:, 0]
    flux = spectrum[:, 1]
    flux_decayed = np.zeros(len(flux))
    for ie in range(len(energy)):
        ratio_this = findRatioForDistance(MH, energy[ie], U2, distance)
        flux_decayed[ie] = flux[ie] * ratio_this
    return np.sum(flux_decayed) / np.sum(flux)


__all__ = [
    'RHN_Gamma_vvv',
    'RHN_Gamma_vll',
    'RHN_TauCM',
    'RHN_BR_vll',
    'RHN_TauF',
    'getRHNSpectrum',
    'getRHNSpectrums',
    'getDecayedRHNSpectrum',
    'findRatioForDistance',
    'findDistanceForRatio',
    'findRatioForDistanceSpectrum',
]
