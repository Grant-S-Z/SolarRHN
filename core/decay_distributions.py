"""
Decay kinematics and distributions for RHN → νe⁺e⁻.

This module provides differential distributions in both CMS and lab frames
for neutrinos from RHN decay.
"""

import math
from .constants import m_electron
from .transformations import cms_to_lab, lab_to_cms


def diff_lambda(x, y, z):
    """Helper function for phase space calculations.
    
    Parameters
    ----------
    x, y, z : float
        Input parameters
    
    Returns
    -------
    float
        λ(x,y,z) = x² + y² + z² - 2(xy + yz + zx)
    """
    return x * x + y * y + z * z - 2 * (x * y + y * z + z * x)


def diff_El_costheta_cms(El, costheta, MH, EH):
    """2D distribution of nuL energy El and costheta in CMS frame.
    
    Parameters
    ----------
    El : float
        Neutrino energy in CMS frame (MeV)
    costheta : float
        cos(θ) in CMS frame
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    float
        Differential distribution dN/dEl/dcostheta in CMS frame
    """
    if costheta > 1.0 or costheta < -1.0:
        return 0.0

    beta = math.sqrt(EH * EH - MH * MH) / EH  # velocity of H in lab frame

    El_l, costheta_l = cms_to_lab(El, costheta, MH, EH)

    El_max = (MH * MH - 4.0 * m_electron * m_electron) / (2.0 * MH)
    costheta_min = (1.0 / beta) * ((MH / El_max) *
                                   (1.0 - (EH - El_l) / EH) - 1.0)
    if El >= El_max or costheta <= costheta_min:
        return 0

    Eb = El / MH
    mi = m_electron / MH
    mj = m_electron / MH
    mb = 0.0
    Q2 = 1.0 - 2.0 * Eb + mb * mb
    if abs(Q2) < 1e-6:
        return 0

    lambda_ij = diff_lambda(1.0, mi * mi / Q2, mj * mj / Q2)
    if lambda_ij <= 0:
        return 0
    lambda_Qb = diff_lambda(1.0, Q2, mb * mb)
    if lambda_Qb <= 0:
        return 0

    Aij = pow(lambda_ij, 1.5)
    Bij = (
        2.0
        * pow(lambda_ij, 0.5)
        * (1.0 + (mi * mi + mj * mj) / Q2 - 2.0 * pow((mi * mi - mj * mj) / Q2, 2.0))
    )
    f1 = math.sqrt(lambda_Qb) * (
        2.0 * Q2 * (1 + mb * mb - Q2) * Aij +
        (pow(1 - mb * mb, 2.0) - Q2 * Q2) * Bij
    )
    fs = lambda_Qb * (2.0 * Q2 * Aij - (1.0 - mb * mb - Q2) * Bij)
    zeta = 1.0

    return f1 + zeta * beta * costheta * fs


def diff_El_costheta_lab(Elp, costhetap, MH, EH):
    """2D distribution of nuL energy El and costheta in lab frame.
    
    Uses Jacobian transformation from CMS to lab frame.
    
    Parameters
    ----------
    Elp : float
        Neutrino energy in lab frame (MeV)
    costhetap : float
        cos(θ) in lab frame
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    float
        Differential distribution dN/dEl/dcostheta in lab frame
    """
    if costhetap > 1.0 or costhetap < -1.0:
        return 0.0

    El, costheta = lab_to_cms(Elp, costhetap, MH, EH)

    diff_cms = diff_El_costheta_cms(El, costheta, MH, EH)

    delta_El = 1e-6
    delta_costheta = 1e-6
    if costhetap > 1.0 - 1e-4:
        delta_costheta = -1e-6

    El_Edelta, costheta_Edelta = lab_to_cms(Elp + delta_El, costhetap, MH, EH)
    El_Tdelta, costheta_Tdelta = lab_to_cms(
        Elp, costhetap + delta_costheta, MH, EH)

    pE_pEp = (El_Edelta - El) / delta_El
    pE_ptp = (El_Tdelta - El) / delta_costheta
    pt_pEp = (costheta_Edelta - costheta) / delta_El
    pt_ptp = (costheta_Tdelta - costheta) / delta_costheta

    Jacob = abs(pE_pEp * pt_ptp - pE_ptp * pt_pEp)
    return diff_cms * Jacob


def diff_El_costheta_lab_wrong(El, costheta, MH, EH):
    """Legacy incorrect implementation (kept for reference).
    
    This version does not properly account for the Jacobian.
    Use diff_El_costheta_lab instead.
    """
    if costheta > 1.0 or costheta < -1.0:
        return 0.0

    El_p, costheta_p = lab_to_cms(El, costheta, MH, EH)

    return diff_El_costheta_cms(El_p, costheta_p, MH, EH)


def diff_costheta(costheta, MH, EH):
    """1D distribution of nuL costheta in lab frame.
    
    Integrated over all neutrino energies.
    
    Parameters
    ----------
    costheta : float
        cos(θ) in lab frame
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    float
        dN/dcostheta integrated over energy
    """
    El_min = 0.0
    El_max = EH - 2.0 * m_electron
    nsteps = 500
    El_step = (El_max - El_min) / nsteps

    int_diff = 0.0
    for istep in range(nsteps):
        int_diff += diff_El_costheta_lab(El_min +
                                         istep * El_step, costheta, MH, EH)
    return int_diff


def diff_El(El, MH, EH):
    """1D distribution of nuL energy El in lab frame.
    
    Integrated over all angles.
    
    Parameters
    ----------
    El : float
        Neutrino energy in lab frame (MeV)
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    float
        dN/dEl integrated over angles
    """
    costheta_min = -1.0
    costheta_max = 1.0
    nsteps = 500
    costheta_step = (costheta_max - costheta_min) / nsteps

    int_diff = 0.0
    for istep in range(nsteps):
        int_diff += diff_El_costheta_lab(
            El, costheta_max - istep * costheta_step, MH, EH
        )

    return int_diff


def diff_Eee(Eee, MH, EH):
    """1D distribution of e⁺e⁻ pair energy in lab frame.
    
    Parameters
    ----------
    Eee : float
        Total energy of e⁺e⁻ pair (MeV)
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    float
        dN/dEee
    """
    El = EH - Eee
    return diff_El(El, MH, EH)


__all__ = [
    'diff_lambda',
    'diff_El_costheta_cms',
    'diff_El_costheta_lab',
    'diff_El_costheta_lab_wrong',
    'diff_costheta',
    'diff_El',
    'diff_Eee',
]
