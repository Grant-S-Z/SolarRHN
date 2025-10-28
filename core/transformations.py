"""
Coordinate transformations between different reference frames.

This module handles Lorentz transformations between:
- Center-of-mass (CMS) frame and lab frame
- Angular transformations (φ ↔ θ) for neutrino propagation
"""

import math
from .constants import distance_SE


def cms_to_lab(El, costheta, MH, EH):
    """Lorentz transformation from CMS frame to lab frame.
    
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
    tuple
        (El_lab, costheta_lab): energy and angle in lab frame
    """
    beta = math.sqrt(EH * EH - MH * MH) / EH  # velocity of RHN in lab frame
    gamma_v = 1.0 / math.sqrt(1.0 - beta * beta)
    beta_l = 1.0  # neutrino, zero mass
    sintheta = math.sqrt(1.0 - costheta * costheta)
    vy_l = beta_l * sintheta  # vy of nuL in CMS frame
    vx_l = beta_l * costheta  # vx of nuL in CMS frame

    costheta_p = math.sqrt(
        pow(gamma_v * (vx_l + beta), 2.0)
        / (pow(gamma_v * (vx_l + beta), 2.0) + vy_l * vy_l)
    )
    if vx_l + beta < 0.0:
        costheta_p = -1.0 * costheta_p

    El_p = gamma_v * (El + beta * El * beta_l * costheta)

    return El_p, costheta_p


def lab_to_cms(El, costheta, MH, EH):
    """Lorentz transformation from lab frame to CMS frame.
    
    Parameters
    ----------
    El : float
        Neutrino energy in lab frame (MeV)
    costheta : float
        cos(θ) in lab frame
    MH : float
        RHN mass (MeV)
    EH : float
        RHN energy in lab frame (MeV)
    
    Returns
    -------
    tuple
        (El_cms, costheta_cms): energy and angle in CMS frame
    """
    beta = math.sqrt(EH * EH - MH * MH) / EH  # velocity of RHN in lab frame
    gamma_v = 1.0 / math.sqrt(1.0 - beta * beta)
    beta_l = 1.0  # neutrino, zero mass
    sintheta = math.sqrt(1.0 - costheta * costheta)
    vy_l = beta_l * sintheta  # vy of nuL in lab frame
    vx_l = beta_l * costheta  # vx of nuL in lab frame

    costheta_p = math.sqrt(
        pow(gamma_v * (vx_l - beta), 2.0)
        / (pow(gamma_v * (vx_l - beta), 2.0) + vy_l * vy_l)
    )
    if vx_l - beta < 0.0:
        costheta_p = -1.0 * costheta_p

    El_p = gamma_v * (El - beta * El * beta_l * costheta)

    return El_p, costheta_p


def transform_phi_to_theta(cosphi, distance):
    """Transform angle φ (in Sun-Earth frame) to θ (lab frame at Earth).
    
    See Yutao's thesis Fig 3-4 for definition of angles.
    
    Parameters
    ----------
    cosphi : float
        cos(φ) where φ is angle in Sun-Earth reference frame
    distance : float
        Distance from Sun to decay point (meters)
    
    Returns
    -------
    float
        cos(θ) in lab frame, or -999.0 if transformation is invalid
    """
    if cosphi < -1.0 or cosphi > 1.0:
        return -999.0
    if cosphi > 0.0 and distance > distance_SE:
        return -999.0

    sinphi = math.sqrt(1.0 - cosphi * cosphi)
    sintheta = distance * sinphi / distance_SE
    if abs(sintheta - 1.0) < 1e-8:
        sintheta = 1.0
    if sintheta > 1.0:
        return -999.0
    costheta = math.sqrt(1.0 - sintheta * sintheta)
    if distance >= distance_SE:
        return -1.0 * costheta
    else:
        return costheta


def transform_theta_to_phi(costheta, distance):
    """Transform angle θ (lab frame) to φ (Sun-Earth frame).
    
    Parameters
    ----------
    costheta : float
        cos(θ) in lab frame at Earth
    distance : float
        Distance from Sun to decay point (meters)
    
    Returns
    -------
    float
        cos(φ) in Sun-Earth frame, or -999.0 if transformation is invalid
    """
    if abs(distance) < 1e-6:
        return -999.0
    if costheta < -1.0 or costheta > 1.0:
        return -999.0
    if costheta < 0.0 and distance < distance_SE:
        return -999.0
    sintheta = math.sqrt(1.0 - costheta * costheta)
    sinphi = distance_SE * sintheta / distance
    if abs(sinphi - 1.0) < 1e-8:
        sinphi = 1.0
    if sinphi > 1.0:
        return -999.0
    cosphi = math.sqrt(1.0 - sinphi * sinphi)
    if distance >= distance_SE:
        return -1.0 * cosphi
    else:
        return cosphi


__all__ = [
    'cms_to_lab',
    'lab_to_cms',
    'transform_phi_to_theta',
    'transform_theta_to_phi',
]
