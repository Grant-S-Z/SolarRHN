import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
import math
import time
from solar import scatter_electron_spectrum
from typing import Any, cast, Optional
import os
from functools import cache
from tools import timer
from ploter import *

pi = 3.141592653
GFermi = 1.1663787e-11  # MeV^-2
m_electron = 0.5109989507  # MeV
hbar = 6.582119569e-22  # MeV second
s2w = 0.22305  # weak mixing angle
distance_SE = 1.4960e11  # meters
speed_of_light = 299792458.0  # m/s


def RHN_Gamma_vvv(MH, U2):
    return pow(GFermi, 2) * pow(MH, 5) * U2 * 1.0 / (192.0 * pow(pi, 3))  # MeV


def RHN_Gamma_vll(MH, U2):
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
        hfactor / (192.0 * pow(pi, 3))  # MeV
    return gamma


def RHN_TauCM(MH, U2):
    return hbar / (RHN_Gamma_vll(MH, U2) + RHN_Gamma_vvv(MH, U2))  # second


def RHN_BR_vll(MH, U2):
    gamma_vll = RHN_Gamma_vll(MH, U2)
    gamma_vvv = RHN_Gamma_vvv(MH, U2)

    return gamma_vll / (gamma_vll + gamma_vvv)


# time of flight for RHN to travel from Sun to Earth in c.m.s
def RHN_TauF(MH, EH):
    if MH >= EH:
        return 0.0

    beta = math.sqrt(EH * EH - MH * MH) / EH
    return MH * distance_SE / (EH * beta * speed_of_light)


def diff_lambda(x, y, z):
    return x * x + y * y + z * z - 2 * (x * y + y * z + z * x)


# 2D distribution of nuL energy El and costheta in c.m.s frame
def diff_El_costheta_cms(El, costheta, MH, EH):
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


# lorentz transformation, input El and costheta in c.m.s frame
def cms_to_lab(El, costheta, MH, EH):
    beta = math.sqrt(EH * EH - MH * MH) / EH  # velocity of H in lab frame
    gamma_v = 1.0 / math.sqrt(1.0 - beta * beta)
    beta_l = 1.0  # neutrino, zero mass
    sintheta = math.sqrt(1.0 - costheta * costheta)  # theta in cms frame
    vy_l = beta_l * sintheta  # vx of nuL in cms frame
    vx_l = beta_l * costheta  # vy of nuL in cms frame

    # tantheta_p = vy_l/(gamma_v*(vx_l - beta))
    costheta_p = math.sqrt(
        pow(gamma_v * (vx_l + beta), 2.0)
        / (pow(gamma_v * (vx_l + beta), 2.0) + vy_l * vy_l)
    )
    if vx_l + beta < 0.0:
        costheta_p = -1.0 * costheta_p

    El_p = gamma_v * (El + beta * El * beta_l * costheta)

    return El_p, costheta_p


# lorentz transformation, input El and costheta in lab frame
def lab_to_cms(El, costheta, MH, EH):
    beta = math.sqrt(EH * EH - MH * MH) / EH  # velocity of H in lab frame
    gamma_v = 1.0 / math.sqrt(1.0 - beta * beta)
    beta_l = 1.0  # neutrino, zero mass
    sintheta = math.sqrt(1.0 - costheta * costheta)  # theta in lab frame
    vy_l = beta_l * sintheta  # vx of nuL in lab frame
    vx_l = beta_l * costheta  # vy of nuL in lab frame

    # tantheta_p = vy_l/(gamma_v*(vx_l - beta))
    costheta_p = math.sqrt(
        pow(gamma_v * (vx_l - beta), 2.0)
        / (pow(gamma_v * (vx_l - beta), 2.0) + vy_l * vy_l)
    )
    if vx_l - beta < 0.0:
        costheta_p = -1.0 * costheta_p

    El_p = gamma_v * (El - beta * El * beta_l * costheta)

    return El_p, costheta_p


# 2D distribution of nuL energy El and costheta in lab frame
def diff_El_costheta_lab(Elp, costhetap, MH, EH):
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
    if costheta > 1.0 or costheta < -1.0:
        return 0.0

    El_p, costheta_p = lab_to_cms(El, costheta, MH, EH)

    return diff_El_costheta_cms(El_p, costheta_p, MH, EH)


# 1D distribution of nuL costheta (in lab frame)
def diff_costheta(costheta, MH, EH):
    El_min = 0.0
    El_max = EH - 2.0 * m_electron
    nsteps = 500
    El_step = (El_max - El_min) / nsteps

    int_diff = 0.0
    for istep in range(nsteps):
        int_diff += diff_El_costheta_lab(El_min +
                                         istep * El_step, costheta, MH, EH)
    return int_diff


# 1D distribution of nuL energy El (in lab frame)
def diff_El(El, MH, EH):
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


# 1D distribution of epem energy Eee (in lab frame)
def diff_Eee(Eee, MH, EH):
    El = EH - Eee
    return diff_El(El, MH, EH)


# given a spectrum in CSV file, interpolate to get the y value for any given x value, return the new spectrum
def interpolateSpectrum(inputCSV, xvalues):
    df = pd.read_csv(inputCSV)
    interp_func = interp1d(
        df["energy"].values,
        df["flux"].values,
        kind="linear",
        fill_value=cast(Any, "extrapolate"),
    )

    spectrum_return = []
    for ix in xvalues:
        iy = interp_func(ix) * 1.0
        spectrum_return.append([ix, iy])
    return np.array(spectrum_return)


def integrateSpectrum(spectrum):
    sum_all = 0.0
    for idx in range(len(spectrum)):
        if idx > 0:
            sum_all += spectrum[idx][1] * \
                (spectrum[idx][0] - spectrum[idx - 1][0])
        else:
            sum_all += spectrum[idx][1] * \
                (spectrum[idx + 1][0] - spectrum[idx][0])
    return sum_all


def integrateSpectrum2D(data, x_centers=None, y_centers=None):
    """Integrate a 2D spectrum over both axes.

    Supports two input formats:
    - data is a (nX, nY, 3) array where data[:,:,0] are x centers,
      data[:,:,1] are y centers and data[:,:,2] are the density values
      (e.g. dN/dx/dy).
    - data is a 2D array Z (shape (nX, nY)) and x_centers, y_centers are
      provided as 1D arrays of centers.

    The function infers bin edges from centers and returns the scalar total
    integral: sum_ij Z[i,j] * dX[i] * dY[j].
    """

    def _edges_from_centers(centers):
        c = np.asarray(centers)
        if c.size == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        edges = np.empty(c.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (c[1:] + c[:-1])
        edges[0] = c[0] - (edges[1] - c[0])
        edges[-1] = c[-1] + (c[-1] - edges[-2])
        return edges

    # decode inputs
    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[2] >= 3:
        x = np.asarray(data[:, 0, 0])
        y = np.asarray(data[0, :, 1])
        Z = np.asarray(data[:, :, 2], dtype=float)
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        if x_centers is None or y_centers is None:
            raise ValueError(
                "x_centers and y_centers must be provided when passing a 2D Z array")
        x = np.asarray(x_centers)
        y = np.asarray(y_centers)
        Z = np.asarray(data, dtype=float)
    else:
        raise ValueError("Unsupported input for integrateSpectrum2D")

    if Z.size == 0:
        return 0.0

    # infer edges and widths
    x_edges = _edges_from_centers(x)
    y_edges = _edges_from_centers(y)
    dX = np.diff(x_edges)
    dY = np.diff(y_edges)

    # ensure shapes
    if Z.shape != (dX.size, dY.size):
        raise ValueError(
            f"Z shape {Z.shape} incompatible with inferred axes sizes {(dX.size, dY.size)}")

    total = float(np.sum(Z * dX[:, None] * dY[None, :]))
    return total


def getRHNSpectrum(spectrum_L, MH, U2):
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


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def getDecayedRHNSpectrum(spectrum_orig, MH, U2, distance, length):
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
            # print(energy[ie], PH, beta, delta_tau, tau_f, tau_cm, flux_orig[ie], math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm)), flux_orig[ie]*math.exp(-1.0*tau_f/tau_cm)*(1.0-math.exp(-1.0*delta_tau/tau_cm)))
    return np.array(spectrum_decayed)


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findRatioForDistance(MH, EH, U2, distance):
    if EH < MH + 0.001:
        return 0.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH * EH - MH * MH)
    beta = PH / EH
    tau_f = MH * distance / (EH * beta * speed_of_light)
    return 1.0 - math.exp(-1.0 * tau_f / tau_cm)


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findDistanceForRatio(MH, EH, U2, ratio):
    if ratio < 0.0 or ratio >= 1.0:
        return -999.0
    tau_cm = RHN_TauCM(MH, U2)
    PH = math.sqrt(EH * EH - MH * MH)
    beta = PH / EH
    tau_f = -1.0 * tau_cm * math.log(1.0 - ratio)
    return tau_f * EH * beta * speed_of_light / MH


# distance: distance from sun to decay point (e.g. earth), unit m
# length: length of detector, unit m
def findRatioForDistanceSpectrum(MH, spectrum, U2, distance):
    energy = spectrum[:, 0]
    flux = spectrum[:, 1]
    flux_decayed = np.zeros(len(flux))
    for ie in range(len(energy)):
        ratio_this = findRatioForDistance(MH, energy[ie], U2, distance)
        flux_decayed[ie] = flux[ie] * ratio_this
    return np.sum(flux_decayed) / np.sum(flux)


# see definition of phi and theta in yutao's thesis fig 3-4
def transform_phi_to_theta(cosphi, distance):
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


# distance: distance from sun to decay point, unit m
# length: length of detector (or flight path to integrate), unit m
# @timer
def getNulEAndAngleFromRHNDecay(spectrum_orig, MH, U2, distance, length, costheta_bins):
    energy = spectrum_orig[:, 0]
    flux_orig = spectrum_orig[:, 1]
    npoints_costheta = len(costheta_bins)

    costheta_step = costheta_bins[2] - costheta_bins[1]
    # 3rd axis: [0]=El value, [1]=costheta/cosphi value, [2]=distribution value
    diff_El_costheta_decayed = np.zeros((len(energy), npoints_costheta, 3))
    diff_El_cosphi_decayed = np.zeros((len(energy), npoints_costheta, 3))
    diff_El_decayed = np.zeros((len(energy), 2))
    diff_costheta_decayed = np.zeros((npoints_costheta, 2))
    diff_cosphi_decayed = np.zeros((npoints_costheta, 2))

    # fill axis metadata
    for iTh in range(npoints_costheta):
        for ie in range(len(energy)):
            diff_El_costheta_decayed[ie][iTh][0] = energy[ie]
            diff_El_costheta_decayed[ie][iTh][1] = costheta_bins[iTh]
            diff_El_cosphi_decayed[ie][iTh][0] = energy[ie]
            diff_El_cosphi_decayed[ie][iTh][1] = costheta_bins[iTh]
        diff_costheta_decayed[iTh][0] = costheta_bins[iTh]
        diff_cosphi_decayed[iTh][0] = costheta_bins[iTh]

    cosphi_needed = np.zeros(npoints_costheta)
    diff_cosphi_needed = np.zeros(npoints_costheta)

    distance_m = distance + 0.5 * length
    # convert cosphi distribution to costheta distribution
    # TODO: convert costheta to cosphi?
    for iTh in range(npoints_costheta):
        cosphi_needed[iTh] = -999.0
        costheta_this = costheta_bins[iTh]
        cosphi_temp = transform_theta_to_phi(costheta_this, distance_m)
        if cosphi_temp < -2.0:
            continue
        delta_costheta = 1e-6
        if costheta_this > 1.0 - 1e-4:
            delta_costheta = -1.0 * 1e-6
        dcosphi_temp = transform_theta_to_phi(
            costheta_this + delta_costheta, distance_m
        )
        if dcosphi_temp < -2.0:
            delta_costheta = -1.0 * delta_costheta
            dcosphi_temp = transform_theta_to_phi(
                costheta_this + delta_costheta, distance_m
            )
        if dcosphi_temp < -2.0:
            continue
        Jacob = abs((dcosphi_temp - cosphi_temp) / delta_costheta)
        # save Jacobian for costheta->cosphi transformation temporarily
        diff_costheta_decayed[iTh][1] = Jacob
        # cosphi_needed includes bins corresponding to costheta
        cosphi_needed[iTh] = cosphi_temp

    # print("cosphi_needed")
    # print(cosphi_needed)
    # integrate over all RHNs that decay in the give length
    nRHN_decayed_total = 0.0
    for ie in range(len(energy)):
        diff_El_decayed[ie][0] = energy[ie]

        EH = energy[ie]

        eStep = 0.0
        if ie > 0.0:
            eStep = energy[ie] - energy[ie - 1]
        else:
            eStep = energy[ie + 1] - energy[ie]
        if EH <= MH:
            continue
        else:
            tau_cm = RHN_TauCM(MH, U2)
            PH = math.sqrt(EH * EH - MH * MH)
            beta = PH / EH
            tau_f = MH * distance / (EH * beta * speed_of_light)
            delta_tau = MH * length / (EH * beta * speed_of_light)
            nRHN_decayed = (
                flux_orig[ie]
                * math.exp(-1.0 * tau_f / tau_cm)
                * (1.0 - math.exp(-1.0 * delta_tau / tau_cm))
            )
            nRHN_decayed_total += nRHN_decayed

            diff_El_cosphi_temp = np.zeros((len(energy), npoints_costheta))
            diff_El_temp = np.zeros(len(energy))
            diff_cosphi_temp = np.zeros(npoints_costheta)
            diff_cosphi_needed_temp = np.zeros(npoints_costheta)

            nTotalDecayed = 0.0
            nReachEarth = 0.0

            # First, compute the 2D distribution in cosphi space
            # iterate over cosphi grid (stored in costheta_bins)
            for icosphi in range(npoints_costheta):
                costheta_temp = transform_phi_to_theta(
                    costheta_bins[icosphi], distance_m)
                for ieL in range(len(energy)):
                    diff_temp = diff_El_costheta_lab(
                        energy[ieL], costheta_bins[icosphi], MH, EH
                    )
                    nTotalDecayed += diff_temp
                    if costheta_temp < -2.0:
                        continue
                    nReachEarth += diff_temp
                    diff_El_cosphi_temp[ieL][icosphi] += diff_temp
                    diff_El_temp[ieL] += diff_temp  # integrate over cosphi
                    # integrate over Elx
                    diff_cosphi_temp[icosphi] += diff_temp
            
            # Then, compute the 2D distribution in costheta space
            # This is done by evaluating at the cosphi values that correspond to each costheta bin
            diff_El_costheta_temp = np.zeros((len(energy), npoints_costheta))
            for iTh in range(npoints_costheta):
                if cosphi_needed[iTh] < -2.0:
                    continue
                costheta_check = transform_phi_to_theta(
                    cosphi_needed[iTh], distance_m)
                if costheta_check < -2.0:
                    continue
                jacobian_this = diff_costheta_decayed[iTh][1]  # dcosphi/dcostheta
                for ieL in range(len(energy)):
                    # Evaluate the physical distribution at the corresponding cosphi value
                    diff_at_cosphi = diff_El_costheta_lab(energy[ieL], cosphi_needed[iTh], MH, EH)
                    # Transform to costheta space: dN/dcostheta = dN/dcosphi * |dcosphi/dcostheta|
                    diff_El_costheta_temp[ieL][iTh] += jacobian_this * diff_at_cosphi
                    # For distances inside Earth orbit, also include the symmetric contribution
                    if distance_m < distance_SE:
                        diff_at_cosphi_neg = diff_El_costheta_lab(
                            energy[ieL], -1.0 * cosphi_needed[iTh], MH, EH
                        )
                        diff_El_costheta_temp[ieL][iTh] += jacobian_this * diff_at_cosphi_neg

            fractionReachEarth = 0.0
            if nTotalDecayed > 0.0:
                fractionReachEarth = nReachEarth / nTotalDecayed

            for icosphi in range(npoints_costheta):
                if cosphi_needed[icosphi] < -2.0:
                    continue
                costheta_temp = transform_phi_to_theta(
                    cosphi_needed[icosphi], distance_m)
                if costheta_temp < -2.0:
                    continue
                for ieL in range(len(energy)):
                    diff_cosphi_needed_temp[icosphi] += (
                        diff_costheta_decayed[icosphi][1]  # Jacob
                        * diff_El_costheta_lab(energy[ieL], cosphi_needed[icosphi], MH, EH)
                    )
                    if distance_m < distance_SE:
                        diff_cosphi_needed_temp[icosphi] += (
                            diff_costheta_decayed[icosphi][1]
                            * diff_El_costheta_lab(
                                energy[ieL], -1.0 *
                                cosphi_needed[icosphi], MH, EH
                            )
                        )

            # ===== Normalize and store the 2D distributions =====
            # Normalize so that when integrated over 2D grid, we get nRHN_decayed × fractionReachEarth
            # The key: we store densities, and integrateSpectrum2D will multiply by bin widths
            # So: ∫∫ ρ(E,angle) dE dangle = Σ ρ[i,j] × ΔE[i] × Δangle[j] = N_total
            # Therefore: ρ[i,j] should be normalized by simple sum, not weighted sum
            
            sum_diff_El_cosphi_temp = np.sum(diff_El_cosphi_temp)
            if sum_diff_El_cosphi_temp > 0.0:
                for ieL in range(len(energy)):
                    for icosphi in range(npoints_costheta):
                        # Store: particles per unit energy per unit cosphi
                        diff_El_cosphi_decayed[ieL][icosphi][2] += (
                            nRHN_decayed
                            * fractionReachEarth
                            * (1.0 / costheta_step)
                            * diff_El_cosphi_temp[ieL][icosphi]
                            / sum_diff_El_cosphi_temp
                        )
            
            sum_diff_El_costheta_temp = np.sum(diff_El_costheta_temp)
            if sum_diff_El_costheta_temp > 0.0:
                for ieL in range(len(energy)):
                    for iTh in range(npoints_costheta):
                        # Store: particles per unit energy per unit costheta
                        diff_El_costheta_decayed[ieL][iTh][2] += (
                            nRHN_decayed
                            * fractionReachEarth
                            * (1.0 / costheta_step)
                            * diff_El_costheta_temp[ieL][iTh]
                            / sum_diff_El_costheta_temp
                        )

            sum_diff_El_temp = np.sum(diff_El_temp)
            if sum_diff_El_temp > 0.0:
                for ieL in range(len(energy)):
                    diff_El_decayed[ieL][1] += (
                        nRHN_decayed
                        * fractionReachEarth
                        * diff_El_temp[ieL]
                        / sum_diff_El_temp
                    )
                    # diff_El_decayed[ieL][1] += nRHN_decayed*1*diff_El_temp[ieL]/sum_diff_El_temp

            sum_diff_cosphi_temp = np.sum(diff_cosphi_temp)
            if sum_diff_cosphi_temp > 0.0:
                for icosphi in range(npoints_costheta):
                    diff_cosphi_decayed[icosphi][1] += (
                        nRHN_decayed
                        * eStep
                        * (1.0 / costheta_step)
                        * fractionReachEarth
                        * diff_cosphi_temp[icosphi]
                        / sum_diff_cosphi_temp
                    )

            sum_diff_cosphi_needed_temp = np.sum(diff_cosphi_needed_temp)
            if sum_diff_cosphi_needed_temp > 0.0:
                for icosphi in range(npoints_costheta):
                    diff_cosphi_needed[icosphi] += (
                        nRHN_decayed
                        * eStep
                        * (1.0 / costheta_step)
                        * fractionReachEarth
                        * diff_cosphi_needed_temp[icosphi]
                        / sum_diff_cosphi_needed_temp
                    )

    # Store diff_cosphi_needed into diff_costheta_decayed for 1D costheta distribution
    for iTh in range(npoints_costheta):
        diff_costheta_decayed[iTh][1] = diff_cosphi_needed[iTh]

    # print("DEBUG: Final integrals check:")
    # print(f"  diff_El_decayed integral: {integrateSpectrum(diff_El_decayed):.6e}")
    # print(f"  diff_cosphi_decayed integral: {integrateSpectrum(diff_cosphi_decayed):.6e}")
    # print(f"  diff_El_cosphi_decayed 2D integral: {integrateSpectrum2D(diff_El_cosphi_decayed):.6e}")
    # print(f"  diff_El_costheta_decayed 2D integral: {integrateSpectrum2D(diff_El_costheta_decayed):.6e}")
    # print(f"  diff_costheta_decayed integral: {integrateSpectrum(diff_costheta_decayed):.6e}")
    
    return diff_El_decayed, diff_costheta_decayed, diff_cosphi_decayed, diff_El_costheta_decayed, diff_El_cosphi_decayed


def saveSpectrums(spectrums, column_names, fileName, labels):
    for i in range(len(spectrums)):
        if column_names is None:
            np.savetxt(
                fileName + labels[i] + ".csv", spectrums[i], delimiter=",", fmt="%f"
            )
        else:
            np.savetxt(
                fileName + labels[i] + ".csv",
                spectrums[i],
                delimiter=",",
                header=column_names,
                fmt="%f",
                comments="",
            )


@timer
def get_and_save_nuL_El_costheta_decay_in_flight(spectrum_L, U2, MH, savepath='./data/simulation/'):
    """Get decayed left-handed neutrino energy and angle distribution

    Parameters
    ----------
    spectrum_L : NDArray
        solar neutrino spectrum
    U2 : float
        mixing parameter square
    MH : float
        right-handed neutrino mass
    Returns
    -------
    NDArray
        decayed left-handed neutrino energy and angle distribution

    """

    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 201
    costheta_step = 2.0 / (npoints_costheta - 1.0)
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * costheta_step

    start_time = time.time()

    spectrum_LD = np.zeros((len(energy), len(costheta_arr), 2))
    spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
    flux_R = spectrum_R[:, 1]
    index_max = np.argmax(flux_R)
    E_max_flux = energy[index_max]

    spectrum_L_left = np.zeros((len(energy), 2))

    diff_El_decayed = np.zeros((len(energy), 2))
    diff_El_decayed_inside = np.zeros((len(energy), 2))
    diff_El_decayed_outside = np.zeros((len(energy), 2))
    diff_costheta_decayed = np.zeros((npoints_costheta, 2))
    diff_costheta_decayed_inside = np.zeros((npoints_costheta, 2))
    diff_costheta_decayed_outside = np.zeros((npoints_costheta, 2))
    diff_cosphi_decayed = np.zeros((npoints_costheta, 2))
    diff_cosphi_decayed_inside = np.zeros((npoints_costheta, 2))
    diff_cosphi_decayed_outside = np.zeros((npoints_costheta, 2))
    # 2D El vs costheta accumulated distributions (and inside/outside buckets)
    diff_El_costheta_decayed = np.zeros((len(energy), len(costheta_arr), 3))
    diff_El_costheta_decayed_inside = np.zeros(
        (len(energy), len(costheta_arr), 3))
    diff_El_costheta_decayed_outside = np.zeros(
        (len(energy), len(costheta_arr), 3))
    for iTh in range(len(costheta_arr)):
        for ie in range(len(energy)):
            diff_El_costheta_decayed[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed[ie, iTh, 1] = costheta_arr[iTh]
            diff_El_costheta_decayed_inside[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed_inside[ie, iTh, 1] = costheta_arr[iTh]
            diff_El_costheta_decayed_outside[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed_outside[ie, iTh, 1] = costheta_arr[iTh]
    for ie in range(len(energy)):
        spectrum_L_left[ie][0] = energy[ie]
        spectrum_L_left[ie][1] = spectrum_L[ie][1] - spectrum_R[ie][1]

        diff_El_decayed[ie][0] = energy[ie]
        diff_El_decayed_inside[ie][0] = energy[ie]
        diff_El_decayed_outside[ie][0] = energy[ie]
    for iTh in range(npoints_costheta):
        diff_costheta_decayed[iTh][0] = costheta_arr[iTh]
        diff_costheta_decayed_inside[iTh][0] = costheta_arr[iTh]
        diff_costheta_decayed_outside[iTh][0] = costheta_arr[iTh]
        diff_cosphi_decayed[iTh][0] = costheta_arr[iTh]
        diff_cosphi_decayed_inside[iTh][0] = costheta_arr[iTh]
        diff_cosphi_decayed_outside[iTh][0] = costheta_arr[iTh]

    print("===============================================")
    print("MH = ", MH, "U2 = ", U2)
    print("Tau_CM = ", RHN_TauCM(MH, U2))

    ratio_orbit = findRatioForDistance(MH, E_max_flux, U2, distance_SE)
    # split decay in steps
    # first, decay before reaching earth orbit
    nsteps_earth = 100
    distance_step = distance_SE * 1.0 / nsteps_earth
    for istep in range(nsteps_earth):
        (
            diff_El_this,
            diff_costheta_this,
            diff_cosphi_this,
            diff_El_costheta_this,
            diff_El_cosphi_this
        ) = getNulEAndAngleFromRHNDecay(
            spectrum_R,
            MH,
            U2,
            istep * 1.0 * distance_step,
            distance_step,
            costheta_arr,
        )
        print(
            "decay inside earth orbit, distance = ",
            "%.2f" % (istep * 1.0 * distance_step / distance_SE),
            " (SE), istep ",
            istep + 1,
            "/",
            nsteps_earth,
            ", decayed flux = ",
            integrateSpectrum(diff_El_this),
            integrateSpectrum(diff_costheta_this),
            integrateSpectrum(diff_cosphi_this),
            integrateSpectrum2D(diff_El_costheta_this),
            integrateSpectrum2D(diff_El_cosphi_this)
        )
        for ie in range(len(energy)):
            diff_El_decayed[ie][1] += diff_El_this[ie][1]
            diff_El_decayed_inside[ie][1] += diff_El_this[ie][1]
        for iTh in range(npoints_costheta):
            diff_costheta_decayed[iTh][1] += diff_costheta_this[iTh][1]
            diff_costheta_decayed_inside[iTh][1] += diff_costheta_this[iTh][1]
            diff_cosphi_decayed[iTh][1] += diff_cosphi_this[iTh][1]
            diff_cosphi_decayed_inside[iTh][1] += diff_cosphi_this[iTh][1]
        # accumulate 2D El vs costheta contributions for this step
        diff_El_costheta_decayed[:, :, 2] += diff_El_costheta_this[:, :, 2]
        diff_El_costheta_decayed_inside[:, :,
                                        2] += diff_El_costheta_this[:, :, 2]
        # print(diff_El_this)
        # print(diff_costheta_this)
        # print(diff_cosphi_this)
        # print("Total flux in this step: ", np.sum(diff_costheta_this[:,1]), np.sum(diff_cosphi_this[:,1]))

    # then, decay after flying outside earth orbit
    distance_start = distance_SE
    ratio_decayed = findRatioForDistanceSpectrum(
        MH, spectrum_R, U2, distance_start)
    distance_next = findDistanceForRatio(
        MH, E_max_flux, U2, ratio_orbit + (1 - ratio_orbit) / 10.0
    )

    print("Ratio decay inside earth orbit: ", ratio_decayed)

    while ratio_decayed < 0.999:
        (
            diff_El_this,
            diff_costheta_this,
            diff_cosphi_this,
            diff_El_costheta_this,
            diff_El_cosphi_this
        ) = getNulEAndAngleFromRHNDecay(
            spectrum_R,
            MH,
            U2,
            distance_start,
            distance_next - distance_start,
            costheta_arr,
        )
        ratio_decayed = findRatioForDistanceSpectrum(
            MH, spectrum_R, U2, distance_next)
        print(
            "decay outside earth orbit, distance = ",
            "%.2f" % (distance_start / distance_SE),
            " (SE), fraction decayed: ",
            "%.3f" % ratio_decayed,
            ", decayed flux = ",
            integrateSpectrum(diff_El_this),
        )

        distance_start = distance_next
        ratio_decayed2 = findRatioForDistance(
            MH, E_max_flux, U2, distance_next)
        ratio_remained2 = 1.0 - ratio_decayed2
        ratio_next = ratio_decayed2 + ratio_remained2 / 10.0
        if ratio_remained2 < 0.01:
            ratio_next = ratio_decayed2 + ratio_remained2 / 2.0
        distance_next = findDistanceForRatio(MH, E_max_flux, U2, ratio_next)

        for ie in range(len(energy)):
            diff_El_decayed[ie][1] += diff_El_this[ie][1]
            diff_El_decayed_outside[ie][1] += diff_El_this[ie][1]
        # accumulate 2D El vs costheta contributions for this outside step
        diff_El_costheta_decayed[:, :, 2] += diff_El_costheta_this[:, :, 2]
        diff_El_costheta_decayed_outside[:, :,
                                         2] += diff_El_costheta_this[:, :, 2]
        for iTh in range(npoints_costheta):
            diff_costheta_decayed[iTh][1] += diff_costheta_this[iTh][1]
            diff_costheta_decayed_outside[iTh][1] += diff_costheta_this[iTh][1]
            diff_cosphi_decayed[iTh][1] += diff_cosphi_this[iTh][1]
            diff_cosphi_decayed_outside[iTh][1] += diff_cosphi_this[iTh][1]

    end_time = time.time()
    elapsed_time = end_time - start_time
    print("Time used: ", elapsed_time, " seconds")

    # Save the 2D distribution (energy vs costheta) to CSV for later inspection.
    # We flatten the (nE, nTh, 3) array into rows [energy, costheta, value].
    os.makedirs(savepath, exist_ok=True)

    nE, nTh, _ = diff_El_costheta_decayed.shape
    flat = diff_El_costheta_decayed.reshape((nE * nTh, 3))
    filename = f"diff_El_costheta_M{MH:.1f}_U{U2:.1e}.csv"
    out_path = os.path.join(savepath, filename)
    np.savetxt(
        out_path,
        flat,
        delimiter=",",
        header="energy,costheta,value",
        fmt="%0.6e",
        comments="",
    )
    print("Saved diff_El_costheta_decayed to", out_path)

    # Also save the 1D energy and costheta distributions alongside the 2D map
    filename_el = f"diff_El_M{MH:.1f}_U{U2:.1e}.csv"
    out_el = os.path.join(savepath, filename_el)
    # diff_El_decayed is shape (nE, 2): [energy, value]
    np.savetxt(
        out_el,
        diff_El_decayed,
        delimiter=",",
        header="energy,value",
        fmt="%0.6e",
        comments="",
    )
    print("Saved diff_El_decayed to", out_el)

    filename_ct = f"diff_costheta_M{MH:.1f}_U{U2:.1e}.csv"
    out_ct = os.path.join(savepath, filename_ct)
    # diff_costheta_decayed is shape (nTh, 2): [costheta, value]
    np.savetxt(
        out_ct,
        diff_costheta_decayed,
        delimiter=",",
        header="costheta,value",
        fmt="%0.6e",
        comments="",
    )
    print("Saved diff_costheta_decayed to", out_ct)

    # return accumulated distributions including 2D map
    return (
        diff_El_decayed,
        diff_costheta_decayed,
        diff_cosphi_decayed,
        diff_El_costheta_decayed,
    )


@timer
def get_and_save_nuL_scatter_electron_El_costheta(diff_El_costheta_decayed, savepath='./data/simulation/', N_int_local=100000):
    """Get left-handed neutrino energy and angle distribution after scattering with electrons
    
    Optimized version: computes scatter spectrum for each incoming angle separately,
    then maps scattering angles to lab frame with azimuthal angle sampling for accuracy.

    Parameters
    ----------
    diff_El_costheta_decayed : NDArray
        left-handed neutrino energy and angle distribution before scattering
        Shape: (nE, nAngle, 3) where [:,:,0] = energy, [:,:,1] = costheta, [:,:,2] = flux
    savepath : str
        Directory to save output CSV files
    N_int_local : int
        Number of Monte Carlo samples for scatter_electron_spectrum

    Returns
    -------
    final_spectrum : ndarray
        Integrated electron 2D spectrum in lab frame (energy × cosθ_lab)
    e_bins : ndarray
        Energy bin edges
    costheta_lab_bins : ndarray
        Lab-frame cosθ bin edges
    spectra_per_angle : ndarray
        Per-incoming-angle spectra (for debugging)
    """
    from solar import scatter_electron_spectrum, bin_mid_array
    
    # Extract source axes
    energy_src = diff_El_costheta_decayed[:, 0, 0]
    costheta_nu = diff_El_costheta_decayed[0, :, 1]
    energy_target = np.array(bin_mid_array)
    
    # Build energy resampling function (bin-conservative)
    def resample_bin_average(x_src, y_src, x_tgt_centers):
        """Conservative bin-averaging resampler"""
        x_src = np.asarray(x_src)
        y_src = np.asarray(y_src)
        x_tgt = np.asarray(x_tgt_centers)
        
        # Infer bin edges from centers
        edges = np.zeros(len(x_tgt) + 1)
        edges[1:-1] = 0.5 * (x_tgt[1:] + x_tgt[:-1])
        edges[0] = x_tgt[0] - (edges[1] - x_tgt[0])
        edges[-1] = x_tgt[-1] + (x_tgt[-1] - edges[-2])
        
        y_tgt = np.zeros(len(x_tgt))
        order = np.argsort(x_src)
        x_sorted = x_src[order]
        y_sorted = y_src[order]
        
        for i in range(len(x_tgt)):
            left, right = edges[i], edges[i + 1]
            mask = (x_sorted > left) & (x_sorted < right)
            xs_local = np.concatenate(([left], x_sorted[mask], [right]))
            ys_local = np.interp(xs_local, x_sorted, y_sorted)
            integral = np.trapezoid(ys_local, xs_local)
            width = right - left
            y_tgt[i] = integral / width if width > 0 else 0.0
        
        return y_tgt
    
    # Step 1: Compute scatter spectrum for each incoming angle
    print("=" * 60)
    print("Computing scattered electrons for each incoming angle...")
    print("=" * 60)
    
    n_in_angles = len(costheta_nu)
    spectra_list = []
    e_bins = None
    a_bins = None
    
    for ia in range(n_in_angles):
        flux_2d = diff_El_costheta_decayed[:, ia, 2]
        
        if np.all(flux_2d == 0):
            spectra_list.append(None)
            continue
        
        # Resample to target energy grid
        flux_target = resample_bin_average(energy_src, flux_2d, energy_target)
        
        # Call scatter function
        print(f"  Angle {ia+1}/{n_in_angles}, cosθ={costheta_nu[ia]:.3f}")
        try:
            time_start = time.perf_counter()
            s2d, s_e, s_a, e_bins, a_bins = scatter_electron_spectrum(
                energy_target, flux_target, N_int_local=N_int_local
            )
            time_end = time.perf_counter()
            print(f"    Scatter completed in {time_end - time_start:.4f} seconds")
            spectra_list.append(s2d)
        except Exception as e:
            print(f"    Warning: Scatter failed: {e}")
            spectra_list.append(None)
    
    # Handle case where no scatter succeeded
    if e_bins is None or a_bins is None:
        print("Warning: No successful scatter calls, returning empty spectrum")
        n_energy_bins = 100
        n_angle_bins = 50
        e_bins = np.linspace(0, max(energy_target.max(), 16.0), n_energy_bins + 1)
        costheta_lab_bins = np.linspace(-1, 1, n_angle_bins + 1)
        final_spectrum = np.zeros((n_energy_bins, n_angle_bins))
        return final_spectrum, e_bins, costheta_lab_bins, []
    
    nE_out = len(e_bins) - 1
    nA_scatter = len(a_bins) - 1
    
    # Replace None with zeros
    for idx, item in enumerate(spectra_list):
        if item is None:
            spectra_list[idx] = np.zeros((nE_out, nA_scatter))
    
    # Step 2: Map scattering angles to lab frame with azimuthal sampling
    print("\n" + "=" * 60)
    print("Mapping scatter angles to lab frame (with azimuthal sampling)...")
    print("=" * 60)
    
    # Define lab-frame cosθ bins
    costheta_lab_bins = np.linspace(-1, 1, 51)
    nCostheta_lab = len(costheta_lab_bins) - 1
    costheta_lab_centers = 0.5 * (costheta_lab_bins[:-1] + costheta_lab_bins[1:])
    
    electron_2d_lab = np.zeros((nE_out, nCostheta_lab))
    
    # Scatter angle centers
    theta_s_centers = 0.5 * (a_bins[:-1] + a_bins[1:])
    
    # Azimuthal angle sampling (increase for better accuracy)
    n_phi = 12  # Sample 12 azimuthal angles
    phi_samples = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for ia in range(n_in_angles):
        if spectra_list[ia] is None or np.all(spectra_list[ia] == 0):
            continue
        
        cos_theta_in = costheta_nu[ia]
        sin_theta_in = np.sqrt(1 - cos_theta_in**2)
        
        print(f"  Processing incoming angle {ia+1}/{n_in_angles}, cosθ_in={cos_theta_in:.3f}")
        
        # For each scatter angle
        for ia_s in range(nA_scatter):
            theta_s = theta_s_centers[ia_s]
            cos_theta_s = np.cos(theta_s)
            sin_theta_s = np.sin(theta_s)
            
            # Sample over azimuthal angles
            for phi in phi_samples:
                # Complete formula: cos(θ_lab) = cos(θ_in)·cos(θ_s) - sin(θ_in)·sin(θ_s)·cos(φ)
                cos_theta_lab = cos_theta_in * cos_theta_s - sin_theta_in * sin_theta_s * np.cos(phi)
                cos_theta_lab = np.clip(cos_theta_lab, -1.0, 1.0)
                
                # Find lab bin
                ilab = np.searchsorted(costheta_lab_bins, cos_theta_lab) - 1
                ilab = np.clip(ilab, 0, nCostheta_lab - 1)
                
                # Distribute flux (divide by n_phi for azimuthal average)
                for ie in range(nE_out):
                    electron_2d_lab[ie, ilab] += spectra_list[ia][ie, ia_s] / n_phi
    
    final_spectrum = electron_2d_lab
    
    # Step 3: Save results
    print("\n" + "=" * 60)
    print("Saving scattered electron spectrum...")
    print("=" * 60)
    
    try:
        os.makedirs(savepath, exist_ok=True)
        e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
        
        rows = []
        for ie in range(nE_out):
            for ilab in range(nCostheta_lab):
                rows.append((e_centers[ie], costheta_lab_centers[ilab], electron_2d_lab[ie, ilab]))
        
        out_path = os.path.join(savepath, "scattered_electrons_2d_lab.csv")
        np.savetxt(out_path, np.array(rows), delimiter=',',
                   header='energy,costheta_lab,value', fmt='%0.6e', comments='')
        print(f"Saved 2D electron spectrum (lab frame) to {out_path}")
    except Exception as e:
        print(f"Warning: failed to save scattered electron spectrum: {e}")
    
    return final_spectrum, e_bins, costheta_lab_bins, spectra_list


def get_and_save_nuL_scatter_electron_El_costheta_from_csv(csv_path, savepath=None, N_int_local=100000, plot=True):
    """Load a flat CSV (energy,costheta,value) produced by
    `get_and_save_nuL_El_costheta_decay_in_flight` and compute the scattered-electron
    2D spectrum using `get_and_save_nuL_scatter_electron_El_costheta`, then optionally plot.

    Parameters
    ----------
    csv_path : str
        Path to CSV file containing neutrino 2D distribution.
        Expected 3 columns: energy,costheta,value (header optional).
    savepath : str, optional
        Directory to save output files. If None, uses directory of csv_path.
    N_int_local : int
        Number of Monte Carlo samples for scatter_electron_spectrum
    plot : bool
        Whether to generate plots (2D map and 1D projections)

    Returns
    -------
    final_spectrum : ndarray
        Electron 2D spectrum in lab frame
    e_bins : ndarray
        Energy bin edges
    costheta_lab_bins : ndarray
        Lab-frame cosθ bin edges
    spectra_per_angle : ndarray
        Per-incoming-angle spectra (for debugging)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    # Determine save directory
    if savepath is None:
        savepath = os.path.dirname(csv_path)
    
    print(f"\n{'='*60}")
    print(f"Loading neutrino distribution from: {csv_path}")
    print(f"{'='*60}")

    # Load CSV and parse
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        df = pd.read_csv(csv_path, header=None)

    if df.shape[1] != 3:
        raise ValueError('CSV must have 3 columns: energy,costheta,value')

    cols = list(df.columns)
    df = df.rename(columns={cols[0]: 'energy',
                   cols[1]: 'costheta', cols[2]: 'value'})

    # Pivot to regular grid (energy x costheta)
    pivot = df.pivot_table(index='energy', columns='costheta',
                           values='value', aggfunc='sum', fill_value=0.0)
    energy = pivot.index.values
    costheta = pivot.columns.values
    Z = pivot.values

    # Build diff_El_costheta_decayed array shape (nE, nCostheta, 3)
    nE = len(energy)
    nTh = len(costheta)
    diff_arr = np.zeros((nE, nTh, 3))
    diff_arr[:, :, 0] = energy[:, None]
    diff_arr[:, :, 1] = costheta[None, :]
    diff_arr[:, :, 2] = Z
    
    print(f"Loaded neutrino distribution: {nE} energy bins × {nTh} angle bins")
    print(f"Energy range: [{energy.min():.2f}, {energy.max():.2f}] MeV")
    print(f"Cosθ range: [{costheta.min():.3f}, {costheta.max():.3f}]")
    print(f"Total neutrino flux: {integrateSpectrum2D(diff_arr):.6e}")

    # Call existing scatter function
    electron_2d, e_bins, costheta_lab_bins, spectra_per_angle = get_and_save_nuL_scatter_electron_El_costheta(
        diff_arr, savepath=savepath, N_int_local=N_int_local
    )
    
    # Compute bin centers for later use
    e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
    costheta_centers = 0.5 * (costheta_lab_bins[:-1] + costheta_lab_bins[1:])
    
    # Generate plots if requested
    if plot:
        print(f"\n{'='*60}")
        print("Generating electron spectrum plots...")
        print(f"{'='*60}")
        
        # Extract parameter info from filename if possible (e.g., diff_El_costheta_M4.0_U1.0e-01.csv)
        basename = os.path.basename(csv_path)
        title_suffix = ""
        if 'M' in basename and 'U' in basename:
            # Try to parse M and U values
            import re
            m_match = re.search(r'M([\d.]+)', basename)
            u_match = re.search(r'U([\d.eE+-]+)', basename)
            if m_match and u_match:
                MH = float(m_match.group(1))
                U2 = float(u_match.group(1))
                title_suffix = f"M={MH:.1f} MeV, U²={U2:.2e}"
        
        # Convert electron_2d to standard format
        nE_e = len(e_centers)
        nA_e = len(costheta_centers)
        diff_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
        diff_El_costheta_electron[:, :, 0] = e_centers[:, None]
        diff_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
        diff_El_costheta_electron[:, :, 2] = electron_2d
        
        # Plot 1: 2D electron spectrum
        plot_El_costheta_map(
            diff_El_costheta_electron,
            savepath,
            filename="electron_2d_from_csv.pdf",
            title_prefix=f"Scattered Electron" + (f" ({title_suffix})" if title_suffix else "")
        )
        
        # Plot 2: 1D energy distribution (integrate over angle)
        diff_El_electron = np.zeros((nE_e, 2))
        diff_El_electron[:, 0] = e_centers
        costheta_widths = np.diff(costheta_lab_bins)
        for ie in range(nE_e):
            diff_El_electron[ie, 1] = np.sum(electron_2d[ie, :] * costheta_widths)
        
        plot_1d_energy_distribution(
            diff_El_electron,
            savepath,
            filename="electron_energy_1d_from_csv.pdf",
            title_prefix=f"Electron Energy" + (f" ({title_suffix})" if title_suffix else ""),
            ylabel="Event rate (MeV⁻¹)"
        )
        
        # Plot 3: 1D angular distribution (integrate over energy)
        diff_costheta_electron = np.zeros((nA_e, 2))
        diff_costheta_electron[:, 0] = costheta_centers
        energy_widths = np.diff(e_bins)
        for ia in range(nA_e):
            diff_costheta_electron[ia, 1] = np.sum(electron_2d[:, ia] * energy_widths)
        
        plot_1d_angle_distribution(
            diff_costheta_electron,
            savepath,
            filename="electron_angle_1d_from_csv.pdf",
            title_prefix=f"Electron Angular" + (f" ({title_suffix})" if title_suffix else ""),
            ylabel="Event rate (sr⁻¹)"
        )
        
        print(f"Plots saved to: {savepath}")
        print(f"  - electron_2d_from_csv.pdf")
        print(f"  - electron_energy_1d_from_csv.pdf")
        print(f"  - electron_angle_1d_from_csv.pdf")
    
    print(f"\n{'='*60}")
    print("Electron spectrum computation complete!")
    print(f"Total electron events: {integrateSpectrum2D(electron_2d, e_centers, costheta_centers):.6e}")
    print(f"{'='*60}\n")

    return electron_2d, e_bins, costheta_lab_bins, spectra_per_angle


def list_neutrino_csv_files(directory='./data/simulation/'):
    """List all neutrino 2D distribution CSV files in a directory.
    
    Parameters
    ----------
    directory : str
        Directory to search for CSV files
    
    Returns
    -------
    list of dict
        List of dictionaries with file info: {'path', 'MH', 'U2', 'filename'}
    """
    import glob
    import re
    
    pattern = os.path.join(directory, 'diff_El_costheta_M*.csv')
    files = glob.glob(pattern)
    
    results = []
    for fpath in sorted(files):
        basename = os.path.basename(fpath)
        
        # Try to extract M and U values
        m_match = re.search(r'M([\d.]+)', basename)
        u_match = re.search(r'U([\d.eE+-]+)', basename)
        
        info = {
            'path': fpath,
            'filename': basename,
            'MH': float(m_match.group(1)) if m_match else None,
            'U2': float(u_match.group(1)) if u_match else None,
        }
        results.append(info)
    
    return results


def batch_compute_electrons_from_csv(directory='./data/simulation/', N_int_local=100000, plot=True):
    """Batch process all neutrino CSV files in a directory to compute electron spectra.
    
    Parameters
    ----------
    directory : str
        Directory containing neutrino CSV files
    N_int_local : int
        Number of Monte Carlo samples for each scattering computation
    plot : bool
        Whether to generate plots for each file
    
    Returns
    -------
    list of dict
        Results for each file processed
    """
    files = list_neutrino_csv_files(directory)
    
    if not files:
        print(f"No neutrino CSV files found in {directory}")
        return []
    
    print(f"\n{'='*70}")
    print(f"Found {len(files)} neutrino distribution files")
    print(f"{'='*70}")
    
    for i, finfo in enumerate(files):
        print(f"{i+1}. {finfo['filename']}")
        if finfo['MH'] is not None and finfo['U2'] is not None:
            print(f"   MH={finfo['MH']:.1f} MeV, U²={finfo['U2']:.2e}")
    
    print(f"\n{'='*70}")
    print("Starting batch processing...")
    print(f"{'='*70}\n")
    
    results = []
    for i, finfo in enumerate(files):
        print(f"\n[{i+1}/{len(files)}] Processing: {finfo['filename']}")
        
        try:
            electron_2d, e_bins, costheta_bins, _ = get_and_save_nuL_scatter_electron_El_costheta_from_csv(
                csv_path=finfo['path'],
                savepath=None,  # Use same directory as input
                N_int_local=N_int_local,
                plot=plot
            )
            
            result = {
                'success': True,
                'file': finfo['filename'],
                'MH': finfo['MH'],
                'U2': finfo['U2'],
                'electron_total': integrateSpectrum2D(electron_2d, 
                                                      0.5*(e_bins[:-1]+e_bins[1:]),
                                                      0.5*(costheta_bins[:-1]+costheta_bins[1:]))
            }
            results.append(result)
            print(f"✓ Success! Total electron events: {result['electron_total']:.6e}")
            
        except Exception as e:
            result = {
                'success': False,
                'file': finfo['filename'],
                'MH': finfo['MH'],
                'U2': finfo['U2'],
                'error': str(e)
            }
            results.append(result)
            print(f"✗ Failed: {e}")
    
    # Print summary
    print(f"\n{'='*70}")
    print("BATCH PROCESSING SUMMARY")
    print(f"{'='*70}")
    
    n_success = sum(1 for r in results if r['success'])
    n_failed = len(results) - n_success
    
    print(f"Total files: {len(results)}")
    print(f"Successful: {n_success}")
    print(f"Failed: {n_failed}")
    
    if n_failed > 0:
        print(f"\nFailed files:")
        for r in results:
            if not r['success']:
                print(f"  - {r['file']}: {r['error']}")
    
    print(f"{'='*70}\n")
    
    return results


def process_single_parameter_set(args):
    """Process a single (U2, MH) parameter set and generate all plots.
    
    Parameters
    ----------
    args : tuple
        (spectrum_nuL_orig, U2, MH, output_dir)
    
    Returns
    -------
    dict
        Summary statistics for this parameter set
    """
    spectrum_nuL_orig, U2, MH, output_dir = args
    
    print(f"\n{'='*60}")
    print(f"Processing U2={U2:.2e}, MH={MH:.1f} MeV")
    print(f"{'='*60}")
    
    # Create subdirectory for this parameter set
    param_dir = os.path.join(output_dir, f"U2_{U2:.2e}_MH_{MH:.1f}")
    os.makedirs(param_dir, exist_ok=True)
    
    # Step 1: Get neutrino distributions from RHN decay
    print("Step 1: Computing neutrino distributions from RHN decay...")
    (
        diff_El_nu,
        diff_costheta_nu,
        diff_cosphi_nu,
        diff_El_costheta_nu,
    ) = get_and_save_nuL_El_costheta_decay_in_flight(
        spectrum_nuL_orig, U2, MH, savepath=param_dir
    )
    
    # Step 2: Plot neutrino 2D distribution
    print("Step 2: Plotting neutrino 2D distribution...")
    plot_El_costheta_map(
        diff_El_costheta_nu, 
        param_dir, 
        filename=f"neutrino_2d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Neutrino: U²={U2:.2e}, M={MH:.1f} MeV"
    )
    
    # Step 3: Plot neutrino 1D energy distribution
    print("Step 3: Plotting neutrino 1D energy distribution...")
    plot_1d_energy_distribution(
        diff_El_nu,
        param_dir,
        filename=f"neutrino_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Neutrino Energy: U²={U2:.2e}, M={MH:.1f} MeV",
        ylabel="Flux (MeV⁻¹ cm⁻² s⁻¹)"
    )
    
    # Step 4: Plot neutrino 1D angular distribution
    print("Step 4: Plotting neutrino 1D angular distribution...")
    plot_1d_angle_distribution(
        diff_costheta_nu,
        param_dir,
        filename=f"neutrino_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
        title_prefix=f"Neutrino Angular: U²={U2:.2e}, M={MH:.1f} MeV",
        ylabel="Flux (sr⁻¹ cm⁻² s⁻¹)"
    )
    
    # Step 5: Compute scattered electron distributions
    print("Step 5: Computing scattered electron distributions...")
    try:
        electron_2d, e_bins, costheta_lab_bins, _ = get_and_save_nuL_scatter_electron_El_costheta(
            diff_El_costheta_nu, 
            savepath=param_dir,
            N_int_local=100000
        )
        
        # Convert electron_2d to same format as neutrino distribution
        e_centers = 0.5 * (e_bins[:-1] + e_bins[1:])
        costheta_centers = 0.5 * (costheta_lab_bins[:-1] + costheta_lab_bins[1:])
        
        # Build diff_El_costheta format for electron
        nE_e = len(e_centers)
        nA_e = len(costheta_centers)
        diff_El_costheta_electron = np.zeros((nE_e, nA_e, 3))
        diff_El_costheta_electron[:, :, 0] = e_centers[:, None]
        diff_El_costheta_electron[:, :, 1] = costheta_centers[None, :]
        diff_El_costheta_electron[:, :, 2] = electron_2d
        
        # Step 6: Plot electron 2D distribution
        print("Step 6: Plotting electron 2D distribution...")
        plot_El_costheta_map(
            diff_El_costheta_electron,
            param_dir,
            filename=f"electron_2d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Electron: U²={U2:.2e}, M={MH:.1f} MeV"
        )
        
        # Step 7: Plot electron 1D energy distribution (integrate over angle)
        print("Step 7: Plotting electron 1D energy distribution...")
        diff_El_electron = np.zeros((nE_e, 2))
        diff_El_electron[:, 0] = e_centers
        # Integrate over angle bins (approximate with simple sum × bin width)
        costheta_widths = np.diff(costheta_lab_bins)
        for ie in range(nE_e):
            diff_El_electron[ie, 1] = np.sum(electron_2d[ie, :] * costheta_widths)
        
        plot_1d_energy_distribution(
            diff_El_electron,
            param_dir,
            filename=f"electron_energy_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Electron Energy: U²={U2:.2e}, M={MH:.1f} MeV",
            ylabel="Event rate (MeV⁻¹)"
        )
        
        # Step 8: Plot electron 1D angular distribution (integrate over energy)
        print("Step 8: Plotting electron 1D angular distribution...")
        diff_costheta_electron = np.zeros((nA_e, 2))
        diff_costheta_electron[:, 0] = costheta_centers
        # Integrate over energy bins
        energy_widths = np.diff(e_bins)
        for ia in range(nA_e):
            diff_costheta_electron[ia, 1] = np.sum(electron_2d[:, ia] * energy_widths)
        
        plot_1d_angle_distribution(
            diff_costheta_electron,
            param_dir,
            filename=f"electron_angle_1d_U2_{U2:.2e}_MH_{MH:.1f}.pdf",
            title_prefix=f"Electron Angular: U²={U2:.2e}, M={MH:.1f} MeV",
            ylabel="Event rate (sr⁻¹)"
        )
        
        electron_success = True
        
    except Exception as e:
        print(f"Warning: Electron scattering failed for U2={U2}, MH={MH}: {e}")
        electron_success = False
    
    # Collect summary statistics
    from utils import integrateSpectrum, integrateSpectrum2D
    summary = {
        'U2': U2,
        'MH': MH,
        'neutrino_total_flux': integrateSpectrum(diff_El_nu),
        'neutrino_2d_integral': integrateSpectrum2D(diff_El_costheta_nu),
        'electron_success': electron_success,
        'output_dir': param_dir
    }
    
    print(f"\nCompleted U2={U2:.2e}, MH={MH:.1f} MeV")
    print(f"  Neutrino total flux: {summary['neutrino_total_flux']:.6e}")
    print(f"  Output directory: {param_dir}")
    
    return summary