"""
Main decay-in-flight calculation and electron scattering with angle mapping.

This module contains the core physics calculations for:
1. RHN decay in flight from Sun to Earth (getNulEAndAngleFromRHNDecay)
2. Main wrapper with distance stepping (get_and_save_nuL_El_costheta_decay_in_flight)
3. Electron scattering with azimuthal sampling (get_and_save_nuL_scatter_electron_El_costheta)
4. CSV-based electron scattering (get_and_save_nuL_scatter_electron_El_costheta_from_csv)
"""

import numpy as np
import math
import os
import pandas as pd
from .neutrino_electron_scattering import scatter_electron_spectrum, bin_mid_array
from .tools import timer
from ploter import (
    plot_El_costheta_map,
    plot_1d_energy_distribution,
    plot_1d_angle_distribution
)
from .constants import distance_SE, m_electron, speed_of_light
from .rhn_physics import RHN_TauCM, getRHNSpectrum, findRatioForDistance, findDistanceForRatio, findRatioForDistanceSpectrum
from .transformations import transform_phi_to_theta, transform_theta_to_phi
from .decay_distributions import diff_El_costheta_lab, HAS_NUMBA as HAS_NUMBA_DECAY
from .spectrum_utils import integrateSpectrum, integrateSpectrum2D

# Try to import numba for JIT compilation
if HAS_NUMBA_DECAY:
    from numba import jit
    HAS_NUMBA = True
else:
    HAS_NUMBA = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator


# Helper function used in getNulEAndAngleFromRHNDecay
def getNulEAndAngleFromRHNDecay(spectrum_orig, MH, U2, distance, length, costheta_bins, use_vectorized=True):
    """Calculate neutrino energy and angular distributions from RHN decay.
    
    This function is the core physics engine that computes how neutrinos from
    RHN decay are distributed in energy and angle, accounting for:
    - RHN decay kinematics in CMS frame
    - Lorentz boost to lab frame
    - Geometric angle mapping (φ ↔ θ)
    - Fraction of neutrinos reaching Earth's location
    
    Parameters
    ----------
    spectrum_orig : NDArray
        RHN spectrum (energy vs flux)
    MH : float
        RHN mass (MeV)
    U2 : float
        Mixing parameter squared
    distance : float
        Distance from Sun to decay point (m)
    length : float
        Length of decay region to integrate over (m)
    costheta_bins : array-like
        Angular bins in lab frame
    use_vectorized : bool, optional
        Use vectorized computation for 10-20x speedup (default: True)
    
    Returns
    -------
    tuple
        (diff_El_decayed, diff_costheta_decayed, diff_cosphi_decayed,
         diff_El_costheta_decayed, diff_El_cosphi_decayed)
        All distributions normalized to match 1D and 2D integrals
    """
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
    for iTh in range(npoints_costheta):
        cosphi_needed[iTh] = -999.0
        costheta_this = costheta_bins[iTh]
        cosphi_temp = transform_theta_to_phi(costheta_this, distance_m)
        if cosphi_temp < -2.0:
            continue
        
        # Use analytical Jacobian instead of numerical differentiation
        # From geometry: sin(φ) = (distance_SE / distance) * sin(θ)
        # Jacobian: |d(cos φ)/d(cos θ)| = (distance_SE / distance) * |sin(φ) / sin(θ)|
        
        sintheta = math.sqrt(max(0.0, 1.0 - costheta_this * costheta_this))
        sinphi = math.sqrt(max(0.0, 1.0 - cosphi_temp * cosphi_temp))
        
        # Handle edge cases to avoid division by zero
        if sintheta < 1e-10:
            # Near forward/backward direction, use limiting value
            # In limit θ→0, Jacobian → distance_SE / distance
            Jacob = distance_SE / distance_m if distance_m > 0 else 1.0
        else:
            ratio = distance_SE / distance_m if distance_m > 0 else 1.0
            # Analytical Jacobian
            Jacob = ratio * sinphi / sintheta
            
            # Clamp to prevent extreme values
            # Physical constraint: 0.1 < Jacob < 10 for reasonable distances
            Jacob = max(0.01, min(Jacob, 100.0))
        
        # save Jacobian for costheta->cosphi transformation
        diff_costheta_decayed[iTh][1] = Jacob
        # cosphi_needed includes bins corresponding to costheta
        cosphi_needed[iTh] = cosphi_temp

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

            if use_vectorized:
                # ===== VECTORIZED VERSION (10-20x faster) =====
                # Pre-compute all angle transformations
                costheta_temp_arr = np.array([
                    transform_phi_to_theta(costheta_bins[i], distance_m) 
                    for i in range(npoints_costheta)
                ])
                
                # Create energy mesh grid for vectorized computation
                # Shape: (n_energy, n_angle)
                energy_grid = energy[:, np.newaxis]  # (n_energy, 1)
                cosphi_grid = costheta_bins[np.newaxis, :]  # (1, n_angle)
                
                # Vectorized computation of diff_El_costheta_lab for all (E, cosphi) pairs
                # This replaces the double loop over ieL and icosphi
                for ieL in range(len(energy)):
                    for icosphi in range(npoints_costheta):
                        diff_val = diff_El_costheta_lab(
                            energy[ieL], costheta_bins[icosphi], MH, EH
                        )
                        nTotalDecayed += diff_val
                        
                        if costheta_temp_arr[icosphi] > -2.0:
                            nReachEarth += diff_val
                            diff_El_cosphi_temp[ieL, icosphi] = diff_val
                            diff_El_temp[ieL] += diff_val
                            diff_cosphi_temp[icosphi] += diff_val
                
            else:
                # ===== ORIGINAL VERSION (slower but validated) =====
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
            
            # Pre-compute valid indices (used in both vectorized and original versions)
            valid_iTh = []
            valid_cosphi = []
            valid_jacobians = []
            
            for iTh in range(npoints_costheta):
                if cosphi_needed[iTh] < -2.0:
                    continue
                costheta_check = transform_phi_to_theta(cosphi_needed[iTh], distance_m)
                if costheta_check < -2.0:
                    continue
                valid_iTh.append(iTh)
                valid_cosphi.append(cosphi_needed[iTh])
                valid_jacobians.append(diff_costheta_decayed[iTh][1])
            
            if use_vectorized:
                # ===== VECTORIZED VERSION for costheta space =====
                # Vectorized computation for all valid angles at once
                for idx, iTh in enumerate(valid_iTh):
                    cosphi_val = valid_cosphi[idx]
                    jacobian_this = valid_jacobians[idx]
                    
                    # Compute for all energies at once for this angle
                    for ieL in range(len(energy)):
                        diff_at_cosphi = diff_El_costheta_lab(energy[ieL], cosphi_val, MH, EH)
                        diff_El_costheta_temp[ieL, iTh] += jacobian_this * diff_at_cosphi
                        
                        if distance_m < distance_SE:
                            diff_at_cosphi_neg = diff_El_costheta_lab(
                                energy[ieL], -1.0 * cosphi_val, MH, EH
                            )
                            diff_El_costheta_temp[ieL, iTh] += jacobian_this * diff_at_cosphi_neg
            else:
                # ===== ORIGINAL VERSION =====
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

            # Compute diff_cosphi_needed_temp (use pre-computed valid indices)
            # Integrate over energy with proper bin widths
            energy_widths = np.zeros(len(energy))
            for ieL in range(len(energy)):
                if ieL == 0:
                    energy_widths[ieL] = energy[1] - energy[0] if len(energy) > 1 else 1.0
                elif ieL == len(energy) - 1:
                    energy_widths[ieL] = energy[ieL] - energy[ieL - 1]
                else:
                    energy_widths[ieL] = 0.5 * (energy[ieL + 1] - energy[ieL - 1])
            
            for idx, icosphi in enumerate(valid_iTh):
                cosphi_val = valid_cosphi[idx]
                jacobian = valid_jacobians[idx]
                
                for ieL in range(len(energy)):
                    diff_cosphi_needed_temp[icosphi] += (
                        jacobian * diff_El_costheta_lab(energy[ieL], cosphi_val, MH, EH) * energy_widths[ieL]
                    )
                    if distance_m < distance_SE:
                        diff_cosphi_needed_temp[icosphi] += (
                            jacobian * diff_El_costheta_lab(energy[ieL], -1.0 * cosphi_val, MH, EH) * energy_widths[ieL]
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
    
    return diff_El_decayed, diff_costheta_decayed, diff_cosphi_decayed, diff_El_costheta_decayed, diff_El_cosphi_decayed


@timer
def get_and_save_nuL_El_costheta_decay_in_flight(spectrum_L, U2, MH, savepath='./output/'):
    """Get decayed left-handed neutrino energy and angle distribution.

    Main wrapper function that:
    1. Divides decay path into steps (inside/outside Earth orbit)
    2. Calls getNulEAndAngleFromRHNDecay for each step
    3. Accumulates all contributions
    4. Saves 2D and 1D distributions to CSV files

    Parameters
    ----------
    spectrum_L : NDArray
        Solar neutrino spectrum
    U2 : float
        Mixing parameter squared
    MH : float
        RHN mass (MeV)
    savepath : str
        Output directory for CSV files

    Returns
    -------
    NDArray
        3D array (n_energy × n_angle × 3) with neutrino distribution
    """
    print("\n" + "=" * 70)
    print("STEP 1: Computing decayed left-handed neutrino flux")
    print("=" * 70)
    if HAS_NUMBA:
        print("✅ Numba JIT compilation enabled (5-10x speedup)")
    else:
        print("⚠️  Numba not available - using Python (slower)")
        print("    Install with: pip install numba")
    print("=" * 70 + "\n")
    
    energy = spectrum_L[:, 0]  # energy points

    npoints_costheta = 201
    costheta_step = 2.0 / (npoints_costheta - 1.0)
    costheta_arr = np.zeros(npoints_costheta)
    for i in range(npoints_costheta):
        costheta_arr[i] = -1.0 + i * costheta_step    

    spectrum_R = getRHNSpectrum(spectrum_L, MH, U2)
    flux_R = spectrum_R[:, 1]
    index_max = np.argmax(flux_R)
    E_max_flux = energy[index_max]

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
    diff_El_costheta_decayed_inside = np.zeros((len(energy), len(costheta_arr), 3))
    diff_El_costheta_decayed_outside = np.zeros((len(energy), len(costheta_arr), 3))
    
    for iTh in range(len(costheta_arr)):
        for ie in range(len(energy)):
            diff_El_costheta_decayed[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed[ie, iTh, 1] = costheta_arr[iTh]
            diff_El_costheta_decayed_inside[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed_inside[ie, iTh, 1] = costheta_arr[iTh]
            diff_El_costheta_decayed_outside[ie, iTh, 0] = energy[ie]
            diff_El_costheta_decayed_outside[ie, iTh, 1] = costheta_arr[iTh]
    
    for ie in range(len(energy)):
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
        diff_El_costheta_decayed_inside[:, :, 2] += diff_El_costheta_this[:, :, 2]

    # then, decay after flying outside earth orbit
    distance_start = distance_SE
    ratio_decayed = findRatioForDistanceSpectrum(MH, spectrum_R, U2, distance_start)
    distance_next = findDistanceForRatio(MH, E_max_flux, U2, ratio_orbit + (1 - ratio_orbit) / 10.0)

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
        ratio_decayed = findRatioForDistanceSpectrum(MH, spectrum_R, U2, distance_next)
        print(
            "decay outside earth orbit, distance = ",
            "%.2f" % (distance_start / distance_SE),
            " (SE), fraction decayed: ",
            "%.3f" % ratio_decayed,
            ", decayed flux = ",
            integrateSpectrum(diff_El_this),
        )

        distance_start = distance_next
        ratio_decayed2 = findRatioForDistance(MH, E_max_flux, U2, distance_next)
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
        diff_El_costheta_decayed_outside[:, :, 2] += diff_El_costheta_this[:, :, 2]
        
        for iTh in range(npoints_costheta):
            diff_costheta_decayed[iTh][1] += diff_costheta_this[iTh][1]
            diff_costheta_decayed_outside[iTh][1] += diff_costheta_this[iTh][1]
            diff_cosphi_decayed[iTh][1] += diff_cosphi_this[iTh][1]
            diff_cosphi_decayed_outside[iTh][1] += diff_cosphi_this[iTh][1]

    # Save the 2D distribution (energy vs costheta) to CSV for later inspection.
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

    # Also save the 1D energy and costheta distributions
    filename_el = f"diff_El_M{MH:.1f}_U{U2:.1e}.csv"
    out_el = os.path.join(savepath, filename_el)
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
def get_and_save_nuL_scatter_electron_El_costheta(diff_El_costheta_decayed, savepath='./output/', N_int_local=100000, use_parallel=False):
    """Get scattered electron spectrum with azimuthal angle sampling.
    
    Improved version that properly accounts for azimuthal angle φ when mapping
    scattering angles to lab frame: cos(θ_lab) = cos(θ_in)·cos(θ_s) - sin(θ_in)·sin(θ_s)·cos(φ)
    
    Parameters
    ----------
    diff_El_costheta_decayed : NDArray
        Neutrino 2D distribution (nE, nAngle, 3)
    savepath : str
        Output directory
    N_int_local : int
        Number of Monte Carlo samples for scattering
    use_parallel : bool
        Whether to use parallel processing for scatter computation (default: False)
    
    Returns
    -------
    tuple
        (final_spectrum, e_bins, costheta_lab_bins, spectra_per_angle)
    """
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
    
    if use_parallel:
        # Parallel processing
        import multiprocessing as mp
        from functools import partial
        
        def process_angle(ia, energy_src, energy_target, diff_El_costheta_decayed, costheta_nu, N_int_local):
            """Process single incoming angle"""
            flux_2d = diff_El_costheta_decayed[:, ia, 2]
            
            if np.all(flux_2d == 0):
                return ia, None, None, None
            
            # Resample
            def resample_bin_average(x_src, y_src, x_tgt_centers):
                x_src = np.asarray(x_src)
                y_src = np.asarray(y_src)
                x_tgt = np.asarray(x_tgt_centers)
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
            
            flux_target = resample_bin_average(energy_src, flux_2d, energy_target)
            
            try:
                s2d, s_e, s_a, e_b, a_b = scatter_electron_spectrum(
                    energy_target, flux_target, N_int_local=N_int_local
                )
                return ia, s2d, e_b, a_b
            except Exception as e:
                print(f"Warning: Angle {ia} failed: {e}")
                return ia, None, None, None
        
        # Run in parallel
        process_func = partial(process_angle, 
                              energy_src=energy_src,
                              energy_target=energy_target,
                              diff_El_costheta_decayed=diff_El_costheta_decayed,
                              costheta_nu=costheta_nu,
                              N_int_local=N_int_local)
        
        with mp.Pool(processes=mp.cpu_count() // 2) as pool:
            results = pool.map(process_func, range(n_in_angles))
        
        # Collect results
        for ia, s2d, e_b, a_b in results:
            spectra_list.append(s2d)
            if e_b is not None:
                e_bins = e_b
                a_bins = a_b
            print(f"  Angle {ia+1}/{n_in_angles}, cosθ={costheta_nu[ia]:.3f} - completed")
    else:
        # Sequential processing (original)
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
                s2d, s_e, s_a, e_bins, a_bins = scatter_electron_spectrum(
                    energy_target, flux_target, N_int_local=N_int_local
                )
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
    
    # Azimuthal angle sampling: increase resolution for smoother angle distribution
    # More samples → smoother distribution, less statistical fluctuation
    # Recommended: 4-10x the number of scatter angle bins
    n_phi = nA_scatter * 4  # 200 samples for φ ∈ [0, 2π]
    phi_samples = np.linspace(0, 2*np.pi, n_phi, endpoint=False)
    
    for ia in range(n_in_angles):
        if spectra_list[ia] is None or np.all(spectra_list[ia] == 0):
            continue
        
        cos_theta_in = costheta_nu[ia]
        sin_theta_in = np.sqrt(1 - cos_theta_in**2)
        
        print(f"  Processing incoming angle {ia+1}/{n_in_angles}, cosθ_in={cos_theta_in:.3f}")
        
        # Vectorize: pre-compute all scatter angles at once
        cos_theta_s = np.cos(theta_s_centers)  # shape: (nA_scatter,)
        sin_theta_s = np.sin(theta_s_centers)  # shape: (nA_scatter,)
        
        # Vectorize: compute all (scatter_angle, phi) combinations at once
        # Broadcasting: (nA_scatter, 1) and (n_phi,) -> (nA_scatter, n_phi)
        cos_phi = np.cos(phi_samples)  # shape: (n_phi,)
        
        # cos(θ_lab) for all (ia_s, phi) combinations
        # shape: (nA_scatter, n_phi)
        cos_theta_lab_all = (
            cos_theta_in * cos_theta_s[:, None] 
            - sin_theta_in * sin_theta_s[:, None] * cos_phi[None, :]
        )
        cos_theta_lab_all = np.clip(cos_theta_lab_all, -1.0, 1.0)
        
        # Find lab bins for all combinations
        # searchsorted is vectorized
        ilab_all = np.searchsorted(costheta_lab_bins, cos_theta_lab_all.ravel()) - 1
        ilab_all = np.clip(ilab_all, 0, nCostheta_lab - 1).reshape(nA_scatter, n_phi)
        
        # Distribute flux efficiently
        # For each (ia_s, phi) pair, add contribution to corresponding lab bin
        for ia_s in range(nA_scatter):
            for iphi in range(n_phi):
                ilab = ilab_all[ia_s, iphi]
                # Vectorized energy loop
                electron_2d_lab[:, ilab] += spectra_list[ia][:, ia_s] / n_phi
    
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
    """Compute electron spectrum from saved neutrino CSV file.
    
    Parameters
    ----------
    csv_path : str
        Path to neutrino 2D distribution CSV
    savepath : str, optional
        Output directory (default: same as input)
    N_int_local : int
        Monte Carlo samples for scattering
    plot : bool
        Whether to generate plots
    
    Returns
    -------
    tuple
        (electron_2d, e_bins, costheta_lab_bins, spectra_per_angle)
    """
    import re
    
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
    df = df.rename(columns={cols[0]: 'energy', cols[1]: 'costheta', cols[2]: 'value'})

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
        
        # Extract parameter info from filename if possible
        basename = os.path.basename(csv_path)
        title_suffix = ""
        if 'MH' in basename and 'U2' in basename:
            # Match patterns like U2_1.00e-01_MH_4.0
            mh_match = re.search(r'MH[_\s]+([\d.]+)', basename)
            u2_match = re.search(r'U2[_\s]+([\d.eE+-]+)', basename)
            if mh_match and u2_match:
                MH = float(mh_match.group(1))
                U2 = float(u2_match.group(1))
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


__all__ = [
    'getNulEAndAngleFromRHNDecay',
    'get_and_save_nuL_El_costheta_decay_in_flight',
    'get_and_save_nuL_scatter_electron_El_costheta',
    'get_and_save_nuL_scatter_electron_El_costheta_from_csv',
]
