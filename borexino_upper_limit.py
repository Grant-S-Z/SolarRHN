"""
Expected (Asimov) upper limit calculation using Cowan et al. 2011 asymptotic formulae.

Computes the median expected exclusion sensitivity and ±1σ/±2σ error bands
under the background-only hypothesis (μ' = 0).

Reference: G. Cowan, K. Cranmer, E. Gross, O. Vitells,
"Asymptotic formulae for likelihood-based tests of new physics"
Eur. Phys. J. C 71 (2011) 1554, arXiv:1007.1727

Method:
  1. Construct Asimov dataset under background-only hypothesis (all observations
     set to their expected background values, with nuisance params profiled at μ=0).
  2. Run the profile-likelihood scan on Asimov data → median expected limit.
  3. Estimate σ_xh from the Δχ² curve: σ_xh = xh_limit / sqrt(2.71).
  4. Compute ±1σ / ±2σ bands in x_h space using Gaussian approximation,
     then convert to u2 space via u2 = u2_ref * sqrt(x_h).

Usage:
  python expected_limit.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# ── Reuse existing code ──────────────────────────────────────────────────────
from reproduce_borexino_fit import (
    load_borexino_data,
    fit_fixed_xh,
    get_signal_template,
    profile_likelihood_scan_u2,
    find_u2_crossings,
    plot_exclusion_2d_u2,
    energy,
    fit_mask,
    estep,
    n_workers,
)
from core.spectrum_utils import interpolateSpectrum

# ── Constants ────────────────────────────────────────────────────────────────
OUTDIR = "./plots/borexino/exclusion/expected"
os.makedirs(OUTDIR, exist_ok=True)

# CL threshold for 90% CL upper limit (1 dof, chi2 quantile)
CL_THRESHOLD = 2.71
# Corresponding one-sided Gaussian significance
Z_ALPHA = np.sqrt(CL_THRESHOLD)  # ≈ 1.645


def construct_asimov_data(
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    sig_ref: np.ndarray,
) -> np.ndarray:
    """Construct the Asimov dataset under background-only hypothesis (μ'=0).

    The Asimov data is the expectation under the background-only model with
    nuisance parameters profiled at the conditional MLE for μ=0.

    Parameters
    ----------
    data : ndarray
        Real experimental data (only used for background-only fit)
    bkg_b8, bkg_be11 : ndarray
        Background templates
    sig_ref : ndarray
        Signal template (not used, but needed for fit_fixed_xh)

    Returns
    -------
    data_asimov : ndarray
        Asimov dataset: background-only expectation
    bestfit_bkg : dict
        The fit result from background-only fit (contains x_b8, x_be11 etc.)
    """
    print(">>> Fitting background-only model to real data...")
    result_bkg = fit_fixed_xh(data, bkg_b8, bkg_be11, sig_ref, x_h=0.0)

    # Asimov data = best-fit background (no Poisson fluctuations)
    data_asimov = (
        result_bkg["x_b8"] * bkg_b8
        + result_bkg["x_be11"] * bkg_be11
    )
    print(f"    x_b8 = {result_bkg['x_b8']:.4f}, x_be11 = {result_bkg['x_be11']:.4f}")
    print(f"    Background-only NLL = {result_bkg['nll']:.4f}")

    return data_asimov, result_bkg


def estimate_sigma_from_curve(
    xh_array: np.ndarray,
    delta_chi2_array: np.ndarray,
) -> tuple[float, float, float]:
    """Estimate σ (std dev of x_h estimator) from the Δχ² profile.

    Uses the Wald approximation: Δχ²(x_h) ≈ (x_h - xh_hat)² / σ_xh².
    For the Asimov data under μ'=0, xh_hat ≈ 0, so Δχ²(x_h) ≈ x_h² / σ_xh².

    We estimate σ_xh from the region where the curve is approximately parabolic,
    using all points with Δχ² < 10 for a robust fit.

    Parameters
    ----------
    xh_array : ndarray
        x_h values scanned
    delta_chi2_array : ndarray
        Corresponding Δχ² values

    Returns
    -------
    sigma : float
        Estimated standard deviation of x̂_h under background-only
    xh_limit : float
        x_h value where Δχ² crosses CL_THRESHOLD (the median expected limit)
    xh_hat : float
        Best-fit x_h (minimum of Δχ²)
    """
    xh = np.asarray(xh_array, dtype=float)
    dchi2 = np.asarray(delta_chi2_array, dtype=float)

    # Sort by xh
    order = np.argsort(xh)
    xh = xh[order]
    dchi2 = dchi2[order]

    # Find minimum (best fit)
    i_min = int(np.argmin(dchi2))
    xh_hat = float(xh[i_min])
    dchi2 = dchi2 - dchi2[i_min]  # ensure minimum is at 0

    # Find where Δχ² crosses threshold (upper limit in x_h)
    crossings = find_u2_crossings(xh, dchi2, threshold=CL_THRESHOLD)
    xh_limit = float(crossings[-1]) if crossings else np.nan

    if not np.isfinite(xh_limit) or xh_limit <= 0:
        print("    WARNING: no crossing found for Δχ² threshold!")
        return np.nan, np.nan, xh_hat

    # Estimate σ from the crossing: Δχ²(xh_limit) ≈ xh_limit² / σ²
    sigma = xh_limit / Z_ALPHA

    # Also do a parabolic fit for robustness
    mask = (dchi2 > 0) & (dchi2 < 10) & (xh > 0)
    if np.sum(mask) >= 3:
        xh_fit = xh[mask]
        dchi2_fit = dchi2[mask]
        # Fit Δχ² = a * xh²  (no linear term since minimum at 0)
        a = np.sum(xh_fit**4) / np.sum(xh_fit**2 * dchi2_fit)  # weighted
        # Actually simpler: a = mean(dchi2_fit / xh_fit²)
        ratios = dchi2_fit / (xh_fit**2)
        a = np.mean(ratios)
        sigma_fit = 1.0 / np.sqrt(a) if a > 0 else sigma
        print(f"    Parabolic fit: a={a:.4f}, sigma_fit={sigma_fit:.4e}")
        sigma = sigma_fit

    print(f"    xh_hat = {xh_hat:.4e}, xh_limit = {xh_limit:.4e}, sigma = {sigma:.4e}")

    return float(sigma), float(xh_limit), float(xh_hat)


def compute_expected_limit_with_bands(
    u2_limits_med: np.ndarray,
    sigma_xh_array: np.ndarray,
    xh_limit_array: np.ndarray,
    u2_ref: float,
) -> dict:
    """Compute ±1σ and ±2σ bands for the expected limit.

    Under the background-only hypothesis:
      - med[xh_up | 0] = σ * Φ^{-1}(1-α)  (which we have from the scan)
      - xh_up ~ N(med, σ) approximately
      - Bands: med ± N×σ (clipped at 0)

    Convert to u2 space: u2 = u2_ref * sqrt(xh)

    Parameters
    ----------
    u2_limits_med : ndarray
        Median expected limit in u2 space (from Asimov scan)
    sigma_xh_array : ndarray
        σ of x̂_h for each mH
    xh_limit_array : ndarray
        x_h limit for each mH
    u2_ref : float
        Reference u² for signal template normalization

    Returns
    -------
    bands : dict
        {'u2_minus2', 'u2_minus1', 'u2_med', 'u2_plus1', 'u2_plus2'}
    """
    n = len(u2_limits_med)
    u2_m2 = np.full(n, np.nan, dtype=float)
    u2_m1 = np.full(n, np.nan, dtype=float)
    u2_med = np.full(n, np.nan, dtype=float)
    u2_p1 = np.full(n, np.nan, dtype=float)
    u2_p2 = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if not np.isfinite(xh_limit_array[i]) or xh_limit_array[i] <= 0:
            continue
        if not np.isfinite(sigma_xh_array[i]) or sigma_xh_array[i] <= 0:
            continue

        xh_lim = float(xh_limit_array[i])
        sig = float(sigma_xh_array[i])

        # Bands in x_h space (Gaussian approximation, clipped at 0)
        xh_m2 = max(0.0, xh_lim - 2.0 * sig)
        xh_m1 = max(0.0, xh_lim - 1.0 * sig)
        xh_p1 = xh_lim + 1.0 * sig
        xh_p2 = xh_lim + 2.0 * sig

        # Convert to u2
        u2_med[i] = u2_limits_med[i]  # already from scan
        u2_m1[i] = u2_ref * np.sqrt(xh_m1)
        u2_m2[i] = u2_ref * np.sqrt(xh_m2)
        u2_p1[i] = u2_ref * np.sqrt(xh_p1)
        u2_p2[i] = u2_ref * np.sqrt(xh_p2)

    return {
        "u2_minus2": u2_m2,
        "u2_minus1": u2_m1,
        "u2_med": u2_med,
        "u2_plus1": u2_p1,
        "u2_plus2": u2_p2,
    }


def plot_expected_exclusion(
    mh_values: np.ndarray,
    observed_u2: np.ndarray | None,
    bands: dict,
    outpath: str,
    borexino_ref_csv: str = "./data/Borexino_exclusion.csv",
):
    """Plot expected exclusion limit with ±1σ and ±2σ bands."""
    mh = np.asarray(mh_values, dtype=float)
    u2_med = bands["u2_med"]
    u2_p1 = bands["u2_plus1"]
    u2_m1 = bands["u2_minus1"]
    u2_p2 = bands["u2_plus2"]
    u2_m2 = bands["u2_minus2"]

    valid = np.isfinite(u2_med) & (u2_med > 0.0)

    plt.figure(figsize=(8, 6))

    # ±2σ band (yellow)
    if np.any(valid):
        plt.fill_between(
            mh[valid],
            u2_m2[valid],
            u2_p2[valid],
            color="yellow",
            alpha=0.3,
            label=r"$\pm 2\sigma$ expected",
        )

    # ±1σ band (green)
    if np.any(valid):
        plt.fill_between(
            mh[valid],
            u2_m1[valid],
            u2_p1[valid],
            color="limegreen",
            alpha=0.3,
            label=r"$\pm 1\sigma$ expected",
        )

    # Median expected limit
    if np.any(valid):
        plt.plot(
            mh[valid],
            u2_med[valid],
            "--",
            lw=2.5,
            color="tab:blue",
            label="Expected (Asimov)",
        )

    # Observed limit (overlay)
    if observed_u2 is not None:
        obs_valid = np.isfinite(observed_u2) & (observed_u2 > 0.0)
        if np.any(obs_valid):
            plt.plot(
                mh[obs_valid],
                observed_u2[obs_valid],
                "o-",
                lw=2,
                ms=4,
                color="tab:red",
                label="Observed (90% C.L.)",
            )
            plt.fill_between(
                mh[obs_valid],
                observed_u2[obs_valid],
                np.max(observed_u2[obs_valid]) * 1.5,
                color="tab:red",
                alpha=0.12,
            )

    # Overlay Borexino published exclusion (for reference)
    if os.path.exists(borexino_ref_csv):
        ref = np.loadtxt(borexino_ref_csv, delimiter=",")
        if ref.ndim == 2 and ref.shape[1] >= 2:
            log10_mh_gev = ref[:, 0]
            log10_u2 = ref[:, 1]

            i_min = int(np.argmin(log10_u2))
            branches = [slice(0, i_min + 1), slice(i_min, None)]

            first = True
            for sl in branches:
                bx = log10_mh_gev[sl]
                by = log10_u2[sl]
                if bx.size < 2:
                    continue
                mh_mev = (10.0 ** bx) * 1e3
                u2_pub = 10.0 ** by
                plt.plot(
                    mh_mev,
                    u2_pub,
                    "-",
                    lw=2.0,
                    color="gray",
                    label="Borexino (published)" if first else None,
                )
                first = False

    plt.yscale("log")
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("Expected exclusion sensitivity (Asimov, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved expected exclusion plot to {outpath}")


def save_expected_csv(
    mh_values: np.ndarray,
    bands: dict,
    outpath: str,
):
    """Save expected limit bands to CSV."""
    header = (
        "mH,"
        "u2_minus2sigma,u2_minus1sigma,"
        "u2_expected,"
        "u2_plus1sigma,u2_plus2sigma"
    )
    np.savetxt(
        outpath,
        np.column_stack([
            mh_values,
            bands["u2_minus2"],
            bands["u2_minus1"],
            bands["u2_med"],
            bands["u2_plus1"],
            bands["u2_plus2"],
        ]),
        delimiter=",",
        header=header,
        fmt="%.8e",
    )
    print(f"Saved expected limit CSV to {outpath}")


def main():
    print("=" * 60)
    print("Expected (Asimov) upper limit computation")
    print("=" * 60)

    # ── 1. Load data and templates ──────────────────────────────────────────
    print("\n>>> Loading Borexino data and templates...")
    exp_data, fit_energy, bkg_b8_full, bkg_be11_full, _ = load_borexino_data()

    data = np.asarray(exp_data[:, 1], dtype=float)
    bkg_b8 = np.asarray(bkg_b8_full[fit_mask], dtype=float)
    bkg_be11 = np.asarray(bkg_be11_full[fit_mask], dtype=float)

    # ── 2. Load solar neutrino spectrum ────────────────────────────────────
    print(">>> Loading 8B neutrino spectrum...")
    from reproduce_borexino_fit import energy as e_grid
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", e_grid)

    # ── 3. Parameters ──────────────────────────────────────────────────────
    u2_ref = 1e-5
    mh_array = np.linspace(2.0, 14.0, 13)
    u2_array = np.logspace(-6, -3, 31)
    if not np.any(np.isclose(u2_array, 0.0)):
        u2_array = np.insert(u2_array, 0, 0.0)

    # ── 4. Observed limit (for comparison, cache from existing scan) ──────
    #    Compute it if the cached file doesn't exist.
    observed_csv = "./plots/borexino/exclusion/exclusion_2d_mh_u2_pyhf_boundary.csv"
    observed_u2 = None
    if os.path.exists(observed_csv):
        obs_data = np.loadtxt(observed_csv, delimiter=",", skiprows=1)
        if obs_data.ndim == 2 and obs_data.shape[1] >= 2:
            # Match to our mh_array
            obs_mh = obs_data[:, 0]
            obs_u2_raw = obs_data[:, 1]
            observed_u2 = np.full(len(mh_array), np.nan, dtype=float)
            for i, mh in enumerate(mh_array):
                idx = np.argmin(np.abs(obs_mh - mh))
                if np.abs(obs_mh[idx] - mh) < 0.01 * mh:
                    observed_u2[i] = obs_u2_raw[idx]
        print(f">>> Loaded observed limit from {observed_csv}")
    else:
        print(f">>> No observed limit found at {observed_csv}, "
              "will compute observed limit too.")
        from reproduce_borexino_fit import run_mh_scan_exclusion
        print(">>> Computing observed exclusion...")
        mh_vals_obs, observed_u2, obs_rows = run_mh_scan_exclusion(
            spectrum_orig=spectrum_nuL_orig,
            data=data,
            bkg_b8=bkg_b8,
            bkg_be11=bkg_be11,
            mh_array=mh_array,
            u2_array=u2_array,
            u2_ref=u2_ref,
            cl_threshold=CL_THRESHOLD,
            n_workers=n_workers,
        )
        # Save
        np.savetxt(
            observed_csv,
            np.column_stack([mh_array, observed_u2]),
            delimiter=",",
            header="mH,u2_90",
            fmt="%.8e",
        )

    # ── 5. Construct Asimov dataset ────────────────────────────────────────
    # Need a signal template for the background-only fit (even though x_h=0)
    sig_ref_m8 = get_signal_template(spectrum_nuL_orig, mH=8.0, u2=u2_ref)
    data_asimov, bkg_fit_result = construct_asimov_data(
        data, bkg_b8, bkg_be11, sig_ref_m8
    )

    # ── 6. Scan over mH on Asimov data ────────────────────────────────────
    print(f"\n>>> Scanning {len(mh_array)} mass points on Asimov data...")
    from reproduce_borexino_fit import run_mh_scan_exclusion

    sigma_xh_array = np.full(len(mh_array), np.nan, dtype=float)
    xh_limit_array = np.full(len(mh_array), np.nan, dtype=float)
    mh_vals, u2_limits_exp, rows_exp = run_mh_scan_exclusion(
        spectrum_orig=spectrum_nuL_orig,
        data=data_asimov,
        bkg_b8=bkg_b8,
        bkg_be11=bkg_be11,
        mh_array=mh_array,
        u2_array=u2_array,
        u2_ref=u2_ref,
        cl_threshold=CL_THRESHOLD,
        n_workers=n_workers,
    )

    # ── 7. For each mH, estimate σ_xh from the profile curve ─────────────
    print("\n>>> Estimating σ_xh from profile likelihood curves...")
    from collections import defaultdict

    rows_by_mh = defaultdict(list)
    for r in rows_exp:
        rows_by_mh[r["mH"]].append(r)

    for i, mH_val in enumerate(mh_array):
        # Skip if scan found no limit at this mH
        if not np.isfinite(u2_limits_exp[i]) or u2_limits_exp[i] <= 0:
            print(f"    mH={mH_val:.1f}: no limit from scan, skipping σ estimation.")
            continue

        mh_rounded = float(np.round(mH_val, 3))
        # Find the matching rows
        candidates = rows_by_mh.get(mH_val, [])
        if not candidates:
            candidates = rows_by_mh.get(mh_rounded, [])
        if not candidates:
            continue

        xh_vals = np.array([r["x_h"] for r in candidates], dtype=float)
        dchi2_vals = np.array([r["delta_chi2"] for r in candidates], dtype=float)

        # Δχ² is already computed relative to the minimum by profile_likelihood_scan_u2.
        # Ensure numerical minimum is exactly at 0 (tiny floating-point adjustment only).
        dchi2_min = np.min(dchi2_vals)
        if dchi2_min > 1e-10:
            dchi2_vals = dchi2_vals - dchi2_min

        sigma, xh_lim, xh_hat = estimate_sigma_from_curve(xh_vals, dchi2_vals)
        sigma_xh_array[i] = sigma
        xh_limit_array[i] = xh_lim

    # ── 8. Compute ±1σ and ±2σ bands ────────────────────────────────────
    print("\n>>> Computing expected limit bands...")
    bands = compute_expected_limit_with_bands(
        u2_limits_med=u2_limits_exp,
        sigma_xh_array=sigma_xh_array,
        xh_limit_array=xh_limit_array,
        u2_ref=u2_ref,
    )

    # ── 9. Plot ──────────────────────────────────────────────────────────
    plot_expected_exclusion(
        mh_values=mh_array,
        observed_u2=observed_u2,
        bands=bands,
        outpath=os.path.join(OUTDIR, "exclusion_expected_vs_observed.pdf"),
    )

    # Additionally plot observed-only for comparison
    plot_exclusion_2d_u2(
        mh_array,
        u2_limits_exp,
        outpath=os.path.join(OUTDIR, "exclusion_expected_only.pdf"),
    )

    # ── 10. Save CSV ────────────────────────────────────────────────────
    save_expected_csv(
        mh_array,
        bands,
        outpath=os.path.join(OUTDIR, "exclusion_expected_bands.csv"),
    )

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Expected 90% CL exclusion limits")
    print("=" * 60)
    print(f"{'mH (MeV)':<12} {'u2_exp':<14} {'u2_-1σ':<14} {'u2_+1σ':<14} {'u2_-2σ':<14} {'u2_+2σ':<14}")
    print("-" * 70)
    for i, mh in enumerate(mh_array):
        if np.isfinite(bands["u2_med"][i]) and bands["u2_med"][i] > 0:
            print(f"{mh:<12.2f} "
                  f"{bands['u2_med'][i]:<14.4e} "
                  f"{bands['u2_minus1'][i]:<14.4e} "
                  f"{bands['u2_plus1'][i]:<14.4e} "
                  f"{bands['u2_minus2'][i]:<14.4e} "
                  f"{bands['u2_plus2'][i]:<14.4e}")

    print(f"\nAll outputs saved to {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
