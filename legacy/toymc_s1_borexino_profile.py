"""Borexino-style exclusion with profiled likelihood and toy-MC calibration.

This script is intentionally standalone and does NOT modify existing S1 workflows.
It reuses the same physics/spectrum machinery, but changes the statistics layer:

1) For each fixed MH, build signal templates over a U2 grid.
2) Use Poisson binned likelihood with one nuisance parameter X_B8 (background norm),
   constrained by Gaussian prior around 1.0.
3) Build profile-likelihood test statistic chi2(U2_j) for observed/Asimov data.
4) Generate toys under H0 (background-only), and compute chi2_crit(U2_j) as the CL quantile.
5) Define exclusion where chi2_data(U2_j) >= chi2_crit(U2_j), and extract crossing(s).

Notes
-----
- This is a practical implementation for expected-exclusion studies.
- Parameter of interest is scanned on the provided U2 grid (discrete profile scan).
- For speed, default grid/toy settings are moderate; increase for final results.
"""

from __future__ import annotations

import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import uproot as ur
from scipy.optimize import minimize_scalar
from tqdm import tqdm

from core import (
    apply_energy_resolution_convolution,
    exposure,
    getRHNSpectrum,
    integrateSpectrum,
    interpolateSpectrum,
)
from workflows import getNuleeInDetector


@dataclass
class ProfileConfig:
    sigma_b8_frac: float = 0.1
    cl: float = 0.90
    n_toys: int = 300
    seed: int = 12345


def _load_original_table_numeric(path: str) -> np.ndarray:
    """Load numeric rows from originalTable-like file with 4 columns.

    Returns an array with columns [energy, best, plus_3sigma, minus_3sigma].
    """
    arr = np.genfromtxt(path, delimiter=",", comments="#", dtype=float)

    if arr.ndim == 1 and arr.size == 0:
        raise ValueError(f"No numeric rows found in {path}")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    arr = arr[:, :4]
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if arr.shape[0] == 0:
        raise ValueError(f"No valid numeric rows found in {path}")

    return arr


def estimate_b8_norm_sigma_from_original_table(path: str) -> float:
    """Estimate global fractional B8 normalization uncertainty from ``originalTable.txt``.

    The input table is expected to contain columns:
    energy, best, plus_3sigma, minus_3sigma.

    We estimate a single normalization nuisance width by integrating each spectrum
    and converting the +3σ/-3σ envelope to an effective 1σ fractional uncertainty:

        sigma_frac = 0.5 * [ (I_plus - I_best) / I_best + (I_best - I_minus) / I_best ] / 3

    Parameters
    ----------
    path : str
        Path to ``originalTable.txt``.

    Returns
    -------
    float
        Estimated 1σ fractional uncertainty for X_B8 normalization.

    Raises
    ------
    ValueError
        If the file has no valid numeric rows or the best-spectrum integral is
        non-positive.
    """
    arr = _load_original_table_numeric(path)

    e = arr[:, 0]
    best = arr[:, 1]
    plus3 = arr[:, 2]
    minus3 = arr[:, 3]

    i_best = float(np.trapezoid(best, e))
    i_plus = float(np.trapezoid(plus3, e))
    i_minus = float(np.trapezoid(minus3, e))

    if i_best <= 0:
        raise ValueError("Best-spectrum integral must be positive")

    frac_plus_3sigma = (i_plus - i_best) / i_best
    frac_minus_3sigma = (i_best - i_minus) / i_best
    sigma_frac = 0.5 * (frac_plus_3sigma + frac_minus_3sigma) / 3.0

    return float(max(sigma_frac, 1e-6))


def estimate_b8_binwise_sigma_from_original_table(
    path: str, target_centers: np.ndarray
) -> np.ndarray:
    """Estimate per-bin fractional B8 uncertainty on analysis energy centers.

    The table provides best and ±3σ spectra versus energy. We convert to an
    effective symmetric 1σ fractional uncertainty per source-energy point,
    then interpolate to ``target_centers``:

        sigma_frac(E) = 0.5 * [ (plus3-best)/best + (best-minus3)/best ] / 3

    Parameters
    ----------
    path : str
        Path to ``originalTable.txt``.
    target_centers : numpy.ndarray
        Analysis bin centers where fractional uncertainties are needed.

    Returns
    -------
    numpy.ndarray
        Per-bin 1σ fractional uncertainties aligned to ``target_centers``.
    """
    arr = _load_original_table_numeric(path)

    e = arr[:, 0]
    best = arr[:, 1]
    plus3 = arr[:, 2]
    minus3 = arr[:, 3]

    sigma_plus = np.clip((plus3 - best) / 3.0, 0.0, None)
    sigma_minus = np.clip((best - minus3) / 3.0, 0.0, None)
    sigma_sym = 0.5 * (sigma_plus + sigma_minus)

    frac = np.zeros_like(best)
    pos = best > 0
    frac[pos] = sigma_sym[pos] / best[pos]

    frac_interp = np.interp(target_centers, e, frac, left=0.0, right=0.0)
    frac_interp = np.nan_to_num(frac_interp, nan=0.0, posinf=0.0, neginf=0.0)
    frac_interp = np.clip(frac_interp, 0.0, None)
    return frac_interp


def draw_background_pseudodata(
    rng: np.random.Generator,
    bkg_counts: np.ndarray,
    bin_sigma_frac: Optional[np.ndarray],
    toy_bin_uncertainty: str,
) -> np.ndarray:
    """Draw background pseudo-data with optional per-bin Gaussian systematics.

    If ``toy_bin_uncertainty == 'independent_gauss'``, each bin expectation is
    fluctuated as ``mu_i = B_i * (1 + z_i * sigma_i)`` with ``z_i ~ N(0, 1)``,
    then Poisson sampled. For ``'none'``, this reduces to standard
    ``Poisson(B_i)`` toy generation.
    """
    mu = np.clip(bkg_counts, 0.0, None).astype(float)

    if toy_bin_uncertainty == "independent_gauss":
        if bin_sigma_frac is None:
            raise ValueError("bin_sigma_frac is required for independent_gauss mode")
        if bin_sigma_frac.shape != mu.shape:
            raise ValueError("bin_sigma_frac shape must match bkg_counts shape")

        z = rng.normal(loc=0.0, scale=1.0, size=mu.shape)
        mu = mu * (1.0 + z * np.clip(bin_sigma_frac, 0.0, None))

    elif toy_bin_uncertainty != "none":
        raise ValueError(f"Unsupported toy_bin_uncertainty mode: {toy_bin_uncertainty}")

    mu = np.clip(mu, 0.0, None)
    return rng.poisson(mu).astype(float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Borexino-style profile-likelihood exclusion scan on S1 signal templates."
    )

    parser.add_argument("--u2-min", type=float, default=1e-6)
    parser.add_argument("--u2-max", type=float, default=1e-1)
    parser.add_argument("--n-u2", type=int, default=6)

    parser.add_argument("--mh-min", type=float, default=2.0)
    parser.add_argument("--mh-max", type=float, default=14.0)
    parser.add_argument("--n-mh", type=int, default=7)

    parser.add_argument("--estep", type=float, default=0.2)
    parser.add_argument("--e-min", type=float, default=0.0)
    parser.add_argument("--e-max", type=float, default=16.0)
    parser.add_argument("--fit-min", type=float, default=4.8)  # fit window
    parser.add_argument("--fit-max", type=float, default=12.8)
    parser.add_argument("--energy-resolution", type=float, default=0.05)

    # parser.add_argument("--sigma-b8-frac", type=float, default=None,
    #                     help="Gaussian prior width for B8 normalization X_B8 around 1.0; if omitted, auto-estimate from originalTable.txt")
    parser.add_argument(
        "--sigma-b8-frac",
        type=float,
        default=0.1,
        help="Gaussian prior width for B8 normalization X_B8",
    )
    parser.add_argument(
        "--sigma-table",
        default="data/originalTable.txt",
        help="Path to originalTable.txt used for auto-estimating sigma_b8_frac",
    )
    parser.add_argument("--cl", type=float, default=0.90)
    parser.add_argument("--n-toys", type=int, default=300)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Number of parallel workers over MH points",
    )

    parser.add_argument(
        "--data-mode",
        choices=["asimov_bkg", "poisson_bkg"],
        default="asimov_bkg",
        help="Observed data used for chi2_data: Asimov bkg or one Poisson bkg toy",
    )
    parser.add_argument(
        "--toy-bin-uncertainty",
        choices=["none", "independent_gauss"],
        default="independent_gauss",
        help="How to include per-bin uncertainty in background pseudo-data generation",
    )

    parser.add_argument("--out-prefix", default="borexino_profile")
    parser.add_argument("--out-dir", default="output/borexino")
    parser.add_argument("--plot-dir", default="plots/borexino/exclusion")

    return parser.parse_args()


def poisson_nll(data: np.ndarray, mu: np.ndarray) -> float:
    """Poisson negative log-likelihood up to additive constant.

    nll = sum(mu - n*log(mu)); term log(n!) omitted (constant in fit comparisons).
    """
    mu_safe = np.clip(mu, 1e-12, None)
    return float(np.sum(mu_safe - data * np.log(mu_safe)))


def profiled_nll_fixed_template(
    data: np.ndarray,
    bkg: np.ndarray,
    sig: np.ndarray,
    sigma_b8_frac: float,
) -> Tuple[float, float]:
    """Profile Poisson NLL over ``X_B8`` for one fixed signal template.

    For a fixed signal hypothesis ``sig`` (i.e. fixed ``U2`` and ``mH``), this
    function minimizes the constrained objective with respect to ``X_B8 >= 0``.

    Parameters
    ----------
    data : numpy.ndarray
        Observed binned counts ``n_i`` in the fit window.
        Shape: ``(n_fit_bins,)``.
    bkg : numpy.ndarray
        Background template counts ``B_i`` in the same bins as ``data``.
        Shape: ``(n_fit_bins,)``.
    sig : numpy.ndarray
        Fixed signal template counts ``S_i`` in the same bins as ``data``.
        Shape: ``(n_fit_bins,)``.
    sigma_b8_frac : float
        1-sigma fractional width of the Gaussian prior on ``X_B8`` centered
        at 1.0. Must be strictly positive.

    Returns
    -------
    nll_min : float
        Profiled minimum value of the constrained NLL at this fixed signal
        template.
    xb8_hat_hat : float
        Conditional best-fit value that minimizes the constrained NLL.

    Notes
    -----
    - Optimization is performed with ``scipy.optimize.minimize_scalar``
      (bounded method) on an adaptive interval ``[0, xb8_upper]``.
    - ``xb8_upper`` is heuristic but data-adaptive to keep the fit stable
      across low/high-count regimes.
    - The Poisson constant term ``log(n_i!)`` is omitted because it cancels in
      likelihood-ratio differences.
    """

    if sigma_b8_frac <= 0:
        raise ValueError("sigma_b8_frac must be > 0")

    # Heuristic bound around nominal X_B8=1. Broad enough for practical scans.
    # Upper bound also adapts to total observed counts.
    bsum = max(np.sum(bkg), 1e-9)
    dsum = max(np.sum(data), 0.0)
    xb8_upper = max(5.0, 3.0 * dsum / bsum + 2.0)

    def objective(x_b8: float) -> float:
        mu = x_b8 * bkg + sig
        nll = poisson_nll(data, mu)
        # penalty = 0.5 * ((x_b8 - 1.0) / sigma_b8_frac) ** 2
        penalty = 0
        return nll + penalty

    res = minimize_scalar(objective, bounds=(0.0, xb8_upper), method="bounded")
    if not res.success:
        raise RuntimeError(f"X_B8 profile fit failed: {res.message}")

    return float(res.fun), float(res.x)


def chi2_curve_for_dataset(
    data: np.ndarray,
    bkg: np.ndarray,
    signal_templates: np.ndarray,
    sigma_b8_frac: float,
) -> Tuple[np.ndarray, np.ndarray, float, int, float]:
    """Compute profile-likelihood curve over scanned templates.

    Parameters
    ----------
    data : numpy.ndarray
        Observed binned counts in fit bins. Shape: ``(n_fit_bins,)``.
    bkg : numpy.ndarray
        Background template in fit bins. Shape: ``(n_fit_bins,)``.
    signal_templates : numpy.ndarray
        Signal template bank with shape ``(n_u2, n_fit_bins)``.
        Row ``j`` corresponds to one scanned ``U2_j``.
    sigma_b8_frac : float
        1-sigma fractional prior width used for profiling ``X_B8``.

    Returns
    -------
    chi2_vals : numpy.ndarray ``(n_u2,)``
    nll_cond : numpy.ndarray ``(n_u2,)``.
        Conditional profiled NLL values for each scanned template.
    nll_hat : float
        Global minimum NLL across the scanned templates.
    ihat : int
        Index of the global best-fit template (best-fit scanned ``U2`` point).
    xb8_hat : float
        Best-fit ``X_B8`` at the global best-fit template.

    Notes
    -----
    - This is a discrete scan in ``U2`` (template index), not a continuous
      optimizer over ``U2``.

    """
    n_u2 = signal_templates.shape[0]
    nll_cond = np.zeros(n_u2, dtype=float)
    xb8_cond = np.zeros(n_u2, dtype=float)

    for j in range(n_u2):
        nll_j, xb8_j = profiled_nll_fixed_template(
            data=data,
            bkg=bkg,
            sig=signal_templates[j],
            sigma_b8_frac=sigma_b8_frac,
        )
        nll_cond[j] = nll_j
        xb8_cond[j] = xb8_j

    ihat = int(np.argmin(nll_cond))
    nll_hat = float(nll_cond[ihat])
    chi2_vals = 2.0 * (nll_cond - nll_hat)
    chi2_vals = np.clip(chi2_vals, 0.0, None)
    return chi2_vals, nll_cond, nll_hat, ihat, float(xb8_cond[ihat])


def build_signal_templates_for_mh(
    spectrum_nuL_orig: np.ndarray,
    mh: float,
    u2_values: np.ndarray,
    energy: np.ndarray,
    estep: float,
    fit_mask: np.ndarray,
    energy_resolution: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build RHN signal template library for one fixed ``m_H``.

    Parameters
    ----------
    spectrum_nuL_orig : numpy.ndarray
        Input solar neutrino spectrum used as RHN production source.
        Expected shape is ``(n_energy_bins, 2)`` where column 0 is energy
        (MeV) and column 1 is flux density.
    mh : float
        Fixed RHN mass (MeV) for this template-building pass.
    u2_values : numpy.ndarray
        1D scan grid of mixing values :math:`U^2`.
        One signal template is built for each entry.
    energy : numpy.ndarray
        1D energy-bin lower edges used by the analysis grid. The function
        expects signal output bins to align with ``energy + 0.5*estep``.
    estep : float
        Energy bin width in MeV.
    fit_mask : numpy.ndarray
        Boolean mask selecting the subset of energy bins used in the
        likelihood fit window (e.g. 4.8--12.8 MeV).
    energy_resolution : float
        Fractional Gaussian energy resolution used in detector smearing,
        i.e. ``sigma(E) = energy_resolution * E``.

    Returns
    -------
    signal_templates_fit : numpy.ndarray
        2D array with shape ``(n_u2, n_fit_bins)``.
        Row ``j`` is the smeared signal template
        :math:`S_i(U^2_j, m_H)` restricted to fit bins.
    signal_totals : numpy.ndarray
        1D array with shape ``(n_u2,)`` containing total smeared signal
        counts summed over the full energy range before fit-window masking.

    """
    n_u2 = u2_values.size
    signal_fit = []
    signal_totals = np.zeros(n_u2, dtype=float)

    for iu2, u2 in enumerate(u2_values):
        spectrum_rhn = getRHNSpectrum(spectrum_nuL_orig, mh, u2)
        _, _, _, diff_eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mh, u2)

        s_bin = np.nan_to_num(
            diff_eee_decayed[:, 1] * exposure, nan=0.0, posinf=0.0, neginf=0.0
        )
        s_bin = np.clip(s_bin, 0.0, None)

        signal_centers = diff_eee_decayed[:, 0] + 0.5 * estep
        expected_centers = energy + 0.5 * estep
        if signal_centers.shape != expected_centers.shape or not np.allclose(
            signal_centers, expected_centers, atol=1e-12
        ):
            raise ValueError("Signal and expected energy bins are not aligned")

        s_bin = apply_energy_resolution_convolution(
            s_bin,
            signal_centers,
            frac_resolution=energy_resolution,
        )

        signal_fit.append(s_bin[fit_mask])
        signal_totals[iu2] = np.sum(s_bin)

    return np.asarray(signal_fit, dtype=float), signal_totals


def load_background(
    energy: np.ndarray,
    estep: float,
    energy_resolution: float | None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load and smear Borexino-like ES background spectrum from ROOT file.

    Parameters
    ----------
    energy : np.ndarray
    estep : float
    energy_resolution : float, optional
        If energy_resolution = None, do no smearing.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Background

    """
    f_bg = ur.open("data/solar_borexino.root")
    h_bg = f_bg["he_es"]
    bg_values = np.asarray(h_bg.values(), dtype=float)
    bg_edges = np.asarray(h_bg.axis().edges(), dtype=float)
    f_bg.close()

    bg_bin_width_src = bg_edges[1] - bg_edges[0]
    bg_centers_src = 0.5 * (bg_edges[:-1] + bg_edges[1:])
    bg_per_mev_src = bg_values / bg_bin_width_src # MeV

    bg_centers = energy + 0.5 * estep
    bg_per_mev = np.interp(
        bg_centers, bg_centers_src, bg_per_mev_src, left=0.0, right=0.0
    )
    b_bin = bg_per_mev * estep # counts / 0.2 MeV (bin width)

    if energy_resolution is not None:
        b_bin = apply_energy_resolution_convolution(
            b_bin,
            bg_centers,
            frac_resolution=energy_resolution,
        )
    return bg_centers, b_bin


def find_crossings_logx(x: np.ndarray, y: np.ndarray) -> List[float]:
    """Find roots of y(x)=0 with linear interpolation in log10(x)."""
    roots: List[float] = []
    lx = np.log10(x)

    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        if y1 == 0:
            roots.append(float(x[i]))
            continue
        if y1 * y2 > 0:
            continue

        # y changes sign (or y2==0): interpolate on (logx, y)
        if y2 == y1:
            roots.append(float(x[i]))
            continue
        t = -y1 / (y2 - y1)
        lx0 = lx[i] + t * (lx[i + 1] - lx[i])
        roots.append(float(10**lx0))

    return roots


def _process_single_mh(task: Tuple[Any, ...]) -> Dict[str, Any]:
    """Worker for one MH point (parallel over MH)."""
    (
        imh,
        mh,
        u2_values,
        spectrum_nuL_orig,
        energy,
        estep,
        fit_mask,
        energy_resolution,
        b_fit,
        data_mode,
        sigma_b8_frac,
        cl,
        n_toys,
        seed,
        toy_bin_uncertainty,
        bkg_bin_sigma_fit,
    ) = task

    rng = np.random.default_rng(seed + 1009 * imh)

    signal_templates_fit, signal_totals = build_signal_templates_for_mh(
        spectrum_nuL_orig=spectrum_nuL_orig,
        mh=mh,
        u2_values=u2_values,
        energy=energy,
        estep=estep,
        fit_mask=fit_mask,
        energy_resolution=energy_resolution,
    )

    if data_mode == "asimov_bkg":
        data = b_fit.copy()
    elif data_mode == "poisson_bkg":
        data = draw_background_pseudodata(
            rng=rng,
            bkg_counts=b_fit,
            bin_sigma_frac=bkg_bin_sigma_fit,
            toy_bin_uncertainty=toy_bin_uncertainty,
        )
    else:
        raise ValueError(f"Unsupported data mode: {data_mode}")

    chi2_data, _, _, ihat_data, xb8_data = chi2_curve_for_dataset(
        data=data,
        bkg=b_fit,
        signal_templates=signal_templates_fit,
        sigma_b8_frac=sigma_b8_frac,
    )

    toy_chi2 = np.zeros((n_toys, u2_values.size), dtype=float)
    for itoy in range(n_toys):
        toy_data = draw_background_pseudodata(
            rng=rng,
            bkg_counts=b_fit,
            bin_sigma_frac=bkg_bin_sigma_fit,
            toy_bin_uncertainty=toy_bin_uncertainty,
        )
        chi2_toy, _, _, _, _ = chi2_curve_for_dataset(
            data=toy_data,
            bkg=b_fit,
            signal_templates=signal_templates_fit,
            sigma_b8_frac=sigma_b8_frac,
        )
        toy_chi2[itoy, :] = chi2_toy

    chi2_crit = np.quantile(toy_chi2, cl, axis=0)
    excluded = chi2_data >= chi2_crit

    delta = chi2_data - chi2_crit
    crossings = find_crossings_logx(u2_values, delta)

    ul_right = np.nan
    if excluded[0]:
        ul_right = float(u2_values[0])
    elif np.any(excluded):
        idx_first = int(np.argmax(excluded))
        if idx_first > 0:
            if len(crossings) > 0:
                ul_right = float(min(crossings))
            else:
                ul_right = float(u2_values[idx_first])
        else:
            ul_right = float(u2_values[idx_first])

    return {
        "imh": imh,
        "mh": float(mh),
        "signal_totals": signal_totals,
        "chi2_data": chi2_data,
        "chi2_crit": chi2_crit,
        "excluded": excluded.astype(int),
        "xb8_hat_data": float(xb8_data),
        "best_u2_data": float(u2_values[ihat_data]),
        "crossings": crossings,
        "ul_right": ul_right,
    }


def main() -> None:
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.plot_dir, exist_ok=True)
    os.makedirs(args.plot_dir + "/chi2_borexino_profile", exist_ok=True)

    u2_values = np.logspace(np.log10(args.u2_min), np.log10(args.u2_max), args.n_u2)
    mh_values = np.linspace(args.mh_min, args.mh_max, args.n_mh)

    energy = np.arange(args.e_min, args.e_max, step=args.estep)
    bg_centers, b_bin = load_background(
        energy=energy,
        estep=args.estep,
        energy_resolution=args.energy_resolution,
    )
    fit_mask = (bg_centers >= args.fit_min) & (bg_centers <= args.fit_max)
    b_fit = b_bin[fit_mask]  # background for fitting

    bkg_bin_sigma_fit: Optional[np.ndarray] = None
    if args.toy_bin_uncertainty == "independent_gauss":
        bkg_bin_sigma = estimate_b8_binwise_sigma_from_original_table(
            args.sigma_table, bg_centers
        )
        bkg_bin_sigma_fit = bkg_bin_sigma[fit_mask]

    # Decide B8 prior sigma: CLI override or automatic estimate from originalTable.
    sigma_b8_frac_value: float
    sigma_source: str
    if args.sigma_b8_frac is None:
        sigma_b8_frac_value = estimate_b8_norm_sigma_from_original_table(
            args.sigma_table
        )
        sigma_source = f"auto from {args.sigma_table}"
    else:
        sigma_b8_frac_value = float(args.sigma_b8_frac)
        sigma_source = "from --sigma-b8-frac"

    print("=" * 72)
    print("BOREXINO-STYLE PROFILE LIKELIHOOD EXCLUSION (S1)")
    print("=" * 72)
    print(f"U2 grid ({len(u2_values)}): {u2_values}")
    print(f"MH grid ({len(mh_values)}): {mh_values}")
    print(f"Energy window used in likelihood: [{args.fit_min}, {args.fit_max}] MeV")
    print(f"B8 prior sigma fraction: {sigma_b8_frac_value:.6g} ({sigma_source})")
    print(f"Toy MC per MH: {args.n_toys}")
    print(f"CL quantile for chi2_crit: {args.cl}")
    print(f"Toy bin uncertainty mode: {args.toy_bin_uncertainty}")
    print(f"Parallel workers over MH: {args.max_workers}")
    print()

    print(">>> Loading 8B neutrino spectrum from CSV...")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    print(f"Integrated B8 flux = {integrateSpectrum(spectrum_nuL_orig):.6e} cm^-2 s^-1")
    print()

    cfg = ProfileConfig(
        sigma_b8_frac=sigma_b8_frac_value,
        cl=args.cl,
        n_toys=args.n_toys,
        seed=args.seed,
    )

    chi2_data_grid = np.full((args.n_mh, args.n_u2), np.nan)
    chi2_crit_grid = np.full((args.n_mh, args.n_u2), np.nan)
    excluded_grid = np.zeros((args.n_mh, args.n_u2), dtype=int)
    signal_total_grid = np.full((args.n_mh, args.n_u2), np.nan)
    xb8_hat_data = np.full(args.n_mh, np.nan)
    best_u2_data = np.full(args.n_mh, np.nan)

    ul_right = np.full(args.n_mh, np.nan)
    all_crossings: List[List[float]] = []

    tasks = [
        (
            imh,
            mh,
            u2_values,
            spectrum_nuL_orig,
            energy,
            args.estep,
            fit_mask,
            args.energy_resolution,
            b_fit,
            args.data_mode,
            cfg.sigma_b8_frac,
            cfg.cl,
            cfg.n_toys,
            cfg.seed,
            args.toy_bin_uncertainty,
            bkg_bin_sigma_fit,
        )
        for imh, mh in enumerate(mh_values)
    ]

    results: Dict[int, Dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(_process_single_mh, task) for task in tasks]
        for fut in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Scan MH (parallel)",
            unit="MH",
        ):
            res = fut.result()
            results[int(res["imh"])] = res

    for imh, mh in enumerate(mh_values):
        res = results[imh]
        signal_total_grid[imh, :] = res["signal_totals"]
        chi2_data_grid[imh, :] = res["chi2_data"]
        chi2_crit_grid[imh, :] = res["chi2_crit"]
        excluded_grid[imh, :] = res["excluded"]
        xb8_hat_data[imh] = res["xb8_hat_data"]
        best_u2_data[imh] = res["best_u2_data"]
        ul_right[imh] = res["ul_right"]
        all_crossings.append(res["crossings"])

        # Per-MH diagnostic plot: chi2_data vs chi2_crit.
        plt.figure(figsize=(7.5, 5.2))
        plt.plot(u2_values, res["chi2_data"], "o-", label="chi2_data")
        plt.plot(
            u2_values,
            res["chi2_crit"],
            "s--",
            label=f"chi2_crit ({int(100 * cfg.cl)}% toys)",
        )
        plt.xscale("log")
        plt.xlabel(r"$U^2$")
        plt.ylabel(r"$\chi^2$ statistic")
        plt.title(rf"$m_H={mh:.2f}$ MeV")
        plt.grid(True, which="both", alpha=0.3)
        plt.legend()
        out_chi2plot = os.path.join(
            args.plot_dir + "/chi2_borexino_profile",
            f"chi2curve_mh_{mh:.2f}_{args.out_prefix}.pdf",
        )
        plt.tight_layout()
        plt.savefig(out_chi2plot, dpi=250)
        plt.close()

    # Final exclusion plot.
    plt.figure(figsize=(8.0, 5.8))
    for imh, mh in enumerate(mh_values):
        excl_mask = excluded_grid[imh].astype(bool)
        if np.any(excl_mask):
            plt.scatter(
                np.full(np.sum(excl_mask), mh), u2_values[excl_mask], s=24, alpha=0.55
            )

    valid = np.isfinite(ul_right)
    if np.any(valid):
        plt.plot(
            mh_values[valid], ul_right[valid], "k-", lw=2.2, label="UL (right branch)"
        )

    plt.yscale("log")
    plt.xlabel(r"$m_H$ (MeV)")
    plt.ylabel(r"$U^2$")
    plt.title(
        f"Borexino-style exclusion ({int(100 * cfg.cl)}% CL), data={args.data_mode}"
    )
    plt.grid(True, which="both", alpha=0.3)
    if np.any(valid):
        plt.legend()
    plt.tight_layout()
    excl_plot_path = os.path.join(args.plot_dir, f"exclusion_{args.out_prefix}.pdf")
    plt.savefig(excl_plot_path, dpi=250)
    plt.close()

    out_npz = os.path.join(args.out_dir, f"{args.out_prefix}.npz")
    np.savez(
        out_npz,
        u2_values=u2_values,
        mh_values=mh_values,
        chi2_data_grid=chi2_data_grid,
        chi2_crit_grid=chi2_crit_grid,
        excluded_grid=excluded_grid,
        signal_total_grid=signal_total_grid,
        xb8_hat_data=xb8_hat_data,
        best_u2_data=best_u2_data,
        ul_right=ul_right,
        fit_mask=fit_mask.astype(int),
        fit_min=args.fit_min,
        fit_max=args.fit_max,
        sigma_b8_frac=cfg.sigma_b8_frac,
        cl=cfg.cl,
        n_toys=cfg.n_toys,
        data_mode=args.data_mode,
        toy_bin_uncertainty=args.toy_bin_uncertainty,
    )

    # Save crossings in a readable text file.
    out_txt = os.path.join(args.out_dir, f"{args.out_prefix}_crossings.txt")
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write("# MH(MeV) crossings_in_U2\n")
        for mh, roots in zip(mh_values, all_crossings):
            if len(roots) == 0:
                f.write(f"{mh:.6f} none\n")
            else:
                roots_str = " ".join(f"{r:.8e}" for r in roots)
                f.write(f"{mh:.6f} {roots_str}\n")

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Saved NPZ: {out_npz}")
    print(f"Saved crossings: {out_txt}")
    print(f"Saved exclusion plot: {excl_plot_path}")
    print("Suggested quick run:")
    print(
        "python toymc_s1_borexino_profile.py --n-u2 7 --n-mh 5 --n-toys 120 "
        "--data-mode asimov_bkg --out-prefix borexino_quick"
    )


if __name__ == "__main__":
    main()
