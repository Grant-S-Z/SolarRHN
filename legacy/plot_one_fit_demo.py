"""Quick diagnostic: plot one profiled fit (background+signal) to one dataset.

This script is intentionally lightweight and does not modify the main scan.
It reuses helpers from `toymc_s1_borexino_profile.py`.

Outputs a PDF showing:
- the observed dataset (Asimov or one toy)
- the best-fit background-only (sig=0) constrained fit
- the best-fit (over scanned U2 grid) signal+background constrained fit

Usage example:

    python plot_one_fit_demo.py --mh 8 --data-mode poisson_bkg --n-toys 1

"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from toymc_s1_borexino_profile import (
    ProfileConfig,
    apply_energy_resolution_convolution,  # re-exported from core in that module
    build_signal_templates_for_mh,
    chi2_curve_for_dataset,
    draw_background_pseudodata,
    estimate_b8_norm_sigma_from_original_table,
    estimate_b8_binwise_sigma_from_original_table,
    load_background,
    poisson_nll,
    profiled_nll_fixed_template,
    interpolateSpectrum,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--mh", type=float, default=8.0)
    p.add_argument("--u2-min", type=float, default=1e-6)
    p.add_argument("--u2-max", type=float, default=1e-1)
    p.add_argument("--n-u2", type=int, default=30)

    p.add_argument("--estep", type=float, default=0.2)
    p.add_argument("--e-min", type=float, default=0.0)
    p.add_argument("--e-max", type=float, default=16.0)
    p.add_argument("--fit-min", type=float, default=4.8)
    p.add_argument("--fit-max", type=float, default=12.8)
    p.add_argument("--energy-resolution", type=float, default=0.05)

    p.add_argument(
        "--data-mode",
        choices=["asimov_bkg", "poisson_bkg", "borexino_data"],
        default="borexino_data",
    )
    p.add_argument(
        "--toy-bin-uncertainty",
        choices=["none", "independent_gauss"],
        default="independent_gauss",
    )

    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--sigma-table", default="data/originalTable.txt")
    p.add_argument("--sigma-b8-frac", type=float, default=0.1)

    p.add_argument("--out", default="plots/borexino/one_fit_demo.pdf")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Grids
    u2_values = np.logspace(np.log10(args.u2_min), np.log10(args.u2_max), args.n_u2)
    energy = np.arange(args.e_min, args.e_max, step=args.estep)

    # Background
    bg_centers, b_bin = load_background(energy=energy, estep=args.estep, energy_resolution=args.energy_resolution)
    fit_mask = (bg_centers >= args.fit_min) & (bg_centers <= args.fit_max)
    b_fit = b_bin[fit_mask]

    bkg_bin_sigma_fit = None
    if args.toy_bin_uncertainty == "independent_gauss":
        bkg_bin_sigma = estimate_b8_binwise_sigma_from_original_table(args.sigma_table, bg_centers)
        bkg_bin_sigma_fit = bkg_bin_sigma[fit_mask]

    if args.sigma_b8_frac is None:
        sigma_b8_frac = float(estimate_b8_norm_sigma_from_original_table(args.sigma_table))
    else:
        sigma_b8_frac = float(args.sigma_b8_frac)

    # Dataset
    rng = np.random.default_rng(args.seed)

    if args.data_mode == "asimov_bkg":
        data = b_fit.copy()
    else:
        data = draw_background_pseudodata(
            rng=rng,
            bkg_counts=b_fit,
            bin_sigma_frac=bkg_bin_sigma_fit,
            toy_bin_uncertainty=args.toy_bin_uncertainty,
        )

    # Signal templates
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    signal_templates_fit, _ = build_signal_templates_for_mh(
        spectrum_nuL_orig=spectrum_nuL_orig,
        mh=args.mh,
        u2_values=u2_values,
        energy=energy,
        estep=args.estep,
        fit_mask=fit_mask,
        energy_resolution=args.energy_resolution,
    )

    # Best scanned point (signal+background)
    chi2_vals, nll_cond, nll_hat, ihat, xb8_hat = chi2_curve_for_dataset(
        data=data,
        bkg=b_fit,
        signal_templates=signal_templates_fit,
        sigma_b8_frac=sigma_b8_frac,
    )
    u2_hat = float(u2_values[ihat])
    sig_hat = signal_templates_fit[ihat]
    mu_hat = xb8_hat * b_fit + sig_hat

    # Background-only constrained fit (sig=0)
    nll0, xb80 = profiled_nll_fixed_template(data=data, bkg=b_fit, sig=np.zeros_like(b_fit), sigma_b8_frac=sigma_b8_frac)
    mu0 = xb80 * b_fit

    # Plot in fit window
    x = bg_centers[fit_mask]
    binw = args.estep

    plt.figure(figsize=(8.2, 5.0))
    plt.step(x, data, where="mid", label=f"data ({args.data_mode})", color="k")
    plt.plot(x, mu0, label=fr"bkg-only fit: $\hat X_{{B8}}={xb80:.3f}$", lw=2.0)
    plt.plot(x, mu_hat, label=fr"best S+B (scan): $U^2={u2_hat:.2e}$, $\hat{{\hat X}}_{{B8}}={xb8_hat:.3f}$", lw=2.0)
    plt.plot(x, sig_hat, label="signal component at best point", lw=1.5, alpha=0.8)

    plt.xlabel("E (MeV)")
    plt.ylabel(f"counts / {binw:.2f} MeV bin")
    plt.title(fr"One profiled fit demo @ $m_H={args.mh:.2f}$ MeV")
    plt.grid(True, alpha=0.25)
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(args.out, dpi=250)
    plt.close()

    print(f"Saved: {args.out}")
    print(f"sigma_b8_frac={sigma_b8_frac:.3g}")
    print(f"bkg-only: xb8={xb80:.4f}, NLL={nll0:.4g}")
    print(f"best S+B: u2={u2_hat:.4e}, xb8={xb8_hat:.4f}, NLL={nll_hat:.4g}")


if __name__ == "__main__":
    main()
