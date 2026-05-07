import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

from core import (
    exposure,
    getRHNSpectrum,
    interpolateSpectrum,
    integrateSpectrum,
)
from workflows import getNuleeInDetector


# ==========================================================================
# HELPERS
# ==========================================================================

def parse_u2_colname(col: str) -> float:
    """Parse 'U2=1.00e-06' → 1e-6."""
    return float(col.split("=")[1])


def compute_signal_count(args):
    """Compute total e+e- pair signal count for a single (U2, MH) point.

    Returns
    -------
    tuple
        (iu2, imh, MH, U2, total_counts)
    """
    iu2, imh, MH, U2, spectrum_nuL_orig = args

    spectrum_RHN = getRHNSpectrum(spectrum_nuL_orig, MH, U2) # / (MeV cm^2 s)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_RHN, MH, U2)
    estep = diff_Eee_decayed[1, 0] - diff_Eee_decayed[0, 0]
    S_bin = np.nan_to_num(
        diff_Eee_decayed[:, 1] * exposure, nan=0.0, posinf=0.0, neginf=0.0
    )
    S_bin = np.clip(S_bin, 0.0, None)
    total_counts = np.sum(S_bin) * estep

    return iu2, imh, MH, U2, total_counts


# ==========================================================================
# MAIN
# ==========================================================================

def main():
    start_time = time.time()

    # ======================================================================
    # READ CUT FRACTION CSV TO DETERMINE GRID
    # ======================================================================

    cut_path = "data/mg5_boost_cut_fraction_smear.csv"
    print(f">>> Reading cut fraction file: {cut_path}")
    df_cut = pd.read_csv(cut_path)

    MH_col = "mass_mev"
    MH_values = df_cut[MH_col].values  # (66,)
    u2_cols = [c for c in df_cut.columns if c.startswith("U2=")]
    U2_values = np.array([parse_u2_colname(c) for c in u2_cols])  # (51,)

    # Cut fraction grid (percentage → fraction)
    cut_grid = df_cut[u2_cols].values  # shape (66, 51), MH rows × U2 columns
    cut_grid = cut_grid / 100.0  # percentage → fraction

    nU2 = len(U2_values)
    nMh = len(MH_values)

    print(f"    MH grid:  {MH_values[0]:.1f} – {MH_values[-1]:.1f} MeV, {nMh} points")
    print(f"    U2 grid:  {U2_values[0]:.1e} – {U2_values[-1]:.1e}, {nU2} points")
    print(f"    Total (MH, U2) points: {nU2 * nMh}")
    print()

    # ======================================================================
    # ENERGY GRID FOR SIGNAL COMPUTATION
    # ======================================================================

    estep = 0.2
    energy = np.arange(0.0, 16.0, step=estep)
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    print(
        ">>> 8B neutrino flux (integrated): "
        f"{integrateSpectrum(spectrum_nuL_orig):.4e} cm⁻² s⁻¹"
    )
    print()

    # ======================================================================
    # COMPUTE RAW SIGNAL COUNTS (PARALLEL)
    # ======================================================================

    compute_tasks = []
    for iu2, U2 in enumerate(U2_values):
        for imh, MH in enumerate(MH_values):
            compute_tasks.append((iu2, imh, MH, U2, spectrum_nuL_orig))

    total_points = len(compute_tasks)
    print(f">>> Computing raw e+e- signal counts ({total_points} points)...")

    count_grid = np.full((nU2, nMh), np.nan)   # U2 rows × MH columns

    max_workers = min(8, os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(
            tqdm(
                executor.map(compute_signal_count, compute_tasks),
                total=total_points,
                desc="Signal counts",
                unit="pt",
            )
        )

    for iu2, imh, MH, U2, total_counts in results:
        count_grid[iu2, imh] = total_counts

    print("    Done.")
    print()

    # ======================================================================
    # SAVE RAW SIGNAL COUNTS CSV
    # ======================================================================

    # count_grid shape: (nU2, nMh) — we want CSV rows = MH, columns = U2
    raw_cols = [f"U2={U2:.6e}" for U2 in U2_values]
    df_raw = pd.DataFrame(count_grid.T, columns=raw_cols)  # shape (nMh, nU2)
    df_raw.insert(0, "mass_mev", MH_values)

    raw_csv = "data/eepair_signal_count_raw.csv"
    df_raw.to_csv(raw_csv, index=False, float_format="%.10e")
    print(f">>> Saved raw signal counts: {raw_csv}")
    print()

    # ======================================================================
    # APPLY CUT FRACTION → SAVE CUT-CORRECTED COUNTS CSV
    # ======================================================================

    # cut_grid:          (66, 51)   MH rows × U2 columns
    # count_grid:        (51, 66)   U2 rows × MH columns
    # We need to align:  cut_grid[mh, iu2] * count_grid[iu2, mh]

    corrected_grid = count_grid * cut_grid.T  # (51, 66) × (51, 66) = (51, 66)

    df_corr = pd.DataFrame(corrected_grid.T, columns=raw_cols)
    df_corr.insert(0, "mass_mev", MH_values)

    corr_csv = "data/eepair_signal_count_cut.csv"
    df_corr.to_csv(corr_csv, index=False, float_format="%.10e")
    print(f">>> Saved cut-corrected signal counts: {corr_csv}")
    print()

    # ======================================================================
    # PLOT: CUT-CORRECTED 2D DISTRIBUTION (log10 color)
    # ======================================================================

    outdir = "plots/ee_pair/"
    os.makedirs(outdir, exist_ok=True)

    # Prepare plotting grid (mask zeros)
    plot_grid = np.where(corrected_grid > 0, corrected_grid, np.nan)

    fig, ax = plt.subplots(figsize=(11, 7))

    # pcolormesh with log norm
    pcm = ax.pcolormesh(
        MH_values,
        U2_values,
        plot_grid,
        shading="auto",
        norm="log",
        cmap="inferno",
    )

    cbar = fig.colorbar(
        pcm, ax=ax, label=r"Expected e$^+$e$^-$ signal counts (after cuts)"
    )

    ax.set_xlabel(r"$M_H$ [MeV]")
    ax.set_ylabel(r"$|U_{eH}|^2$")
    ax.set_yscale("log")
    ax.set_title(
        "Solar RHN → e⁺e⁻ in detector (after MG5 boost cuts)"
    )

    # Contour lines
    valid = plot_grid[np.isfinite(plot_grid)]
    if len(valid) > 1:
        log_min = np.floor(np.log10(valid.min()))
        log_max = np.ceil(np.log10(valid.max()))
        if log_max > log_min:
            levels = np.logspace(log_min, log_max, 10)
            cs = ax.contour(
                MH_values,
                U2_values,
                plot_grid,
                levels=levels,
                colors="white",
                linewidths=0.6,
                alpha=0.5,
            )
            ax.clabel(cs, inline=True, fontsize=8, fmt="%.1e")

    ax.set_xlim(MH_values[0], MH_values[-1])
    ax.set_ylim(U2_values[0], U2_values[-1])

    fig.tight_layout()

    plot_pdf = os.path.join(outdir, "eepair_signal_count_cut_2D.pdf")
    fig.savefig(plot_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f">>> Saved plot: {plot_pdf}")

    # ======================================================================
    # ALSO SAVE RAW 2D PLOT FOR COMPARISON
    # ======================================================================

    raw_plot_grid = np.where(count_grid > 0, count_grid, np.nan)

    fig2, ax2 = plt.subplots(figsize=(11, 7))

    pcm2 = ax2.pcolormesh(
        MH_values,
        U2_values,
        raw_plot_grid,
        shading="auto",
        norm="log",
        cmap="inferno",
    )

    cbar2 = fig2.colorbar(
        pcm2, ax=ax2, label=r"Expected e$^+$e$^-$ signal counts (raw)"
    )

    ax2.set_xlabel(r"$M_H$ [MeV]")
    ax2.set_ylabel(r"$|U_{eH}|^2$")
    ax2.set_yscale("log")
    ax2.set_title("Solar RHN → e⁺e⁻ in detector (raw, before cuts)")

    valid2 = raw_plot_grid[np.isfinite(raw_plot_grid)]
    if len(valid2) > 1:
        log_min2 = np.floor(np.log10(valid2.min()))
        log_max2 = np.ceil(np.log10(valid2.max()))
        if log_max2 > log_min2:
            levels2 = np.logspace(log_min2, log_max2, 10)
            cs2 = ax2.contour(
                MH_values,
                U2_values,
                raw_plot_grid,
                levels=levels2,
                colors="white",
                linewidths=0.6,
                alpha=0.5,
            )
            ax2.clabel(cs2, inline=True, fontsize=8, fmt="%.1e")

    ax2.set_xlim(MH_values[0], MH_values[-1])
    ax2.set_ylim(U2_values[0], U2_values[-1])

    fig2.tight_layout()

    raw_plot_pdf = os.path.join(outdir, "s1_eepair_signal_count_raw_2D.pdf")
    fig2.savefig(raw_plot_pdf, dpi=200, bbox_inches="tight")
    plt.close(fig2)
    print(f">>> Saved plot: {raw_plot_pdf}")

    # ======================================================================
    # TIMING
    # ======================================================================

    end_time = time.time()
    elapsed = end_time - start_time
    hours, remainder = divmod(elapsed, 3600)
    minutes, seconds = divmod(remainder, 60)
    print()
    print(f"{'=' * 60}")
    print(
        f"Total runtime: {int(hours):d}h {int(minutes):d}m {seconds:.1f}s ({elapsed:.1f}s)"
    )
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
