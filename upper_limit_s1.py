"""
Expected exclusion sensitivity — scan μ (signal strength) per (mH, u2).

Approach:
  1. Asimov data = nominal B8 background (Solar.root, no fluctuations)
  2. For each (mH, u2): generate signal → scan μ → fit xb only
  3. Δχ²(μ) = 2·(NLL(μ) − NLL_min)
  4. μ_up = μ where Δχ² = 2.71
  5. μ_up < 1 → excluded at this (mH, u2)
  6. For each mH: find u2 where μ_up crosses 1 → u2_90

Usage:
  python upper_limit_new.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot as ur
import pyhf
import matplotlib.pyplot as plt
from tqdm import tqdm

from borexino_data_exclusion import (
    get_signal_template,
    energy,
    fit_mask,
    estep,
)
from core import *
from core.stats import apply_energy_resolution_convolution

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
Z_ALPHA = np.sqrt(CL_THRESHOLD)

MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 141

U2_MIN = 1e-6
U2_MAX = 1e-1
N_U2 = 101

N_WORKERS = 10

if boolBorexino:
    OUTDIR = "./plots/borexino/upper_limit_new"
else:
    OUTDIR = "./plots/upper_limit_new"
os.makedirs(OUTDIR, exist_ok=True)


# ── Background ───────────────────────────────────────────────────────────────
def load_b8_background():
    if boolBorexino:
        f = ur.open("data/solar_borexino.root")
    else:
        f = ur.open("data/Solar.root")
    h = f["he_es"]
    bg_values = np.asarray(h.values(), dtype=float)
    bg_edges = np.asarray(h.axis().edges(), dtype=float)
    f.close()
    src_width = bg_edges[1] - bg_edges[0]
    src_centers = 0.5 * (bg_edges[:-1] + bg_edges[1:])
    src_per_mev = bg_values / src_width
    our_centers = energy + 0.5 * estep
    b_per_mev = np.interp(our_centers, src_centers, src_per_mev, left=0.0, right=0.0)
    return b_per_mev * estep


# ── pyhf fit (xb only, signal fixed) ────────────────────────────────────────
def fit_cond(data, bkg, signal):
    """Fit xb with signal fixed.  Returns (xb, NLL)."""
    bkg_safe = np.clip(bkg, 1e-12, None)
    sig_safe = np.clip(signal, 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "borexino",
                "samples": [
                    {"name": "signal", "data": sig_safe.tolist(), "modifiers": []},
                    {
                        "name": "bkg",
                        "data": bkg_safe.tolist(),
                        "modifiers": [
                            {"name": "xb", "type": "normfactor", "data": None}
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "borexino", "data": data.tolist()}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "xb",
                    "parameters": [
                        {"name": "xb", "inits": [1.0], "bounds": [[0.3, 3.0]]},
                    ],
                },
            }
        ],
    }
    ws = pyhf.Workspace(spec)
    model = ws.model()
    data_full = ws.data(model)
    bestfit, twice_nll = pyhf.infer.mle.fit(data_full, model, return_fitted_val=True)
    xb = float(bestfit[model.config.par_order.index("xb")])
    nll = 0.5 * float(twice_nll)
    return xb, nll


# ── μ=1 test for one (mH, u2) ───────────────────────────────────────────────
def test_one_point(spectrum_orig, data, bkg, nll_0, mH, u2):
    """Check if nominal signal at (mH, u2) is excluded at 90% CL.

    nll_0 should be the NLL at μ=0 (pre-computed once).
    Returns True if excluded (Δχ² > 2.71).
    """
    signal_nominal = get_signal_template(spectrum_orig, float(mH), float(u2))
    if signal_nominal.sum() < 1e-12:
        return False
    _, nll_1 = fit_cond(data, bkg, signal_nominal)
    dchi2 = 2.0 * max(0.0, nll_1 - nll_0)
    return bool(dchi2 > CL_THRESHOLD)


# ── One mH: test μ=1 at each u2, find excluded region boundary ─────────────
def analyze_one_mh(spectrum_orig, data, bkg, nll_0, mH, u2_arr):
    """For one mH: test μ=1 at each u2, find excluded region boundary.

    Always returns {"u2_low", "u2_high", "mH"}.
    """
    excluded = []
    for u2 in u2_arr:
        excluded.append(test_one_point(spectrum_orig, data, bkg, nll_0, mH, u2))

    excluded = np.array(excluded, dtype=bool)
    u2_arr = np.asarray(u2_arr)

    # Find transitions: True=excluded, diffs = +1 (enter), -1 (leave)
    diffs = np.diff(np.concatenate(([False], excluded, [False])).astype(int))
    enter = np.where(diffs == 1)[0]
    leave = np.where(diffs == -1)[0]

    if len(enter) == 0:
        return {"u2_low": U2_MAX, "u2_high": U2_MAX, "mH": float(mH)}

    i0 = enter[0] - 1 if enter[0] > 0 else 0  # last NOT-excluded before enter
    i1 = leave[0] - 1  # last excluded index

    # Interpolate lower crossing
    if i0 < len(u2_arr) - 1 and excluded[i0+1]:
        low = 0.5 * (u2_arr[i0] + u2_arr[i0+1])
    else:
        low = u2_arr[0]

    # Interpolate upper crossing
    if i1 < len(u2_arr) - 1:
        high = 0.5 * (u2_arr[i1] + u2_arr[i1+1])
    else:
        high = U2_MAX

    return {"u2_low": float(low), "u2_high": float(high), "mH": float(mH)}


# ── Worker & parallel ───────────────────────────────────────────────────────
def _worker(args):
    spectrum_orig, data, bkg, nll_0, mH, u2_arr = args
    return analyze_one_mh(spectrum_orig, data, bkg, nll_0, float(mH), u2_arr)

def scan_parallel(spectrum_orig, data, bkg, nll_0, mh_arr, u2_arr, *, label="scan"):
    tasks = [(spectrum_orig, data, bkg, nll_0, float(mH), u2_arr) for mH in mh_arr]
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, t): t[4] for t in tasks}
        with tqdm(total=len(tasks), desc=label) as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results[r["mH"]] = r
                pbar.update(1)
    return [results.get(float(mH)) for mH in mh_arr]


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_result(mh, results, outpath, *, show_bands=True):
    """Clean plot: boundary curve + optional ±1σ band, no fill."""
    n = len(mh)
    mh_arr = np.asarray(mh)
    y_top = U2_MAX * 10

    u2_low = np.array([r["u2_low"] for r in results])
    u2_high = np.array([r["u2_high"] for r in results])

    plt.figure(figsize=(8, 6))

    # ±1σ band (optional), symmetric around lower boundary
    if show_bands:
        band_low = np.full(n, np.nan)
        band_high = np.full(n, np.nan)
        for i in range(n):
            if u2_low[i] >= U2_MAX:
                continue
            s = u2_low[i] / (2.0 * Z_ALPHA)   # signal ∝ u2² → n=2
            band_low[i] = max(0.0, u2_low[i] - s)
            band_high[i] = u2_low[i] + s
        m = np.isfinite(band_low) & np.isfinite(band_high)
        if np.any(m):
            plt.fill_between(mh_arr, band_low, band_high, where=m,
                             color="limegreen", alpha=0.25,
                             label=r"$\pm 1\sigma$ expected")

    # Boundary: u2_low line
    boundary = np.where(u2_low < U2_MAX, u2_low, np.nan)
    plt.plot(mh_arr, boundary, "-", lw=2.2, color="tab:blue", label="Expected limit")

    # Upper edge for window points
    upper_mask = (u2_high < U2_MAX) & (u2_low < y_top)
    if np.any(upper_mask):
        plt.plot(mh_arr[upper_mask], u2_high[upper_mask], "-", lw=2.2, color="tab:blue")

    # Borexino published
    ref_csv = "./data/Borexino_exclusion.csv"
    if os.path.exists(ref_csv):
        ref = np.loadtxt(ref_csv, delimiter=",")
        if ref.ndim == 2 and ref.shape[1] >= 2:
            log10_mh, log10_u2 = ref[:, 0], ref[:, 1]
            i_min = int(np.argmin(log10_u2))
            for first, sl in enumerate([slice(0, i_min + 1), slice(i_min, None)]):
                bx, by = log10_mh[sl], log10_u2[sl]
                if bx.size < 2:
                    continue
                plt.plot(
                    (10**bx) * 1e3,
                    10**by,
                    "-",
                    lw=2.0,
                    color="gray",
                    label="Borexino (published)" if first == 0 else None,
                )

    plt.yscale("log")
    plt.xlim(MH_MIN, MH_MAX)
    plt.ylim(U2_MIN, y_top)
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("Expected exclusion (Asimov, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Expected exclusion — μ=1 test per (mH, u2)")
    print("=" * 60)

    print("\n>>> Loading B8 background ...")
    bkg_full = load_b8_background()
    bkg_full = apply_energy_resolution_convolution(
        bkg_full, energy + 0.5 * estep, frac_resolution=0.05,
    )
    bkg = bkg_full[fit_mask]
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    data_asimov = bkg.copy()
    print(f"    bins: {len(bkg)}, counts: {bkg.sum():.2f}")

    # Compute NLL at μ=0 once (same for all mH, u2)
    print("\n>>> Computing NLL(μ=0) ...")
    sig_zero = np.zeros(len(bkg))
    _, nll_0 = fit_cond(data_asimov, bkg, sig_zero)
    print(f"    NLL(μ=0) = {nll_0:.6f}")

    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    print(
        f"\n>>> Scan: {len(mh_arr)} mH × {len(u2_arr)} u2"
        f" on {N_WORKERS} workers"
    )
    results = scan_parallel(
        spectrum_orig, data_asimov, bkg, nll_0, mh_arr, u2_arr, label="expected"
    )

    plot_result(
        mh_arr,
        results,
        outpath=os.path.join(OUTDIR, "exclusion_upper_limit.pdf"),
        show_bands=True,
    )

    csv_path = os.path.join(OUTDIR, "upper_limit_bands.csv")
    with open(csv_path, "w") as f:
        f.write("mH,u2_low,u2_high\n")
        for i, r in enumerate(results):
            f.write(f"{mh_arr[i]:.6f},{r['u2_low']:.4e},{r['u2_high']:.4e}\n")
    print(f"Saved: {csv_path}")

    print("\n" + "=" * 60)
    print(f"{'mH':<8} {'u2_low':<14} {'u2_high':<14}")
    print("-" * 60)
    for i, r in enumerate(results):
        lo = r["u2_low"]
        hi = r["u2_high"]
        slo = f"{lo:<14.4e}" if lo < U2_MAX else "no limit"
        shi = f"{hi:<14.4e}" if hi < U2_MAX else ""
        print(f"{mh_arr[i]:<8.1f} {slo:<14} {shi:<14}")
    print(f"\nOutputs: {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
