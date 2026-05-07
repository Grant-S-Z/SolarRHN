"""
Expected (Asimov) upper limit — B8 background only (Solar.root), signal per u2.

Approach (Cowan et al. 2011):
  1. Asimov data = nominal B8 background from Solar.root (no real data)
  2. For each u2, generate signal template → fit xb only → NLL
  3. Δχ²(u2) = 2·(NLL(u2) − NLL_min)
  4. Find u2 where Δχ² = 2.71 → median expected limit
  5. ±1σ / ±2σ bands from σ = u2_limit / √(2.71)

Usage:
  python upper_limit.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot as ur
import matplotlib.pyplot as plt
from tqdm import tqdm

from borexino_data_exclusion import (
    get_signal_template,
    fit_xb_with_pyhf,
    find_u2_crossings,
    energy,
    fit_mask,
    estep,
)
from core import interpolateSpectrum, exposure

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
Z_ALPHA = np.sqrt(CL_THRESHOLD)

MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 7    # quick test

U2_MIN = 1e-6
U2_MAX = 1e-3
N_U2 = 16   # quick test

N_WORKERS = min(os.cpu_count() or 4, 8)

OUTDIR = "./plots/upper_limit"
os.makedirs(OUTDIR, exist_ok=True)


# ── Background from Solar.root ───────────────────────────────────────────────
def load_b8_background():
    """Load B8 ES background from Solar.root."""
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
    b_bin = b_per_mev * estep              # counts / bin
    return b_bin[fit_mask]                 # only fit range


# ── Single-mH scan ───────────────────────────────────────────────────────────
def scan_one_mh(spectrum_orig, data_asimov, bkg, mH, u2_arr):
    """Scan u2 for one mH: signal per u2 → fit xb → NLL → Δχ²."""
    rows = []
    for u2 in np.asarray(u2_arr, dtype=float):
        s = get_signal_template(spectrum_orig, float(mH), u2)
        xb, nll = fit_xb_with_pyhf(data_asimov, bkg, s)
        rows.append({"u2": u2, "xb": xb, "nll": nll})

    nll_arr = np.array([r["nll"] for r in rows])
    nll_min = float(np.min(nll_arr))
    for r in rows:
        r["delta_chi2"] = 2.0 * (r["nll"] - nll_min)
    return rows


# ── Band calculation ────────────────────────────────────────────────────────
def compute_bands(u2_arr, dchi2_arr):
    """From a Δχ²(u2) curve, find 90% CL excluded region.

    Returns:
        None  — no exclusion
        dict  — 'type' in {'upper_limit', 'window'}, always includes ±Nσ bands
    """
    crossings = find_u2_crossings(u2_arr, dchi2_arr, threshold=CL_THRESHOLD)
    if not crossings:
        return None

    if len(crossings) >= 2:
        low, high = float(crossings[0]), float(crossings[-1])
        s = low / Z_ALPHA
        return {
            "type": "window",
            "u2_low": low,
            "u2_high": high,
            "u2_low_m1": max(0.0, low - s),
            "u2_low_p1": low + s,
            "u2_low_m2": max(0.0, low - 2*s),
            "u2_low_p2": low + 2*s,
            "u2_high_m1": max(0.0, high - s),
            "u2_high_p1": high + s,
            "u2_high_m2": max(0.0, high - 2*s),
            "u2_high_p2": high + 2*s,
        }

    u2_lim = float(crossings[-1])
    sigma_u2 = u2_lim / Z_ALPHA
    return {
        "type": "upper_limit",
        "u2_med": u2_lim,
        "u2_m1": max(0.0, u2_lim - sigma_u2),
        "u2_p1": u2_lim + sigma_u2,
        "u2_m2": max(0.0, u2_lim - 2 * sigma_u2),
        "u2_p2": u2_lim + 2 * sigma_u2,
    }


# ── Worker ───────────────────────────────────────────────────────────────────
def _worker(args):
    spectrum_orig, data, bkg, mH, u2_arr = args
    rows = scan_one_mh(spectrum_orig, data, bkg, float(mH), u2_arr)
    dchi2 = np.array([r["delta_chi2"] for r in rows])
    return float(mH), compute_bands(u2_arr, dchi2)


# ── Parallel scanner ────────────────────────────────────────────────────────
def scan_parallel(spectrum_orig, data, bkg, mh_arr, u2_arr, *, label="scan"):
    tasks = [(spectrum_orig, data, bkg, float(mH), u2_arr) for mH in mh_arr]
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, t): t[3] for t in tasks}
        with tqdm(total=len(tasks), desc=label) as pbar:
            for fut in as_completed(futures):
                mH, bands = fut.result()
                results[mH] = bands
                pbar.update(1)
    return [results.get(float(mH)) for mH in mh_arr]


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_result(mh, bands, outpath):
    """Unified exclusion plot: one continuous fill + one boundary curve.
    
    - upper_limit → excluded above u2_med
    - window      → excluded between u2_low and u2_high
    - no limit    → nothing (gap)
    """
    mh = np.asarray(mh)
    n = len(mh)

    # Build lower / upper arrays for the fill
    y_ceil = U2_MAX * 10
    lower = np.full(n, np.nan)
    upper = np.full(n, np.nan)
    
    for i, b in enumerate(bands):
        if b is None:
            continue
        if b["type"] == "window":
            lower[i] = b["u2_low"]
            upper[i] = b["u2_high"]
        else:
            lower[i] = b["u2_med"]
            upper[i] = y_ceil

    # ── Draw ──
    plt.figure(figsize=(8, 6))

    # ±Nσ bands — continuous across both upper_limit and window points
    band1_low = np.full(n, np.nan)
    band1_high = np.full(n, np.nan)
    band2_low = np.full(n, np.nan)
    band2_high = np.full(n, np.nan)

    for i, b in enumerate(bands):
        if b is None:
            continue
        if b["type"] == "window":
            band2_low[i] = b["u2_low_m2"]
            band2_high[i] = b["u2_high_p2"]
            band1_low[i] = b["u2_low_m1"]
            band1_high[i] = b["u2_high_p1"]
        else:
            band2_low[i] = b["u2_m2"]
            band2_high[i] = b["u2_p2"]
            band1_low[i] = b["u2_m1"]
            band1_high[i] = b["u2_p1"]

    # Only fill where defined and finite
    mask2 = np.isfinite(band2_low) & np.isfinite(band2_high)
    mask1 = np.isfinite(band1_low) & np.isfinite(band1_high)
    if np.any(mask2):
        plt.fill_between(mh, band2_low, band2_high, where=mask2,
                         color="yellow", alpha=0.25, label=r"$\pm 2\sigma$ expected")
    if np.any(mask1):
        plt.fill_between(mh, band1_low, band1_high, where=mask1,
                         color="limegreen", alpha=0.25, label=r"$\pm 1\sigma$ expected")

    # Excluded region fill
    valid = np.isfinite(lower)
    if np.any(valid):
        plt.fill_between(mh, lower, upper, where=valid,
                         color="tab:blue", alpha=0.15, label="Excluded (90% C.L.)")

    # Boundary curve: only draw for upper-limit points (one solid curve)
    lim_mh, lim_u2 = [], []
    for i, b in enumerate(bands):
        if b and b["type"] != "window":
            lim_mh.append(mh[i])
            lim_u2.append(b["u2_med"])
    if lim_mh:
        plt.plot(lim_mh, lim_u2, "-", lw=2.2, color="tab:blue",
                 label="Expected limit")

    # Borexino published exclusion
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
                plt.plot((10 ** bx) * 1e3, 10 ** by, "-", lw=2.0,
                         color="gray", label="Borexino (published)" if first == 0 else None)

    plt.yscale("log")
    plt.xlim(MH_MIN, MH_MAX)
    plt.ylim(U2_MIN, U2_MAX * 10)
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("Expected exclusion sensitivity (Asimov, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Expected (Asimov) upper limit — B8 only, signal per u2")
    print("=" * 60)

    # 1. Load
    print("\n>>> Loading B8 background from Solar.root ...")
    bkg = load_b8_background()
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    data_asimov = bkg.copy()
    print(f"    B8 bins in fit range: {len(bkg)}, total counts: {bkg.sum():.2f}")

    # 2. Scan parameters
    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    # # 3. Debug: NLL curves + Δχ² plots for key mH
    # print("\n>>> DEBUG: NLL curves + Δχ²")
    # from borexino_data_exclusion import fit_xb_with_pyhf, get_signal_template

    # e_centers = energy[fit_mask] + 0.5 * estep

    # for mH_debug in [10.0, 12.0, 14.0]:
    #     rows = scan_one_mh(spectrum_orig, data_asimov, bkg, mH_debug, u2_arr)
    #     u2_vals = np.array([r["u2"] for r in rows])
    #     nll_vals = np.array([r["nll"] for r in rows])
    #     xb_vals = np.array([r["xb"] for r in rows])
    #     dchi2_vals = np.array([r["delta_chi2"] for r in rows])
    #     imin = int(np.argmin(nll_vals))

    #     print(f"\n--- mH={mH_debug:.1f} ---")
    #     crossings = find_u2_crossings(u2_vals, dchi2_vals, threshold=CL_THRESHOLD)
    #     print(f"  n_crossings = {len(crossings)}  ->  {crossings}")
    #     print(f"  {'u2':<12} {'xb':<8} {'NLL':<14} {'Δχ²':<12}")
    #     for i in range(len(u2_arr)):
    #         m = " <-- min" if i == imin else ""
    #         print(f"  {u2_vals[i]:<12.4e} {xb_vals[i]:<8.4f} {nll_vals[i]:<14.6f} {dchi2_vals[i]:<12.6f}{m}")
    #     s1 = get_signal_template(spectrum_orig, mH_debug, u2=1e-5).sum()
    #     print(f"  signal sum @ u2=1e-5: {s1:.4e}")

    #     # ── Δχ² curve plot ──
    #     fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    #     ax1.plot(u2_vals[1:], dchi2_vals[1:], "o-", lw=2, ms=5)
    #     ax1.axhline(CL_THRESHOLD, color="tab:orange", ls="--", lw=1.5, label=f"90% CL ({CL_THRESHOLD})")
    #     ax1.scatter(u2_vals[imin], dchi2_vals[imin], color="tab:red", s=80, zorder=3)
    #     for c in crossings:
    #         ax1.axvline(c, color="tab:red", ls=":", lw=1.5)
    #     ax1.set_xscale("log")
    #     ax1.set_xlabel(r"$|U_{eH}|^2$")
    #     ax1.set_ylabel(r"$\Delta\chi^2$")
    #     ax1.set_title(f"mH={mH_debug:.1f} MeV  —  Δχ²")
    #     ax1.grid(True, ls=":", alpha=0.5)
    #     ax1.legend(fontsize=9)

    #     ax2.plot(u2_vals[1:], xb_vals[1:], "s-", lw=2, ms=5, color="tab:purple")
    #     ax2.axhline(1.0, color="gray", ls="--")
    #     ax2.set_xscale("log")
    #     ax2.set_xlabel(r"$|U_{eH}|^2$")
    #     ax2.set_ylabel(r"$\hat{x}_b$")
    #     ax2.set_title(f"mH={mH_debug:.1f} MeV  —  xb")
    #     ax2.grid(True, ls=":", alpha=0.5)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(OUTDIR, f"debug_chi2_mH{mH_debug:.1f}.pdf"), dpi=150)
    #     plt.close()

    #     # ── Fit spectra at key u2 ──
    #     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    #     fig.suptitle(f"mH={mH_debug:.1f} MeV  —  fit spectra", fontsize=14)
    #     for ax, u2_show in zip(axes, [0.0, 1e-5, 1e-3]):
    #         s = get_signal_template(spectrum_orig, mH_debug, u2=u2_show)
    #         xb, nll = fit_xb_with_pyhf(data_asimov, bkg, s)
    #         model = xb * bkg + s
    #         ax.step(e_centers, data_asimov, where='mid', color='black', lw=1.0, label='Asimov data')
    #         ax.plot(e_centers, model, '-', lw=2, color='tab:blue', label=f'fit: xb={xb:.4f}, NLL={nll:.3f}')
    #         ax.plot(e_centers, xb * bkg, '--', lw=1.5, color='tab:orange', label='xb·bkg')
    #         ax.plot(e_centers, s, ':', lw=1.5, color='tab:green', label=f'signal (u2={u2_show:.0e})')
    #         ax.set_title(f"u2={u2_show:.0e}")
    #         ax.set_xlabel("E (MeV)")
    #         ax.set_ylabel("Counts / bin")
    #         ax.legend(fontsize=7)
    #         ax.grid(True, ls=':', alpha=0.5)
    #     plt.tight_layout()
    #     plt.savefig(os.path.join(OUTDIR, f"debug_fit_mH{mH_debug:.1f}.pdf"), dpi=150)
    #     plt.close()
    #     print(f"  Saved debug plots for mH={mH_debug:.1f}")

    # 4. Scan
    print(f"\n>>> Expected limit: {len(mh_arr)} mH × {len(u2_arr)} u2 on {N_WORKERS} workers")
    bands_list = scan_parallel(
        spectrum_orig, data_asimov, bkg, mh_arr, u2_arr,
        label="expected",
    )

    # 5. Plot
    plot_result(mh_arr, bands_list,
                outpath=os.path.join(OUTDIR, "exclusion_upper_limit.pdf"))

    # 6. CSV
    csv_path = os.path.join(OUTDIR, "upper_limit_bands.csv")
    with open(csv_path, "w") as f:
        f.write("mH,type,u2_low,u2_med,u2_high,"
                "u2_minus2sigma,u2_minus1sigma,u2_plus1sigma,u2_plus2sigma\n")
        for i, mH in enumerate(mh_arr):
            b = bands_list[i]
            if b is None:
                f.write(f"{mH:.6f},none,,,,,,\n")
            elif b["type"] == "window":
                f.write(f"{mH:.6f},window,{b['u2_low']:.8e},,{b['u2_high']:.8e},,,,\n")
            else:
                f.write(f"{mH:.6f},upper_limit,,{b['u2_med']:.8e},,"
                        f"{b['u2_m2']:.8e},{b['u2_m1']:.8e},"
                        f"{b['u2_p1']:.8e},{b['u2_p2']:.8e}\n")
    print(f"Saved: {csv_path}")

    # 7. Summary
    hdr = f"{'mH':<8} {'type':<13} {'lower/med':<14} {'upper/+1σ':<14} {'notes':<20}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, mH in enumerate(mh_arr):
        b = bands_list[i]
        if b is None:
            print(f"{mH:<8.1f} {'no limit':<13}")
        elif b["type"] == "window":
            print(f"{mH:<8.1f} {'window':<13} {b['u2_low']:<14.4e} {b['u2_high']:<14.4e} {'':<20}")
        else:
            print(f"{mH:<8.1f} {'upper limit':<13} {b['u2_med']:<14.4e} {b['u2_p1']:<14.4e} {'+1σ':<20}")
    print(f"\nOutputs: {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
