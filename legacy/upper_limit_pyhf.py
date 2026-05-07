"""
Expected exclusion sensitivity — pyhf upper_limit (Cowan et al. 2011 asymptotics).

For each (mH, u2):
  1. Build workspace: signal (POI=xh) + background (nuisance=xb)
  2. Asimov data = nominal background
  3. pyhf.infer.intervals.upper_limits.upper_limit → expected xh_up
  4. xh_up < 1 → excluded at this u2

Usage:
  python upper_limit_pyhf.py
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
from core import interpolateSpectrum

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
Z_ALPHA = np.sqrt(CL_THRESHOLD)

MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 13

U2_MIN = 1e-6
U2_MAX = 1e-3
N_U2 = 31

N_WORKERS = min(os.cpu_count() or 4, 8)

OUTDIR = "./plots/upper_limit_pyhf"
os.makedirs(OUTDIR, exist_ok=True)


# ── Background ───────────────────────────────────────────────────────────────
def load_b8_background():
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


# ── Test one (mH, u2) via pyhf upper_limit ──────────────────────────────────
def test_one_point(spectrum_orig, data_asimov, bkg, mH, u2):
    """Using pyhf.infer.intervals.upper_limits.upper_limit, compute expected xh_up.
    Return True if xh_up < 1 (excluded at 90% CL)."""
    signal = get_signal_template(spectrum_orig, float(mH), float(u2))
    if signal.sum() < 1e-12:
        return False

    bkg_safe = np.clip(bkg, 1e-12, None)
    sig_safe = np.clip(signal, 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [{
            "name": "borexino",
            "samples": [
                {"name": "signal", "data": sig_safe.tolist(),
                 "modifiers": [{"name": "xh", "type": "normfactor", "data": None}]},
                {"name": "bkg",   "data": bkg_safe.tolist(),
                 "modifiers": [{"name": "xb", "type": "normfactor", "data": None}]},
            ],
        }],
        "observations": [{"name": "borexino", "data": data_asimov.tolist()}],
        "measurements": [{
            "name": "Measurement",
            "config": {
                "poi": "xh",
                "parameters": [
                    {"name": "xh", "inits": [1.0], "bounds": [[0.0, 10.0]]},
                    {"name": "xb", "inits": [1.0], "bounds": [[0.3, 3.0]]},
                ],
            },
        }],
    }

    ws = pyhf.Workspace(spec)
    model = ws.model()
    data = ws.data(model)

    obs, exp = pyhf.infer.intervals.upper_limits.upper_limit(
        data, model, scan=np.logspace(-2, 1, 31), level=0.10
    )
    # exp: [median-2σ, median-1σ, median, median+1σ, median+2σ]
    xh_up_expected = exp[2]
    return bool(xh_up_expected < 1.0)


# ── One mH ───────────────────────────────────────────────────────────────────
def analyze_one_mh(spectrum_orig, data_asimov, bkg, mH, u2_arr):
    """For one mH: test μ=1 at each u2, find excluded region boundary."""
    excluded = []
    for u2 in u2_arr:
        excluded.append(test_one_point(spectrum_orig, data_asimov, bkg, mH, u2))

    excluded = np.array(excluded, dtype=bool)
    u2_arr = np.asarray(u2_arr)

    diffs = np.diff(np.concatenate(([False], excluded, [False])).astype(int))
    enter = np.where(diffs == 1)[0]
    leave = np.where(diffs == -1)[0]

    if len(enter) == 0:
        return {"u2_low": U2_MAX, "u2_high": U2_MAX, "mH": float(mH)}

    i0 = enter[0] - 1 if enter[0] > 0 else 0
    i1 = leave[0] - 1

    low = 0.5 * (u2_arr[i0] + u2_arr[i0+1]) if i0 < len(u2_arr)-1 else u2_arr[0]
    high = 0.5 * (u2_arr[i1] + u2_arr[i1+1]) if i1 < len(u2_arr)-1 else U2_MAX

    return {"u2_low": float(low), "u2_high": float(high), "mH": float(mH)}


# ── Worker & parallel ───────────────────────────────────────────────────────
def _worker(args):
    return analyze_one_mh(*args)

def scan_parallel(spectrum_orig, data_asimov, bkg, mh_arr, u2_arr, *, label="scan"):
    tasks = [(spectrum_orig, data_asimov, bkg, float(mH), u2_arr) for mH in mh_arr]
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, t): t[3] for t in tasks}
        with tqdm(total=len(tasks), desc=label) as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results[r["mH"]] = r
                pbar.update(1)
    return [results.get(float(mH)) for mH in mh_arr]


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_result(mh, results, outpath, *, show_bands=True):
    n = len(mh)
    mh_arr = np.asarray(mh)
    y_top = U2_MAX * 10

    u2_low = np.array([r["u2_low"] for r in results])
    u2_high = np.array([r["u2_high"] for r in results])

    plt.figure(figsize=(8, 6))

    fill_mask = (u2_low < y_top)
    if np.any(fill_mask):
        plt.fill_between(mh_arr, u2_low, u2_high, where=fill_mask,
                         color="tab:blue", alpha=0.15, label="Excluded (90% C.L.)")

    boundary = np.where(u2_low < U2_MAX, u2_low, np.nan)
    plt.plot(mh_arr, boundary, "-", lw=2.2, color="tab:blue", label="Expected limit")

    upper_mask = (u2_high < U2_MAX) & (u2_low < y_top)
    if np.any(upper_mask):
        plt.plot(mh_arr[upper_mask], u2_high[upper_mask], "-", lw=2.2, color="tab:blue")

    if show_bands:
        band1_low = np.full(n, np.nan)
        band1_high = np.full(n, np.nan)
        band2_low = np.full(n, np.nan)
        band2_high = np.full(n, np.nan)
        for i in range(n):
            if u2_low[i] >= U2_MAX: continue
            s = u2_low[i] / Z_ALPHA
            band2_low[i] = max(0.0, u2_low[i] - 2*s)
            band2_high[i] = u2_high[i] + 2*s
            band1_low[i] = max(0.0, u2_low[i] - s)
            band1_high[i] = u2_high[i] + s
        m2 = np.isfinite(band2_low) & np.isfinite(band2_high)
        m1 = np.isfinite(band1_low) & np.isfinite(band1_high)
        if np.any(m2):
            plt.fill_between(mh_arr, band2_low, band2_high, where=m2,
                             color="yellow", alpha=0.25, label=r"$\pm 2\sigma$ expected")
        if np.any(m1):
            plt.fill_between(mh_arr, band1_low, band1_high, where=m1,
                             color="limegreen", alpha=0.25, label=r"$\pm 1\sigma$ expected")

    ref_csv = "./data/Borexino_exclusion.csv"
    if os.path.exists(ref_csv):
        ref = np.loadtxt(ref_csv, delimiter=",")
        if ref.ndim == 2 and ref.shape[1] >= 2:
            log10_mh, log10_u2 = ref[:, 0], ref[:, 1]
            i_min = int(np.argmin(log10_u2))
            for first, sl in enumerate([slice(0, i_min+1), slice(i_min, None)]):
                bx, by = log10_mh[sl], log10_u2[sl]
                if bx.size < 2: continue
                plt.plot((10**bx)*1e3, 10**by, "-", lw=2.0, color="gray",
                         label="Borexino (published)" if first == 0 else None)

    plt.yscale("log")
    plt.xlim(MH_MIN, MH_MAX)
    plt.ylim(U2_MIN, y_top)
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("Expected exclusion (pyhf upper_limit, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Expected exclusion — pyhf upper_limit")
    print("=" * 60)

    print("\n>>> Loading B8 background ...")
    bkg_full = load_b8_background()
    bkg = bkg_full[fit_mask]
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    data_asimov = bkg.copy()
    print(f"    bins: {len(bkg)}, counts: {bkg.sum():.2f}")

    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    print(f"\n>>> Scan: {len(mh_arr)} mH × {len(u2_arr)} u2"
          f" on {N_WORKERS} workers")
    results = scan_parallel(spectrum_orig, data_asimov, bkg, mh_arr, u2_arr,
                            label="expected")

    plot_result(mh_arr, results,
                outpath=os.path.join(OUTDIR, "exclusion_upper_limit_asym.pdf"),
                show_bands=False)

    csv_path = os.path.join(OUTDIR, "upper_limit_bands_asym.csv")
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
