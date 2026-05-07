"""
Expected exclusion sensitivity — Toy Monte Carlo (direct xh_up approach).

For each (mH, u2):
  1. Generate signal
  2. N toys = Poisson(bkg), xb=1.0
  3. For each toy:
       free fit (xb, xh) → xh_hat, NLL_min
       cond fit at xh=1    → NLL_1
       σ = |1 − xh_hat| / √(2(NLL_1 − NLL_min))
       xh_up = xh_hat + σ × 1.645
  4. xh_up_median = median({xh_up_k})
  5. xh_up_median < 1 → excluded

Usage:
  python upper_limit_mc.py
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
MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 5

U2_MIN = 1e-6
U2_MAX = 1e-3
N_U2 = 12

N_TOYS = 200
N_WORKERS = min(os.cpu_count() or 4, 8)
Z_ALPHA = 1.645

OUTDIR = "./plots/upper_limit_mc"
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


# ── pyhf fits ────────────────────────────────────────────────────────────────
def fit_free(data, bkg, signal):
    """Free fit (xb, xh).  Returns (xb, xh, NLL)."""
    spec = {
        "version": "1.0.0",
        "channels": [{
            "name": "borexino",
            "samples": [
                {"name": "signal", "data": np.clip(signal, 0.0, None).tolist(),
                 "modifiers": [{"name": "xh", "type": "normfactor", "data": None}]},
                {"name": "bkg", "data": np.clip(bkg, 1e-12, None).tolist(),
                 "modifiers": [{"name": "xb", "type": "normfactor", "data": None}]},
            ],
        }],
        "observations": [{"name": "borexino", "data": data.tolist()}],
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
    d = ws.data(model)
    bestfit, twice_nll = pyhf.infer.mle.fit(d, model, return_fitted_val=True)
    xh = float(bestfit[model.config.par_order.index("xh")])
    xb = float(bestfit[model.config.par_order.index("xb")])
    nll = 0.5 * float(twice_nll)
    return xb, xh, nll


def fit_cond(data, bkg, signal):
    """Fit xb only, signal fixed.  Returns (xb, NLL)."""
    spec = {
        "version": "1.0.0",
        "channels": [{
            "name": "borexino",
            "samples": [
                {"name": "signal", "data": np.clip(signal, 0.0, None).tolist(),
                 "modifiers": []},
                {"name": "bkg", "data": np.clip(bkg, 1e-12, None).tolist(),
                 "modifiers": [{"name": "xb", "type": "normfactor", "data": None}]},
            ],
        }],
        "observations": [{"name": "borexino", "data": data.tolist()}],
        "measurements": [{
            "name": "Measurement",
            "config": {
                "poi": "xb",
                "parameters": [{"name": "xb", "inits": [1.0], "bounds": [[0.3, 3.0]]}],
            },
        }],
    }
    ws = pyhf.Workspace(spec)
    model = ws.model()
    d = ws.data(model)
    bestfit, twice_nll = pyhf.infer.mle.fit(d, model, return_fitted_val=True)
    xb = float(bestfit[model.config.par_order.index("xb")])
    nll = 0.5 * float(twice_nll)
    return xb, nll


# ── MC for one (mH, u2) ─────────────────────────────────────────────────────
def mc_test_one_point(spectrum_orig, bkg, mH, u2, *, n_toys=N_TOYS, seed=42):
    """MC at one (mH, u2): median xh_up < 1 → excluded."""
    signal = get_signal_template(spectrum_orig, float(mH), float(u2))
    if signal.sum() < 1e-12:
        return None
    sig_zero = np.zeros_like(signal)

    rng = np.random.default_rng(seed + int(mH * 1000) + int(u2 * 1e10))
    xh_up_vals = np.empty(n_toys)

    for k in range(n_toys):
        toy = rng.poisson(bkg)

        # Free fit
        _, xh_hat, nll_min = fit_free(toy, bkg, signal)

        # Cond fit at xh=1
        _, nll_1 = fit_cond(toy, bkg, signal)

        dchi2 = 2.0 * max(0.0, nll_1 - nll_min)
        sigma = abs(1.0 - xh_hat) / np.sqrt(max(1e-10, dchi2))
        xh_up_vals[k] = xh_hat + Z_ALPHA * sigma

    xh_up_med = float(np.median(xh_up_vals))
    return {
        "xh_up_med": xh_up_med,
        "excluded": bool(xh_up_med < 1.0),
        "xh_up_vals": xh_up_vals,
    }


# ── One mH ───────────────────────────────────────────────────────────────────
def analyze_one_mh(spectrum_orig, bkg, mH, u2_arr):
    excluded = []
    for u2 in u2_arr:
        r = mc_test_one_point(spectrum_orig, bkg, mH, u2)
        excluded.append(r["excluded"] if r else False)

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

def scan_parallel(spectrum_orig, bkg, mh_arr, u2_arr, *, label="scan"):
    tasks = [(spectrum_orig, bkg, float(mH), u2_arr) for mH in mh_arr]
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, t): t[2] for t in tasks}
        with tqdm(total=len(tasks), desc=label) as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results[r["mH"]] = r
                pbar.update(1)
    return [results.get(float(mH)) for mH in mh_arr]


# ── Plot ─────────────────────────────────────────────────────────────────────
def plot_result(mh, results, outpath):
    n = len(mh)
    mh_arr = np.asarray(mh)
    y_top = U2_MAX * 10
    u2_low = np.array([r["u2_low"] for r in results])
    u2_high = np.array([r["u2_high"] for r in results])

    plt.figure(figsize=(8, 6))
    fill_mask = (u2_low < y_top)
    if np.any(fill_mask):
        plt.fill_between(mh_arr, u2_low, u2_high, where=fill_mask,
                         color="tab:blue", alpha=0.15, label="Excluded (MC 90% C.L.)")
    boundary = np.where(u2_low < U2_MAX, u2_low, np.nan)
    plt.plot(mh_arr, boundary, "-", lw=2.2, color="tab:blue", label="Expected (MC)")
    upper_mask = (u2_high < U2_MAX) & (u2_low < y_top)
    if np.any(upper_mask):
        plt.plot(mh_arr[upper_mask], u2_high[upper_mask], "-", lw=2.2, color="tab:blue")

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
    plt.title("Expected exclusion (MC, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print(f"Expected exclusion — MC toys (N={N_TOYS})")
    print("=" * 60)

    print("\n>>> Loading B8 background ...")
    bkg_full = load_b8_background()
    bkg = bkg_full[fit_mask]
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    print(f"    bins: {len(bkg)}, counts: {bkg.sum():.2f}")

    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    print(f"\n>>> MC scan: {len(mh_arr)} mH × {len(u2_arr)} u2"
          f" × {N_TOYS} toys on {N_WORKERS} workers")

    # Debug
    print("\n>>> Debug: mH=8, u2=1e-5")
    r = mc_test_one_point(spectrum_orig, bkg, 8.0, 1e-5)
    if r:
        print(f"    xh_up_median = {r['xh_up_med']:.4e}  →  "
              f"{'EXCLUDED' if r['excluded'] else 'NOT excluded'}")

    results = scan_parallel(spectrum_orig, bkg, mh_arr, u2_arr, label="MC")

    plot_result(mh_arr, results,
                outpath=os.path.join(OUTDIR, "exclusion_mc.pdf"))

    print("\n" + "=" * 60)
    for i, r in enumerate(results):
        lo = r["u2_low"]
        s = f"{lo:.4e}" if lo < U2_MAX else "no limit"
        print(f"  mH={mh_arr[i]:.1f}  u2_low={s}")
    print(f"\nOutputs: {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
