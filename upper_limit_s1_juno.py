"""
JUNO expected exclusion sensitivity — μ=1 test per (mH, u2).

Backgrounds: B8 (ES), B12 (cosmogenic β⁻), C10 (cosmogenic β⁺)
Simultaneous pyhf fit with one normfactor per background component.

Approach:
  1. Asimov data = nominal B8 + B12 + C10 (no fluctuations)
  2. For each (mH, u2): generate signal → fit xb_B8, xb_B12, xb_C10
  3. Δχ²(μ) = 2·(NLL(μ) − NLL(μ=0))
  4. μ=1 excluded if Δχ² > 2.71  →  μ_up < 1 excluded

Usage:
  python upper_limit_s1_juno.py
"""

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot as ur
import pyhf
import matplotlib.pyplot as plt
from tqdm import tqdm

# Suppress pyhf's internal traceback when minimizer hits bounds
logging.getLogger("pyhf.optimize.mixins").setLevel(logging.CRITICAL)

from core import *
from core.stats import apply_energy_resolution_convolution
from workflows import getNuleeInDetector

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
Z_ALPHA = np.sqrt(CL_THRESHOLD)

MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 71

U2_MIN = 1e-6
U2_MAX = 1e-1
N_U2 = 51

N_WORKERS = 100

# JUNO-specific
ENERGY_RESOLUTION = 0.03   # 3% fractional energy resolution
ESTEP = 0.2
FIT_E_MIN = 5.0
FIT_E_MAX = 12.8

# B12 / C10 CSVs are digitised from a 10-year figure; B8 ROOT + signal
# use exposure_time = 2 yr from core.constants.  Scale to match.
CSV_EXPOSURE_SCALE = 2.0 / 10.0   # 10 yr → 2 yr

OUTDIR = f'./plots/{detector_name}/upper_limit_s1'
os.makedirs(OUTDIR, exist_ok=True)

# Energy grid
energy_full = np.arange(0.0, 16.0, ESTEP)
fit_mask = (energy_full >= FIT_E_MIN) & (energy_full <= FIT_E_MAX)
energy_fit = energy_full[fit_mask]
n_fit_bins = int(np.count_nonzero(fit_mask))


# ── Background loading ──────────────────────────────────────────────────────
def load_b8_background():
    """Load B8 ES background from ROOT, rebin to analysis grid (counts/bin)."""
    f = ur.open('data/juno/solar_juno_fv_5mev.root')
    h = f['he_es']
    bg_vals = np.asarray(h.values(), dtype=float)
    bg_edges = np.asarray(h.axis().edges(), dtype=float)
    f.close()

    src_width = bg_edges[1] - bg_edges[0]          # 0.02 MeV
    src_centers = 0.5 * (bg_edges[:-1] + bg_edges[1:])
    src_per_mev = bg_vals / src_width

    our_centers = energy_full + 0.5 * ESTEP
    b_per_mev = np.interp(our_centers, src_centers, src_per_mev, left=0.0, right=0.0)
    return b_per_mev * ESTEP  # counts per 0.2 MeV bin


def load_csv_background(path):
    """Load digitised background CSV → counts per 0.2 MeV analysis bin.

    CSV columns: energy (MeV), counts per 0.1 MeV bin.
    """
    raw = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('#') or line.startswith('"#') or line == 'x,y':
                continue
            parts = line.split(',')
            if len(parts) >= 2 and parts[1].strip():
                raw.append((float(parts[0]), float(parts[1])))
    arr = np.array(raw)
    csv_e = arr[:, 0]
    csv_cts = arr[:, 1]               # counts per 0.1 MeV bin
    csv_per_mev = csv_cts / 0.1       # counts per MeV

    our_centers = energy_full + 0.5 * ESTEP
    our_per_mev = np.interp(our_centers, csv_e, csv_per_mev, left=0.0, right=0.0)
    return our_per_mev * ESTEP * CSV_EXPOSURE_SCALE   # scale 10 yr → 2 yr


# ── Signal template ─────────────────────────────────────────────────────────
def get_signal_template(spectrum_orig, mH, u2):
    """Compute e⁺e⁻ signal counts per bin on the analysis grid (JUNO)."""
    if u2 <= 0.0:
        return np.zeros(n_fit_bins, dtype=float)

    spectrum_rhn = getRHNSpectrum(spectrum_orig, mH, u2)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mH, u2)

    s_bin = np.nan_to_num(
        diff_Eee_decayed[:, 1] * exposure * ESTEP,
        nan=0.0, posinf=0.0, neginf=0.0,
    )
    s_bin = np.clip(s_bin, 0.0, None)
    s_bin = apply_energy_resolution_convolution(
        s_bin, diff_Eee_decayed[:, 0], frac_resolution=ENERGY_RESOLUTION,
    )
    return s_bin[fit_mask]


# ── pyhf fit (3 bkg normfactors, signal fixed) ─────────────────────────────
def fit_cond_three_bkg(data, b8, b12, c10, signal):
    """Fit xb_b8, xb_b12, xb_c10 with signal fixed. Returns (dict_of_xb, NLL).

    On fit failure (e.g. signal overwhelms backgrounds → bounds incompatible),
    returns a sentinel NLL = 1e300 so that Δχ² → ∞ and the point is excluded.
    """
    b8_s = np.clip(np.asarray(b8, dtype=float), 1e-12, None)
    b12_s = np.clip(np.asarray(b12, dtype=float), 1e-12, None)
    c10_s = np.clip(np.asarray(c10, dtype=float), 1e-12, None)
    sig_s = np.clip(np.asarray(signal, dtype=float), 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "juno",
                "samples": [
                    {"name": "signal", "data": sig_s.tolist(), "modifiers": []},
                    {
                        "name": "b8",
                        "data": b8_s.tolist(),
                        "modifiers": [
                            {"name": "xb_b8", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "b12",
                        "data": b12_s.tolist(),
                        "modifiers": [
                            {"name": "xb_b12", "type": "normfactor", "data": None}
                        ],
                    },
                    {
                        "name": "c10",
                        "data": c10_s.tolist(),
                        "modifiers": [
                            {"name": "xb_c10", "type": "normfactor", "data": None}
                        ],
                    },
                ],
            }
        ],
        "observations": [{"name": "juno", "data": data.tolist()}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "xb_b8",
                    "parameters": [
                        {"name": "xb_b8",  "inits": [1.0], "bounds": [[0.3, 3.0]]},
                        {"name": "xb_b12", "inits": [1.0], "bounds": [[0.0, 5.0]]},
                        {"name": "xb_c10", "inits": [1.0], "bounds": [[0.0, 5.0]]},
                    ],
                },
            }
        ],
    }
    ws = pyhf.Workspace(spec)
    model = ws.model()
    data_full = ws.data(model)

    try:
        bestfit, twice_nll = pyhf.infer.mle.fit(
            data_full, model, return_fitted_val=True,
        )
    except (pyhf.exceptions.FailedMinimization, RuntimeError, ValueError) as exc:
        # Fit failed — signal incompatible with data even at boundary.
        # Return sentinel large NLL so the point is counted as excluded.
        return {"xb_b8": 0.3, "xb_b12": 0.0, "xb_c10": 0.0}, 1e300

    par_order = model.config.par_order
    xb = {
        "xb_b8":  float(bestfit[par_order.index("xb_b8")]),
        "xb_b12": float(bestfit[par_order.index("xb_b12")]),
        "xb_c10": float(bestfit[par_order.index("xb_c10")]),
    }
    nll = 0.5 * float(twice_nll)
    return xb, nll


# ── μ=1 test for one (mH, u2) ───────────────────────────────────────────────
def test_one_point(spectrum_orig, data, b8, b12, c10, nll_0, mH, u2):
    """Return True if nominal signal at (mH, u2) is excluded at 90% CL."""
    signal_nominal = get_signal_template(spectrum_orig, float(mH), float(u2))
    if signal_nominal.sum() < 1e-12:
        return False
    _, nll_1 = fit_cond_three_bkg(data, b8, b12, c10, signal_nominal)
    dchi2 = 2.0 * max(0.0, nll_1 - nll_0)
    return bool(dchi2 > CL_THRESHOLD)


# ── One mH: test μ=1 at each u2, find excluded region boundary ─────────────
def analyze_one_mh(spectrum_orig, data, b8, b12, c10, nll_0, mH, u2_arr):
    excluded = []
    for u2 in u2_arr:
        excluded.append(
            test_one_point(spectrum_orig, data, b8, b12, c10, nll_0, mH, u2)
        )
    excluded = np.array(excluded, dtype=bool)
    u2_arr = np.asarray(u2_arr)

    diffs = np.diff(np.concatenate(([False], excluded, [False])).astype(int))
    enter = np.where(diffs == 1)[0]
    leave = np.where(diffs == -1)[0]

    if len(enter) == 0:
        return {"u2_low": U2_MAX, "u2_high": U2_MAX, "mH": float(mH)}

    i0 = enter[0] - 1 if enter[0] > 0 else 0
    i1 = leave[0] - 1

    low = 0.5 * (u2_arr[i0] + u2_arr[i0 + 1]) if i0 < len(u2_arr) - 1 and excluded[i0 + 1] else u2_arr[0]
    high = 0.5 * (u2_arr[i1] + u2_arr[i1 + 1]) if i1 < len(u2_arr) - 1 else U2_MAX

    return {"u2_low": float(low), "u2_high": float(high), "mH": float(mH)}


# ── Worker & parallel ───────────────────────────────────────────────────────
def _worker(args):
    spectrum_orig, data, b8, b12, c10, nll_0, mH, u2_arr = args
    return analyze_one_mh(spectrum_orig, data, b8, b12, c10, nll_0, float(mH), u2_arr)


def scan_parallel(spectrum_orig, data, b8, b12, c10, nll_0, mh_arr, u2_arr, *, label="scan"):
    tasks = [
        (spectrum_orig, data, b8, b12, c10, nll_0, float(mH), u2_arr)
        for mH in mh_arr
    ]
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
    n = len(mh)
    mh_arr = np.asarray(mh)
    y_top = U2_MAX * 10

    u2_low = np.array([r["u2_low"] for r in results])
    u2_high = np.array([r["u2_high"] for r in results])

    plt.figure(figsize=(8, 6))

    # ±1σ band
    if show_bands:
        band_low = np.full(n, np.nan)
        band_high = np.full(n, np.nan)
        for i in range(n):
            if u2_low[i] >= U2_MAX:
                continue
            s = u2_low[i] / (2.0 * Z_ALPHA)
            band_low[i] = max(0.0, u2_low[i] - s)
            band_high[i] = u2_low[i] + s
        m = np.isfinite(band_low) & np.isfinite(band_high)
        if np.any(m):
            plt.fill_between(
                mh_arr, band_low, band_high, where=m,
                color="limegreen", alpha=0.25, label=r"$\pm 1\sigma$ expected",
            )

    # Lower boundary
    boundary = np.where(u2_low < U2_MAX, u2_low, np.nan)
    plt.plot(mh_arr, boundary, "-", lw=2.2, color="tab:blue", label="Expected limit (JUNO)")

    # Upper edge (window points)
    upper_mask = (u2_high < U2_MAX) & (u2_low < y_top)
    if np.any(upper_mask):
        plt.plot(mh_arr[upper_mask], u2_high[upper_mask], "-", lw=2.2, color="tab:blue")

    # Borexino published (reference)
    ref_csv = "./data/Borexino_exclusion.csv"
    if os.path.exists(ref_csv):
        ref = np.loadtxt(ref_csv, delimiter=",")
        if ref.ndim == 2 and ref.shape[1] >= 2:
            log10_mh, log10_u2 = ref[:, 0], ref[:, 1]
            # Convert from Borexino paper: mH in GeV→MeV, split into two branches
            i_min = int(np.argmin(log10_u2))
            for first, sl in enumerate([slice(0, i_min + 1), slice(i_min, None)]):
                bx, by = log10_mh[sl], log10_u2[sl]
                if bx.size < 2:
                    continue
                plt.plot(
                    (10**bx) * 1e3,
                    10**by,
                    "-", lw=2.0, color="gray",
                    label="Borexino (published)" if first == 0 else None,
                )

    plt.yscale("log")
    plt.xlim(MH_MIN, MH_MAX)
    plt.ylim(U2_MIN, y_top)
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("JUNO expected exclusion (Asimov, 90% C.L.)\nB8 + B12 + C10")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved: {outpath}")


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("JUNO expected exclusion — μ=1 test per (mH, u2)")
    print("Backgrounds: B8 (ES) + B12 (β⁻) + C10 (β⁺)")
    print("=" * 60)

    # ── Load backgrounds ─────────────────────────────────────────────────
    print("\n>>> Loading B8 background (ROOT) ...")
    b8_full = load_b8_background()
    b8_full = apply_energy_resolution_convolution(
        b8_full, energy_full + 0.5 * ESTEP, frac_resolution=ENERGY_RESOLUTION,
    )

    print(">>> Loading B12 background (CSV) ...")
    b12_full = load_csv_background('data/juno/b12_bkg.csv')
    b12_full = apply_energy_resolution_convolution(
        b12_full, energy_full + 0.5 * ESTEP, frac_resolution=ENERGY_RESOLUTION,
    )

    print(">>> Loading C10 background (CSV) ...")
    c10_full = load_csv_background('data/juno/c10_bkg.csv')
    c10_full = apply_energy_resolution_convolution(
        c10_full, energy_full + 0.5 * ESTEP, frac_resolution=ENERGY_RESOLUTION,
    )

    b8  = b8_full[fit_mask]
    b12 = b12_full[fit_mask]
    c10 = c10_full[fit_mask]

    print(f"    Fit bins: {n_fit_bins}  ({FIT_E_MIN}–{FIT_E_MAX} MeV, {ESTEP} MeV step)")
    print(f"    B8  total = {b8.sum():.1f}")
    print(f"    B12 total = {b12.sum():.1f}")
    print(f"    C10 total = {c10.sum():.1f}")
    print(f"    Total bkg = {b8.sum() + b12.sum() + c10.sum():.1f}")

    # ── Asimov data = sum of nominal backgrounds ────────────────────────
    data_asimov = b8 + b12 + c10

    # ── Signal spectrum ─────────────────────────────────────────────────
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy_full)

    # ── NLL at μ=0 ──────────────────────────────────────────────────────
    print("\n>>> Computing NLL(μ=0) ...")
    sig_zero = np.zeros(n_fit_bins)
    xb_0, nll_0 = fit_cond_three_bkg(data_asimov, b8, b12, c10, sig_zero)
    print(f"    xb (μ=0): b8={xb_0['xb_b8']:.4f}  b12={xb_0['xb_b12']:.4f}  c10={xb_0['xb_c10']:.4f}")
    print(f"    NLL(μ=0) = {nll_0:.6f}")

    # ── Scan ────────────────────────────────────────────────────────────
    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    print(f"\n>>> Scan: {len(mh_arr)} mH × {len(u2_arr)} u2  on {N_WORKERS} workers")
    results = scan_parallel(
        spectrum_orig, data_asimov, b8, b12, c10, nll_0,
        mh_arr, u2_arr, label="JUNO expected",
    )

    # ── Plot ────────────────────────────────────────────────────────────
    plot_result(
        mh_arr, results,
        outpath=os.path.join(OUTDIR, "exclusion_upper_limit_juno.pdf"),
        show_bands=True,
    )

    # ── CSV output ──────────────────────────────────────────────────────
    csv_path = os.path.join(OUTDIR, "upper_limit_bands_juno.csv")
    with open(csv_path, "w") as f:
        f.write("mH,u2_low,u2_high\n")
        for i, r in enumerate(results):
            f.write(f"{mh_arr[i]:.6f},{r['u2_low']:.4e},{r['u2_high']:.4e}\n")
    print(f"Saved: {csv_path}")

    # ── Summary table ───────────────────────────────────────────────────
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
