"""
Expected (Asimov) upper limit — two-component background model (B8 + Be11).

Approach (Cowan et al. 2011):
  1. Asimov data = bkg_b8 + bkg_be11  (nominal background, no real data)
  2. profile_likelihood_scan_u2 on Asimov data → Δχ²(u2) for each mH
  3. Find u2 where Δχ² = 2.71 → median expected limit
  4. ±1σ / ±2σ bands from σ = u2_limit / √(2.71)

Usage:
  python upper_limit.py
"""

import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from reproduce_borexino_fit import (
    load_borexino_data,
    profile_likelihood_scan_u2,
    find_u2_crossings,
    energy,
    fit_mask,
)
from core.spectrum_utils import interpolateSpectrum

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
Z_ALPHA = np.sqrt(CL_THRESHOLD)

MH_MIN = 2.0
MH_MAX = 14.0
N_MH = 13

U2_MIN = 1e-6
U2_MAX = 1e-3
N_U2 = 31
U2_REF = 1e-5

N_WORKERS = min(os.cpu_count() or 4, 8)

OUTDIR = "./plots/borexino/upper_limit"
os.makedirs(OUTDIR, exist_ok=True)
fname = 'upper_limit_borexino_bkg'


# ── Band calculation ────────────────────────────────────────────────────────
def compute_bands(u2_arr, dchi2_arr):
    crossings = find_u2_crossings(u2_arr, dchi2_arr, threshold=CL_THRESHOLD)
    if not crossings:
        return None

    u2_lim = float(crossings[-1])
    sigma_u2 = u2_lim / Z_ALPHA

    return {
        "u2_med": u2_lim,
        "u2_m1": max(0.0, u2_lim - sigma_u2),
        "u2_p1": u2_lim + sigma_u2,
        "u2_m2": max(0.0, u2_lim - 2 * sigma_u2),
        "u2_p2": u2_lim + 2 * sigma_u2,
    }


# ── Worker ───────────────────────────────────────────────────────────────────
def _scan_one_mh(args):
    spectrum_orig, data, bkg_b8, bkg_be11, mH, u2_arr, u2_ref = args
    rows = profile_likelihood_scan_u2(
        spectrum_orig, data, bkg_b8, bkg_be11, float(mH), u2_arr, u2_ref,
    )
    dchi2 = np.array([r["delta_chi2"] for r in rows])
    result = compute_bands(u2_arr, dchi2)
    return float(mH), result


# ── Parallel scanner ────────────────────────────────────────────────────────
def scan_mh_parallel(spectrum_orig, data, bkg_b8, bkg_be11, mh_arr, u2_arr, *, label="scan"):
    tasks = [(spectrum_orig, data, bkg_b8, bkg_be11, float(mH), u2_arr, U2_REF)
             for mH in mh_arr]
    results = {}

    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_scan_one_mh, t): t[4] for t in tasks}
        with tqdm(total=len(tasks), desc=label) as pbar:
            for fut in as_completed(futures):
                mH, bands = fut.result()
                results[mH] = bands
                pbar.update(1)

    return [results.get(float(mH)) for mH in mh_arr]


# ── Plotting ────────────────────────────────────────────────────────────────
def plot_result(mh, bands, outpath):
    valid = np.array([b is not None for b in bands])

    u2_med = np.array([b["u2_med"] if b else np.nan for b in bands])
    u2_m1  = np.array([b["u2_m1"]  if b else np.nan for b in bands])
    u2_p1  = np.array([b["u2_p1"]  if b else np.nan for b in bands])
    u2_m2  = np.array([b["u2_m2"]  if b else np.nan for b in bands])
    u2_p2  = np.array([b["u2_p2"]  if b else np.nan for b in bands])

    plt.figure(figsize=(8, 6))

    if np.any(valid):
        plt.fill_between(mh[valid], u2_m2[valid], u2_p2[valid],
                         color="yellow", alpha=0.3, label=r"$\pm 2\sigma$ expected")
        plt.fill_between(mh[valid], u2_m1[valid], u2_p1[valid],
                         color="limegreen", alpha=0.3, label=r"$\pm 1\sigma$ expected")
        plt.plot(mh[valid], u2_med[valid], "--", lw=2.5,
                 color="tab:blue", label="Expected (Asimov)")

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
    print("Expected (Asimov) upper limit")
    print("=" * 60)

    # 1. Load templates
    print("\n>>> Loading background templates and spectrum...")
    _, _, bkg_b8_full, bkg_be11_full, _ = load_borexino_data()
    bkg_b8 = np.asarray(bkg_b8_full[fit_mask], dtype=float)
    bkg_be11 = np.asarray(bkg_be11_full[fit_mask], dtype=float)
    spectrum_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)

    # 2. Asimov data = nominal background (no real data)
    # print(">>> Asimov data = bkg_b8 + bkg_be11")
    # data_asimov = bkg_b8 + bkg_be11
    print(">>> Asimov data = bkg_b8")
    data_asimov = bkg_b8

    # 3. Scan parameters
    mh_arr = np.linspace(MH_MIN, MH_MAX, N_MH)
    u2_arr = np.logspace(np.log10(U2_MIN), np.log10(U2_MAX), N_U2)
    if not np.any(np.isclose(u2_arr, 0.0)):
        u2_arr = np.insert(u2_arr, 0, 0.0)

    # 4. Scan
    print(f"\n>>> Expected limit: {len(mh_arr)} mH × {len(u2_arr)} u2"
          f" on {N_WORKERS} workers")
    bands_list = scan_mh_parallel(
        spectrum_orig, data_asimov, bkg_b8, bkg_be11, mh_arr, u2_arr,
        label="expected",
    )

    # 5. Plot
    plot_result(
        mh_arr, bands_list,
        outpath=os.path.join(OUTDIR, f"exclusion_{fname}.pdf"),
    )

    # 6. CSV
    csv_path = os.path.join(OUTDIR, f"{fname}_bands.csv")
    with open(csv_path, "w") as f:
        f.write("mH,u2_minus2sigma,u2_minus1sigma,"
                "u2_expected,u2_plus1sigma,u2_plus2sigma\n")
        for i, mH in enumerate(mh_arr):
            b = bands_list[i]
            if b:
                f.write(f"{mH:.6f},{b['u2_m2']:.8e},{b['u2_m1']:.8e},"
                        f"{b['u2_med']:.8e},{b['u2_p1']:.8e},{b['u2_p2']:.8e}\n")
            else:
                f.write(f"{mH:.6f},,,,,\n")
    print(f"Saved: {csv_path}")

    # 7. Summary
    hdr = f"{'mH':<8} {'Expected':<14} {'-1σ':<14} {'+1σ':<14} {'-2σ':<14} {'+2σ':<14}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("-" * len(hdr))
    for i, mH in enumerate(mh_arr):
        b = bands_list[i]
        if b:
            print(f"{mH:<8.1f} {b['u2_med']:<14.4e} {b['u2_m1']:<14.4e} "
                  f"{b['u2_p1']:<14.4e} {b['u2_m2']:<14.4e} "
                  f"{b['u2_p2']:<14.4e}")
        else:
            print(f"{mH:<8.1f} {'no limit':<14}")
    print(f"\nOutputs: {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
