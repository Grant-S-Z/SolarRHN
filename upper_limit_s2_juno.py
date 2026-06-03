"""
JUNO s2 exclusion — reads toymc_s2 NPZ data, projects to 1D, pyhf fit.

Signal: scattered e⁻ 2D (E_e, cosθ) → sum over cosθ → 1D energy
Background: B8 + B12 + C10 (same as s1)

Usage:
  python upper_limit_s2_juno.py [base_dir]

  base_dir defaults to "plots_grid_scan_s2"
"""

import logging
import os
import sys
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import uproot as ur
import pyhf
from tqdm import tqdm

logging.getLogger("pyhf.optimize.mixins").setLevel(logging.CRITICAL)

from core import *
from core.stats import apply_energy_resolution_convolution

# ── Settings ────────────────────────────────────────────────────────────────
CL_THRESHOLD = 2.71
N_WORKERS = 100
U2_MAX = 1

ENERGY_RESOLUTION = 0.03
ESTEP = 0.2
FIT_E_MIN = 5.0
FIT_E_MAX = 12.8
CSV_EXPOSURE_SCALE = 2.0 / 10.0

OUTDIR = f'./plots/{detector_name}/upper_limit_s2'
os.makedirs(OUTDIR, exist_ok=True)

energy_full = np.arange(0.0, 16.0, ESTEP)
fit_mask = (energy_full >= FIT_E_MIN) & (energy_full <= FIT_E_MAX)
n_fit_bins = int(np.count_nonzero(fit_mask))
bin_centers_full = energy_full + 0.5 * ESTEP


# ── Backgrounds ─────────────────────────────────────────────────────────────
def load_b8():
    f = ur.open('data/juno/solar_juno_fv_5mev.root')
    h = f['he_es']
    v = np.asarray(h.values(), dtype=float)
    e = np.asarray(h.axis().edges(), dtype=float)
    f.close()
    c = 0.5 * (e[:-1] + e[1:])
    return np.interp(bin_centers_full, c, v / (e[1] - e[0]), 0, 0) * ESTEP


def load_csv(path):
    raw = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line.startswith('#') or line.startswith('"#') or line == 'x,y':
                continue
            p = line.split(',')
            if len(p) >= 2 and p[1].strip():
                raw.append((float(p[0]), float(p[1])))
    a = np.array(raw)
    return np.interp(bin_centers_full, a[:, 0], a[:, 1] / 0.1, 0, 0) * ESTEP * CSV_EXPOSURE_SCALE


# ── pyhf ────────────────────────────────────────────────────────────────────
def fit_cond(data, b8, b12, c10, sig):
    b8_s  = np.clip(np.asarray(b8), 1e-12, None)
    b12_s = np.clip(np.asarray(b12), 1e-12, None)
    c10_s = np.clip(np.asarray(c10), 1e-12, None)
    sig_s = np.clip(np.asarray(sig), 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [{
            "name": "juno",
            "samples": [
                {"name": "sig", "data": sig_s.tolist(), "modifiers": []},
                {"name": "b8",  "data": b8_s.tolist(),
                 "modifiers": [{"name": "xb_b8",  "type": "normfactor", "data": None}]},
                {"name": "b12", "data": b12_s.tolist(),
                 "modifiers": [{"name": "xb_b12", "type": "normfactor", "data": None}]},
                {"name": "c10", "data": c10_s.tolist(),
                 "modifiers": [{"name": "xb_c10", "type": "normfactor", "data": None}]},
            ],
        }],
        "observations": [{"name": "juno", "data": np.asarray(data).tolist()}],
        "measurements": [{
            "name": "M", "config": {
                "poi": "xb_b8",
                "parameters": [
                    {"name": "xb_b8",  "inits": [1.0], "bounds": [[0.3, 5.0]]},
                    {"name": "xb_b12", "inits": [1.0], "bounds": [[0.0, 5.0]]},
                    {"name": "xb_c10", "inits": [1.0], "bounds": [[0.0, 5.0]]},
                ],
            },
        }],
    }
    ws = pyhf.Workspace(spec)
    m = ws.model()
    d = ws.data(m)
    try:
        bf, t2 = pyhf.infer.mle.fit(d, m, return_fitted_val=True)
    except (pyhf.exceptions.FailedMinimization, RuntimeError, ValueError):
        return None, 1e300
    po = m.config.par_order
    xb = {k: float(bf[po.index(k)]) for k in ["xb_b8", "xb_b12", "xb_c10"]}
    return xb, 0.5 * float(t2)


# ── Signal reader ───────────────────────────────────────────────────────────
def load_signal_1d(npz_path):
    """Read NPZ, sum 2D over cosθ → 1D, rebin to analysis grid, smear."""
    d = np.load(npz_path)
    sig_2d = d["counts_2d"]           # (nE, nCos)
    e_bins = d["e_bins"]
    ct_bins = d["costheta_lab_bins"]

    # Integrate over cosθ → 1D counts per energy bin
    e_centers_native = 0.5 * (e_bins[:-1] + e_bins[1:])
    sig_1d_native = sig_2d.sum(axis=1)

    # Rebin to analysis grid (simple interp of cumulative)
    sig_1d = np.interp(bin_centers_full, e_centers_native, sig_1d_native, 0, 0)

    # Apply JUNO energy resolution (overwrites the 5% already in NPZ)
    sig_1d = apply_energy_resolution_convolution(
        sig_1d, bin_centers_full, frac_resolution=ENERGY_RESOLUTION,
    )
    return sig_1d[fit_mask]


# ── Scan NPZ directories ────────────────────────────────────────────────────
def find_npz_dirs(base_dir):
    dirs = sorted(glob.glob(os.path.join(base_dir, "U2_*_MH_*")))
    points = []
    for d in dirs:
        name = os.path.basename(d)
        parts = name.split("_")
        try:
            u2_idx = parts.index("U2")
            mh_idx = parts.index("MH")
            u2 = float(parts[u2_idx + 1])
            mh = float(parts[mh_idx + 1])
            npz = os.path.join(d, "electron_data.npz")
            if os.path.exists(npz):
                points.append((mh, u2, npz))
        except (ValueError, IndexError):
            continue
    return points


# ── Worker (module-level, required for pickle) ─────────────────────────────
def _worker(args):
    mH, b8, b12, c10, data, nll_0, u2_list, sig_cache, cl = args
    excluded = []
    for u2 in u2_list:
        sig = sig_cache.get((mH, u2))
        if sig is None or sig.sum() < 1e-12:
            excluded.append(False)
        else:
            _, nll_1 = fit_cond(data, b8, b12, c10, sig)
            dchi2 = 2.0 * max(0.0, nll_1 - nll_0)
            excluded.append(dchi2 > cl)
    excluded = np.array(excluded, dtype=bool)
    u2a = np.asarray(u2_list)
    diffs = np.diff(np.concatenate(([False], excluded, [False])).astype(int))
    enter = np.where(diffs == 1)[0]
    leave = np.where(diffs == -1)[0]
    if len(enter) == 0:
        return {"mH": mH, "u2_low": U2_MAX, "u2_high": U2_MAX}
    i0 = enter[0] - 1 if enter[0] > 0 else 0
    i1 = leave[0] - 1
    lo = 0.5 * (u2a[i0] + u2a[i0 + 1]) if i0 < len(u2a) - 1 else u2a[0]
    hi = 0.5 * (u2a[i1] + u2a[i1 + 1]) if i1 < len(u2a) - 1 else u2_list[-1]
    return {"mH": mH, "u2_low": float(lo), "u2_high": float(hi)}


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    base_dir = sys.argv[1] if len(sys.argv) > 1 else "plots_grid_scan_s2"

    print("=" * 60)
    print("JUNO s2 exclusion — from toymc_s2 NPZ data")
    print(f"Base dir: {base_dir}")
    print("=" * 60)

    # Backgrounds
    print("\n>>> Loading backgrounds ...")
    b8 = apply_energy_resolution_convolution(load_b8(), bin_centers_full, ENERGY_RESOLUTION)[fit_mask]
    b12 = apply_energy_resolution_convolution(load_csv('data/juno/b12_bkg.csv'), bin_centers_full, ENERGY_RESOLUTION)[fit_mask]
    c10 = apply_energy_resolution_convolution(load_csv('data/juno/c10_bkg.csv'), bin_centers_full, ENERGY_RESOLUTION)[fit_mask]
    data_asimov = b8 + b12 + c10
    print(f"    B8={b8.sum():.0f}  B12={b12.sum():.0f}  C10={c10.sum():.0f}  total={data_asimov.sum():.0f}")

    # NLL(μ=0)
    sig0 = np.zeros(n_fit_bins)
    _, nll_0 = fit_cond(data_asimov, b8, b12, c10, sig0)
    print(f"    NLL(μ=0) = {nll_0:.6f}")

    # Find NPZ points
    points = find_npz_dirs(base_dir)
    print(f"\n>>> Found {len(points)} NPZ points")

    if not points:
        print("No data found!")
        return

    # Group by mH: {mH: [(u2, npz_path), ...]}
    mh_vals = sorted(set(p[0] for p in points))
    u2_vals = sorted(set(p[1] for p in points))
    u2_arr = np.asarray(u2_vals)
    mh_groups = {mh: [(u2, npz) for m, u2, npz in points if m == mh] for mh in mh_vals}

    # Pre-load all signals (fast I/O)
    print(">>> Loading signals from NPZ ...")
    sig_cache = {}
    for mh, u2, npz_path in tqdm(points, desc="Loading NPZ"):
        sig_cache[(mh, u2)] = load_signal_1d(npz_path)

    tasks = [(mh, b8, b12, c10, data_asimov, nll_0, u2_vals, sig_cache, CL_THRESHOLD)
             for mh in mh_vals]

    print(f">>> Scanning {len(tasks)} mH × {len(u2_vals)} u2 on {N_WORKERS} workers")
    results = {}
    with ProcessPoolExecutor(max_workers=N_WORKERS) as ex:
        futures = {ex.submit(_worker, t): t[0] for t in tasks}
        with tqdm(total=len(tasks), desc="JUNO s2") as pbar:
            for fut in as_completed(futures):
                r = fut.result()
                results[r["mH"]] = r
                pbar.update(1)
    results = [results[mh] for mh in mh_vals]

    # Save
    csv_path = os.path.join(OUTDIR, "upper_limit_bands_s2.csv")
    with open(csv_path, "w") as f:
        f.write("mH,u2_low,u2_high\n")
        for r in results:
            f.write(f"{r['mH']:.6f},{r['u2_low']:.4e},{r['u2_high']:.4e}\n")
    print(f"Saved: {csv_path}")

    print("\n" + "=" * 60)
    for r in results:
        lo = f"{r['u2_low']:.4e}" if r['u2_low'] < U2_MAX else "no limit"
        hi = f"{r['u2_high']:.4e}" if r['u2_high'] < U2_MAX else ""
        print(f"mH={r['mH']:.1f}  u2_low={lo}  u2_high={hi}")
    print(f"\nOutputs: {OUTDIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
