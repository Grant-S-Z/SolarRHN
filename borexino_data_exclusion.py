import argparse
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np
import matplotlib.pyplot as plt
import pyhf

from core import *
from core.stats import apply_energy_resolution_convolution
from toymc_s1_borexino_profile import load_background
from workflows import getNuleeInDetector


# Analysis constants (same convention as borexino_data_fit_pyhf.py)
estep: float = 0.2
e_min: float = 0.0
e_max: float = 16.0
fit_e_min: float = 4.8
fit_e_max: float = 12.8
n_all = int((e_max - e_min) / estep) + 1
energy = np.linspace(e_min, e_max, n_all)
fit_mask = (energy >= fit_e_min) & (energy <= fit_e_max)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Borexino 2D exclusion scan with pyhf")
    p.add_argument("--mh-min", type=float, default=2.0)
    p.add_argument("--mh-max", type=float, default=15.0)
    p.add_argument("--n-mh", type=int, default=14)

    p.add_argument("--u2-min", type=float, default=1e-6)
    p.add_argument("--u2-max", type=float, default=1e-1)
    p.add_argument("--n-u2", type=int, default=25)
    p.add_argument("--include-u2-zero", action="store_true")

    p.add_argument("--cl-threshold", type=float, default=2.71)
    p.add_argument("--use-mc-threshold", action="store_true")
    p.add_argument("--n-toys", type=int, default=200)
    p.add_argument("--seed", type=int, default=12345)

    p.add_argument("--n-workers", type=int, default=8)
    p.add_argument("--outdir", default="./plots/borexino/fit")
    p.add_argument("--out-prefix", default="exclusion2d_pyhf")
    return p.parse_args()


def load_borexino_data() -> np.ndarray:
    data = np.loadtxt("./data/borexino_data.csv", delimiter=",", skiprows=1)
    return data[:, 1]


def get_signal_template(spectrum_orig: np.ndarray, mH: float, u2: float) -> np.ndarray:
    if u2 <= 0.0:
        return np.zeros(np.count_nonzero(fit_mask), dtype=float)

    spectrum_rhn = getRHNSpectrum(spectrum_orig, mH, u2)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mH, u2)

    s_bin = np.nan_to_num(
        diff_Eee_decayed[:, 1] * exposure * estep,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    s_bin = np.clip(s_bin, 0.0, None)
    s_bin = apply_energy_resolution_convolution(
        s_bin, diff_Eee_decayed[:, 0], frac_resolution=0.05,
    )
    return s_bin[fit_mask]


def fit_xb_with_pyhf(data: np.ndarray, bkg: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    data = np.asarray(data, dtype=float)
    bkg = np.clip(np.asarray(bkg, dtype=float), 1e-12, None)
    sig = np.clip(np.asarray(sig, dtype=float), 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "borexino",
                "samples": [
                    {"name": "signal", "data": sig.tolist(), "modifiers": []},
                    {
                        "name": "background",
                        "data": bkg.tolist(),
                        "modifiers": [{"name": "xb", "type": "normfactor", "data": None}],
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
                    "parameters": [{"name": "xb", "inits": [1.0], "bounds": [[0.3, 3.0]]}],
                },
            }
        ],
    }

    ws = pyhf.Workspace(spec)
    model = ws.model()
    data_full = ws.data(model)
    bestfit_pars, twice_nll = pyhf.infer.mle.fit(data_full, model, return_fitted_val=True)

    xb_hat = float(bestfit_pars[model.config.par_order.index("xb")])
    nll_hat = 0.5 * float(twice_nll)
    return xb_hat, nll_hat


def profile_scan_mh(
    spectrum_orig: np.ndarray,
    data: np.ndarray,
    bkg: np.ndarray,
    mH: float,
    u2_values: np.ndarray,
) -> list[dict]:
    rows = []
    for u2 in np.asarray(u2_values, dtype=float):
        s = get_signal_template(spectrum_orig, mH, u2)
        xb_hat, nll_hat = fit_xb_with_pyhf(data, bkg, s)
        rows.append({"mH": float(mH), "u2": float(u2), "xb": xb_hat, "nll": nll_hat})

    nll = np.array([r["nll"] for r in rows], dtype=float)
    nll_min = float(np.min(nll))
    for r in rows:
        r["delta_chi2"] = 2.0 * (r["nll"] - nll_min)
    return rows


def infer_excluded_bounds(
    u2_values: np.ndarray,
    delta_chi2: np.ndarray,
    threshold: float,
    crossings: list[float],
) -> tuple[float, float]:
    """Infer excluded interval [low, high] from crossings and endpoint status."""
    x = np.asarray(u2_values, dtype=float)
    y = np.asarray(delta_chi2, dtype=float) - float(threshold)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if len(crossings) == 0:
        return np.nan, np.nan

    if len(crossings) >= 2:
        return float(crossings[0]), float(crossings[-1])

    c = float(crossings[0])
    left_excluded = bool(y[0] >= 0.0)
    right_excluded = bool(y[-1] >= 0.0)

    if right_excluded and not left_excluded:
        # excluded region is [c, x_max]
        return c, float(x[-1])
    if left_excluded and not right_excluded:
        # excluded region is [x_min, c]
        return float(x[0]), c

    # ambiguous edge case: tangent / numerical tie
    return c, c


def run_one_mh_task(args_tuple) -> tuple[float, list[dict], list[float], float, float]:
    """Worker task for one mH point."""
    if len(args_tuple) == 7:
        mH, u2_values, data, b_fit, spectrum_nuL_orig, cl_threshold, mc_config = args_tuple
        use_mc = mc_config.get("use_mc", False)
        n_toys = mc_config.get("n_toys", 200)
        seed = mc_config.get("seed", 12345)
    else:
        # backward compatibility
        mH, u2_values, data, b_fit, spectrum_nuL_orig, cl_threshold = args_tuple
        use_mc = False
        n_toys = 200
        seed = 12345

    rows = profile_scan_mh(
        spectrum_orig=spectrum_nuL_orig,
        data=data,
        bkg=b_fit,
        mH=float(mH),
        u2_values=u2_values,
    )
    dchi2 = np.array([r["delta_chi2"] for r in rows], dtype=float)
    
    if use_mc:
        # Compute MC thresholds
        thresholds = compute_mc_thresholds(
            spectrum_orig=spectrum_nuL_orig,
            data=data,
            bkg=b_fit,
            mH=float(mH),
            u2_values=u2_values,
            n_toys=n_toys,
            seed=seed,
        )
        # Find crossings where dchi2 >= thresholds (i.e., dchi2 - thresholds >= 0)
        diff = dchi2 - thresholds
        crossings = find_u2_crossings(u2_values, diff, threshold=0.0)
        low, high = infer_excluded_bounds(u2_values, diff, 0.0, crossings)
    else:
        crossings = find_u2_crossings(u2_values, dchi2, threshold=cl_threshold)
        low, high = infer_excluded_bounds(u2_values, dchi2, cl_threshold, crossings)

    return float(mH), rows, crossings, float(low), float(high)


def find_u2_crossings(u2_values: np.ndarray, delta_chi2: np.ndarray, threshold: float = 2.71) -> list[float]:
    x = np.asarray(u2_values, dtype=float)
    y = np.asarray(delta_chi2, dtype=float) - float(threshold)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    crossings = []
    for i in range(len(x) - 1):
        y1, y2 = y[i], y[i + 1]
        x1, x2 = x[i], x[i + 1]

        if y1 == 0.0:
            crossings.append(float(x1))
            continue

        if y1 * y2 < 0.0:
            xc = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            crossings.append(float(xc))

    # Deduplicate (can happen if exact points lie on threshold)
    crossings = sorted(crossings)
    uniq = []
    for c in crossings:
        if len(uniq) == 0 or not np.isclose(c, uniq[-1], rtol=1e-10, atol=0.0):
            uniq.append(c)
    return uniq


def compute_mc_thresholds(
    spectrum_orig: np.ndarray,
    data: np.ndarray,
    bkg: np.ndarray,
    mH: float,
    u2_values: np.ndarray,
    n_toys: int = 200,
    seed: int = 12345,
) -> np.ndarray:
    """
    Compute MC thresholds (90% quantile of Δχ²) for each u2 point.
    
    Strategy:
    1. Fit background-only (u2=0) to get best-fit xb and expected counts.
    2. Generate n_toys Poisson toys from the background-only expectation.
    3. For each toy, perform full u2 scan and compute Δχ²(u2) profile.
    4. For each u2 point, collect Δχ² values from all toys, compute 90% quantile.
    5. Return thresholds array (same length as u2_values).
    """
    # Combine seed with mH to get different sequences for different masses
    local_seed = seed + int(mH * 1000)
    np.random.seed(local_seed)
    
    # 1. Background-only fit
    s_zero = get_signal_template(spectrum_orig, mH, 0.0)
    xb_b, _ = fit_xb_with_pyhf(data, bkg, s_zero)
    mu_bkg_only = xb_b * bkg  # expected counts under background-only
    
    # Precompute signal templates for all u2 values
    s_templates = {}
    for u2 in u2_values:
        s_templates[u2] = get_signal_template(spectrum_orig, mH, u2)
    
    # 2. Generate toy datasets
    thresholds = np.zeros(len(u2_values), dtype=float)
    
    # We'll collect Δχ² values for each u2 point
    # Initialize list of lists
    dchi2_collector = [[] for _ in range(len(u2_values))]
    
    print(f"[MC] mH={mH:.2f} MeV: generating {n_toys} toys from background-only...")
    
    for _ in range(n_toys):
        # Generate Poisson toy
        toy_data = np.random.poisson(mu_bkg_only)
        
        # Scan u2_values for this toy
        toy_rows = []
        for i, u2 in enumerate(u2_values):
            s = s_templates[u2]
            xb_hat, nll_hat = fit_xb_with_pyhf(toy_data, bkg, s)
            toy_rows.append({"u2": u2, "nll": nll_hat, "xb": xb_hat})
        
        # Compute Δχ² for this toy
        nll_vals = np.array([r["nll"] for r in toy_rows], dtype=float)
        nll_min = float(np.min(nll_vals))
        for i, r in enumerate(toy_rows):
            dchi2 = 2.0 * (r["nll"] - nll_min)
            dchi2_collector[i].append(dchi2)
    
    # 4. Compute 90% quantile for each u2 point
    for i in range(len(u2_values)):
        if dchi2_collector[i]:
            thresholds[i] = np.percentile(dchi2_collector[i], 90.0)
        else:
            thresholds[i] = 2.71  # fallback to asymptotic
    
    print(f"[MC] mH={mH:.2f} MeV: thresholds computed")
    return thresholds


def plot_exclusion_2d(
    mh_values: np.ndarray,
    u2_low: np.ndarray,
    u2_high: np.ndarray,
    u2_min: float,
    u2_max: float,
    outpath: str,
    use_mc: bool = False,
) -> None:
    mh_values = np.asarray(mh_values, dtype=float)
    u2_low = np.asarray(u2_low, dtype=float)
    u2_high = np.asarray(u2_high, dtype=float)

    plt.figure(figsize=(7.2, 5.2))

    valid_low = np.isfinite(u2_low)
    valid_high = np.isfinite(u2_high)

    if np.any(valid_low):
        plt.plot(mh_values[valid_low], u2_low[valid_low], "-", lw=2, ms=4, color="tab:blue", label="lower crossing")
    if np.any(valid_high):
        plt.plot(mh_values[valid_high], u2_high[valid_high], "-", lw=2, ms=4, color="tab:purple", label="upper crossing")

    both = valid_low & valid_high
    if np.any(both):
        plt.fill_between(
            mh_values[both],
            u2_low[both],
            u2_high[both],
            color="tab:red",
            alpha=0.18,
            label="excluded band",
        )

    # if np.any(~valid_low & ~valid_high):
    #     plt.scatter(
    #         mh_values[~valid_low & ~valid_high],
    #         np.full(np.sum(~valid_low & ~valid_high), np.sqrt(u2_min * u2_max)),
    #         marker="x",
    #         color="gray",
    #         label="no crossing",
    #     )

    plt.yscale("log")
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    if use_mc:
        plt.title("Borexino 2D exclusion (pyhf, MC calibrated threshold)")
    else:
        plt.title("Borexino 2D exclusion (pyhf, asymptotic threshold)")
    plt.ylim(u2_min, u2_max)
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_borexino_original_exclusion():
    ex_array = np.loadtxt('./data/Borexino_exclusion.csv')



def main() -> None:
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    mh_values = np.linspace(args.mh_min, args.mh_max, args.n_mh)
    u2_values = np.logspace(np.log10(args.u2_min), np.log10(args.u2_max), args.n_u2)
    if not np.any(np.isclose(u2_values, 0.0)):
        u2_values = np.insert(u2_values, 0, 0.0)

    data = load_borexino_data()
    _, bg = load_background(energy=energy, estep=estep, energy_resolution=None)
    b_fit = bg[fit_mask]

    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)

    all_rows = []
    crossing_map = {}
    bound_map = {}

    # Prepare MC configuration if needed
    mc_config = {}
    if args.use_mc_threshold:
        mc_config = {
            "use_mc": True,
            "n_toys": args.n_toys,
            "seed": args.seed,
        }
    
    tasks = []
    for mH in mh_values:
        task = (
            float(mH),
            u2_values,
            data,
            b_fit,
            spectrum_nuL_orig,
            float(args.cl_threshold),
        )
        if args.use_mc_threshold:
            # Append MC config as 7th element
            task = task + (mc_config,)
        tasks.append(task)

    # Adjust output prefix for MC results
    out_prefix = args.out_prefix
    if args.use_mc_threshold:
        out_prefix = f"{out_prefix}_mc"

    n_workers = max(1, int(args.n_workers))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(run_one_mh_task, t) for t in tasks]
        for fut in as_completed(futures):
            mH_val, rows, crossings, low, high = fut.result()
            all_rows.extend(rows)
            crossing_map[mH_val] = crossings
            bound_map[mH_val] = (low, high)

            if len(crossings) == 0:
                print(f"mH={mH_val:.2f} MeV: no crossing at delta_chi2={args.cl_threshold:.2f}")
            elif len(crossings) == 1:
                print(f"mH={mH_val:.2f} MeV: crossing={crossings[0]:.4e}, excluded=[{low:.4e}, {high:.4e}]")
            else:
                print(f"mH={mH_val:.2f} MeV: crossings={crossings[0]:.4e}, {crossings[-1]:.4e}")

    u2_low = np.array([bound_map.get(float(mH), (np.nan, np.nan))[0] for mH in mh_values], dtype=float)
    u2_high = np.array([bound_map.get(float(mH), (np.nan, np.nan))[1] for mH in mh_values], dtype=float)

    # Save boundary table
    boundary_path = os.path.join(args.outdir, f"{out_prefix}_boundary.csv")
    with open(boundary_path, "w", encoding="utf-8") as f:
        f.write("mH,u2_limit_low,u2_limit_high\n")
        for mH, lo, hi in zip(mh_values, u2_low, u2_high):
            lo_str = "nan" if not np.isfinite(lo) else f"{lo:.8e}"
            hi_str = "nan" if not np.isfinite(hi) else f"{hi:.8e}"
            f.write(f"{mH:.6f},{lo_str},{hi_str}\n")

    # Save full scan grid
    grid_path = os.path.join(args.outdir, f"{out_prefix}_grid.csv")
    with open(grid_path, "w", encoding="utf-8") as f:
        f.write("mH,u2,xb,nll,delta_chi2\n")
        for r in all_rows:
            f.write(
                f"{r['mH']:.6f},{r['u2']:.8e},{r['xb']:.8f},{r['nll']:.8f},{r['delta_chi2']:.8f}\n"
            )

    fig_path = os.path.join(args.outdir, f"{out_prefix}.pdf")
    plot_exclusion_2d(mh_values, u2_low, u2_high, args.u2_min, args.u2_max, fig_path, use_mc=args.use_mc_threshold)

    print(f"Saved boundary: {boundary_path}")
    print(f"Saved grid:     {grid_path}")
    print(f"Saved figure:   {fig_path}")


if __name__ == "__main__":
    main()
