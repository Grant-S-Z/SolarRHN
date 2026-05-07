import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import matplotlib.pyplot as plt
import pyhf

from core import *
from workflows import getNuleeInDetector


# Constants
estep: float = 0.2
e_min: float = 0.0
e_max: float = 16.0
fit_e_min: float = 4.8
fit_e_max: float = 12.8
n_all = int((e_max - e_min) / estep) + 1
energy = np.linspace(e_min, e_max, n_all)
fit_mask = (energy >= fit_e_min) & (energy <= fit_e_max)
n_workers = 8


def load_borexino_data():
    data = np.loadtxt("./data/borexino_data.csv", delimiter=",", skiprows=1)
    fit_data = np.loadtxt("./data/borexino_fit.csv", delimiter=",", skiprows=1)

    # CSV columns: energy, b8, be11, signal
    fit_energy = fit_data[:, 0]
    fit_bkg_b8 = fit_data[:, 1]
    fit_bkg_be11 = fit_data[:, 2]
    fit_signal = fit_data[:, 3]

    # Interpolate fit templates onto the global analysis energy grid
    bkg_b8 = np.interp(energy, fit_energy, fit_bkg_b8, left=0.0, right=0.0)
    bkg_be11 = np.interp(energy, fit_energy, fit_bkg_be11, left=0.0, right=0.0)
    signal = np.interp(energy, fit_energy, fit_signal, left=0.0, right=0.0)

    return data, fit_energy, bkg_b8, bkg_be11, signal


def poisson_deviance(data: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance (Baker-Cousins) for goodness-of-fit."""
    data = np.asarray(data, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)

    term = np.zeros_like(mu, dtype=float)
    pos = data > 0.0
    term[pos] = data[pos] * np.log(data[pos] / mu[pos])
    dev = 2.0 * np.sum(mu - data + term)
    return float(dev)


def fit(
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    signal: np.ndarray,
) -> dict:
    """Fit x_b8, x_be11, x_h for one signal template."""
    data = np.asarray(data, dtype=float)
    bkg_b8 = np.asarray(bkg_b8, dtype=float)
    bkg_be11 = np.asarray(bkg_be11, dtype=float)
    signal = np.asarray(signal, dtype=float)

    b8_safe = np.clip(bkg_b8, 1e-12, None)
    be11_safe = np.clip(bkg_be11, 1e-12, None)
    sig_safe = np.clip(signal, 1e-12, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "borexino",
                "samples": [
                    {
                        "name": "b8",
                        "data": b8_safe.tolist(),
                        "modifiers": [{"name": "x_b8", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "be11",
                        "data": be11_safe.tolist(),
                        "modifiers": [{"name": "x_be11", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "signal",
                        "data": sig_safe.tolist(),
                        "modifiers": [{"name": "x_h", "type": "normfactor", "data": None}],
                    },
                ],
            }
        ],
        "observations": [{"name": "borexino", "data": data.tolist()}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "x_h",
                    "parameters": [
                        {"name": "x_b8", "inits": [1.0], "bounds": [[0.8, 1.0]]}, # BS05(AGS, OP) -> BS05(OP)
                        {"name": "x_be11", "inits": [1.0], "bounds": [[0.0, 10.0]]},
                        {"name": "x_h", "inits": [1.0], "bounds": [[0.0, 10.0]]},
                    ],
                },
            }
        ],
    }

    ws = pyhf.Workspace(spec)
    model = ws.model()
    data_full = ws.data(model)

    bestfit_pars, twice_nll = pyhf.infer.mle.fit(
        data_full,
        model,
        return_fitted_val=True,
    )

    x_b8 = float(bestfit_pars[model.config.par_order.index("x_b8")])
    x_be11 = float(bestfit_pars[model.config.par_order.index("x_be11")])
    x_h = float(bestfit_pars[model.config.par_order.index("x_h")])
    nll_hat = 0.5 * float(twice_nll)

    mu_hat = x_b8 * bkg_b8 + x_be11 * bkg_be11 + x_h * signal
    n_par = 3
    ndf = int(data.size) - n_par
    chi2_dev = poisson_deviance(data, mu_hat)
    chi2_ndf = chi2_dev / ndf if ndf > 0 else np.nan

    result = {
        "x_b8": x_b8,
        "x_be11": x_be11,
        "x_h": x_h,
        "nll": nll_hat,
        "chi2_dev": chi2_dev,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "mu": mu_hat,
        "bkg_b8": bkg_b8,
        "bkg_be11": bkg_be11,
        "signal": signal,
    }

    print(
        f"x_b8={x_b8:.4f}, x_be11={x_be11:.4f}, x_h={x_h:.4f}, "
        f"NLL={nll_hat:.4f}, chi2/ndf={chi2_dev:.2f}/{ndf}={chi2_ndf:.3f}"
    )
    return result


def fit_fixed_xh(
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    signal_ref: np.ndarray,
    x_h: float,
) -> dict:
    """Fit x_b8 and x_be11 with x_h fixed."""
    data = np.asarray(data, dtype=float)
    bkg_b8 = np.asarray(bkg_b8, dtype=float)
    bkg_be11 = np.asarray(bkg_be11, dtype=float)
    signal_ref = np.asarray(signal_ref, dtype=float)

    s = np.clip(signal_ref * float(x_h), 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "borexino",
                "samples": [
                    {
                        "name": "b8",
                        "data": np.clip(bkg_b8, 1e-12, None).tolist(),
                        "modifiers": [{"name": "x_b8", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "be11",
                        "data": np.clip(bkg_be11, 1e-12, None).tolist(),
                        "modifiers": [{"name": "x_be11", "type": "normfactor", "data": None}],
                    },
                    {
                        "name": "signal",
                        "data": s.tolist(),
                        "modifiers": [],
                    },
                ],
            }
        ],
        "observations": [{"name": "borexino", "data": data.tolist()}],
        "measurements": [
            {
                "name": "Measurement",
                "config": {
                    "poi": "x_b8",
                    "parameters": [
                        {"name": "x_b8", "inits": [1.0], "bounds": [[0.8, 1.0]]},
                        {"name": "x_be11", "inits": [1.0], "bounds": [[0.0, 10.0]]},
                    ],
                },
            }
        ],
    }

    ws = pyhf.Workspace(spec)
    model = ws.model()
    data_full = ws.data(model)

    bestfit_pars, twice_nll = pyhf.infer.mle.fit(
        data_full,
        model,
        return_fitted_val=True,
    )

    x_b8_hat = float(bestfit_pars[model.config.par_order.index("x_b8")])
    x_be11_hat = float(bestfit_pars[model.config.par_order.index("x_be11")])
    nll_hat = 0.5 * float(twice_nll)

    n_par = 2
    ndf = int(data.size) - n_par
    mu_hat = x_b8_hat * bkg_b8 + x_be11_hat * bkg_be11 + s
    chi2_dev = poisson_deviance(data, mu_hat)
    chi2_ndf = chi2_dev / ndf if ndf > 0 else np.nan

    print(
        f"[fixed-x_h] x_h={x_h:.4e}, x_b8={x_b8_hat:.4f}, x_be11={x_be11_hat:.4f}, "
        f"NLL={nll_hat:.4f}, chi2/ndf={chi2_dev:.2f}/{ndf}={chi2_ndf:.3f}"
    )

    return {
        "x_h": float(x_h),
        "x_b8": x_b8_hat,
        "x_be11": x_be11_hat,
        "nll": nll_hat,
        "chi2_dev": chi2_dev,
        "ndf": ndf,
        "chi2_ndf": chi2_ndf,
        "mu": mu_hat,
        "signal": s,
    }


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
    return s_bin[fit_mask]


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

    crossings = sorted(crossings)
    uniq = []
    for c in crossings:
        if len(uniq) == 0 or not np.isclose(c, uniq[-1], rtol=1e-10, atol=0.0):
            uniq.append(c)
    return uniq


def profile_likelihood_scan(
    signal_ref: np.ndarray,
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    mH: float,
    x_h_array: np.ndarray,
) -> list[dict]:
    results = []

    for x_h in np.asarray(x_h_array, dtype=float):
        result = fit_fixed_xh(data, bkg_b8, bkg_be11, signal_ref, x_h)

        results.append(
            {
                "mH": float(mH),
                "x_h": float(x_h),
                "x_b8": float(result["x_b8"]),
                "x_be11": float(result["x_be11"]),
                "nll": float(result["nll"]),
                "chi2_dev": float(result["chi2_dev"]),
                "ndf": int(result["ndf"]),
                "chi2_ndf": float(result["chi2_ndf"]),
            }
        )

    nll_array = np.array([r["nll"] for r in results], dtype=float)
    nll_min = np.min(nll_array)
    for r in results:
        dchi2 = 2.0 * (r["nll"] - nll_min)
        r["delta_chi2"] = float(dchi2)
        r["excluded_90"] = bool(dchi2 > 2.71)
        r["excluded_95"] = bool(dchi2 > 3.84)

    return results


def profile_likelihood_scan_u2(
    spectrum_orig: np.ndarray,
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    mH: float,
    u2_array: np.ndarray,
    u2_ref: float,
) -> list[dict]:
    signal_ref = get_signal_template(spectrum_orig, mH, u2_ref)
    u4_ref = float(u2_ref) ** 2

    rows = []
    for u2 in np.asarray(u2_array, dtype=float):
        u4 = float(u2) ** 2
        x_h = 0.0 if u4_ref <= 0.0 else u4 / u4_ref # fix x_h corresponding to u2 and u2_ref
        r = fit_fixed_xh(data, bkg_b8, bkg_be11, signal_ref, x_h)
        rows.append(
            {
                "mH": float(mH),
                "u2": float(u2),
                "u4": u4,
                "x_h": x_h,
                "x_b8": float(r["x_b8"]),
                "x_be11": float(r["x_be11"]),
                "nll": float(r["nll"]),
                "chi2_dev": float(r["chi2_dev"]),
                "ndf": int(r["ndf"]),
                "chi2_ndf": float(r["chi2_ndf"]),
            }
        )

    nll_array = np.array([r["nll"] for r in rows], dtype=float)
    nll_min = np.min(nll_array)
    for r in rows:
        dchi2 = 2.0 * (r["nll"] - nll_min)
        r["delta_chi2"] = float(dchi2)

    return rows


def _run_one_mh_task(args_tuple) -> tuple[float, float, list[dict]]:
    spectrum_orig, data, bkg_b8, bkg_be11, mH, u2_array, u2_ref, cl_threshold = args_tuple

    rows = profile_likelihood_scan_u2(
        spectrum_orig=spectrum_orig,
        data=data,
        bkg_b8=bkg_b8,
        bkg_be11=bkg_be11,
        mH=float(mH),
        u2_array=u2_array,
        u2_ref=u2_ref,
    )

    dchi2 = np.array([r["delta_chi2"] for r in rows], dtype=float)
    crossings = find_u2_crossings(np.asarray(u2_array, dtype=float), dchi2, threshold=cl_threshold)

    u2_limit = np.nan
    if len(crossings) > 0:
        u2_limit = float(crossings[-1])

    return float(mH), float(u2_limit), rows


def run_mh_scan_exclusion(
    spectrum_orig: np.ndarray,
    data: np.ndarray,
    bkg_b8: np.ndarray,
    bkg_be11: np.ndarray,
    mh_array: np.ndarray,
    u2_array: np.ndarray,
    u2_ref: float,
    cl_threshold: float = 2.71,
    n_workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    mh_vals = np.asarray(mh_array, dtype=float)
    u2_limits = np.full_like(mh_vals, np.nan, dtype=float)
    all_rows = []

    tasks = [
        (
            spectrum_orig,
            data,
            bkg_b8,
            bkg_be11,
            float(mH),
            np.asarray(u2_array, dtype=float),
            float(u2_ref),
            float(cl_threshold),
        )
        for mH in mh_vals
    ]

    if int(n_workers) <= 1:
        for i, task in enumerate(tasks):
            mH_val, u2_limit, rows = _run_one_mh_task(task)
            u2_limits[i] = u2_limit
            all_rows.extend(rows)
            if np.isfinite(u2_limit):
                print(f"[2D exclusion] mH={mH_val:.3f} MeV -> u2_90={u2_limit:.4e}")
            else:
                print(f"[2D exclusion] mH={mH_val:.3f} MeV -> no crossing at dchi2={cl_threshold:.2f}")
        return mh_vals, u2_limits, all_rows

    n_workers = max(1, int(n_workers))
    idx_map = {float(mh): i for i, mh in enumerate(mh_vals)}

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = [ex.submit(_run_one_mh_task, task) for task in tasks]
        for fut in as_completed(futures):
            mH_val, u2_limit, rows = fut.result()
            i = idx_map[float(mH_val)]
            u2_limits[i] = u2_limit
            all_rows.extend(rows)

            if np.isfinite(u2_limit):
                print(f"[2D exclusion] mH={mH_val:.3f} MeV -> u2_90={u2_limit:.4e}")
            else:
                print(f"[2D exclusion] mH={mH_val:.3f} MeV -> no crossing at dchi2={cl_threshold:.2f}")

    return mh_vals, u2_limits, all_rows


def plot_fit_result(result: dict, data: np.ndarray, dir: str):
    x_b8 = float(result["x_b8"])
    x_be11 = float(result["x_be11"])
    x_h = float(result["x_h"])

    b8 = np.asarray(result["bkg_b8"], dtype=float)
    be11 = np.asarray(result["bkg_be11"], dtype=float)
    sig = np.asarray(result["signal"], dtype=float)
    data_fit = np.asarray(data, dtype=float)

    comp_b8 = x_b8 * b8
    comp_be11 = x_be11 * be11
    comp_sig = x_h * sig
    total = comp_b8 + comp_be11 + comp_sig

    x = energy[fit_mask]
    x_edges = np.concatenate(([x[0] - 0.5 * estep], x + 0.5 * estep))

    plt.figure(figsize=(8, 5.5))
    plt.stairs(data_fit, x_edges, color="black", lw=1.0, label="Data (hist)")
    plt.plot(x, total, "-", lw=2.2, color="tab:blue", label="Best fit total")
    plt.plot(x, b8, "--", lw=2, color="tab:orange", label=r"$B8$")
    plt.plot(x, be11, "--", lw=2, color="tab:purple", label=r"$Be11$")
    plt.plot(x, sig, "--", lw=2, color="tab:green", label=r"$S$")

    plt.xlabel("Energy (MeV)")
    plt.ylabel("Counts / bin")
    plt.xlim(4.8, 12.8)
    plt.title(
        f"Borexino fit: x_B8={x_b8:.3f}, x_Be11={x_be11:.3f}, x_H={x_h:.3f}, chi2/ndf={result['chi2_ndf']:.3f}"
    )
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/fit_optimal.pdf")
    plt.close()


def plot_nll_curve(u4_values, nll, mH, dir):
    u4_values = np.asarray(u4_values, dtype=float)
    nll = np.asarray(nll, dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(u4_values, nll, "o-", lw=2, ms=5, label="NLL")

    i_min = int(np.argmin(nll))
    plt.scatter(
        u4_values[i_min],
        nll[i_min],
        color="tab:red",
        zorder=3,
        label=f"Min: {nll[i_min]:.3f} at {u4_values[i_min]:.2e}",
    )

    plt.xscale("log")
    plt.xlabel(r"$|U_{eH}|^4$")
    plt.ylabel("NLL")
    plt.title(f"Borexino NLL Scan (pyhf, mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/nll_pyhf_u4_mH_{mH:g}.pdf")
    plt.close()


def plot_profile_curve(u4_values, delta_chi2, mH, cl_crit=2.71, dir='./plots/borexino/fit'):
    """Plot Δχ² profile and mark exclusion crossing at cl_crit."""
    u4 = np.asarray(u4_values, dtype=float)
    dchi2 = np.asarray(delta_chi2, dtype=float)

    order = np.argsort(u4)
    u4 = u4[order]
    dchi2 = dchi2[order]

    # Find crossing at cl_crit
    du = dchi2 - float(cl_crit)
    u4_cross = np.nan
    for i in range(len(u4) - 1):
        y1, y2 = du[i], du[i + 1]
        if y1 == 0.0:
            u4_cross = u4[i]
            break
        if y1 * y2 < 0.0:
            x1, x2 = u4[i], u4[i + 1]
            u4_cross = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            break

    plt.figure(figsize=(7, 5))
    plt.plot(u4, dchi2, "o-", lw=2, ms=5, label=r"$\Delta\chi^2=-2\Delta\ln\mathcal{L}$")

    plt.axhline(2.71, color="tab:orange", ls="--", lw=1.5, label="90% C.L. (1 dof)")
    plt.axhline(3.84, color="tab:red", ls="--", lw=1.5, label="95% C.L. (1 dof)")

    if np.isfinite(u4_cross):
        plt.axvline(u4_cross, color='tab:red', ls='--', lw=1.5, label=fr'Exclusion: $U^4_{{90}}={u4_cross:.2e}$')
        mask_excl = u4 >= u4_cross
        if np.any(mask_excl):
            plt.fill_between(u4[mask_excl], dchi2[mask_excl], cl_crit, color='tab:red', alpha=0.18)
        print(f'[exclusion] mH={mH:g} MeV, U4_90 = {u4_cross:.4e}')
    else:
        print(f'[exclusion] mH={mH:g} MeV, no crossing found for threshold {cl_crit:.2f}')

    plt.xscale("log")
    plt.xlabel(r"$|U_{eH}|^4$")
    plt.ylabel(r"$\Delta\chi^2$")

    plt.title(f"Borexino Profile Likelihood (pyhf, mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/profile_pyhf_u4_mH_{mH:g}.pdf")
    plt.close()


def plot_likelihood_ratio_curve(u4_values, nll, mH, dir='./plots/borexino/fit'):
    """Plot normalized likelihood L/Lmax as a function of U^4.

    Also mark U4_90 defined by right-tail area = 10% of total area.
    """
    u4_values = np.asarray(u4_values, dtype=float)
    nll = np.asarray(nll, dtype=float)

    nll_min = float(np.min(nll))
    ll = np.exp(-(nll - nll_min))
    ll_norm = ll / np.max(ll)

    order = np.argsort(u4_values)
    u4 = u4_values[order]
    y = ll_norm[order]

    # Compute area-based 90% CL: integral_{u4_90}^{max} y du4 = 10% * integral y du4
    total_area = float(np.trapezoid(y, u4))
    u4_90 = np.nan
    if total_area > 0.0 and len(u4) >= 2:
        tail_area = np.zeros_like(u4)
        for i in range(len(u4) - 1, -1, -1):
            tail_area[i] = float(np.trapezoid(y[i:], u4[i:])) if i < len(u4) - 1 else 0.0

        frac = tail_area / total_area
        target = 0.10

        if np.any(frac <= target):
            i_hi = int(np.argmax(frac <= target))
            if i_hi == 0:
                u4_90 = u4[0]
            else:
                i_lo = i_hi - 1
                f1, f2 = frac[i_lo], frac[i_hi]
                x1, x2 = u4[i_lo], u4[i_hi]
                if f1 == f2:
                    u4_90 = x2
                else:
                    u4_90 = x1 + (target - f1) * (x2 - x1) / (f2 - f1)

    plt.figure(figsize=(7, 5))
    plt.plot(u4, y, "o-", lw=2, ms=5, color="tab:green", label=r"$L/L_{\max}$")

    i_max = int(np.argmax(y))
    plt.scatter(
        u4[i_max],
        y[i_max],
        color="tab:red",
        zorder=3,
        label=f"Max at {u4[i_max]:.2e}",
    )

    if np.isfinite(u4_90):
        plt.axvline(u4_90, color="tab:purple", ls="--", lw=1.5, label=fr"Area 90% C.L.: $U^4_{{90}}={u4_90:.2e}$")
        mask_right = u4 >= u4_90
        if np.any(mask_right):
            plt.fill_between(u4[mask_right], y[mask_right], 0.0, color="tab:purple", alpha=0.15)
        print(f"[likelihood-area] mH={mH:g} MeV, U4_90(area)= {u4_90:.4e}")

    plt.xscale("log")
    plt.ylim(0.0, 1.05)
    plt.xlabel(r"$|U_{eH}|^4$")
    plt.ylabel(r"$L/L_{\max}$")
    plt.title(f"Borexino Normalized Likelihood (pyhf, mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(dir, exist_ok=True)
    plt.savefig(f"{dir}/likelihood_norm_pyhf_u4_mH_{mH:g}.pdf")
    plt.close()


def plot_exclusion_2d_u2(
    mh_values: np.ndarray,
    u2_limits: np.ndarray,
    outpath: str,
    borexino_ref_csv: str = "./data/Borexino_exclusion.csv",
):
    mh = np.asarray(mh_values, dtype=float)
    u2 = np.asarray(u2_limits, dtype=float)

    valid = np.isfinite(u2) & (u2 > 0.0)

    plt.figure(figsize=(7.2, 5.2))
    if np.any(valid):
        plt.plot(mh[valid], u2[valid], "o-", lw=2, ms=4, color="tab:red", label=r"This work (90% C.L.)")
        plt.fill_between(mh[valid], u2[valid], np.max(u2[valid]), color="tab:red", alpha=0.15, label="excluded")

    # Overlay Borexino published exclusion: columns are log10(mH/GeV), log10(u2)
    if os.path.exists(borexino_ref_csv):
        ref = np.loadtxt(borexino_ref_csv, delimiter=",")
        if ref.ndim == 2 and ref.shape[1] >= 2:
            log10_mh_gev = ref[:, 0]
            log10_u2 = ref[:, 1]

            # Split at the deepest point into two branches, plot each in original order
            i_min = int(np.argmin(log10_u2))
            branches = [
                slice(0, i_min + 1),
                slice(i_min, None),
            ]

            first = True
            for sl in branches:
                bx = log10_mh_gev[sl]
                by = log10_u2[sl]
                if bx.size < 2:
                    continue
                mh_mev = (10.0 ** bx) * 1e3
                u2 = 10.0 ** by

                plt.plot(
                    mh_mev,
                    u2,
                    "-",
                    lw=2.0,
                    color="tab:blue",
                    label="Borexino (published)" if first else None,
                )
                first = False

    plt.yscale("log")
    plt.xlabel(r"$m_H$ [MeV]")
    plt.ylabel(r"$|U_{eH}|^2$")
    plt.title("Borexino 2D exclusion (profile likelihood, 90% C.L.)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def plot_profile_curve_u2(u2_values, delta_chi2, mH, outpath):
    u2_values = np.asarray(u2_values, dtype=float)
    delta_chi2 = np.asarray(delta_chi2, dtype=float)

    order = np.argsort(u2_values)
    u2_values = u2_values[order]
    delta_chi2 = delta_chi2[order]

    plt.figure(figsize=(7, 5))
    plt.plot(
        u2_values,
        delta_chi2,
        "o-",
        lw=2,
        ms=4,
        label=r"$\Delta\chi^2=-2\Delta\ln\mathcal{L}$",
    )
    plt.axhline(2.71, color="tab:orange", ls="--", lw=1.5, label="90% C.L. (1 dof)")
    plt.axhline(3.84, color="tab:red", ls="--", lw=1.5, label="95% C.L. (1 dof)")

    plt.xscale("log")
    plt.xlabel(r"$|U_{eH}|^2$")
    plt.ylabel(r"$\Delta\chi^2$")
    plt.title(f"Borexino profile likelihood (mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath)
    plt.close()


def main():
    # Borexino data + bkg + signal templates from file
    exp_data, fit_energy, bkg_b8, bkg_be11, signal_ref_raw = load_borexino_data()

    data = np.asarray(exp_data[:, 1], dtype=float)
    bkg_b8 = np.asarray(bkg_b8[fit_mask], dtype=float)
    bkg_be11 = np.asarray(bkg_be11[fit_mask], dtype=float)

    # # Use Borexino signal as one reference simulated template at u2_ref.
    # # Treat this template as the reference normalization; x_h only rescales its amplitude.
    # signal_ref = np.interp(energy, fit_energy, signal_ref_raw, left=0.0, right=0.0)
    # sig_ref = np.asarray(signal_ref[fit_mask], dtype=float)

    # result = fit(data, bkg_b8, bkg_be11, sig_ref)
    # plot_fit_result(result, data)

    # # If signal yield scales linearly with u2, convert fitted x_h to effective u2.
    # mH = 8
    # u2_ref = 8e-6
    # u4_ref = 8e-6**2
    # u4_eff = float(result["x_h"]) * u4_ref
    # u2_eff = u4_ref**0.5

    # print(
    #     f"[fit @ u2_ref={u2_ref:.1e}] x_b8={result['x_b8']:.4f}, x_be11={result['x_be11']:.4f}, "
    #     f"x_h={result['x_h']:.4f}, u2_eff={u2_eff:.4e}, NLL={result['nll']:.4f}, chi2/ndf={result['chi2_ndf']:.3f}"
    # )

    # x_h_array = np.linspace(0.0, 3.5, 36)
    # results = profile_likelihood_scan(sig_ref, data, bkg_b8, bkg_be11, mH, x_h_array)

    # u4_array = np.asarray(x_h_array, dtype=float) * u4_ref
    # nll_array = np.array([r["nll"] for r in results], dtype=float)
    # delta_chi2_array = np.array([r["delta_chi2"] for r in results], dtype=float)

    # plot_nll_curve(u4_array, nll_array, mH)
    # plot_likelihood_ratio_curve(u4_array, nll_array, mH)
    # plot_profile_curve(u4_array, delta_chi2_array, mH)
    # plot_profile_curve(u4_array, delta_chi2_array, mH, cl_crit=2.71)

    # Use My signal as reference
    dir = './plots/borexino/fit/my_signal'
    print(">>> Loading 8B neutrino spectrum from csv file...")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)

    mH = 8
    u2_ref = 8e-6
    u4_ref = 8e-6**2
    sig_ref = get_signal_template(spectrum_nuL_orig, mH, u2_ref)

    result = fit(data, bkg_b8, bkg_be11, sig_ref)
    plot_fit_result(result, data, dir)

    x_h_array = np.linspace(0.0, 3.5, 36)
    results = profile_likelihood_scan(sig_ref, data, bkg_b8, bkg_be11, mH, x_h_array)

    u4_array = np.asarray(x_h_array, dtype=float) * u4_ref
    nll_array = np.array([r["nll"] for r in results], dtype=float)
    delta_chi2_array = np.array([r["delta_chi2"] for r in results], dtype=float)

    plot_nll_curve(u4_array, nll_array, mH, dir)
    plot_likelihood_ratio_curve(u4_array, nll_array, mH, dir)
    plot_profile_curve(u4_array, delta_chi2_array, mH, dir=dir)


    # Scan MH to get exclusion
    # Since MHs are different, use simulation signal templates for each (mH, u2)
    u2_ref = 1e-5

    # mh_array = np.linspace(2.0, 14.0, 76)
    mh_array = np.linspace(2.0, 14.0, 13)
    u2_array = np.logspace(-6, -3, 31)
    if not np.any(np.isclose(u2_array, 0.0)):
        u2_array = np.insert(u2_array, 0, 0.0)

    mh_vals, u2_limits_90, all_rows = run_mh_scan_exclusion(
        spectrum_orig=spectrum_nuL_orig,
        data=data,
        bkg_b8=bkg_b8,
        bkg_be11=bkg_be11,
        mh_array=mh_array,
        u2_array=u2_array,
        u2_ref=u2_ref,
        cl_threshold=2.71,
        n_workers=os.cpu_count() or 1,
    )

    os.makedirs("./plots/borexino/exclusion", exist_ok=True)
    plot_exclusion_2d_u2(
        mh_vals,
        u2_limits_90,
        outpath="./plots/borexino/exclusion/exclusion_2d_mh_u2_pyhf.pdf",
    )

    with open("./plots/borexino/exclusion/exclusion_2d_mh_u2_pyhf_boundary.csv", "w", encoding="utf-8") as f:
        f.write("mH,u2_90\n")
        for mh, u2lim in zip(mh_vals, u2_limits_90):
            u2s = "nan" if not np.isfinite(u2lim) else f"{u2lim:.8e}"
            f.write(f"{mh:.6f},{u2s}\n")

    # # Save separate 1D profile-likelihood chi2 curves for each mH
    # for mh in mh_vals:
    #     rows_mh = [r for r in all_rows if np.isclose(float(r["mH"]), float(mh))]
    #     if len(rows_mh) == 0:
    #         continue
    #     u2_this = np.array([r["u2"] for r in rows_mh], dtype=float)
    #     dchi2_this = np.array([r["delta_chi2"] for r in rows_mh], dtype=float)
    #     out_profile = f"./plots/borexino/fit/profile_u2_mH_{float(mh):.3f}.pdf"
    #     plot_profile_curve_u2(u2_this, dchi2_this, float(mh), out_profile)


if __name__ == "__main__":
    main()
