import os
import numpy as np
import matplotlib.pyplot as plt
import pyhf

from core import *
from toymc_s1_borexino_profile import load_background
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


def load_borexino_data() -> np.ndarray:
    data = np.loadtxt("./data/borexino_data.csv", delimiter=",", skiprows=1)
    print(f"Borexino data: {data}")
    return data


def get_signal_template(spectrum_orig: np.ndarray, mH: float, u2: float) -> np.ndarray:
    if u2 <= 0.0:
        return np.zeros(np.count_nonzero(fit_mask), dtype=float)

    spectrum_rhn = getRHNSpectrum(spectrum_orig, mH, u2)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mH, u2)

    # Convert to counts per analysis bin (0.2 MeV)
    s_bin = np.nan_to_num(
        diff_Eee_decayed[:, 1] * exposure * estep,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    s_bin = np.clip(s_bin, 0.0, None)
    return s_bin[fit_mask]


def poisson_deviance(data: np.ndarray, mu: np.ndarray) -> float:
    """Poisson deviance (Baker-Cousins) for goodness-of-fit."""
    data = np.asarray(data, dtype=float)
    mu = np.clip(np.asarray(mu, dtype=float), 1e-12, None)

    term = np.zeros_like(mu, dtype=float)
    pos = data > 0.0
    term[pos] = data[pos] * np.log(data[pos] / mu[pos])
    dev = 2.0 * np.sum(mu - data + term)
    return float(dev)


def _fit_xb_with_pyhf(data: np.ndarray, bkg: np.ndarray, sig: np.ndarray) -> tuple[float, float]:
    """Fit background scale xb for fixed signal template using pyhf.

    Returns
    -------
    xb_hat : float
    nll_hat : float
        Poisson NLL value (not 2*NLL)
    """
    data = np.asarray(data, dtype=float)
    bkg = np.asarray(bkg, dtype=float)
    sig = np.asarray(sig, dtype=float)

    # pyhf prefers non-negative expected counts
    bkg_safe = np.clip(bkg, 1e-12, None)
    sig_safe = np.clip(sig, 0.0, None)

    spec = {
        "version": "1.0.0",
        "channels": [
            {
                "name": "borexino",
                "samples": [
                    {
                        "name": "signal",
                        "data": sig_safe.tolist(),
                        "modifiers": [],
                    },
                    {
                        "name": "background",
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
                        {
                            "name": "xb",
                            "inits": [1.0],
                            "bounds": [[0.3, 3.0]],
                        }
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

    xb_hat = float(bestfit_pars[model.config.par_order.index("xb")])
    nll_hat = 0.5 * float(twice_nll)
    return xb_hat, nll_hat


def profile_likelihood_scan_pyhf(
    spectrum_orig: np.ndarray,
    data: np.ndarray,
    bkg: np.ndarray,
    mH: float,
    u2_array: np.ndarray,
) -> list[dict]:
    results = []

    n_bins = int(np.asarray(data).size)
    n_par = 1  # xb only for fixed-u2 scan
    ndf = n_bins - n_par

    for u2 in np.asarray(u2_array, dtype=float):
        s = get_signal_template(spectrum_orig, mH, float(u2))
        xb_hat, nll_hat = _fit_xb_with_pyhf(data, bkg, s)

        mu_hat = xb_hat * np.asarray(bkg, dtype=float) + s
        chi2_dev = poisson_deviance(data, mu_hat)
        chi2_ndf = chi2_dev / ndf if ndf > 0 else np.nan

        results.append(
            {
                "u2": float(u2),
                "S": s.copy(),
                "xb": xb_hat,
                "nll": nll_hat,
                "chi2_dev": chi2_dev,
                "ndf": ndf,
                "chi2_ndf": chi2_ndf,
            }
        )
        print(
            f"U2 = {u2:.2e}, X_B8 = {xb_hat:.4f}, NLL = {nll_hat:.4f}, "
            f"chi2/ndf = {chi2_dev:.2f}/{ndf} = {chi2_ndf:.3f}"
        )

    nll_array = np.array([r["nll"] for r in results], dtype=float)
    nll_min = np.min(nll_array)
    for r in results:
        r["delta_chi2"] = 2.0 * (r["nll"] - nll_min)

    return results


def plot_fit_results(results, data, B, u2, mH):
    if len(results) == 0:
        raise ValueError("results is empty")

    u2_scan = np.array([r["u2"] for r in results], dtype=float)
    idx = int(np.argmin(np.abs(u2_scan - float(u2))))
    r = results[idx]

    xb = float(r["xb"])
    S = np.asarray(r["S"], dtype=float)
    B = np.asarray(B, dtype=float)
    data_fit = np.asarray(data, dtype=float)

    bg = xb * B
    total = bg + S
    x = energy[fit_mask]
    x_edges = np.concatenate(([x[0] - 0.5 * estep], x + 0.5 * estep))

    plt.figure(figsize=(8, 5.5))
    plt.stairs(data_fit, x_edges, color="black", lw=1.0, label="Data (hist)")
    plt.plot(x, total, "-", lw=2.2, color="tab:blue", label="Best fit (B+S)")
    plt.plot(x, bg, "--", lw=2, color="tab:orange", label=r"Background: $X_B$")
    plt.plot(x, S, "--", lw=2, color="tab:green", label="Signal: S")

    plt.xlabel("Energy (MeV)")
    plt.ylabel("Counts / bin")
    plt.xlim(4.8, 12.8)
    plt.title(
        f"Borexino fit (mH={mH:g} MeV, U2={r['u2']:.2e}, xB={xb:.3f}, chi2/ndf={r['chi2_ndf']:.3f})"
    )
    plt.grid(True, ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("./plots/borexino/fit", exist_ok=True)
    plt.savefig(f"./plots/borexino/fit/fit_pyhf_u2_{r['u2']:.2e}_mH_{mH:g}.pdf")
    plt.close()


def plot_nll_curve(u2_values, nll, mH):
    u2_values = np.asarray(u2_values, dtype=float)
    nll = np.asarray(nll, dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(u2_values, nll, "o-", lw=2, ms=5, label="NLL")

    i_min = int(np.argmin(nll))
    plt.scatter(
        u2_values[i_min],
        nll[i_min],
        color="tab:red",
        zorder=3,
        label=f"Min: {nll[i_min]:.3f} at {u2_values[i_min]:.2e}",
    )

    plt.xscale("log")
    plt.xlabel(r"$|U_{eH}|^2$")
    plt.ylabel("NLL")
    plt.title(f"Borexino NLL Scan (pyhf, mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("./plots/borexino/fit", exist_ok=True)
    plt.savefig(f"./plots/borexino/fit/nll_pyhf_u2_mH_{mH:g}.pdf")
    plt.close()


def plot_profile_curve(u2_values, delta_chi2, mH):
    plt.figure(figsize=(7, 5))
    plt.plot(
        u2_values,
        delta_chi2,
        "o-",
        lw=2,
        ms=5,
        label=r"$\Delta\chi^2=-2\Delta\ln\mathcal{L}$",
    )

    plt.axhline(2.71, color="tab:orange", ls="--", lw=1.5, label="90% C.L. (1 dof)")
    plt.axhline(3.84, color="tab:red", ls="--", lw=1.5, label="95% C.L. (1 dof)")

    plt.xscale("log")
    plt.xlabel(r"$|U_{eH}|^2$")
    plt.ylabel(r"$\Delta\chi^2$")

    plt.title(f"Borexino Profile Likelihood (pyhf, mH={mH:g} MeV)")
    plt.grid(True, which="both", ls=":", alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs("./plots/borexino/fit", exist_ok=True)
    plt.savefig(f"./plots/borexino/fit/profile_pyhf_u2_mH_{mH:g}.pdf")
    plt.close()


def plot_exclusion_from_profile(u2_values, delta_chi2, mH, cl_crit=2.71):
    """Plot 90%CL-style exclusion from profile curve using fixed threshold."""
    u2 = np.asarray(u2_values, dtype=float)
    dchi2 = np.asarray(delta_chi2, dtype=float)

    order = np.argsort(u2)
    u2 = u2[order]
    dchi2 = dchi2[order]

    du = dchi2 - float(cl_crit)
    u2_cross = np.nan

    for i in range(len(u2) - 1):
        y1, y2 = du[i], du[i + 1]
        if y1 == 0.0:
            u2_cross = u2[i]
            break
        if y1 * y2 < 0.0:
            x1, x2 = u2[i], u2[i + 1]
            u2_cross = x1 + (0.0 - y1) * (x2 - x1) / (y2 - y1)
            break

    plt.figure(figsize=(7, 5))
    plt.plot(u2, dchi2, 'o-', lw=2, ms=5, color='tab:blue', label=r'$\Delta\chi^2$')
    plt.axhline(cl_crit, color='tab:orange', ls='--', lw=1.5, label=f'90% C.L. threshold ({cl_crit:.2f})')

    if np.isfinite(u2_cross):
        plt.axvline(u2_cross, color='tab:red', ls='--', lw=1.5, label=fr'Exclusion: $U^2_{{90}}={u2_cross:.2e}$')
        mask_excl = u2 >= u2_cross
        if np.any(mask_excl):
            plt.fill_between(u2[mask_excl], dchi2[mask_excl], cl_crit, color='tab:red', alpha=0.18)
        print(f'[exclusion] mH={mH:g} MeV, U2_90 = {u2_cross:.4e}')
    else:
        print(f'[exclusion] mH={mH:g} MeV, no crossing found for threshold {cl_crit:.2f}')

    plt.xscale('log')
    plt.xlabel(r'$|U_{eH}|^2$')
    plt.ylabel(r'$\Delta\chi^2$')
    plt.title(f'Borexino exclusion (pyhf, mH={mH:g} MeV)')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    os.makedirs('./plots/borexino/fit', exist_ok=True)
    plt.savefig(f'./plots/borexino/fit/exclusion_pyhf_u2_mH_{mH:g}.pdf')
    plt.close()


def main():
    data = load_borexino_data()[:, 1]

    _, bg = load_background(energy=energy, estep=estep, energy_resolution=None)
    B = bg[fit_mask]
    print(f"{len(bg)} bins → {len(B)} bins for fit")

    print(">>> Loading 8B neutrino spectrum from csv file...")
    spectrum_nuL_orig = interpolateSpectrum("data/8BSpectrum.csv", energy)
    print("B8 neutrino flux (integrated): ", integrateSpectrum(spectrum_nuL_orig), "cm^-2 s^-1")
    print()

    mH = 8.0
    u2_array = np.linspace(1e-6, 2e-5, 20)
    # u2_array = np.linspace(1e-7, 1e-6, 10)
    # u2_array = np.linspace(0, 1e-4, 11)
    u2_array = np.insert(u2_array, 0, 0.0)

    results = profile_likelihood_scan_pyhf(spectrum_nuL_orig, data, B, mH, u2_array)
    nll_array = np.array([r["nll"] for r in results], dtype=float)
    delta_chi2_array = np.array([r["delta_chi2"] for r in results], dtype=float)

# for u2 in u2_array:
    #     plot_fit_results(results, data, B, u2, mH)
    # plot_nll_curve(u2_array, nll_array, mH)
    # plot_profile_curve(u2_array, delta_chi2_array, mH)
    plot_exclusion_from_profile(u2_array, delta_chi2_array, mH, cl_crit=2.71)


if __name__ == "__main__":
    main()
