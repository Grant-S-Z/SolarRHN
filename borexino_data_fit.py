from dataclasses import dataclass, field
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit

from core import *
from toymc_s1_borexino_profile import load_background
from workflows import getNuleeInDetector


# Constants
u_max = 1e-1
u_min = 1e-6

estep: float = 0.2
e_min: float = 0.0
e_max: float = 16.0
fit_e_min: float = 4.8
fit_e_max: float = 12.8
n_all = int((e_max - e_min) / estep) + 1
n_fit = int((fit_e_max - fit_e_min) / estep) + 1
energy = np.linspace(e_min, e_max, n_all)
fit_energy = np.linspace(e_min, e_max, n_fit)
fit_mask = (energy >= fit_e_min) & (energy <= fit_e_max)


# Load Data
def load_borexino_data():
    data = np.loadtxt("./data/borexino_data.csv", delimiter=",", skiprows=1)
    print(f"Borexino data: {data}")
    return data


# Stats
def nll_poisson(data: np.ndarray, mu: np.ndarray) -> float:
    """Poisson negative loss likelihood.

    Parameters
    ----------
    data : np.ndarray
        Borexino experiment data
    mu : np.ndarray
        Predicted data by theory

    Returns
    -------
    float

    """
    mu_safe = np.clip(mu, 1e-12, None)
    nll = float(np.sum(mu_safe - data * np.log(mu_safe)))
    return nll

# def make_nll(data, b, s):
#     def nll(x_b, x_s):
#         mu = x_b * b + x_s * s
#         return nll_poisson(data, mu)

def make_nll(data, b, s):
    def nll(xb):
        mu = xb * b + s
        return nll_poisson(data, mu)
    return nll


# Simulation and Scan
def get_signal_template(spectrum_orig, mH, u2):
    spectrum_rhn = getRHNSpectrum(spectrum_orig, mH, u2)
    _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mH, u2)
    S = np.nan_to_num(diff_Eee_decayed[:, 1] * exposure * estep, nan=0.0, posinf=0.0, neginf=0.0) # 0.2 MeV^-1
    S = np.clip(S, 0.0, None)
    S = S[fit_mask]
    return S

def profile_likelihood_scan(spectrum_orig: np.ndarray, data: np.ndarray, B: np.ndarray, mH: float, u2_array: np.ndarray):
    """Scan U2 to get the likelihood minimum and corresponding best parameters.

    Parameters
    ----------

    """
    results = []

    for u2 in u2_array:
        S = get_signal_template(spectrum_orig, mH, u2)

        nll = make_nll(data, B, S)
        m = Minuit(nll, xb=1.0)
        m.limits['xb'] = (0.3, 3)
        m.errordef = Minuit.LIKELIHOOD
        m.migrad()

        if m.valid:
            results.append({
                'u2': u2,
                'S': S.copy(),
                'xb': m.values['xb'],
                'xb_err': m.errors['xb'],
                'nll': m.fval
            })
            print(f'U2 = {u2:.2e}, X_B8 = {m.values['xb']:.4f}, NLL = {m.fval:.4f}')
        else:
            print(f'U2 = {u2:.2e}: fit failed')

    nll_array = np.array([r['nll'] for r in results])
    nll_min = np.min(nll_array)
    for r in results:
        r['delta_chi2'] = 2.0 * (r['nll'] - nll_min)

    return results


# Plot
def plot_S_for_u2(spectrum_nuL_orig, u2_array, mh=8):
    plt.figure()
    for u2 in u2_array:
        spectrum_rhn = getRHNSpectrum(spectrum_nuL_orig, mh, u2)
        _, _, _, diff_Eee_decayed, _, _ = getNuleeInDetector(spectrum_rhn, mh, u2)
        S = np.nan_to_num(diff_Eee_decayed[:, 1] * exposure, nan=0.0, posinf=0.0, neginf=0.0) # MeV^-1
        S = np.clip(S, 0.0, None)
        S_centers = diff_Eee_decayed[:, 0] + 0.5 * estep

        plt.plot(S_centers, S, '-', linewidth=2, label=f'U2={u2}')
        plt.xlabel('Energy (MeV)')
        plt.ylabel('Counts')
        plt.legend()

    plt.savefig(f'./plots/borexino/Eee/S_u2_{u2_array[0]}_{u2_array[-1]}_mh_{mh}.pdf')
    plt.close()


def plot_fit_results(results, data, B, u2, mH):
    """Use scan results directly to plot data/fit/background/signal for a selected U2."""
    if len(results) == 0:
        raise ValueError('results is empty')

    u2_scan = np.array([r['u2'] for r in results], dtype=float)
    idx = int(np.argmin(np.abs(u2_scan - float(u2))))
    r = results[idx]

    xb = float(r['xb'])
    S = np.asarray(r['S'], dtype=float)
    B = np.asarray(B, dtype=float)

    if len(data) == len(B):
        data_fit = np.asarray(data, dtype=float)
    elif len(data) == len(energy):
        data_fit = np.asarray(data[fit_mask], dtype=float)
    else:
        raise ValueError('data length does not match fit bins or full bins')

    bg = xb * B
    total = bg + S
    x = energy[fit_mask]
    x_edges = np.concatenate(([x[0] - 0.5 * estep], x + 0.5 * estep))
    # yerr = np.sqrt(np.clip(data_fit, 0.0, None))

    plt.figure(figsize=(8, 5.5))
    # plt.errorbar(x, data_fit, yerr=yerr, fmt='o', ms=4, capsize=2, color='black', label='Data')
    plt.stairs(data_fit, x_edges, color='black', lw=1.0, label='Data (hist)')
    plt.plot(x, total, '-', lw=2.2, color='tab:blue', label='Best fit (B+S)')
    plt.plot(x, bg, '--', lw=2, color='tab:orange', label=r'Background: $X_B$')
    plt.plot(x, S, '--', lw=2, color='tab:green', label='Signal: S')

    plt.xlabel('Energy (MeV)')
    plt.ylabel('Counts / bin')
    plt.xlim(4.8, 12.8)
    plt.title(f'Borexino fit (mH={mH:g} MeV, U2={r["u2"]:.2e}, xB={xb:.3f})')
    plt.grid(True, ls=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./plots/borexino/fit/fit_u2_{r["u2"]:.2e}_mH_{mH:g}.pdf')
    plt.close()


def plot_nll_curve(u2_values, nll, mH):
    u2_values = np.asarray(u2_values, dtype=float)
    nll = np.asarray(nll, dtype=float)

    plt.figure(figsize=(7, 5))
    plt.plot(u2_values, nll, 'o-', lw=2, ms=5, label='NLL')

    i_min = int(np.argmin(nll))
    plt.scatter(u2_values[i_min], nll[i_min], color='tab:red', zorder=3,
                label=f'Min: {nll[i_min]:.3f} at {u2_values[i_min]:.2e}')

    plt.xscale('log')
    plt.xlabel(r'$|U_{eH}|^2$')
    plt.ylabel('NLL')
    plt.title(f'Borexino NLL Scan (mH={mH:g} MeV)')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./plots/borexino/fit/nll_u2_mH_{mH:g}.pdf')
    plt.close()


def plot_profile_curve(u2_values, delta_chi2, mH):
    plt.figure(figsize=(7, 5))
    plt.plot(u2_values, delta_chi2, 'o-', lw=2, ms=5, label=r'$\Delta\chi^2=-2\Delta\ln\mathcal{L}$')

    plt.axhline(2.71, color='tab:orange', ls='--', lw=1.5, label='90% C.L. (1 dof)')
    plt.axhline(3.84, color='tab:red', ls='--', lw=1.5, label='95% C.L. (1 dof)')

    plt.xscale('log')
    plt.xlabel(r'$|U_{eH}|^2$')
    plt.ylabel(r'$\Delta\chi^2$')
    plt.title(f'Borexino Profile Likelihood (mH={mH:g} MeV)')
    plt.grid(True, which='both', ls=':', alpha=0.5)
    plt.legend()
    plt.tight_layout()

    plt.savefig(f'./plots/borexino/fit/profile_u2_mH_{mH:g}.pdf')
    plt.close()


def main():
    # Borexino data
    data = load_borexino_data()[:, 1]

    # B8 background data
    _, bg = load_background(energy=energy, estep=estep, energy_resolution=None)
    B = bg[fit_mask]
    print(f"{len(bg)} bins → {len(B)} bins for fit")

    # Signal data
    print(">>> Loading 8B neutrino spectrum from csv file...")
    spectrum_nuL_orig = interpolateSpectrum(
        "data/8BSpectrum.csv", energy) # MeV^-1 cm^-2 s^-1

    print("B8 neutrino flux (integrated): ",
          integrateSpectrum(spectrum_nuL_orig), "cm^-2 s^-1")
    print()

    # Likelihood
    mH = 8.
    # u2_array = np.logspace(-6, -5, 11)
    u2_array = np.linspace(1e-6, 1e-5, 10)
    results = profile_likelihood_scan(spectrum_nuL_orig, data, B, mH, u2_array)
    nll_array = np.array([r['nll'] for r in results])
    delta_chi2_array = np.array([r['delta_chi2'] for r in results])

    # Plot
    # plot_S_for_u2(spectrum_nuL_orig, u2_array, mh=8)
    for u2 in u2_array:
        plot_fit_results(results, data, B, u2, mH)
    plot_nll_curve(u2_array, nll_array, mH)
    plot_profile_curve(u2_array, delta_chi2_array, mH)


if __name__ == "__main__":
    main()
