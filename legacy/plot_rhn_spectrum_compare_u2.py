import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from core import (
    attenuation_length,
    distance_SE,
    getDecayedRHNSpectrum_vll,
    getRHNSpectrum,
    interpolateSpectrum,
)


def parse_u2_values(text: str):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("U2 list is empty. Use e.g. --u2 1e-7,1e-6,1e-5,1e-4")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare decayed RHN spectra at Earth for fixed MH and multiple U2, "
            "using getDecayedRHNSpectrum_vll."
        )
    )
    parser.add_argument("--mh", type=float, default=10.0, help="Fixed RHN mass MH in MeV")
    parser.add_argument("--emin", type=float, default=0.0, help="Minimum energy (MeV)")
    parser.add_argument("--emax", type=float, default=16.0, help="Maximum energy (MeV)")
    parser.add_argument("--estep", type=float, default=0.05, help="Energy step (MeV)")
    parser.add_argument("--input", type=str, default="data/8BSpectrum.csv", help="Input solar neutrino spectrum CSV")
    parser.add_argument("--outdir", type=str, default="plots/earth_RHN_decay_spectrum/", help="Output directory")
    parser.add_argument("--tag", type=str, default="", help="Optional output filename tag")
    args = parser.parse_args()

    u2_values = [1e-5, 1e-4, 1e-3, 1e-2]
    if args.estep <= 0:
        raise ValueError("--estep must be positive")

    energy = np.arange(args.emin, args.emax + 0.5 * args.estep, args.estep)
    spectrum_nuL_orig = interpolateSpectrum(args.input, energy)

    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for u2 in u2_values:
        spectrum_rhn = getRHNSpectrum(spectrum_nuL_orig, args.mh, u2)
        spectrum_decayed = getDecayedRHNSpectrum_vll(
            spectrum_rhn, args.mh, u2, distance_SE, attenuation_length
        )

        x = spectrum_decayed[:, 0]
        y = spectrum_decayed[:, 1]

        mask = y > 0
        if np.any(mask):
            plt.plot(x[mask], y[mask], linewidth=2, label=fr"$U^2={u2:.1e}$")

    plt.yscale("log")
    plt.xlabel(r"$E_H$ [MeV]")
    plt.ylabel(r"Decayed RHN spectrum $d\Phi/dE$ [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title(fr"Same $M_H={args.mh:.1f}\,\mathrm{{MeV}}$, different $U^2$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    tag = f"_{args.tag}" if args.tag else ""
    outfile = os.path.join(args.outdir, f"decayed_rhn_spectrum_compare_mh_{args.mh:.1f}{tag}.pdf")
    plt.savefig(outfile)
    plt.close()

    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
