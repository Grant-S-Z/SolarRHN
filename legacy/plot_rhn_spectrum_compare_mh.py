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


def parse_mh_values(text: str):
    vals = [float(x.strip()) for x in text.split(",") if x.strip()]
    if len(vals) == 0:
        raise ValueError("MH list is empty. Use e.g. --mh 2,4,6,8,10")
    return vals


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compare decayed RHN spectra at Earth for fixed U2 and multiple MH, "
            "using getDecayedRHNSpectrum_vll."
        )
    )
    parser.add_argument("--u2", type=float, default=1e-1, help="Fixed mixing |U_eH|^2")
    parser.add_argument("--mh", type=str, default="2,4,6,8,10,12", help="Comma-separated MH values in MeV")
    parser.add_argument("--emin", type=float, default=0.0, help="Minimum energy (MeV)")
    parser.add_argument("--emax", type=float, default=16.0, help="Maximum energy (MeV)")
    parser.add_argument("--estep", type=float, default=0.05, help="Energy step (MeV)")
    parser.add_argument("--input", type=str, default="data/8BSpectrum.csv", help="Input solar neutrino spectrum CSV")
    parser.add_argument("--outdir", type=str, default="plots/earth_RHN_decay_spectrum/", help="Output directory")
    parser.add_argument("--tag", type=str, default="", help="Optional output filename tag")
    args = parser.parse_args()

    mh_values = parse_mh_values(args.mh)
    if args.estep <= 0:
        raise ValueError("--estep must be positive")

    energy = np.arange(args.emin, args.emax + 0.5 * args.estep, args.estep)
    spectrum_nuL_orig = interpolateSpectrum(args.input, energy)

    os.makedirs(args.outdir, exist_ok=True)

    plt.figure(figsize=(8, 6))

    for mh in mh_values:
        spectrum_rhn = getRHNSpectrum(spectrum_nuL_orig, mh, args.u2)
        spectrum_decayed = getDecayedRHNSpectrum_vll(
            spectrum_rhn, mh, args.u2, distance_SE, attenuation_length
        )

        x = spectrum_decayed[:, 0]
        y = spectrum_decayed[:, 1]

        mask = y > 0
        if np.any(mask):
            plt.plot(x[mask], y[mask], linewidth=2, label=fr"$M_H={mh:.1f}\,\mathrm{{MeV}}$")

    plt.yscale("log")
    plt.xlabel(r"$E_H$ [MeV]")
    plt.ylabel(r"Decayed RHN spectrum $d\Phi/dE$ [cm$^{-2}$ s$^{-1}$ MeV$^{-1}$]")
    plt.title(fr"Same $U^2={args.u2:.1e}$, different $M_H$")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    tag = f"_{args.tag}" if args.tag else ""
    outfile = os.path.join(args.outdir, f"decayed_rhn_spectrum_compare_u2_{args.u2:.1e}{tag}.pdf")
    plt.savefig(outfile)
    plt.close()

    print(f"Saved: {outfile}")


if __name__ == "__main__":
    main()
