#!/usr/bin/env python3
"""Load chi2 grid from an NPZ file and plot exclusion region directly.

Usage examples:
    # Basic
    python plot_exclusion_from_npz.py output/chi2_grid_s1_xxx.npz --ylog

    # Specify output name/format and axis scales
    python plot_exclusion_from_npz.py output/chi2_grid_s1_xxx.npz \
        --out-dir plots/exclusion/ --file-name s1_exclusion --type png --xlog --ylog
"""

import argparse
import os

import numpy as np

from pytools.rt_ploter import rt_plot_exclusion_region


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read NPZ and call rt_plot_exclusion_region to draw exclusion plot."
    )
    parser.add_argument("npz", help="Path to input .npz file")
    parser.add_argument(
        "--out-dir",
        default="plots/exclusion/",
        help="Output directory for the figure (default: plots/exclusion/)",
    )
    parser.add_argument(
        "--file-name",
        default=None,
        help="Output file name without extension (default: derived from NPZ file name)",
    )
    parser.add_argument("--cl", type=float, default=0.90, help="Confidence level (default: 0.90)")
    parser.add_argument("--ndof", type=int, default=2, help="Degrees of freedom (default: 2)")
    parser.add_argument("--xlog", action="store_true", help="Use log scale on x-axis (MH)")
    parser.add_argument("--ylog", action="store_true", help="Use log scale on y-axis (U2)")
    parser.add_argument("--type", default="pdf", help="Output file type, e.g. pdf/png (default: pdf)")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.npz):
        raise FileNotFoundError(f"Input NPZ not found: {args.npz}")

    data = np.load(args.npz)
    required = ["U2_values", "MH_values", "chi2_grid"]
    missing = [k for k in required if k not in data]
    if missing:
        raise KeyError(
            f"NPZ file is missing required keys: {missing}. "
            f"Available keys: {list(data.keys())}"
        )

    U2_values = data["U2_values"]
    MH_values = data["MH_values"]
    chi2_grid = data["chi2_grid"]

    file_name = args.file_name
    if file_name is None:
        stem = os.path.splitext(os.path.basename(args.npz))[0]
        file_name = f"exclusion_{stem}"

    chi2_crit, out_path = rt_plot_exclusion_region(
        U2_values,
        MH_values,
        chi2_grid,
        file_name=file_name,
        dir=args.out_dir,
        cl=args.cl,
        ndof=args.ndof,
        xlog=args.xlog,
        ylog=args.ylog,
        type=args.type,
    )

    print(f"Saved exclusion plot to: {out_path}")
    print(f"chi2 critical value: {chi2_crit:.6f}")


if __name__ == "__main__":
    main()
