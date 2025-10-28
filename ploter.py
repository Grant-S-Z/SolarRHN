import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_El_costheta_map(diff_El_costheta, outpath, filename="El_costheta.png", cmap="viridis", costheta_min=None, costheta_max=None, title_prefix=""):
    """Save a heatmap of diff_El_costheta[:,:,2].

    diff_El_costheta: array shape (nE, nCostheta, 3)
    outpath: directory path
    filename: output image file name
    title_prefix: optional prefix for plot title
    """
    Z = diff_El_costheta[:, :, 2]
    X = diff_El_costheta[:, 0, 0]
    Y = diff_El_costheta[0, :, 1]
    # Optionally restrict costheta range
    if costheta_min is not None or costheta_max is not None:
        cmin = -np.inf if costheta_min is None else costheta_min
        cmax = np.inf if costheta_max is None else costheta_max
        mask = (Y >= cmin) & (Y <= cmax)
        if not np.any(mask):
            raise ValueError(f"No costheta values in requested range [{cmin}, {cmax}]")
        Y_sub = Y[mask]
        Z_sub = Z[:, mask]
    else:
        Y_sub = Y
        Z_sub = Z

    # Create meshgrid for pcolormesh: need X and Y bin edges (approximate by midpoints)
    if len(X) > 1:
        x_edges = np.concatenate(([X[0] - (X[1] - X[0]) / 2.0], (X[:-1] + X[1:]) / 2.0, [X[-1] + (X[-1] - X[-2]) / 2.0]))
    else:
        x_edges = np.array([X[0] - 0.5, X[0] + 0.5])

    if len(Y_sub) > 1:
        y_edges = np.concatenate(([Y_sub[0] - (Y_sub[1] - Y_sub[0]) / 2.0], (Y_sub[:-1] + Y_sub[1:]) / 2.0, [Y_sub[-1] + (Y_sub[-1] - Y_sub[-2]) / 2.0]))
    else:
        y_edges = np.array([Y_sub[0] - 0.5, Y_sub[0] + 0.5])

    plt.figure(figsize=(8, 6))
    pcm = plt.pcolormesh(x_edges, y_edges, Z_sub.T, cmap=cmap)
    plt.colorbar(pcm, label="Flux density")
    plt.xlabel("Energy (MeV)")
    plt.ylabel("cos(θ)")
    title = title_prefix if title_prefix else "Energy vs cos(θ)"
    if costheta_min is not None or costheta_max is not None:
        cmin_val = costheta_min if costheta_min is not None else Y_sub.min()
        cmax_val = costheta_max if costheta_max is not None else Y_sub.max()
        title += f" [cos(θ): {cmin_val:.2f},{cmax_val:.2f}]"
    plt.title(title)
    plt.tight_layout()
    plt.ylim(Y_sub.min(), Y_sub.max())
    # ensure outpath exists and join cleanly
    os.makedirs(outpath, exist_ok=True)
    out_file = os.path.join(outpath, filename)
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 2D map to {out_file}")


def plot_1d_energy_distribution(diff_El, outpath, filename="energy_1d.png", title_prefix="", xlabel="Energy (MeV)", ylabel="Flux"):
    """Plot 1D energy distribution.
    
    diff_El: array shape (nE, 2) where [:, 0] is energy and [:, 1] is flux
    """
    plt.figure(figsize=(8, 6))
    plt.plot(diff_El[:, 0], diff_El[:, 1], 'b-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_prefix if title_prefix else "Energy Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outpath, exist_ok=True)
    out_file = os.path.join(outpath, filename)
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 1D energy distribution to {out_file}")


def plot_1d_angle_distribution(diff_costheta, outpath, filename="angle_1d.png", title_prefix="", xlabel="cos(θ)", ylabel="Flux"):
    """Plot 1D angle distribution.
    
    diff_costheta: array shape (nTheta, 2) where [:, 0] is cos(theta) and [:, 1] is flux
    """
    plt.figure(figsize=(8, 6))
    plt.plot(diff_costheta[:, 0], diff_costheta[:, 1], 'r-', linewidth=2)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_prefix if title_prefix else "Angular Distribution")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    os.makedirs(outpath, exist_ok=True)
    out_file = os.path.join(outpath, filename)
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 1D angular distribution to {out_file}")
