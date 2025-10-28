import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

def plot_El_costheta_map(diff_El_costheta, outpath, filename="El_costheta.png", cmap="viridis", 
                         costheta_min=None, costheta_max=None, title_prefix="", 
                         log_scale=None, vmin=None, vmax=None):
    """Save a heatmap of diff_El_costheta[:,:,2].
    
    Automatically saves both linear and log scale versions.

    Parameters
    ----------
    diff_El_costheta : ndarray
        Array shape (nE, nCostheta, 3)
    outpath : str
        Directory path for output
    filename : str
        Output image file name (will add suffixes for linear/log versions)
    cmap : str
        Matplotlib colormap name
    costheta_min : float, optional
        Minimum cos(θ) to plot
    costheta_max : float, optional
        Maximum cos(θ) to plot
    title_prefix : str
        Optional prefix for plot title
    log_scale : bool, optional
        If specified, only save that version. If None, save both.
    vmin : float, optional
        Minimum value for color scale (useful with log_scale)
    vmax : float, optional
        Maximum value for color scale
    """
    # Determine which versions to plot
    if log_scale is None:
        plot_versions = [False, True]  # Both linear and log
    else:
        plot_versions = [log_scale]  # Only specified version
    
    for use_log in plot_versions:
        _plot_2d_map_single(diff_El_costheta, outpath, filename, cmap, 
                           costheta_min, costheta_max, title_prefix, 
                           use_log, vmin, vmax)


def _plot_2d_map_single(diff_El_costheta, outpath, filename, cmap, 
                        costheta_min, costheta_max, title_prefix, 
                        log_scale, vmin, vmax):
    """Internal function to plot a single version (linear or log)."""
    Z = diff_El_costheta[:, :, 2]
    X = diff_El_costheta[:, 0, 0]
    Y = diff_El_costheta[0, :, 1]
    
    # Filter out energy bins that are zero or very close to zero
    energy_mask = X > 1e-6  # Keep only energy > 0
    if not np.any(energy_mask):
        raise ValueError("All energy values are zero or negative")
    X_filtered = X[energy_mask]
    Z_filtered = Z[energy_mask, :]
    
    # Optionally restrict costheta range
    if costheta_min is not None or costheta_max is not None:
        cmin = -np.inf if costheta_min is None else costheta_min
        cmax = np.inf if costheta_max is None else costheta_max
        mask = (Y >= cmin) & (Y <= cmax)
        if not np.any(mask):
            raise ValueError(f"No costheta values in requested range [{cmin}, {cmax}]")
        Y_sub = Y[mask]
        Z_sub = Z_filtered[:, mask]
    else:
        Y_sub = Y
        Z_sub = Z_filtered
    
    # Use filtered X for plotting
    X = X_filtered

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
    
    # Apply log scale if requested
    if log_scale:
        from matplotlib.colors import LogNorm
        Z_plot = Z_sub.T
        # Handle zeros/negatives for log scale
        Z_plot = np.where(Z_plot > 0, Z_plot, np.nan)
        vmin_use = vmin if vmin is not None else np.nanmin(Z_plot[Z_plot > 0]) if np.any(Z_plot > 0) else 1e-10
        vmax_use = vmax if vmax is not None else np.nanmax(Z_plot)
        pcm = plt.pcolormesh(x_edges, y_edges, Z_plot, cmap=cmap, norm=LogNorm(vmin=vmin_use, vmax=vmax_use))
    else:
        vmin_use = vmin if vmin is not None else None
        vmax_use = vmax if vmax is not None else None
        pcm = plt.pcolormesh(x_edges, y_edges, Z_sub.T, cmap=cmap, vmin=vmin_use, vmax=vmax_use)
    
    plt.colorbar(pcm, label="Flux density" + (" (log scale)" if log_scale else ""))
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
    
    # Add suffix to filename
    base, ext = os.path.splitext(filename)
    suffix = "_log" if log_scale else "_linear"
    out_file = os.path.join(outpath, f"{base}{suffix}{ext}")
    
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 2D map ({'log' if log_scale else 'linear'}) to {out_file}")


def plot_1d_energy_distribution(diff_El, outpath, filename="energy_1d.png", 
                                title_prefix="", xlabel="Energy (MeV)", ylabel="Flux",
                                logx=None, logy=None, xlim=None, ylim=None):
    """Plot 1D energy distribution.
    
    Automatically saves both linear and log scale versions.
    
    Parameters
    ----------
    diff_El : ndarray
        Array shape (nE, 2) where [:, 0] is energy and [:, 1] is flux
    outpath : str
        Directory path for output
    filename : str
        Output image file name (will add suffixes for linear/log versions)
    title_prefix : str
        Optional prefix for plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    logx : bool, optional
        If specified, only use that x-scale. If None, save both versions.
    logy : bool, optional
        If specified, only use that y-scale. If None, save both versions.
    xlim : tuple, optional
        (xmin, xmax) for x-axis limits
    ylim : tuple, optional
        (ymin, ymax) for y-axis limits
    """
    # If not specified, plot both linear and log versions
    if logy is None:
        plot_versions = [False, True]  # linear and logy
    else:
        plot_versions = [logy]
    
    for use_logy in plot_versions:
        _plot_1d_energy_single(diff_El, outpath, filename, title_prefix, 
                              xlabel, ylabel, logx if logx is not None else False, 
                              use_logy, xlim, ylim)


def _plot_1d_energy_single(diff_El, outpath, filename, title_prefix, 
                           xlabel, ylabel, logx, logy, xlim, ylim):
    """Internal function to plot a single version (linear or log)."""
    # Filter out energy = 0 or very close to 0
    energy_mask = diff_El[:, 0] > 1e-6
    if not np.any(energy_mask):
        print(f"Warning: All energy values are zero or negative, skipping plot {filename}")
        return
    
    energy_filtered = diff_El[energy_mask, 0]
    flux_filtered = diff_El[energy_mask, 1]
    
    plt.figure(figsize=(8, 6))
    plt.plot(energy_filtered, flux_filtered, 'b-', linewidth=2)
    
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title_prefix if title_prefix else "Energy Distribution")
    plt.grid(True, alpha=0.3)
    
    if xlim is not None:
        plt.xlim(xlim)
    else:
        # Auto-set xlim: start from minimum non-zero energy
        plt.xlim(energy_filtered.min(), energy_filtered.max())
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    os.makedirs(outpath, exist_ok=True)
    
    # Add suffix to filename
    base, ext = os.path.splitext(filename)
    suffix = "_log" if logy else "_linear"
    out_file = os.path.join(outpath, f"{base}{suffix}{ext}")
    
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 1D energy distribution ({'log' if logy else 'linear'}) to {out_file}")


def plot_1d_angle_distribution(diff_costheta, outpath, filename="angle_1d.png", 
                               title_prefix="", xlabel="cos(θ)", ylabel="Flux",
                               logx=None, logy=None, xlim=None, ylim=None,
                               costheta_min=None, costheta_max=None):
    """Plot 1D angle distribution.
    
    Automatically saves both linear and log scale versions.
    
    Parameters
    ----------
    diff_costheta : ndarray
        Array shape (nTheta, 2) where [:, 0] is cos(theta) and [:, 1] is flux
    outpath : str
        Directory path for output
    filename : str
        Output image file name (will add suffixes for linear/log versions)
    title_prefix : str
        Optional prefix for plot title
    xlabel : str
        X-axis label
    ylabel : str
        Y-axis label
    logx : bool, optional
        If specified, only use that x-scale. If None, save both versions.
    logy : bool, optional
        If specified, only use that y-scale. If None, save both versions.
    xlim : tuple, optional
        (xmin, xmax) for x-axis limits
    ylim : tuple, optional
        (ymin, ymax) for y-axis limits
    costheta_min : float, optional
        Minimum cos(θ) to plot
    costheta_max : float, optional
        Maximum cos(θ) to plot
    """
    # If not specified, plot both linear and log versions
    if logy is None:
        plot_versions = [False, True]  # linear and logy
    else:
        plot_versions = [logy]
    
    for use_logy in plot_versions:
        _plot_1d_angle_single(diff_costheta, outpath, filename, title_prefix,
                             xlabel, ylabel, logx if logx is not None else False,
                             use_logy, xlim, ylim, costheta_min, costheta_max)


def _plot_1d_angle_single(diff_costheta, outpath, filename, title_prefix,
                          xlabel, ylabel, logx, logy, xlim, ylim,
                          costheta_min, costheta_max):
    """Internal function to plot a single version (linear or log)."""
    # Apply costheta range filter (default to full range [-1, 1])
    costheta = diff_costheta[:, 0]
    flux = diff_costheta[:, 1]
    
    # Default range for forward-peaked distributions (can be overridden)
    # Relaxed criteria: only zoom if extremely forward-peaked (99% of flux in cos(θ) > 0.95)
    if costheta_min is None and costheta_max is None:
        # Auto-detect: only restrict if distribution is EXTREMELY forward-peaked
        # Check if 99% of flux is in cos(θ) > 0.95
        high_angle_mask = costheta > 0.95
        if np.any(high_angle_mask):
            high_angle_fraction = np.sum(flux[high_angle_mask]) / np.sum(flux) if np.sum(flux) > 0 else 0
            if high_angle_fraction > 0.99:
                # Extremely forward-peaked, zoom in
                costheta_min = 0.9
                costheta_max = 1.0
    
    # Apply user-specified or auto-detected range
    if costheta_min is not None or costheta_max is not None:
        cmin = -np.inf if costheta_min is None else costheta_min
        cmax = np.inf if costheta_max is None else costheta_max
        angle_mask = (costheta >= cmin) & (costheta <= cmax)
        if not np.any(angle_mask):
            print(f"Warning: No data in costheta range [{cmin}, {cmax}], using full range")
            angle_mask = np.ones(len(costheta), dtype=bool)
    else:
        angle_mask = np.ones(len(costheta), dtype=bool)
    
    costheta_filtered = costheta[angle_mask]
    flux_filtered = flux[angle_mask]
    
    plt.figure(figsize=(8, 6))
    plt.plot(costheta_filtered, flux_filtered, 'r-', linewidth=2)
    
    if logx:
        plt.xscale('log')
    if logy:
        plt.yscale('log')
    
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = title_prefix if title_prefix else "Angular Distribution"
    if costheta_min is not None or costheta_max is not None:
        cmin_val = costheta_min if costheta_min is not None else costheta_filtered.min()
        cmax_val = costheta_max if costheta_max is not None else costheta_filtered.max()
        title += f" [cos(θ): {cmin_val:.2f},{cmax_val:.2f}]"
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    if xlim is not None:
        plt.xlim(xlim)
    else:
        # Default to [-1, 1] range
        plt.xlim(-1.0, 1.0)
    
    if ylim is not None:
        plt.ylim(ylim)
    
    plt.tight_layout()
    
    os.makedirs(outpath, exist_ok=True)
    
    # Add suffix to filename
    base, ext = os.path.splitext(filename)
    suffix = "_log" if logy else "_linear"
    out_file = os.path.join(outpath, f"{base}{suffix}{ext}")
    
    plt.savefig(out_file, dpi=200)
    plt.close()
    print(f"Saved 1D angular distribution ({'log' if logy else 'linear'}) to {out_file}")
