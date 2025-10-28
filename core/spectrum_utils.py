"""
Spectrum utilities for interpolation, integration, and file operations.
"""

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Any, cast


def interpolateSpectrum(inputCSV, xvalues):
    """Interpolate spectrum from CSV file.
    
    Parameters
    ----------
    inputCSV : str
        Path to CSV file with 'energy' and 'flux' columns
    xvalues : array-like
        Energy values to interpolate at
    
    Returns
    -------
    NDArray
        2D array with shape (len(xvalues), 2) containing [energy, flux]
    """
    df = pd.read_csv(inputCSV)
    interp_func = interp1d(
        df["energy"].values,
        df["flux"].values,
        kind="linear",
        fill_value=cast(Any, "extrapolate"),
    )

    spectrum_return = []
    for ix in xvalues:
        iy = interp_func(ix) * 1.0
        spectrum_return.append([ix, iy])
    return np.array(spectrum_return)


def integrateSpectrum(spectrum):
    """Integrate 1D spectrum using trapezoidal rule.
    
    Parameters
    ----------
    spectrum : NDArray
        2D array with shape (N, 2) containing [x, y] values
    
    Returns
    -------
    float
        Integral of spectrum
    """
    sum_all = 0.0
    for idx in range(len(spectrum)):
        if idx > 0:
            sum_all += spectrum[idx][1] * \
                (spectrum[idx][0] - spectrum[idx - 1][0])
        else:
            sum_all += spectrum[idx][1] * \
                (spectrum[idx + 1][0] - spectrum[idx][0])
    return sum_all


def integrateSpectrum2D(data, x_centers=None, y_centers=None):
    """Integrate a 2D spectrum over both axes.

    Supports two input formats:
    - data is a (nX, nY, 3) array where data[:,:,0] are x centers,
      data[:,:,1] are y centers and data[:,:,2] are the density values
      (e.g. dN/dx/dy).
    - data is a 2D array Z (shape (nX, nY)) and x_centers, y_centers are
      provided as 1D arrays of centers.

    The function infers bin edges from centers and returns the scalar total
    integral: sum_ij Z[i,j] * dX[i] * dY[j].
    
    Parameters
    ----------
    data : NDArray
        Either (nX, nY, 3) array or (nX, nY) array of density values
    x_centers : array-like, optional
        1D array of x bin centers (required if data is 2D)
    y_centers : array-like, optional
        1D array of y bin centers (required if data is 2D)
    
    Returns
    -------
    float
        Total integral over 2D grid
    """

    def _edges_from_centers(centers):
        c = np.asarray(centers)
        if c.size == 1:
            return np.array([c[0] - 0.5, c[0] + 0.5])
        edges = np.empty(c.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (c[1:] + c[:-1])
        edges[0] = c[0] - (edges[1] - c[0])
        edges[-1] = c[-1] + (c[-1] - edges[-2])
        return edges

    # decode inputs
    if isinstance(data, np.ndarray) and data.ndim == 3 and data.shape[2] >= 3:
        x = np.asarray(data[:, 0, 0])
        y = np.asarray(data[0, :, 1])
        Z = np.asarray(data[:, :, 2], dtype=float)
    elif isinstance(data, np.ndarray) and data.ndim == 2:
        if x_centers is None or y_centers is None:
            raise ValueError(
                "x_centers and y_centers must be provided when passing a 2D Z array")
        x = np.asarray(x_centers)
        y = np.asarray(y_centers)
        Z = np.asarray(data, dtype=float)
    else:
        raise ValueError("Unsupported input for integrateSpectrum2D")

    if Z.size == 0:
        return 0.0

    # infer edges and widths
    x_edges = _edges_from_centers(x)
    y_edges = _edges_from_centers(y)
    dX = np.diff(x_edges)
    dY = np.diff(y_edges)

    # ensure shapes
    if Z.shape != (dX.size, dY.size):
        raise ValueError(
            f"Z shape {Z.shape} incompatible with inferred axes sizes {(dX.size, dY.size)}")

    total = float(np.sum(Z * dX[:, None] * dY[None, :]))
    return total


def saveSpectrums(spectrums, column_names, fileName, labels):
    """Save multiple spectra to CSV files.
    
    Parameters
    ----------
    spectrums : list of NDArray
        List of spectra to save
    column_names : str or None
        Header string for CSV file
    fileName : str
        Base filename (without extension)
    labels : list of str
        Labels to append to fileName for each spectrum
    """
    for i in range(len(spectrums)):
        if column_names is None:
            np.savetxt(
                fileName + labels[i] + ".csv", spectrums[i], delimiter=",", fmt="%f"
            )
        else:
            np.savetxt(
                fileName + labels[i] + ".csv",
                spectrums[i],
                delimiter=",",
                header=column_names,
                fmt="%f",
                comments="",
            )


__all__ = [
    'interpolateSpectrum',
    'integrateSpectrum',
    'integrateSpectrum2D',
    'saveSpectrums',
]
