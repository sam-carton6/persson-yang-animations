"""
psd.py — Power spectral density functions.

idealized_PSD : analytical self-affine fractal PSD (no surface needed)
psd_2D        : radially-averaged PSD of a generated surface (used by
                estimate_hurst_exponent)
"""

from dataclasses import dataclass

import numpy as np

from .parameters import Params


@dataclass
class PSD:
    """Radially-averaged power spectral density."""
    q: np.ndarray  # wavevector array [1/m]
    C: np.ndarray  # C(q) [m^4]


def idealized_PSD(p: Params, N: int = 2000) -> PSD:
    """
    Build the analytical self-affine fractal PSD used by Persson-Yang theory.

    C(q) = C0 * q^{-2(1+H)}   for q >= qr  (power-law fractal region)
    C(q) = C0 * qr^{-2(1+H)}  for q <  qr  (flat roll-off plateau)

    Normalised so that:
        sigma^2 = 2*pi * integral_{q0}^{q1}  q * C(q) dq   [eq. 4 in paper]

    Parameters
    ----------
    p : Params
    N : int  number of log-spaced wavevector points (default 2000)

    Returns
    -------
    PSD with .q [1/m] and .C [m^4]
    """
    q = np.logspace(np.log10(p.q0), np.log10(p.q1), N)

    C_raw = np.where(q < p.qr,
                     p.qr ** (-2.0 * (1.0 + p.H)),
                     q   ** (-2.0 * (1.0 + p.H)))

    # Normalise: sigma^2 = 2*pi * int q C(q) dq
    raw_rms_sq = 2.0 * np.pi * np.trapezoid(q * C_raw, q)
    C0 = p.sigma ** 2 / raw_rms_sq

    return PSD(q=q, C=C0 * C_raw)


def psd_2D(z: np.ndarray, pixel_width: float) -> PSD:
    """
    Compute the radially-averaged 2-D power spectral density of surface z.

    Mirrors the MATLAB psd_2D.m approach: FFT → |H|^2 scaled to C(q) units,
    then bin-average over annular rings in q-space.

    Parameters
    ----------
    z           : 2-D surface height array [m]
    pixel_width : spatial resolution [m/pixel]

    Returns
    -------
    PSD with .q [1/m] and .C [m^4]
    """
    n, m = z.shape
    Lx = m * pixel_width
    Ly = n * pixel_width

    # 2-D FFT → power spectrum scaled to C(q) [m^4]
    Z = np.fft.fft2(z)
    Cq2d = (np.abs(Z) ** 2) * (pixel_width ** 2) / ((2.0 * np.pi) ** 2 * m * n)

    # Frequency axes (physical units, shifted so DC is at centre)
    qx = np.fft.fftshift(np.fft.fftfreq(m)) * (2.0 * np.pi / pixel_width)
    qy = np.fft.fftshift(np.fft.fftfreq(n)) * (2.0 * np.pi / pixel_width)
    QX, QY = np.meshgrid(qx, qy)
    rho = np.sqrt(QX ** 2 + QY ** 2)
    Cq2d_shift = np.fft.fftshift(Cq2d)

    # Radial bin edges (log-spaced)
    q_min = np.sqrt((2.0 * np.pi / Lx) ** 2 + (2.0 * np.pi / Ly) ** 2)
    q_max = np.sqrt(qx[-1] ** 2 + qy[-1] ** 2)
    J = 1000
    q_edges = np.floor(10.0 ** np.linspace(np.log10(q_min), np.log10(q_max), J))

    C_ave = np.full(len(q_edges), np.nan)
    rho_floor = np.floor(rho)

    for j in range(len(q_edges) - 1):
        mask = (rho_floor > q_edges[j]) & (rho_floor <= q_edges[j + 1])
        if np.any(mask):
            C_ave[j] = np.nanmean(Cq2d_shift[mask])

    valid = ~np.isnan(C_ave)
    return PSD(q=q_edges[valid].astype(float), C=C_ave[valid])
