"""
surface.py — Stochastic rough-surface generation and analysis.

artificial_surf         : generate a fractal rough surface (port of artificial_surf.m)
calculate_rms           : measure RMS roughness of a surface
estimate_hurst_exponent : estimate H from the slope of the log PSD
"""

from typing import Optional, Tuple

import numpy as np

from .parameters import Params
from .psd import PSD, psd_2D


# ---------------------------------------------------------------------------
#  artificial_surf
# ---------------------------------------------------------------------------

def artificial_surf(
    sigma: float,
    H: float,
    Lx: float,
    m: int,
    n: int,
    qr: Optional[float] = None,
) -> Tuple[np.ndarray, float, PSD]:
    """
    Generate a randomly rough surface with prescribed PSD statistics.

    Mirrors MATLAB artificial_surf.m exactly, including conjugate-symmetry
    enforcement so that ifft2 yields a real surface.

    Parameters
    ----------
    sigma : float  RMS roughness [m]
    H     : float  Hurst exponent (0 <= H <= 1)
    Lx    : float  Physical length in x [m]
    m     : int    Pixels in x  (power of 2 recommended)
    n     : int    Pixels in y  (power of 2 recommended)
    qr    : float  Roll-off wavevector [1/m]  (optional; default = no roll-off)

    Returns
    -------
    z          : (n, m) float array  surface height [m]
    pixel_width: float               spatial resolution [m/pixel]
    psd        : PSD                 radially-averaged PSD of the generated surface
    """
    # Make m and n even
    if m % 2 != 0:
        m -= 1
    if n % 2 != 0:
        n -= 1

    pixel_width = Lx / m
    Lx = m * pixel_width
    Ly = n * pixel_width

    # Nyquist frequency
    q_max = np.pi / pixel_width

    if qr is not None and qr > q_max:
        qr = q_max

    # ---- Wavevectors (physical units, fftshift convention) -----------------
    # qx[k] = (2*pi/m)*k for k=0..m-1, then fftshift, subtract 2*pi, unwrap,
    # divide by pixel_width.  This matches MATLAB's convention exactly.
    # Equivalent to: fftshift(fftfreq(m)) * 2*pi / pixel_width
    qx = np.fft.fftshift(np.fft.fftfreq(m)) * (2.0 * np.pi / pixel_width)
    qy = np.fft.fftshift(np.fft.fftfreq(n)) * (2.0 * np.pi / pixel_width)
    QX, QY = np.meshgrid(qx, qy)  # shape (n, m)
    rho = np.sqrt(QX ** 2 + QY ** 2)  # radial wavevector

    # ---- 2-D PSD matrix Cq -------------------------------------------------
    if qr is None or qr == 0:
        # No roll-off: pure power law everywhere
        with np.errstate(divide="ignore", invalid="ignore"):
            Cq = np.where(rho > 0, rho ** (-2.0 * (H + 1.0)), 0.0)
    else:
        Cq = np.where(rho < qr,
                      qr  ** (-2.0 * (H + 1.0)),
                      rho ** (-2.0 * (H + 1.0)))
        Cq[rho == 0] = 0.0

    # ---- Normalise to prescribed RMS ---------------------------------------
    # DC (centre of fftshifted array) = zero
    Cq[n // 2, m // 2] = 0.0

    RMS_F2D = np.sqrt(np.sum(Cq) * (2.0 * np.pi) ** 2 / (Lx * Ly))
    alfa = sigma / RMS_F2D
    Cq = Cq * alfa ** 2

    # ---- Radially-averaged PSD (for output / verification) -----------------
    rho_floor = np.floor(rho).astype(int)
    J = 1000
    qrmin = np.log10(np.sqrt((2.0 * np.pi / Lx) ** 2 + (2.0 * np.pi / Ly) ** 2))
    qrmax = np.log10(np.sqrt(float(qx[-1]) ** 2 + float(qy[-1]) ** 2))
    q_bins = np.floor(10.0 ** np.linspace(qrmin, qrmax, J)).astype(int)

    C_ave = np.full(len(q_bins), np.nan)
    for j in range(len(q_bins) - 1):
        mask = (rho_floor > q_bins[j]) & (rho_floor <= q_bins[j + 1])
        if np.any(mask):
            C_ave[j] = float(np.nanmean(Cq[mask]))

    valid = ~np.isnan(C_ave)
    out_psd = PSD(q=q_bins[valid].astype(float), C=C_ave[valid])

    # ---- Bq: amplitude in Fourier space ------------------------------------
    # Bq = sqrt(Cq / (pixel_width^2 / (n*m*(2*pi)^2)))
    Bq = np.sqrt(Cq * (n * m * (2.0 * np.pi) ** 2) / pixel_width ** 2)

    # ---- Enforce conjugate symmetry on magnitude ---------------------------
    # Four special points must be real (zero imaginary part → phase = 0)
    Bq[0,      0]      = 0.0
    Bq[0,      m // 2] = 0.0
    Bq[n // 2, m // 2] = 0.0
    Bq[n // 2, 0]      = 0.0

    # Lower-left quadrant mirrors upper-right (rot180)
    Bq[1:,       1:m // 2]  = np.rot90(Bq[1:,       m // 2 + 1:], 2)
    # Top row (between corners) mirrors itself rot180
    Bq[0,        1:m // 2]  = Bq[0,        m // 2 + 1:][::-1]
    # Left column below centre mirrors above centre (flipped)
    Bq[n // 2 + 1:, 0]      = Bq[1:n // 2, 0][::-1]
    # Centre column below centre mirrors above centre (flipped)
    Bq[n // 2 + 1:, m // 2] = Bq[1:n // 2, m // 2][::-1]

    # ---- Random phase in (-pi, pi) -----------------------------------------
    rng = np.random.default_rng()
    phi = rng.uniform(-np.pi, np.pi, size=(n, m))

    # ---- Enforce conjugate symmetry on phase (negate for Hermitian FFT) ----
    phi[0,      0]      = 0.0
    phi[0,      m // 2] = 0.0
    phi[n // 2, m // 2] = 0.0
    phi[n // 2, 0]      = 0.0

    phi[1:,       1:m // 2]  = -np.rot90(phi[1:,       m // 2 + 1:], 2)
    phi[0,        1:m // 2]  = -phi[0,        m // 2 + 1:][::-1]
    phi[n // 2 + 1:, 0]      = -phi[1:n // 2, 0][::-1]
    phi[n // 2 + 1:, m // 2] = -phi[1:n // 2, m // 2][::-1]

    # ---- Generate surface --------------------------------------------------
    # Hm = Bq * exp(i*phi)  (complex Fourier amplitudes with prescribed PSD)
    Hm = Bq * np.exp(1j * phi)
    z = np.fft.ifft2(np.fft.ifftshift(Hm)).real

    return z, pixel_width, out_psd


# ---------------------------------------------------------------------------
#  calculate_rms
# ---------------------------------------------------------------------------

def calculate_rms(z: np.ndarray) -> float:
    """
    Compute RMS roughness Rq of surface z.

    Parameters
    ----------
    z : 2-D surface height array [m]

    Returns
    -------
    Rq : float  [m]
    """
    return float(np.sqrt(np.mean(z ** 2)))


# ---------------------------------------------------------------------------
#  estimate_hurst_exponent
# ---------------------------------------------------------------------------

def estimate_hurst_exponent(z: np.ndarray, pixel_width: float) -> float:
    """
    Estimate the Hurst exponent H from the slope of the log PSD.

    Fits a line to the middle portion (40–99 %) of log10(C) vs log10(q)
    to avoid the roll-off (low q) and aliasing (high q) regions.

    Parameters
    ----------
    z           : 2-D surface height array [m]
    pixel_width : spatial resolution [m/pixel]

    Returns
    -------
    H_est : float  estimated Hurst exponent
    """
    psd = psd_2D(z, pixel_width)

    q = psd.q
    C = psd.C
    valid = (q > 0) & (C > 0)
    log_q = np.log10(q[valid])
    log_C = np.log10(C[valid])

    n_pts = len(log_q)
    idx_start = round(n_pts * 0.40)
    idx_end   = round(n_pts * 0.99)
    idx_fit   = slice(idx_start, idx_end)

    coeffs = np.polyfit(log_q[idx_fit], log_C[idx_fit], 1)
    slope  = coeffs[0]
    H_est  = -slope / 2.0 - 1.0

    return float(H_est)
