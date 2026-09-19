"""
zoom.py — Spatial low-pass filter for surface zoom visualisation.

zoom_surface_spatial : low-pass filter z in Fourier space at magnification
                       zeta_vis, returning the filtered surface and the
                       corresponding Persson magnification zeta_P.
"""

from typing import Tuple

import numpy as np

from .parameters import Params


def zoom_surface_spatial(
    z: np.ndarray,
    zeta_vis: float,
    p: Params,
) -> Tuple[np.ndarray, float]:
    """
    Low-pass filter the surface z to retain only wavelengths down to
    lambda = Lx / zeta_vis.

    The filter is a circular top-hat in Fourier space:
        keep all modes with |q| <= zeta_vis * q_L / 2
    where q_L = 2*pi/Lx is the fundamental wavevector.

    Parameters
    ----------
    z        : (n, m) surface height array [m]
    zeta_vis : int/float  visual magnification (integer zoom level)
    p        : Params  (uses p.Lx, p.q0)

    Returns
    -------
    z_zoom  : (n, m) float array  filtered surface [m]
    zeta_P  : float  Persson magnification = zeta_vis * q_L / q0
              (< 1 means we are below the fractal onset)
    """
    n_rows, n_cols = z.shape
    pixel_width = p.Lx / n_cols

    q_L      = 2.0 * np.pi / p.Lx        # fundamental wavevector [1/m]
    q_nyq    = np.pi / pixel_width        # Nyquist frequency [1/m]

    # Normalised frequency coordinates (centred)
    fx = np.fft.fftshift(np.fft.fftfreq(n_cols))  # -0.5 .. 0.5-1/n_cols
    fy = np.fft.fftshift(np.fft.fftfreq(n_rows))
    FX, FY = np.meshgrid(fx, fy)
    R = np.sqrt(FX ** 2 + FY ** 2)        # normalised radial frequency

    # Fractional cut-off: keep R <= frac/2
    # (matches make_fig14_gif:  frac = zeta * q_L / q_Nyquist,  mask R <= frac/2)
    frac = zeta_vis * q_L / q_nyq         # = 2 * zeta_vis / n_cols
    mask = R <= (frac / 2.0)

    Z_fft  = np.fft.fftshift(np.fft.fft2(z))
    z_zoom = np.fft.ifft2(np.fft.ifftshift(Z_fft * mask)).real

    zeta_P = float(zeta_vis) * q_L / p.q0

    return z_zoom, zeta_P
