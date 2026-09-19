"""
contact.py — Persson contact mechanics: relative contact area A(zeta)/A0.

eval_contact_area : compute A(zeta)/A0 and G(zeta) via Persson's erf formula.
zoom_surface      : legacy alias for eval_contact_area (MATLAB zoom_surface.m).
"""

from typing import Tuple

import numpy as np
from scipy.special import erf

from .parameters import Params
from .psd import PSD


def eval_contact_area(psd: PSD, zeta: float, p: Params) -> Tuple[float, float]:
    """
    Compute the apparent relative contact area A(zeta)/A0 at magnification zeta.

    Theory (paper Section 2):

        G(zeta) = (pi/4) * (E / (1 - nu^2))^2
                  * integral_{q0}^{zeta*q0}  q^3 * C(q) dq

        A(zeta)/A0 = erf( P0 / (2 * sqrt(G(zeta))) )

    Parameters
    ----------
    psd  : PSD   analytical PSD from idealized_PSD()
    zeta : float magnification (dimensionless, >= 1)
    p    : Params

    Returns
    -------
    A_ratio : float  A(zeta)/A0  in [0, 1]
    G_zeta  : float  G(zeta) [Pa^2]
    """
    E_star = p.E / (1.0 - p.nu ** 2)  # plane-strain modulus

    mask  = (psd.q >= p.q0) & (psd.q <= zeta * p.q0)
    q_sub = psd.q[mask]
    C_sub = psd.C[mask]

    if len(q_sub) < 2:
        return 1.0, 0.0

    G_zeta = (np.pi / 4.0) * E_star ** 2 * np.trapezoid(q_sub ** 3 * C_sub, q_sub)

    if G_zeta <= 0.0:
        return 1.0, G_zeta

    A_ratio = float(erf(p.P0 / (2.0 * np.sqrt(G_zeta))))
    return A_ratio, G_zeta


# Legacy alias — matches MATLAB zoom_surface.m (identical physics)
zoom_surface = eval_contact_area
