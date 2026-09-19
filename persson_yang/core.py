"""
core.py — Core Persson-Yang quantities: u_c, Q_dot, and critical-magnification search.

compute_uc    : RMS roughness in the unresolved band [zeta_c*q0, q1]
compute_Qdot  : seal leak rate
find_critical : binary search for zeta_c where A(zeta)/A0 first drops to target_A
"""

from typing import Optional, Tuple

import numpy as np

from .contact import eval_contact_area
from .parameters import Params
from .psd import PSD


def compute_uc(psd: PSD, zeta_c: float, p: Params) -> float:
    """
    Compute u_c = h_rms(zeta_c), the RMS roughness in the band [zeta_c*q0, q1].

    Persson & Yang use this as an estimate for the gap height at the critical
    constriction (the narrowest point of the leak channel).

    From paper equation (4):
        h_rms^2(zeta) = 2*pi * integral_{zeta*q0}^{q1}  q * C(q) dq

    Parameters
    ----------
    psd    : PSD  analytical PSD
    zeta_c : float  critical magnification
    p      : Params

    Returns
    -------
    uc : float  [m]
    """
    q_min = zeta_c * p.q0

    mask  = psd.q >= q_min
    q_sub = psd.q[mask]
    C_sub = psd.C[mask]

    if len(q_sub) < 2:
        return 0.0

    h_rms_sq = 2.0 * np.pi * np.trapezoid(q_sub * C_sub, q_sub)
    return float(np.sqrt(max(h_rms_sq, 0.0)))


def compute_Qdot(uc: float, p: Params) -> float:
    """
    Compute the seal leak rate Q_dot [m^3/s].

    From paper equations (2)-(3):
        M      = alpha * u_c^3 / (12 * eta)
        Q_dot  = (Ly / Lx) * M * deltaP

    Parameters
    ----------
    uc : float  critical gap height [m]
    p  : Params

    Returns
    -------
    Qdot : float  [m^3/s]
    """
    M    = p.alpha * uc ** 3 / (12.0 * p.eta)
    Qdot = (p.Ly / p.Lx) * M * p.deltaP
    return float(Qdot)


def find_critical(
    psd: PSD,
    p: Params,
    zeta_search: Optional[np.ndarray] = None,
    target_A: float = 0.4,
) -> Tuple[float, float, float]:
    """
    Find the critical magnification zeta_c where A(zeta)/A0 first drops to
    target_A, then compute the corresponding lambda_c and u_c.

    Parameters
    ----------
    psd          : PSD  analytical PSD (built with this p)
    p            : Params  (P0 already set to the desired squeezing pressure)
    zeta_search  : 1-D array of magnifications to scan (default: logspace(0,6,4000))
    target_A     : float  percolation threshold (default 0.4)

    Returns
    -------
    zeta_c   : float  critical magnification  (NaN if not found)
    lambda_c : float  critical wavelength [m] (NaN if not found)
    uc       : float  critical gap height [m] (NaN if not found)
    """
    if zeta_search is None:
        zeta_search = np.logspace(0, 6, 4000)

    for zeta in zeta_search:
        A_ratio, _ = eval_contact_area(psd, zeta, p)
        if A_ratio <= target_A:
            zeta_c   = float(zeta)
            lambda_c = (2.0 * np.pi) / (zeta_c * p.q0)
            uc       = compute_uc(psd, zeta_c, p)
            return zeta_c, lambda_c, uc

    return float("nan"), float("nan"), float("nan")
