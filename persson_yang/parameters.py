"""
parameters.py — Physical/numerical parameters for the Persson-Yang model.

Paper values (Section 3):
  E = 10 MPa, nu = 0.5, q0 = 1e4 m^-1, q1 = 7.8e9 m^-1,
  deltaP = 0.01 MPa, eta = 1e-3 Pa.s, Ly/Lx = 1
"""

import copy
from dataclasses import dataclass


@dataclass
class Params:
    """All physical/numerical parameters for the Persson-Yang seal leak-rate model."""

    # --- Rubber elastic properties ---
    E: float = 10e6     # Young's modulus [Pa]  (low-freq rubber modulus)
    nu: float = 0.5     # Poisson ratio

    # --- Surface roughness ---
    sigma: float = 2e-6  # RMS roughness [m]  (default; swept in plots)
    H: float = 0.8       # Hurst exponent  (fractal dim Df = 3 - H = 2.2)

    # --- Wavevectors (paper Section 3) ---
    q0: float = 1.0e4   # Long-wavelength cut-off [1/m]
    qr: float = 1.0e4   # Roll-off wavevector [1/m]  (= q0 → no flat region)
    q1: float = 7.8e9   # Short-wavelength cut-off [1/m]

    # --- Squeezing pressure (default; swept in figure loops) ---
    P0: float = 0.5e6   # [Pa]

    # --- Fluid properties (paper Section 3) ---
    eta: float = 1e-3       # Dynamic viscosity [Pa.s]  (water)
    deltaP: float = 0.01e6  # Pressure difference across seal [Pa]

    # --- Seal geometry ---
    Lx: float = 1e-2   # Seal width in x-direction [m]  (1 cm)
    Ly: float = 1e-2   # Seal width in y-direction [m]  (Ly/Lx = 1)

    # --- Numerical surface generation ---
    m: int = 512        # Pixels in x
    n: int = 512        # Pixels in y

    # --- Leak-rate pre-factor (order unity, Section 2) ---
    alpha: float = 1.0

    # --- Percolation threshold (site percolation on 2D grid) ---
    pc: float = 0.6     # → A(zeta_c)/A0 = 1 - pc ≈ 0.4

    @property
    def A0(self) -> float:
        """Nominal contact area [m^2]."""
        return self.Lx * self.Ly

    def copy(self) -> "Params":
        """Return a shallow copy (safe since all fields are scalars)."""
        return copy.copy(self)


def load_parameters() -> Params:
    """Return default Persson-Yang parameters (matches MATLAB load_parameters.m)."""
    return Params()
