"""
plots.py — Reproduce Persson & Yang (2008) Figures 6-11.

make_plots() sweeps P0 for four sigma values (Figs 6-8) and four H values
(Figs 9-11), then saves IOP-journal-style PNGs.

Formatting notes vs MATLAB make_plots.m:
  - log10(y) plotted on a linear axis  →  same approach, manual ytick labels
  - exportgraphics 300 dpi             →  fig.savefig(..., dpi=300)
  - Inline end-of-curve labels         →  ax.text with clip_on=False
  - ax.Position [left bottom w h]      →  ax.set_position(...)
  - Font 'Helvetica'                   →  rcParam sans-serif (DejaVu Sans)
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FixedFormatter

from .parameters import load_parameters, Params
from .psd import idealized_PSD
from .core import find_critical, compute_Qdot


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def make_plots(out_dir: str = ".") -> None:
    """
    Reproduce Persson & Yang (2008) Figures 6-11 and save as PNG files.

    Figures 6-8 : lambda_c, u_c, Q_dot vs P0  (sigma = 1,2,4,6 um, H=0.8)
    Figures 9-11: lambda_c, u_c, Q_dot vs P0  (H = 0.6,0.7,0.8,0.9, sigma=2um)

    Parameters
    ----------
    out_dir : str  directory to save PNG files (default: current directory)
    """
    import os
    os.makedirs(out_dir, exist_ok=True)

    print("Generating Figures 6-11 (Persson & Yang style)...")

    p_base      = load_parameters()
    P0_vec      = np.linspace(0.05e6, 1.0e6, 60)   # 0.05 – 1 MPa
    target_A    = 0.4
    zeta_search = np.logspace(0, 6, 4000)

    # ---- Sweep 1: vary sigma, H = 0.8 ----
    sigmas     = [1e-6, 2e-6, 4e-6, 6e-6]
    sig_labels = [r'rms = 1 $\mu$m', r'rms = 2 $\mu$m',
                  r'rms = 4 $\mu$m', r'rms = 6 $\mu$m']

    lc_s  = np.full((4, len(P0_vec)), np.nan)
    uc_s  = np.full((4, len(P0_vec)), np.nan)
    Qd_s  = np.full((4, len(P0_vec)), np.nan)

    for i, sig in enumerate(sigmas):
        p = p_base.copy()
        p.sigma = sig
        p.H = 0.8
        psd = idealized_PSD(p)
        for j, P0 in enumerate(P0_vec):
            p.P0 = P0
            zc, lc, uc = find_critical(psd, p, zeta_search, target_A)
            if not np.isnan(zc):
                lc_s[i, j] = lc
                uc_s[i, j] = uc
                Qd_s[i, j] = compute_Qdot(uc, p)
        print(f"  sigma sweep {i+1}/4 done")

    # ---- Sweep 2: vary H, sigma = 2 um ----
    H_vals    = [0.6, 0.7, 0.8, 0.9]
    H_labels  = ['H = 0.6', 'H = 0.7', 'H = 0.8', 'H = 0.9']

    lc_H  = np.full((4, len(P0_vec)), np.nan)
    uc_H  = np.full((4, len(P0_vec)), np.nan)
    Qd_H  = np.full((4, len(P0_vec)), np.nan)

    for i, Hv in enumerate(H_vals):
        p = p_base.copy()
        p.sigma = 2e-6
        p.H = Hv
        psd = idealized_PSD(p)
        for j, P0 in enumerate(P0_vec):
            p.P0 = P0
            zc, lc, uc = find_critical(psd, p, zeta_search, target_A)
            if not np.isnan(zc):
                lc_H[i, j] = lc
                uc_H[i, j] = uc
                Qd_H[i, j] = compute_Qdot(uc, p)
        print(f"  H sweep {i+1}/4 done")

    P0_MPa = P0_vec / 1e6

    # Colour sequences matching paper (largest sigma / highest H on top)
    # cyan / green / orange / blue  (top → bottom curve order)
    clr = [
        [0.00, 0.75, 0.75],   # cyan   (6 um — top)
        [0.13, 0.55, 0.13],   # green  (4 um)
        [0.85, 0.33, 0.10],   # orange (2 um)
        [0.00, 0.00, 0.80],   # blue   (1 um — bottom)
    ]
    clrH = clr  # same colour sequence for H sweep

    # Plot order: highest curve first (index [3,2,1,0] of the sweep arrays)
    order = [3, 2, 1, 0]

    # ---- Figure 6 ----
    _paper_fig(
        P0_MPa, lc_s[order, :], clr,
        [sig_labels[k] for k in order],
        xlims=(0, 1), ylims=(-8, -3),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,\lambda_c$ (m)',
        fname=f"{out_dir}/Fig6_lambda_c_sigma.png",
    )

    # ---- Figure 7 ----
    _paper_fig(
        P0_MPa, uc_s[order, :], clr,
        [sig_labels[k] for k in order],
        xlims=(0, 1), ylims=(-10, -5),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,u_c$ (m)',
        fname=f"{out_dir}/Fig7_uc_sigma.png",
    )

    # ---- Figure 8 ----
    _paper_fig(
        P0_MPa, Qd_s[order, :], clr,
        [sig_labels[k] for k in order],
        xlims=(0, 1), ylims=(-25, -9),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,\dot{Q}$ (m$^3$/s)',
        fname=f"{out_dir}/Fig8_Qdot_sigma.png",
    )

    # ---- Figure 9 ----
    _paper_fig(
        P0_MPa, lc_H[order, :], clrH,
        [H_labels[k] for k in order],
        xlims=(0, 1), ylims=(-8, -3),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,\lambda_c$ (m)',
        fname=f"{out_dir}/Fig9_lambda_c_H.png",
    )

    # ---- Figure 10 ----
    _paper_fig(
        P0_MPa, uc_H[order, :], clrH,
        [H_labels[k] for k in order],
        xlims=(0, 1), ylims=(-10, -6),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,u_c$ (m)',
        fname=f"{out_dir}/Fig10_uc_H.png",
    )

    # ---- Figure 11 ----
    _paper_fig(
        P0_MPa, Qd_H[order, :], clrH,
        [H_labels[k] for k in order],
        xlims=(0, 1), ylims=(-27, -11),
        xlabel='squeezing pressure (MPa)',
        ylabel=r'$\log\,\dot{Q}$ (m$^3$/s)',
        fname=f"{out_dir}/Fig11_Qdot_H.png",
    )

    print(f"\nDone. PNGs saved to: {out_dir}")


# ---------------------------------------------------------------------------
#  Private helper: one IOP-style paper panel
# ---------------------------------------------------------------------------

def _paper_fig(
    x: np.ndarray,
    Y: np.ndarray,
    clr: list,
    labels: list,
    xlims: tuple,
    ylims: tuple,
    xlabel: str,
    ylabel: str,
    fname: str,
) -> None:
    """
    Draw one paper-style figure panel and save to fname.

    Y is (nCurves, nPoints).  Curves are plotted in row order (row 0 = top).
    Y values are log10-transformed before plotting (paper y-axis is log).
    """
    # Figure size matching MATLAB: 9 x 8 cm
    fig, ax = plt.subplots(figsize=(9 / 2.54, 8 / 2.54), facecolor="white")
    ax.set_facecolor("white")

    n_curves = Y.shape[0]

    for i in range(n_curves):
        y = np.log10(Y[i, :])
        y[~np.isfinite(y)] = np.nan
        ax.plot(x, y, "-", color=clr[i], linewidth=1.6)

        # Inline end-of-curve label (rightmost finite point)
        finite = np.where(np.isfinite(y))[0]
        if len(finite) == 0:
            continue
        idx = finite[-1]
        ax.text(
            x[idx] + 0.01, y[idx],
            labels[i],
            color=clr[i],
            fontsize=7,
            fontfamily="sans-serif",
            va="center", ha="left",
            clip_on=False,
        )

    # Axes formatting
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)

    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ytick_vals = list(range(int(ylims[0]), int(ylims[1]) + 1))
    ax.yaxis.set_major_locator(FixedLocator(ytick_vals))
    ax.yaxis.set_major_formatter(FixedFormatter([str(v) for v in ytick_vals]))

    ax.tick_params(direction="in", which="both", top=True, right=True,
                   labelsize=8)
    ax.spines[:].set_linewidth(0.8)
    ax.spines[:].set_color("black")

    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    # Tight layout matching MATLAB ax.Position = [0.14 0.13 0.72 0.80]
    ax.set_position([0.14, 0.13, 0.72, 0.80])

    fig.savefig(fname, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
