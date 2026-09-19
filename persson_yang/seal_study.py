"""
seal_study.py — Applied example: rubber seal on a rough substrate.

analyze_seal() produces:
  Figure 1 — three sweep panels (Q vs roughness level, Q vs squeezing force,
             Q vs fluid pressure difference)
  Figure 2 — scenario dashboard: baseline roughness through worst case

Roughness model: each roughness level n adds a fixed RMS increment in
quadrature, sigma = sqrt(sigma0^2 + n * sigma_inc^2), where sigma_inc is set
by a characteristic asperity length scale and the Hertz contact area.
"""

import os
from typing import Tuple

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from .psd import idealized_PSD
from .contact import eval_contact_area
from .core import compute_uc, compute_Qdot
from .parameters import Params


# ---------------------------------------------------------------------------
#  Model constants (SI)
# ---------------------------------------------------------------------------

D_SEAT   = 30e-3      # contact-line diameter [m]
R_TIP    = 0.4e-3     # cylinder tip radius [m]
C_SEAT   = np.pi * D_SEAT   # contact-line length [m]

E_RUB    = 10e6       # rubber Young's modulus [Pa]
NU_RUB   = 0.47
E_STAR   = E_RUB / (1.0 - NU_RUB ** 2)
ETA      = 1e-3       # fluid viscosity [Pa.s]  (water)
Q1       = 7.8e9      # short-wavelength cut-off [1/m]

SIGMA0   = 1e-6       # baseline RMS roughness [m]
H_EXP    = 0.8        # Hurst exponent
L_ASP    = 500e-6     # asperity length scale for the roughness increment [m]

F_NOM    = 13.0       # nominal squeezing force [N]
DP_NOM   = 35e3       # nominal fluid pressure difference [Pa]

M3S_TO_MLMIN = 60.0 * 1e6   # m^3/s -> mL/min


# ---------------------------------------------------------------------------
#  Public entry point
# ---------------------------------------------------------------------------

def analyze_seal(out_dir: str = ".",
                 portfolio_dir: str | None = None,
                 portfolio_orig_dir: str | None = None) -> None:
    """
    Run the rough-substrate seal study and save plots + results text.

    Parameters
    ----------
    out_dir            : directory for all output files (default: current directory)
    portfolio_dir      : if set, outputs are copied to this folder (img/projects/…)
    portfolio_orig_dir : if set, outputs are also copied here (img/originals/…)
    """
    os.makedirs(out_dir, exist_ok=True)
    print("\n=== Rubber Seal on a Rough Substrate  (Persson & Yang 2008) ===\n")

    a_n, P0_n, A_n, _ = _hertz_contact(F_NOM, R_TIP, C_SEAT, E_STAR)
    si_n = _roughness_increment(L_ASP, A_n)

    print("--- Geometry ---")
    print(f"  D={D_SEAT*1e3:.1f} mm  R_tip={R_TIP*1e3:.2f} mm  C={C_SEAT*1e3:.1f} mm")
    print(f"  Hertz (F={F_NOM:.0f} N): a={a_n*1e6:.0f} um  "
          f"P0={P0_n/1e6:.3f} MPa  P0/E*={P0_n/E_STAR:.4f}")
    print(f"  Roughness increment per level: {si_n*1e6:.3f} um")
    for n in range(6):
        print(f"    n={n}: sigma={np.sqrt(SIGMA0**2 + n*si_n**2)*1e6:.3f} um")
    print()

    # ---- Sweep 1: roughness level (n = 0..5) ----
    print(f"Sweep 1: roughness level (n=0..5, F={F_NOM:.0f} N, dP={DP_NOM/1e3:.0f} kPa)")
    n_sw = list(range(6))
    Q_s1 = np.zeros(len(n_sw))
    s_s1 = np.zeros(len(n_sw))
    for k, nv in enumerate(n_sw):
        Q_s1[k], s_s1[k] = _seal_Q(nv, F_NOM, DP_NOM)
        print(f"  n={nv}  sigma={s_s1[k]*1e6:.3f} um  Q={Q_s1[k]:.2e} m3/s")

    # ---- Sweep 2: squeezing force (n = 1,2,3) ----
    print(f"\nSweep 2: squeezing force (n=1,2,3, dP={DP_NOM/1e3:.0f} kPa)")
    F_sw    = [2, 5, 10, 13, 20, 35, 45, 65]            # [N]
    n_for23 = [1, 2, 3]
    Q_s2 = np.array([[_seal_Q(nv, Fv, DP_NOM)[0] for Fv in F_sw] for nv in n_for23])

    # ---- Sweep 3: fluid pressure difference (n = 1,2,3) ----
    print(f"Sweep 3: pressure difference (n=1,2,3, F={F_NOM:.0f} N)")
    dP_sw = [5, 35, 70, 140, 200, 350, 500, 700]        # [kPa]
    Q_s3 = np.array([[_seal_Q(nv, F_NOM, dPv * 1e3)[0] for dPv in dP_sw] for nv in n_for23])

    # ---- Scenario dashboard ----
    print("\nScenario dashboard...")
    scenes = [                       # (name, n, F [N], dP [kPa])
        ("Baseline",        0, 13,  35),
        ("Level 1",         1, 13,  35),
        ("Level 2",         2, 13,  35),
        ("Level 3",         3, 13,  35),
        ("Low force",       1,  4,  35),
        ("High pressure",   1, 13, 200),
        ("Worst case",      3,  4, 200),
    ]
    Q_sc = np.array([_seal_Q(nv, Fv, dPv * 1e3)[0] for _, nv, Fv, dPv in scenes])
    for (name, nv, Fv, dPv), Q in zip(scenes, Q_sc):
        print(f"  {name:<16}  n={nv} F={Fv} N dP={dPv} kPa  Q={Q:.2e} m3/s")

    # ---- Plots + results ----
    with plt.rc_context({"mathtext.fontset": "stix"}):
        _plot_sweeps(n_sw, Q_s1, s_s1, F_sw, Q_s2, n_for23, dP_sw, Q_s3, out_dir)
        _plot_dashboard(scenes, Q_sc, out_dir)
    _print_results(n_sw, Q_s1, s_s1, F_sw, Q_s2, n_for23,
                   dP_sw, Q_s3, scenes, Q_sc, out_dir)

    import pathlib, shutil
    out_files = [
        "roughness_sweep.png", "force_sweep.png", "pressure_sweep.png",
        "seal_param_sweeps.png", "seal_scenarios.png", "seal_results.txt",
    ]

    print("\nSaved files:")
    for fname in out_files:
        p = pathlib.Path(out_dir).resolve() / fname
        print(f"  {p}  ({'EXISTS' if p.exists() else 'MISSING'})")

    # Auto-copy outputs to portfolio directories
    copy_targets = []
    if portfolio_dir      is not None: copy_targets.append(pathlib.Path(portfolio_dir))
    if portfolio_orig_dir is not None: copy_targets.append(pathlib.Path(portfolio_orig_dir))
    if copy_targets:
        print(f"\nCopying outputs to portfolio:")
        for fname in out_files:
            src = pathlib.Path(out_dir).resolve() / fname
            # Results text goes to the served folder only, not originals/
            targets = copy_targets[:1] if fname.endswith(".txt") else copy_targets
            if src.exists():
                for dest_dir in targets:
                    dest_dir.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(src, dest_dir / fname)
                print(f"  {fname}  →  {len(targets)} location(s)")
            else:
                print(f"  {fname}  MISSING — skipped")


# ---------------------------------------------------------------------------
#  Local physics helpers
# ---------------------------------------------------------------------------

def _hertz_contact(
    F_N: float, R_tip: float, C_seat: float, E_star: float
) -> Tuple[float, float, float, float]:
    """Hertz cylinder contact: returns (a, P0, A_contact, q0)."""
    q  = F_N / C_seat
    a  = np.sqrt(4.0 * q * R_tip / (np.pi * E_star))
    P0 = q / (2.0 * a)
    A_cont = 2.0 * a * C_seat
    q0 = 2.0 * np.pi / (2.0 * a)
    return a, P0, A_cont, q0


def _roughness_increment(L_asp: float, A_cont: float) -> float:
    """RMS roughness added per roughness level (asperity scale L_asp over A_cont)."""
    R = L_asp / 2.0
    return R ** 2 * np.sqrt(np.pi / (6.0 * A_cont))


def _seal_Q(n_level: int, F_N: float, dP: float) -> Tuple[float, float]:
    """Full-seal leak rate [m^3/s] and RMS roughness [m] for one operating point."""
    _, P0, A, q0 = _hertz_contact(F_N, R_TIP, C_SEAT, E_STAR)
    sig = np.sqrt(SIGMA0 ** 2 + n_level * _roughness_increment(L_ASP, A) ** 2)
    _, _, _, Q_raw = _persson_solve(sig, H_EXP, P0, q0, Q1, A,
                                    dP, ETA, E_RUB, NU_RUB)
    return Q_raw * (C_SEAT / (2.0 * np.pi / q0)), sig


def _persson_solve(
    sigma: float, H: float, P0: float, q0: float,
    q1: float, A0: float, dP: float, eta: float, E: float, nu: float
) -> Tuple[float, float, float, float]:
    """Persson solve → (zeta_c, lambda_c, u_c, Q_dot)."""
    p = Params(
        E=E, nu=nu, sigma=sigma, H=H,
        q0=q0, qr=q0, q1=q1,
        P0=P0, eta=eta, deltaP=dP,
        Lx=2.0 * np.pi / q0, Ly=2.0 * np.pi / q0,
        alpha=1.0, pc=0.6, m=512, n=512,
    )
    psd = idealized_PSD(p)

    zeta_v = np.logspace(0, 8, 12000)
    for zeta in zeta_v:
        Ar, _ = eval_contact_area(psd, zeta, p)
        if Ar <= 0.4:
            zc = float(zeta)
            lc = 2.0 * np.pi / (zc * q0)
            uc = compute_uc(psd, zc, p)
            Qdot = compute_Qdot(uc, p)
            return zc, lc, uc, Qdot

    return float("nan"), float("nan"), float("nan"), 0.0


# ---------------------------------------------------------------------------
#  Plot helpers
# ---------------------------------------------------------------------------

def _style_ax(ax, fsz: int = 10) -> None:
    """Apply shared axis style (matches MATLAB style_ax helper)."""
    ax.set_facecolor("white")
    ax.tick_params(direction="in", which="both", labelsize=fsz,
                   top=True, right=True, colors="black")
    ax.spines[:].set_color("black")
    ax.grid(True, linestyle=":", alpha=0.25)
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
    # Times New Roman — matches MATLAB ax.FontName = 'Times New Roman'
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label]
                 + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontfamily("serif")
        try:
            item.set_fontname("Times New Roman")
        except Exception:
            pass


_C_RAMP = [
    [0.00, 0.45, 0.74],   # blue   (n=1)
    [0.93, 0.51, 0.08],   # orange (n=2)
    [0.80, 0.10, 0.10],   # red    (n=3)
]
_DOT_CLR = [
    [0.20, 0.70, 0.20],
    [0.00, 0.45, 0.74],
    [0.93, 0.51, 0.08],
    [0.80, 0.10, 0.10],
    [0.55, 0.00, 0.55],
    [0.20, 0.00, 0.00],
]
_Q_MIN = 1e-20   # sentinel for "sealed" cases

_COND_F  = f"$F = {F_NOM:.0f}$ N"

# Compact canvases (inches) so text reads large on the site; saved tight-cropped
_FIG_SINGLE = (5.4, 3.85)
_FIG_COMBO  = (6.2, 3.8)
_FIG_DASH   = (6.0, 3.6)
_SAVE = dict(dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.03)


def _legend(ax, loc, fsz, standalone) -> None:
    """Serif legend; tighter spacing on the combined figure's small panels."""
    kw = {} if standalone else dict(handlelength=1.4, borderpad=0.3,
                                    labelspacing=0.2, handletextpad=0.4)
    leg = ax.legend(loc=loc, prop={"family": "serif",
                                   "size": fsz - (1 if standalone else 1.5)}, **kw)
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_edgecolor("black")
_COND_DP = f"$\\Delta P = {DP_NOM/1e3:.0f}$ kPa"


def _plot_sweeps(n_sw, Q_s1, sig_s1, F_sw, Q_s2, n_for23,
                 dP_sw, Q_s3, out_dir: str) -> None:
    """Three individual sweep figures + one combined 1×3 figure."""
    n_sw   = np.array(n_sw)
    Q_s1   = np.array(Q_s1)
    sig_s1 = np.array(sig_s1)

    Q1_plot  = np.maximum(Q_s1, _Q_MIN)
    real_idx = np.where(Q_s1 > 0)[0]
    x_F  = np.array(F_sw,  dtype=float)
    x_dP = np.array(dP_sw, dtype=float)

    # ---------- Figure 1: Q vs roughness level ----------
    fig1, ax1 = plt.subplots(figsize=_FIG_SINGLE, facecolor="white")
    _draw_sweep1(ax1, n_sw, Q_s1, Q1_plot, sig_s1, real_idx, fsz=10, standalone=True)
    fig1.savefig(os.path.join(out_dir, "roughness_sweep.png"), **_SAVE)
    plt.close(fig1)

    # ---------- Figure 2: Q vs squeezing force ----------
    fig2, ax2 = plt.subplots(figsize=_FIG_SINGLE, facecolor="white")
    _draw_sweep2(ax2, x_F, Q_s2, n_for23, fsz=10, standalone=True)
    fig2.savefig(os.path.join(out_dir, "force_sweep.png"), **_SAVE)
    plt.close(fig2)

    # ---------- Figure 3: Q vs fluid pressure difference ----------
    fig3, ax3 = plt.subplots(figsize=_FIG_SINGLE, facecolor="white")
    _draw_sweep3(ax3, x_dP, Q_s3, n_for23, fsz=10, standalone=True)
    fig3.savefig(os.path.join(out_dir, "pressure_sweep.png"), **_SAVE)
    plt.close(fig3)

    # ---------- Combined 1×3 ----------
    fig_c = plt.figure(figsize=_FIG_COMBO, facecolor="white")
    axs = fig_c.subplots(1, 3)
    fig_c.suptitle("Seal Leak Rate  —  Persson & Yang (2008)",
                   fontsize=11, fontweight="bold", fontfamily="serif", y=1.0)

    _draw_sweep1(axs[0], n_sw, Q_s1, Q1_plot, sig_s1, real_idx, fsz=8, standalone=False)
    _draw_sweep2(axs[1], x_F, Q_s2, n_for23, fsz=8, standalone=False)
    _draw_sweep3(axs[2], x_dP, Q_s3, n_for23, fsz=8, standalone=False)

    fig_c.subplots_adjust(top=0.80, wspace=0.55)
    fig_c.savefig(os.path.join(out_dir, "seal_param_sweeps.png"), **_SAVE)
    plt.close(fig_c)


def _draw_sweep1(ax, n_sw, Q_s1, Q1_plot, sig_s1, real_idx, fsz,
                 standalone: bool = True) -> None:
    if len(real_idx) > 0:
        if real_idx[0] > 0:
            ax.semilogy(n_sw[:real_idx[0]+1], Q1_plot[:real_idx[0]+1],
                        "--", color=(0.65, 0.65, 0.65), lw=1.0)
        ax.semilogy(n_sw[real_idx[0]:], Q1_plot[real_idx[0]:],
                    "-", color=(0.40, 0.40, 0.40), lw=1.5)

    for k, nv in enumerate(n_sw):
        c  = _DOT_CLR[min(k, len(_DOT_CLR)-1)]
        dn = f"n={nv}, σ={sig_s1[k]*1e6:.2f} μm"
        if Q_s1[k] <= 0:
            ax.semilogy(nv, _Q_MIN, "v", ms=0.6*fsz, lw=0.8,
                        mec=c, mfc="white", label=dn + " (sealed)")
        else:
            ax.semilogy(nv, Q1_plot[k], "o", ms=0.6*fsz, lw=0.8,
                        mec=[x*0.75 for x in c], mfc=c, label=dn)

    ax.set_xlim(-0.4, max(n_sw) + 1.0)
    ax.set_ylim(_Q_MIN * 0.4, max(Q1_plot) * 8e3)
    ax.set_xticks(n_sw)
    _style_ax(ax, fsz)
    ax.set_xlabel("Surface roughness level  $n$", fontsize=fsz+1)
    ax.set_ylabel("$\\dot{Q}$ (m$^3$/s)", fontsize=fsz+1)
    cond = f"{_COND_F},  {_COND_DP}"
    if standalone:
        ax.set_title(f"Surface roughness sweep  —  Persson & Yang (2008)\n{cond}",
                     fontsize=fsz)
    else:
        ax.set_title(f"Surface roughness sweep\n{cond}", fontsize=fsz)
    _legend(ax, "lower right", fsz, standalone)


def _draw_sweep2(ax, F_sw, Q_s2, n_for23, fsz, standalone: bool = True) -> None:
    for ni, nv in enumerate(n_for23):
        Q2 = Q_s2[ni].copy().astype(float)
        Q2[Q2 <= 0] = np.nan
        if np.any(np.isfinite(Q2)):
            ax.plot(F_sw, Q2, "-o", color=_C_RAMP[ni], lw=1.6,
                    mfc=_C_RAMP[ni], ms=0.45*fsz,
                    label=f"Roughness level {nv}" if standalone else f"n = {nv}")

    ax.axvline(F_NOM, ls="--", color=(0.5, 0.5, 0.5), lw=0.8)
    ax.plot([], [], "--", color=(0.5, 0.5, 0.5), lw=0.8,
            label=f"Nominal force ({F_NOM:.0f} N)" if standalone else "Nominal")
    if standalone:
        ax.plot([], [], "v", mec=_DOT_CLR[0], mfc="white", ms=4,
                ls="none", label="Baseline roughness (n=0): sealed")

    ax.set_xlim(-1.0, max(F_sw) * 1.15)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set(fontsize=fsz, fontfamily="serif")
    _style_ax(ax, fsz)
    ax.set_xlabel("Squeezing force  $F$ (N)", fontsize=fsz+1)
    ax.set_ylabel("$\\dot{Q}$ (m$^3$/s)", fontsize=fsz+1)
    if standalone:
        ax.set_title(f"Squeezing force sweep  —  Persson & Yang (2008)\n{_COND_DP}",
                     fontsize=fsz)
        leg_loc = "best"
    else:
        ax.set_title(f"Squeezing force sweep\n{_COND_DP}", fontsize=fsz)
        leg_loc = "upper right"
    _legend(ax, leg_loc, fsz, standalone)


def _draw_sweep3(ax, dP_sw, Q_s3, n_for23, fsz, standalone: bool = True) -> None:
    for ni, nv in enumerate(n_for23):
        Q3 = Q_s3[ni].copy().astype(float)
        Q3[Q3 <= 0] = np.nan
        if np.any(np.isfinite(Q3)):
            ax.plot(dP_sw, Q3, "-s", color=_C_RAMP[ni], lw=1.6,
                    mfc=_C_RAMP[ni], ms=0.45*fsz,
                    label=f"Roughness level {nv}" if standalone else f"n = {nv}")

    ax.axvline(DP_NOM / 1e3, ls="--", color=(0.5, 0.5, 0.5), lw=0.8)
    ax.plot([], [], "--", color=(0.5, 0.5, 0.5), lw=0.8,
            label=f"Nominal ΔP ({DP_NOM/1e3:.0f} kPa)" if standalone else "Nominal")

    ax.set_xlim(min(dP_sw) * 0.5, max(dP_sw) * 1.12)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    ax.yaxis.get_offset_text().set(fontsize=fsz, fontfamily="serif")
    _style_ax(ax, fsz)
    ax.set_xlabel("Fluid pressure difference  $\\Delta P$ (kPa)", fontsize=fsz+1)
    ax.set_ylabel("$\\dot{Q}$ (m$^3$/s)", fontsize=fsz+1)
    if standalone:
        ax.set_title(f"Pressure sweep  —  Persson & Yang (2008)\n{_COND_F}",
                     fontsize=fsz)
        leg_loc = "best"
    else:
        ax.set_title(f"Pressure sweep\n{_COND_F}", fontsize=fsz)
        leg_loc = "upper left"
    _legend(ax, leg_loc, fsz, standalone)


def _plot_dashboard(scenes, Q_sc, out_dir: str) -> None:
    """Scenario dashboard bar chart (baseline roughness → worst case)."""
    n_sc = len(scenes)
    t    = np.linspace(0, 1, n_sc)
    clr  = np.column_stack([t**0.5, (1-t)**0.5, np.zeros(n_sc)]) * 0.80 + 0.05

    fig, ax = plt.subplots(figsize=_FIG_DASH, facecolor="white")

    Q_plot  = np.maximum(Q_sc, _Q_MIN)
    xlabels = [f"{name}  (n={nv}, {Fv} N, {dPv} kPa)" for name, nv, Fv, dPv in scenes]

    ylo = _Q_MIN * 0.8
    yhi = max(Q_plot) * 5e3
    for s in range(n_sc):
        ax.bar(s + 1, Q_plot[s], width=0.65, color=clr[s], edgecolor="none")
        lbl = f"{Q_sc[s]:.2e} m$^3$/s"
        if np.log10(Q_plot[s] / ylo) > 9:     # tall bar: label inside
            y_lbl, va = np.exp(0.5 * (np.log(ylo) + np.log(Q_plot[s]))), "center"
        else:                                   # short bar: label above
            y_lbl, va = Q_plot[s] * 3.0, "bottom"
        ax.text(s + 1, y_lbl, lbl, rotation=90, ha="center", va=va,
                fontsize=8, fontfamily="serif")

    ax.fill_between([0.4, 1.6], ylo, yhi, color=(0.85, 0.97, 0.85), alpha=0.35)
    ax.fill_between([1.6, n_sc + 0.6], ylo, yhi, color=(0.97, 0.85, 0.85), alpha=0.35)
    y_zone = np.exp(np.log(ylo) * 0.10 + np.log(yhi) * 0.90)
    ax.text(1, y_zone, "SEALING", ha="center", fontsize=9, fontweight="bold")
    ax.text((1.6 + n_sc + 0.6) / 2, y_zone, "LEAKING",
            ha="center", fontsize=9, fontweight="bold")

    ax.set_yscale("log")
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(0.4, n_sc + 0.6)
    ax.set_xticks(range(1, n_sc + 1))
    ax.set_xticklabels(xlabels, rotation=40, ha="right", fontsize=8.5)

    _style_ax(ax, 9)
    ax.set_ylabel("$\\dot{Q}$ (m$^3$/s)", fontsize=10)
    ax.set_title(
        "Rubber Seal on a Rough Substrate  —  Persson & Yang (2008)\n"
        "Percolation threshold: contact area < 40 %  →  connected gap  →  leakage",
        fontsize=9,
    )

    fig.savefig(os.path.join(out_dir, "seal_scenarios.png"), **_SAVE)
    plt.close(fig)


def _print_results(
    n_sw, Q_s1, sig_s1,
    F_sw, Q_s2, n_for23,
    dP_sw, Q_s3,
    scenes, Q_sc, out_dir: str,
) -> None:
    """Write seal_results.txt and mirror to console."""
    SEP = "-" * 80

    lines = [
        "=== RUBBER SEAL ON A ROUGH SUBSTRATE -- LEAK RATE RESULTS ===",
        "Persson & Yang (2008)  |  Full seal contact length (patch × C/Lx)",
        "Q = 0 reported as 'SEALED' (no percolation path)",
        "",
        f"--- SURFACE ROUGHNESS SWEEP  (F = {F_NOM:.0f} N,  dP = {DP_NOM/1e3:.0f} kPa) ---",
        f"  {'n':<5}  {'sigma(um)':<10}  {'Q (m^3/s)':<16}  Q (mL/min)",
        "  " + SEP,
    ]
    for k, nv in enumerate(n_sw):
        Q = Q_s1[k]
        if Q > 1e-19:
            lines.append(f"  {nv:<5d}  {sig_s1[k]*1e6:<10.3f}  "
                         f"{Q:<16.3e}  {Q*M3S_TO_MLMIN:.3g}")
        else:
            lines.append(f"  {nv:<5d}  {sig_s1[k]*1e6:<10.3f}  SEALED")

    lines += ["", f"--- SQUEEZING FORCE SWEEP  (dP = {DP_NOM/1e3:.0f} kPa) ---"]
    hdr = f"  {'F (N)':<12}" + "".join(
        f"  {'n='+str(nv)+'  Q(m^3/s)':<22}" for nv in n_for23)
    lines += [hdr, "  " + SEP]
    for k, Fv in enumerate(F_sw):
        row = f"  {Fv:<12.0f}"
        for ni in range(len(n_for23)):
            Q = Q_s2[ni, k]
            row += f"  {Q:<22.3e}" if Q > 1e-19 else f"  {'SEALED':<22}"
        if Fv == F_NOM:
            row += "  <-- nominal"
        lines.append(row)

    lines += ["", f"--- PRESSURE DIFFERENCE SWEEP  (F = {F_NOM:.0f} N) ---"]
    hdr = f"  {'dP (kPa)':<12}" + "".join(
        f"  {'n='+str(nv)+'  Q(m^3/s)':<22}" for nv in n_for23)
    lines += [hdr, "  " + SEP]
    for k, dPv in enumerate(dP_sw):
        row = f"  {dPv:<12.0f}"
        for ni in range(len(n_for23)):
            Q = Q_s3[ni, k]
            row += f"  {Q:<22.3e}" if Q > 1e-19 else f"  {'SEALED':<22}"
        if dPv * 1e3 == DP_NOM:
            row += "  <-- nominal"
        lines.append(row)

    lines += [
        "",
        "--- SCENARIO DASHBOARD ---",
        f"  {'Scenario':<40}  {'Q (m^3/s)':<16}  Q (mL/min)",
        "  " + SEP,
    ]
    for (name, nv, Fv, dPv), Q in zip(scenes, Q_sc):
        nm = f"{name}  (n={nv}, {Fv} N, {dPv} kPa)"
        if Q > 1e-19:
            lines.append(f"  {nm:<40}  {Q:<16.3e}  {Q*M3S_TO_MLMIN:.3g}")
        else:
            lines.append(f"  {nm:<40}  SEALED")
    lines.append("")

    fpath = os.path.join(out_dir, "seal_results.txt")
    with open(fpath, "w", encoding="utf-8") as fid:
        for line in lines:
            print(line)
            fid.write(line + "\n")
    print(f"Results written to: {fpath}\n")
