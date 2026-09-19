"""
fig14.py — Recreate Figure 14 physics: animated GIF + two MP4 videos.

make_fig14_gif : produce fig14.gif, fig14_binary.mp4, fig14_sidebyside.mp4
                 with the same frame-by-frame logic as MATLAB make_fig14_gif.m.
"""

import os
from datetime import datetime
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless backend for frame rendering
import matplotlib.pyplot as plt

from .contact import eval_contact_area
from .parameters import Params
from .psd import PSD


def make_fig14_gif(
    p: Params,
    z: np.ndarray,
    psd_theory: PSD,
    outfile: str = "fig14.gif",
    frame_rate: int = 5,
) -> None:
    """
    Recreate Figure 14: animated contact maps at zeta = 1 .. m/2.

    Outputs (written alongside *outfile*):
        fig14.gif              — looping GIF for web embedding
        fig14_binary.mp4       — single-panel binary contact map
        fig14_sidebyside.mp4   — mesh (left) + contact map (right)
        fig14_YYYYMMDD_HHMMSS.*— timestamped copies of all three

    Color convention: BLACK = contact, WHITE = gap  (paper Fig 14).

    Parameters
    ----------
    p          : Params  (Lx, q0, sigma, H, m, n, P0 used)
    z          : (n, m)  generated rough surface [m]
    psd_theory : PSD     analytical PSD for Persson contact-area formula
    outfile    : str     path of the primary GIF output
    frame_rate : int     frames per second for MP4 and GIF  (default 5)
    """
    try:
        import imageio
    except ImportError:
        raise ImportError(
            "imageio is required for GIF/MP4 output.  "
            "Install with:  pip install imageio imageio-ffmpeg"
        )

    gif_delay = 1.0 / frame_rate

    out_dir    = os.path.dirname(os.path.abspath(outfile))
    base       = os.path.splitext(os.path.basename(outfile))[0]
    gif_file   = outfile
    mp4_bin    = os.path.join(out_dir, f"{base}_binary.mp4")
    mp4_side   = os.path.join(out_dir, f"{base}_sidebyside.mp4")

    for f in (gif_file, mp4_bin, mp4_side):
        if os.path.exists(f):
            os.remove(f)

    n_rows, n_cols = z.shape
    pixel_width    = p.Lx / n_cols
    q_L            = 2.0 * np.pi / p.Lx
    q_nyq          = np.pi / pixel_width
    zeta_max       = int(q_nyq / q_L)   # = n_cols // 2

    p_viz    = p.copy()
    p_viz.P0 = 0.2e6

    print(f"\n--- Fig14 outputs ---")
    print(f"  GIF:          {gif_file}")
    print(f"  MP4 binary:   {mp4_bin}")
    print(f"  MP4 side:     {mp4_side}")
    print(f"  zeta = 1..{zeta_max}  |  {frame_rate} fps  |  P0 = {p_viz.P0/1e6:.2f} MPa\n")

    # ---- Pre-compute FFT once ----
    fx = np.fft.fftshift(np.fft.fftfreq(n_cols))
    fy = np.fft.fftshift(np.fft.fftfreq(n_rows))
    FX, FY = np.meshgrid(fx, fy)
    R      = np.sqrt(FX ** 2 + FY ** 2)
    Z_fft  = np.fft.fftshift(np.fft.fft2(z))

    # Fixed z-limits for mesh (um, 10 % headroom)
    z_lim = float(np.max(np.abs(z))) * 1e6 * 1.1

    # Downsampled mesh grid (4x)
    ds   = 4
    x_mm = np.linspace(0, (n_cols - 1) * pixel_width * 1e3, n_cols)
    y_mm = np.linspace(0, (n_rows - 1) * pixel_width * 1e3, n_rows)
    Xmm, Ymm = np.meshgrid(x_mm, y_mm)
    Xds = Xmm[::ds, ::ds]
    Yds = Ymm[::ds, ::ds]

    # Figure sizes matching MATLAB (cm → inches)
    # Binary: 18 x 19 cm → 7.09 x 7.48 in
    # Side:   36 x 16 cm → 14.17 x 6.30 in
    fig_bin  = plt.figure(figsize=(18 / 2.54, 19 / 2.54), facecolor="white")
    fig_side = plt.figure(figsize=(36 / 2.54, 16 / 2.54), facecolor="white")

    gif_frames: list = []

    # Open MP4 writers
    writer_bin  = imageio.get_writer(mp4_bin,  fps=frame_rate, codec="libx264",
                                     quality=8)
    writer_side = imageio.get_writer(mp4_side, fps=frame_rate, codec="libx264",
                                     quality=8)

    for zeta in range(1, zeta_max + 1):

        # ---- Low-pass filter ----
        frac   = zeta * q_L / q_nyq   # = 2*zeta / n_cols
        mask   = R <= (frac / 2.0)
        z_zoom = np.fft.ifft2(np.fft.ifftshift(Z_fft * mask)).real

        # ---- Contact area (Persson formula) ----
        zeta_P = zeta * q_L / p.q0
        if zeta_P < 1.0:
            A_ratio = 1.0
        else:
            A_ratio, _ = eval_contact_area(psd_theory, zeta_P, p_viz)
            A_ratio = max(0.01, A_ratio)

        thresh     = np.quantile(z_zoom, 1.0 - A_ratio)
        bw_contact = z_zoom >= thresh        # True = in contact
        bw_display = ~bw_contact             # True = gap (white in paper)
        ttl = f'$\\zeta = {zeta}$     $A/A_0 = {A_ratio:.3f}$'

        # ================================================================
        # FIGURE 1: single-panel binary  → GIF + binary MP4
        # ================================================================
        fig_bin.clf()
        ax_b = fig_bin.add_axes([0.05, 0.12, 0.90, 0.78])
        ax_b.imshow(bw_display.astype(float), cmap="gray", vmin=0, vmax=1,
                    origin="upper", aspect="equal")
        ax_b.set_xticks([])
        ax_b.set_yticks([])
        ax_b.set_title(ttl, fontsize=13, fontweight="bold")

        ax_ann = fig_bin.add_axes([0, 0, 1, 0.10])
        ax_ann.axis("off")
        ax_ann.text(0.5, 0.72,
                    "Black = contact  |  White = gap  |  Persson & Yang (2008)",
                    ha="center", fontsize=8, color=(0.3, 0.3, 0.3),
                    transform=ax_ann.transAxes)
        ax_ann.text(0.5, 0.18,
                    (f"$P_0 = {p_viz.P0/1e6:.1f}$ MPa   "
                     f"$\\sigma = {p.sigma*1e6:.1f}\\,\\mu$m   "
                     f"$H = {p.H:.1f}$   {n_cols}×{n_rows} grid"),
                    ha="center", fontsize=7, color=(0.5, 0.5, 0.5),
                    transform=ax_ann.transAxes)

        frame_bin = _fig_to_rgb(fig_bin)
        writer_bin.append_data(frame_bin)
        gif_frames.append(frame_bin)

        # ================================================================
        # FIGURE 2: side-by-side mesh + binary  → side MP4
        # ================================================================
        fig_side.clf()
        ax_m = fig_side.add_subplot(1, 2, 1, projection="3d")
        zds  = z_zoom[::ds, ::ds] * 1e6   # um
        ax_m.plot_surface(Xds, Yds, zds, cmap="turbo",
                          linewidth=0, antialiased=False)
        ax_m.set_zlim(-z_lim, z_lim)
        ax_m.set_xlabel("X (mm)", fontsize=8)
        ax_m.set_ylabel("Y (mm)", fontsize=8)
        ax_m.set_zlabel("Height (µm)", fontsize=8)
        ax_m.set_title(f"Filtered surface  $\\zeta = {zeta}$", fontsize=10)
        ax_m.view_init(elev=30, azim=45)

        ax_c = fig_side.add_subplot(1, 2, 2)
        ax_c.imshow(bw_display.astype(float), cmap="gray", vmin=0, vmax=1,
                    origin="upper", aspect="equal")
        ax_c.set_xticks([])
        ax_c.set_yticks([])
        ax_c.set_title(f"Contact map   $A/A_0 = {A_ratio:.3f}$", fontsize=10)
        fig_side.suptitle(ttl, fontsize=13, fontweight="bold")

        writer_side.append_data(_fig_to_rgb(fig_side))

        if zeta % 32 == 0 or zeta == zeta_max:
            print(f"  Frame {zeta}/{zeta_max}  zeta={zeta}  A/A0={A_ratio:.3f}")

    writer_bin.close()
    writer_side.close()
    plt.close(fig_bin)
    plt.close(fig_side)

    # ---- Save GIF ----
    imageio.mimsave(gif_file, gif_frames, fps=frame_rate, loop=0)

    print("\nAll outputs saved.\n")


def _fig_to_rgb(fig: plt.Figure) -> np.ndarray:
    """Render a matplotlib figure to an (H, W, 3) uint8 RGB array."""
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    # buffer_rgba() replaced tostring_rgb() in matplotlib 3.8+
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).copy()
    return buf.reshape(h, w, 4)[:, :, :3]  # drop alpha
