"""
main.py — Persson-Yang seal leak-rate simulation  (Python port of v6)

Based on: Persson & Yang, J. Phys.: Condens. Matter 20 (2008) 315011

HOW TO RUN:
  1.  pip install -r requirements.txt
  2.  python main.py

QUICK START — just the paper figures:
  Set do_make_plots = True, leave everything else False, run.
  Six PNG files will be saved to the current directory.
"""

import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # change to "Qt5Agg" or "Agg" if TkAgg unavailable
import matplotlib.pyplot as plt

from persson_yang import (
    load_parameters,
    idealized_PSD,
    artificial_surf,
    calculate_rms,
    estimate_hurst_exponent,
    eval_contact_area,
    zoom_surface_spatial,
    check_percolation_path,
    compute_uc,
    compute_Qdot,
    find_critical,
    make_plots,
    make_fig14_gif,
    analyze_seal,
)

# =============================================================================
#  Control Toggles — set to True to enable each section
# =============================================================================
do_plot_surface  = False   # 3-D mesh of the generated rough surface
do_plot_PSD      = False   # log-log PSD of the generated surface
do_rms_calc      = False   # measure Rq of the generated surface
do_Hurst_calc    = False   # estimate H from the generated surface PSD
do_fig14_gif     = False   # GIF + MP4: contact maps at zeta=1..zeta_max
do_path_search   = False   # GIF: BFS leak-path at the critical zoom level
do_ratio_test    = False   # plot A/A0 vs zeta (sanity check)
do_make_plots    = True    # reproduce paper Figures 6-11  ← main result
do_seal_study    = False   # applied example: rubber seal on a rough substrate

# =============================================================================
#  Parameters
# =============================================================================
p = load_parameters()

# =============================================================================
#  Surface Generation  (needed for all visual/GIF sections)
# =============================================================================
z = None
pixel_width = None
psd_surf = None

if any([do_plot_surface, do_plot_PSD, do_rms_calc, do_Hurst_calc,
        do_fig14_gif, do_path_search, do_ratio_test]):
    print(f"Generating rough surface ({p.m}x{p.n}, "
          f"sigma={p.sigma*1e6:.1f} um, H={p.H:.2f})...")
    z, pixel_width, psd_surf = artificial_surf(p.sigma, p.H, p.Lx, p.m, p.n, p.qr)
    x_arr = np.linspace(0, (p.m - 1) * pixel_width, p.m)
    y_arr = np.linspace(0, (p.n - 1) * pixel_width, p.n)
    X, Y = np.meshgrid(x_arr, y_arr)
    print(f"  Done. PixelWidth = {pixel_width:.3e} m")

# Analytical PSD (used for contact-area threshold in all GIF sections)
psd_theory = idealized_PSD(p)

# =============================================================================
#  Plot Surface
# =============================================================================
if do_plot_surface:
    fig = plt.figure()
    ax  = fig.add_subplot(111, projection="3d")
    # Downsample 4x for rendering speed
    ds = 4
    ax.plot_surface(X[::ds, ::ds] * 1e3, Y[::ds, ::ds] * 1e3,
                    z[::ds, ::ds] * 1e6, cmap="jet")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Height (µm)")
    ax.set_title("Generated Rough Surface")
    plt.tight_layout()
    plt.show()

# =============================================================================
#  Plot PSD
# =============================================================================
if do_plot_PSD:
    fig, ax = plt.subplots()
    ax.loglog(psd_surf.q, psd_surf.C, "b-", linewidth=1.5)
    ax.set_xlabel("Wavevector q (1/m)")
    ax.set_ylabel("C(q)  (m⁴)")
    ax.set_title("Radially-Averaged PSD of Generated Surface")
    ax.grid(True, which="both")
    plt.tight_layout()
    plt.show()

# =============================================================================
#  Surface Analysis
# =============================================================================
if do_rms_calc:
    Rq = calculate_rms(z)
    print(f"Measured RMS roughness: {Rq:.3e} m  (input sigma = {p.sigma:.3e} m)")

if do_Hurst_calc:
    H_est = estimate_hurst_exponent(z, pixel_width)
    print(f"Estimated Hurst exponent: {H_est:.3f}  (input H = {p.H:.2f})")

# =============================================================================
#  Figure 14 GIF
# =============================================================================
if do_fig14_gif:
    # Resize Lx so q0 = 3 * q_L  (matches the paper's MD simulation setup)
    p_f14 = p.copy()
    p_f14.Lx = 3.0 * (2.0 * np.pi / p.q0)   # ~1.885 mm
    p_f14.Ly = p_f14.Lx
    print(f"Fig14: regenerating surface with Lx = {p_f14.Lx*1e3:.3f} mm ...")
    z_f14, _, _ = artificial_surf(p_f14.sigma, p_f14.H, p_f14.Lx,
                                  p_f14.m, p_f14.n, p_f14.qr)
    psd_f14 = idealized_PSD(p_f14)
    make_fig14_gif(p_f14, z_f14, psd_f14, "fig14.gif")

# =============================================================================
#  Percolation Path Search  (BFS GIF at the critical zoom level)
# =============================================================================
if do_path_search:
    pixel_width_loc = p.Lx / p.m
    q_L_ps  = 2.0 * np.pi / p.Lx
    q_nyq   = np.pi / pixel_width_loc
    zeta_vis_max = int(q_nyq / q_L_ps)   # = m // 2

    p_viz = p.copy()
    p_viz.P0 = 0.2e6

    perc_zoom       = None
    perc_bw_contact = None
    perc_bw_gap     = None

    for zeta_vis in range(1, zeta_vis_max + 1):
        z_zoom, zeta_P = zoom_surface_spatial(z, zeta_vis, p)
        if zeta_P < 1.0:
            A_th = 1.0
        else:
            A_th, _ = eval_contact_area(psd_theory, zeta_P, p_viz)
        A_th = max(0.01, A_th)
        thresh        = np.quantile(z_zoom, 1.0 - A_th)
        bw_contact    = z_zoom >= thresh
        bw_gap        = ~bw_contact
        found, _, _   = check_percolation_path(bw_gap, do_frames=False)
        if found:
            perc_zoom       = zeta_vis
            perc_bw_contact = bw_contact
            perc_bw_gap     = bw_gap
            print(f"Path search: percolation at zeta_vis={zeta_vis}  "
                  f"zeta_P={zeta_P:.1f}  (A/A0={A_th:.3f})")
            break

    if perc_zoom is not None:
        import imageio
        out_path = "perc_path_search.gif"
        _, path_mask, bfs_frames = check_percolation_path(perc_bw_gap, do_frames=True)

        gif_frames = []
        for img in bfs_frames:
            gif_frames.append(img)

        # Final frame: contact map with leak path in red
        rgb_final = np.stack([
            np.where(path_mask, 255, perc_bw_contact.astype(np.uint8) * 200),
            np.where(path_mask, 0,   perc_bw_contact.astype(np.uint8) * 200),
            np.where(path_mask, 0,   perc_bw_contact.astype(np.uint8) * 200),
        ], axis=-1).astype(np.uint8)

        # Convert grayscale BFS frames to RGB for imageio
        rgb_bfs = [np.stack([f, f, f], axis=-1) for f in gif_frames]
        rgb_bfs.append(rgb_final)

        imageio.mimsave(out_path, rgb_bfs, fps=20, loop=0)
        print(f"Saved: {out_path}")
    else:
        print("No percolation channel found in path search.")

# =============================================================================
#  A/A0 Ratio Scan  (sanity check)
# =============================================================================
if do_ratio_test:
    p_rt = p.copy()
    p_rt.Lx = 3.0 * (2.0 * np.pi / p.q0)
    p_rt.Ly = p_rt.Lx
    q_L_rt  = 2.0 * np.pi / p_rt.Lx
    pw_rt   = p_rt.Lx / p_rt.m
    q_nyq_rt = np.pi / pw_rt
    zeta_vis_max_rt = int(q_nyq_rt / q_L_rt)
    psd_rt = idealized_PSD(p_rt)

    P0_list = [0.2e6, 0.587e6]
    labels  = ["$P_0 = 0.2$ MPa", "$P_0 = 0.587$ MPa (paper MD)"]
    colors  = ["b", "r"]

    fig, ax = plt.subplots()
    for pi_idx, P0v in enumerate(P0_list):
        p_rt.P0 = P0v
        A_vals = np.zeros(zeta_vis_max_rt)
        for k in range(1, zeta_vis_max_rt + 1):
            zeta_P_rt = k * q_L_rt / p_rt.q0
            if zeta_P_rt < 1.0:
                A_vals[k-1] = 1.0
            else:
                A_vals[k-1], _ = eval_contact_area(psd_rt, zeta_P_rt, p_rt)

        ax.plot(range(1, zeta_vis_max_rt + 1), A_vals,
                colors[pi_idx], linewidth=2, label=labels[pi_idx])

    ax.axhline(0.4, color="k", linestyle="--", linewidth=1.5,
               label="Percolation threshold (0.4)")
    ax.set_xlabel("Magnification ζ (= q/q_L)")
    ax.set_ylabel("$A(\\zeta) / A_0$")
    ax.set_title("$A/A_0$ vs magnification")
    ax.legend(loc="lower left")
    ax.grid(True)
    plt.tight_layout()
    plt.show()

# =============================================================================
#  Reproduce Paper Figures 6-11
# =============================================================================
if do_make_plots:
    make_plots(out_dir=".")

# =============================================================================
#  Applied Example: Rubber Seal on a Rough Substrate
# =============================================================================
if do_seal_study:
    analyze_seal(out_dir=".")
