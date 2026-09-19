"""
grain_zoomout.py — Close-up of the roughness, pull back to reveal a trapped grain.

One continuous take (PyVista, offscreen), true lateral scale throughout:
  1. close-up on the micro-roughness next to the grain (~60 um field of view)
  2. the camera pulls back ~100x; the grain (500 um) rises into frame and ends up
     towering over a nearly flat plain (roughness rolls off above 2*pi/q0)
  3. the rubber comes down, tents over the grain and lifts off the substrate
     around it (non-contact halo); contact patches darken elsewhere
  4. hold with a slow orbit

Two height maps cover the zoom range, both drawn from the Persson & Yang PSD
(flat below q0, power law q^-2(1+H) above):
  * coarse: whole field (default 8 mm, 2048^2)
  * fine:   high-q band only, around the start point, added to the interpolated
            coarse map (the zoom-in counterpart of zoom.py's low-pass filter)

Visual choices, not physics results:
  * surface heights are exaggerated; the factor eases from --exag-start to
    --exag-end during the pull-back and is shown on screen. The grain is never
    exaggerated.
  * the rubber is a geometric drape (low-pass of the substrate + offset, linear
    skirt around the grain), not an elastic solution

Outputs (animations/out/, gitignored): grain_zoomout.mp4, grain_zoomout.gif

Run (from Python/):
  .venv/Scripts/python animations/grain_zoomout.py            # full render
  .venv/Scripts/python animations/grain_zoomout.py --preview  # quick low-res check

Real inputs belong in a local JSON of Params overrides passed with --config
(files matching animations/local_* are gitignored), e.g.
  {"sigma": 2e-6, "H": 0.8, "q0": 1e4}
"""

import argparse
import json
import os
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from persson_yang import load_parameters  # noqa: E402

HERE    = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "out")


# ---------------------------------------------------------------------------
#  Surfaces (all lengths in um from here on)
# ---------------------------------------------------------------------------

def band_rms(p, q_lo, q_hi):
    """RMS height carried by wavevectors q_lo..q_hi [1/m] for the P&Y PSD at p.sigma."""
    H, q0 = p.H, p.q0
    qL = min(q_lo, q0)

    def cum(q):   # 2*pi * integral_0^q  q' C(q') dq'  with C0 = 1
        q = np.asarray(q, float)
        flat = np.minimum(q, q0) ** 2 / 2
        tail = np.where(q > q0, q0 ** 2 / (2 * H) * (1 - (q / q0) ** (-2 * H)), 0.0)
        return 2 * np.pi * (flat + tail)

    C0 = p.sigma ** 2 / (cum(np.inf) - cum(qL))
    return float(np.sqrt(C0 * (cum(q_hi) - cum(q_lo))))


def synth(n, dx_um, p, q_lo, q_hi, rng):
    """Random surface [um] on an n x n grid with the P&Y PSD restricted to q_lo..q_hi."""
    q = 2 * np.pi * np.sqrt(np.fft.fftfreq(n)[:, None] ** 2
                            + np.fft.rfftfreq(n)[None, :] ** 2) / (dx_um * 1e-6)
    C = np.where(q < p.q0, 1.0, (np.maximum(q, 1e-30) / p.q0) ** (-2 * (1 + p.H)))
    C[(q < q_lo) | (q > q_hi)] = 0.0
    noise = rng.standard_normal((n, n))
    z = np.fft.irfft2(np.fft.rfft2(noise) * np.sqrt(C), s=(n, n))
    return z * (band_rms(p, q_lo, q_hi) * 1e6 / z.std())


def ease(t):
    """Smoothstep 0..1."""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


class Terrain:
    """Coarse field plus a fine patch; sample() returns a block-averaged window."""

    def __init__(self, p, args, rng):
        self.L  = args.field_mm * 1e3                   # field width [um]
        self.n  = args.n
        self.dx = self.L / self.n
        q_nyq_c = np.pi / (self.dx * 1e-6)
        self.z  = synth(self.n, self.dx, p, 2 * np.pi / (self.L * 1e-6), q_nyq_c, rng)

        # Grain at field centre; fine patch centred on the start point
        self.r  = args.grain_um / 2
        self.gx = self.gy = self.L / 2
        az0 = np.deg2rad(args.azimuth)
        self.sx = self.gx + 1.3 * self.r * np.cos(az0)
        self.sy = self.gy + 1.3 * self.r * np.sin(az0)

        nf, k = args.fine_n, args.fine_factor
        self.fdx = self.dx / k
        self.fL  = nf * self.fdx
        self.fx0 = self.sx - self.fL / 2
        self.fy0 = self.sy - self.fL / 2
        coords = np.mgrid[0:nf, 0:nf] * self.fdx
        ci = (self.fy0 + coords[0]) / self.dx
        cj = (self.fx0 + coords[1]) / self.dx
        base = map_coordinates(self.z, [ci, cj], order=3, mode="wrap")
        self.zf = base + synth(nf, self.fdx, p, q_nyq_c, np.pi / (self.fdx * 1e-6), rng)

        # Rubber: follows long wavelengths, touches the top `contact` fraction
        zl = gaussian_filter(self.z, args.rubber_px, mode="wrap")
        self.base = zl + np.quantile(self.z - zl, 1.0 - args.contact)

        print(f"  coarse {self.n}^2 @ {self.dx:.2f} um, fine {nf}^2 @ {self.fdx:.2f} um "
              f"({self.fL:.0f} um patch), rms {self.z.std():.2f} um, "
              f"peak-valley {np.ptp(self.z):.1f} um")

    def sample(self, cx, cy, W, cells):
        """Heights [um] on a <= cells x cells grid covering W um centred at (cx, cy)."""
        in_fine = (cx - W / 2 >= self.fx0 and cx + W / 2 <= self.fx0 + self.fL and
                   cy - W / 2 >= self.fy0 and cy + W / 2 <= self.fy0 + self.fL)
        if in_fine:
            arr, dx, x0, y0 = self.zf, self.fdx, self.fx0, self.fy0
            extra = []
        else:
            arr, dx, x0, y0 = self.z, self.dx, 0.0, 0.0
            extra = [self.base]
        N  = arr.shape[0]
        Wp = int(min(round(W / dx), N))
        s  = max(1, int(np.ceil(Wp / cells)))
        m  = (Wp // s) * s
        i0 = int(np.clip(round((cy - y0) / dx) - Wp // 2, 0, N - m))
        j0 = int(np.clip(round((cx - x0) / dx) - Wp // 2, 0, N - m))

        def blk(a):
            a = a[i0:i0 + m, j0:j0 + m]
            return a.reshape(m // s, s, m // s, s).mean(axis=(1, 3))

        xs = x0 + (j0 + np.arange(m // s) * s + (s - 1) / 2) * dx
        ys = y0 + (i0 + np.arange(m // s) * s + (s - 1) / 2) * dx
        X, Y = np.meshgrid(xs, ys)
        base = blk(extra[0]) if extra else None
        return X, Y, blk(arr), base

    def grain_rest(self, exag):
        """Grain centre height with the substrate exaggerated by `exag`."""
        w  = int(np.ceil(self.r / self.dx)) + 1
        gi = gj = self.n // 2
        yy, xx = np.mgrid[-w:w + 1, -w:w + 1] * self.dx
        d2  = xx ** 2 + yy ** 2
        cap = np.where(d2 <= self.r ** 2, np.sqrt(np.maximum(self.r ** 2 - d2, 0)), -np.inf)
        return float(np.max(exag * self.z[gi - w:gi + w + 1, gj - w:gj + w + 1] + cap))


# ---------------------------------------------------------------------------
#  Rendering
# ---------------------------------------------------------------------------

SUBSTRATE = np.array([0.74, 0.71, 0.66])
CONTACT   = np.array([0.20, 0.19, 0.18])


def render(args):
    import pyvista as pv
    import imageio

    p = load_parameters()
    if args.config:
        with open(args.config) as f:
            for k, v in json.load(f).items():
                setattr(p, k, v)
    if args.preview:
        args.n, args.fine_n = 1024, 512

    print("Building surfaces...")
    T_ = Terrain(p, args, np.random.default_rng(args.seed))

    fps = args.fps
    res = (640, 360) if args.preview else (1280, 720)
    cells = 160 if args.preview else 360

    # Timeline [s]
    t_hold0, t_zoom, t_drape, t_end = 1.0, 7.0, 2.5, 2.0
    T = t_hold0 + t_zoom + t_drape + t_end
    n_frames = int(round(T * fps))
    W0, W1 = args.start_um, args.end_mm * 1e3

    pl = pv.Plotter(off_screen=True, window_size=res)
    pl.set_background("#0d1117", top="#2b3645")
    if not args.preview:
        pl.enable_anti_aliasing("ssaa")
    sphere = pv.Sphere(radius=T_.r, theta_resolution=96, phi_resolution=96)

    os.makedirs(OUT_DIR, exist_ok=True)
    mp4 = os.path.join(OUT_DIR, "grain_zoomout.mp4")
    gif = os.path.join(OUT_DIR, "grain_zoomout.gif")
    writer = imageio.get_writer(mp4, fps=fps, codec="libx264", quality=8,
                                macro_block_size=8)
    gif_frames, gif_every = [], max(1, fps // 15)

    print(f"Rendering {n_frames} frames at {res[0]}x{res[1]}, {fps} fps...")
    for f in range(n_frames):
        t = f / fps
        u_zoom  = ease((t - t_hold0) / t_zoom)
        u_drape = ease((t - t_hold0 - t_zoom) / t_drape)

        W    = W0 * (W1 / W0) ** u_zoom
        exag = args.exag_start * (args.exag_end / args.exag_start) ** u_zoom
        # Focal point drifts from the start spot to the grain; stays low at first
        # so the grain rises into frame from behind
        cx = T_.sx + (T_.gx - T_.sx) * u_zoom
        cy = T_.sy + (T_.gy - T_.sy) * u_zoom

        X, Y, Z, base = T_.sample(cx, cy, W, cells)
        Zs = Z * exag
        gz = T_.grain_rest(exag)

        # ---- contact (only once the rubber is down, coarse field only) ----
        rgb = np.broadcast_to(SUBSTRATE, Zs.shape + (3,)).copy()
        top, skirt = gz + T_.r, 3.0 * T_.r
        dist = np.hypot(X - T_.gx, Y - T_.gy)
        if base is not None:
            B = base * exag
            tent = top - np.clip(dist - 0.6 * T_.r, 0, None) * (top - B) / skirt
            con = (Zs >= np.maximum(B, tent)).astype(float)
            a = ease((u_drape - 0.6) / 0.4)
            rgb += a * con[..., None] * (CONTACT - SUBSTRATE)
            rubber = np.maximum.reduce([Zs, B, tent])

        sub = pv.StructuredGrid(X, Y, Zs)
        rgb8 = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        sub.point_data["rgb"] = np.stack([rgb8[..., c].ravel(order="F") for c in range(3)], axis=1)
        pl.add_mesh(sub, scalars="rgb", rgb=True, smooth_shading=True,
                    specular=0.25, name="substrate")

        pl.add_mesh(sphere.translate((T_.gx, T_.gy, gz), inplace=False),
                    color="#d99a3e", smooth_shading=True, specular=0.5,
                    specular_power=20, name="grain")

        # ---- rubber: descends from above after the pull-back ----
        if base is not None and u_drape > 0:
            lift = (1.0 - u_drape) * 3.0 * T_.r
            Zr = rubber + lift + 0.003 * W
            pl.add_mesh(pv.StructuredGrid(X, Y, Zr), color="#7fb8ff",
                        opacity=0.45 * ease(u_drape / 0.2), smooth_shading=True,
                        specular=0.5, name="rubber")

        # ---- camera: low and close at first, rising as it pulls back ----
        az = np.deg2rad(args.azimuth + 30 * t / T)
        el = np.deg2rad(22 + 10 * u_zoom)
        d  = 1.1 * W
        fz = float(np.median(Zs)) + (gz * 0.5) * u_zoom
        pl.camera.focal_point = (cx, cy, fz)
        pl.camera.position = (cx + d * np.cos(el) * np.cos(az),
                              cy + d * np.cos(el) * np.sin(az),
                              fz + d * np.sin(el))
        pl.camera.up = (0, 0, 1)
        pl.camera.view_angle = 40
        pl.reset_camera_clipping_range()

        if args.labels:
            pl.add_text(f"field of view  {_fmt_len(W)}", position="lower_left",
                        font_size=10, color="#c9d1d9", name="fov")
            pl.add_text(f"surface heights x{exag:.0f}  (grain to scale)",
                        position="lower_right", font_size=8, color="#8b949e", name="exag")

        img = pl.screenshot(return_img=True)
        writer.append_data(img)
        if f % gif_every == 0:
            gif_frames.append(img[::2, ::2])
        if f % 60 == 0:
            print(f"  frame {f}/{n_frames}  FOV {_fmt_len(W)}  x{exag:.0f}")

    writer.close()
    pl.close()
    imageio.mimsave(gif, gif_frames, duration=1000 * gif_every / fps, loop=0)
    print(f"Saved:\n  {mp4}\n  {gif}")


def _fmt_len(um):
    return f"{um:.0f} um" if um < 1000 else f"{um/1e3:.1f} mm"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", action="store_true", help="smaller grids, 640x360")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--field-mm", type=float, default=8.0, help="coarse field width")
    ap.add_argument("--n", type=int, default=2048, help="coarse grid size")
    ap.add_argument("--fine-n", type=int, default=1024, help="fine patch grid size")
    ap.add_argument("--fine-factor", type=int, default=8, help="fine/coarse resolution ratio")
    ap.add_argument("--grain-um", type=float, default=500, help="grain diameter")
    ap.add_argument("--start-um", type=float, default=60, help="opening field of view")
    ap.add_argument("--end-mm", type=float, default=5.0, help="final field of view")
    ap.add_argument("--exag-start", type=float, default=12, help="height exaggeration, close-up")
    ap.add_argument("--exag-end", type=float, default=5, help="height exaggeration, wide")
    ap.add_argument("--azimuth", type=float, default=-60, help="camera azimuth [deg]")
    ap.add_argument("--rubber-px", type=float, default=10,
                    help="rubber drape low-pass length (coarse px)")
    ap.add_argument("--contact", type=float, default=0.3, help="contact area fraction A/A0")
    ap.add_argument("--no-labels", dest="labels", action="store_false")
    ap.add_argument("--config", help="local JSON of Params overrides (not committed)")
    render(ap.parse_args())
