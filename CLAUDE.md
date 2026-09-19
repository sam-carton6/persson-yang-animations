# Persson & Yang seal model + 3D animations

Python implementation of B.N.J. Persson & C. Yang, "Theory of the leak-rate of seals",
J. Phys.: Condens. Matter 20 (2008) 315011, plus PyVista animations built on it.
The paper is not included (copyright); get it from the journal.

## Setup
- Python 3.11+ (developed on 3.13). `python -m venv .venv`, then
  `.venv/Scripts/python -m pip install -r requirements.txt` (Windows) or `.venv/bin/python ...` (macOS/Linux).
- PyVista renders offscreen; no display needed.

## Run
- Animation: `python animations/grain_zoomout.py` (about 40 s, writes `animations/out/`).
  `--preview` gives a 640x360 check in about 15 s. `--help` lists every knob.
- Paper figures / model: `python main.py` (toggles at the top; Figs 6–11 on by default).
  `main.py` uses the TkAgg backend for interactive plots; switch to "Agg" if Tk is missing.
- Reference render: `media/grain_zoomout.mp4`.

## Layout
- `persson_yang/`: the model.
  `parameters` (`Params` dataclass, paper Section 3 values), `psd` (idealized PSD C(q)),
  `surface` (`artificial_surf`: random self-affine surface with a prescribed PSD), `contact`
  (A(ζ)/A0 from Persson contact mechanics), `zoom` (low-pass a surface to magnification ζ),
  `percolation` (BFS leak-path search), `core` (critical magnification, u_c, leak rate Q),
  `plots` (paper Figs 6–11), `fig14` (Fig 14 contact-map GIF/MP4), `seal_study` (worked example:
  rubber seal on a rough substrate).
- `animations/grain_zoomout.py`: the 3D shot (see below).
- `main.py`: driver script.

## The grain animation: how it works
One continuous take: close-up on the micro-roughness (60 µm field of view), pull back ~80x to 5 mm,
and a 500 µm grain looms into frame and ends up as a boulder on a nearly flat plain. Then the rubber
comes down, tents over the grain and lifts off the substrate around it (non-contact halo). Contact
patches darken elsewhere.

Scale facts that drive the design (default Params):
- RMS roughness σ = 2 µm, Hurst exponent H = 0.8, roll-off q0 = 1e4 1/m, so no roughness grows
  beyond a wavelength of 2π/q0 ≈ 0.63 mm. Peak-to-valley over an 8 mm field is about 10–20 µm.
- A 500 µm grain is ~250x σ, so at true scale it's a boulder on a plain. Don't draw it as comparable
  to the bumps. (An earlier version turned off the roll-off and made fake mountains. That was wrong.)
- True-scale bumps are invisible, so **surface heights are exaggerated** (x12 in the close-up easing to
  x5 wide, shown on screen). **The grain is never exaggerated.** Keep it that way and keep the label.

Two height maps cover the zoom range. One height map can't span 60 µm to 8 mm with enough detail:
- coarse: 8 mm field, 2048², the whole P&Y PSD band down to its Nyquist wavelength
- fine: 1024² at 1/8 of the coarse pixel, only the higher-q band, added to the cubic-interpolated
  coarse map around the start point (the zoom-in counterpart of `zoom.py`'s low-pass filter)
`Terrain.sample()` picks whichever map covers the current window and block-averages it to the render
grid. Both come from the same PSD normalisation (`band_rms`), so the switch doesn't pop.

Honest approximations (say so if the output is used for anything quantitative):
- the rubber is a geometric drape (Gaussian low-pass of the substrate + offset, linear skirt of 3r
  around the grain), not an elastic solution. Contact fraction is a flag (`--contact`, default 0.3)
  rather than coming from `eval_contact_area`
- the grain sits at its rest height from the start. It doesn't fall.

Ideas not done yet: grain falls and lands mid-zoom, a counting scale bar, contact fraction from
`eval_contact_area` at the current magnification, leak channel via `percolation.check_percolation_path`
on the drape's gap map, and a real elastic tent (the lift-off radius around a particle).

## Conventions
- SI units in the model (m, Pa); the animation works in µm internally.
- Renders (`*.mp4`, `*.gif`, `animations/out/`) are gitignored. Only `media/` holds a reference copy.
- Real/proprietary inputs go in a local JSON passed with `--config` (e.g. `{"sigma": 2e-6, "H": 0.8}`).
  Files matching `animations/local_*` are gitignored so they never get committed.
