# Persson & Yang seal model + animations

Python implementation of Persson & Yang (2008), *Theory of the leak-rate of seals*
(J. Phys.: Condens. Matter 20, 315011), plus a 3D animation of a 500 µm grain trapped under a rubber
seal on a rough substrate.

Reference render: [`media/grain_zoomout.mp4`](media/grain_zoomout.mp4)

## Quick start
```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # macOS/Linux: .venv/bin/python
.venv/Scripts/python animations/grain_zoomout.py --preview
```
Full-quality render: drop `--preview`. Options: `--help`.

## Using Claude Code
`CLAUDE.md` explains the model, the scale choices behind the animation, its approximations, and
ideas for next steps. Open this folder in Claude Code and it picks all of that up.
