"""
percolation.py — BFS percolation path search.

check_percolation_path : find a top-to-bottom connected path through
                         passable (True) pixels using breadth-first search.
"""

from collections import deque
from typing import List, Optional, Tuple

import numpy as np


def check_percolation_path(
    bw_image: np.ndarray,
    do_frames: bool = False,
    max_bfs_steps: int = 1_000_000,
    max_frames: int = 500,
    frame_every: int = 1000,
) -> Tuple[bool, np.ndarray, List[np.ndarray]]:
    """
    BFS search for a top-to-bottom connected path through True pixels.

    Used to detect when the gap channel (non-contact region) first
    percolates across the seal, allowing fluid to leak through.

    Parameters
    ----------
    bw_image       : (n, m) bool array   True = passable (gap pixels)
    do_frames      : bool  if True, collect intermediate uint8 images for GIF
    max_bfs_steps  : int   safety limit on BFS iterations
    max_frames     : int   max number of animation frames to collect
    frame_every    : int   collect one frame every N BFS steps

    Returns
    -------
    path_exists : bool           True if a top-to-bottom path was found
    path_mask   : (n, m) bool    pixels on the found path
    frames      : list of uint8  animation frames (empty if do_frames=False)
    """
    rows, cols = bw_image.shape
    visited   = np.zeros((rows, cols), dtype=bool)
    path_mask = np.zeros((rows, cols), dtype=bool)

    # Seed from all passable top-row pixels
    queue: deque = deque()
    for c in range(cols):
        if bw_image[0, c]:
            queue.append((0, c))
            visited[0, c]   = True
            path_mask[0, c] = True

    path_exists = False
    bfs_steps   = 0
    frames: List[np.ndarray] = []

    # 4-connected neighbours: (Δrow, Δcol)
    dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()
        bfs_steps += 1

        if bfs_steps > max_bfs_steps:
            import warnings
            warnings.warn(f"BFS exceeded {max_bfs_steps} steps; aborting.")
            break

        if r == rows - 1:
            path_exists = True
            break

        for dr, dc in dirs:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                if bw_image[nr, nc] and not visited[nr, nc]:
                    visited[nr, nc]   = True
                    path_mask[nr, nc] = True
                    queue.append((nr, nc))

        if do_frames and (bfs_steps % frame_every == 0) and len(frames) < max_frames:
            img = (bw_image.astype(np.uint8)) * 255
            img[path_mask] = 128
            frames.append(img.copy())

    return path_exists, path_mask, frames
