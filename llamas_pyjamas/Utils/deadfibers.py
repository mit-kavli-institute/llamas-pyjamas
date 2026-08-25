"""Dead-fibre bookkeeping helpers.

Convention (post-2026-07): each camera's spectra are stored as LIVE-indexed
per-fibre arrays — one row per traced/live fibre, dead fibres absent — for
EVERY per-fibre array (counts, wave, xshift, sky, throughput, errors, ...).
This keeps the arrays mutually aligned: row ``k`` is the same physical fibre in
all of them, so per-fibre operations (arcTransfer, skyModel) pair the right
rows. The camera's ``dead_fibers`` list holds the FIBERMAP positions of the
dead fibres; from it we can recover each live row's physical fibre id, or
expand a live array to fibermap indexing (dead rows re-inserted) when a
fibermap-aligned product is wanted.

Before this convention, only ``counts`` (and a few arrays derived from it) were
padded to fibermap indexing while ``wave``/``xshift``/``sky`` stayed live, so
``counts[i]`` and ``wave[i]`` described different fibres after the first dead
fibre — a silent misalignment. These helpers make the two representations
explicit and round-trippable.
"""

import numpy as np


def live_fibre_ids(n_live, dead_fibers):
    """Fibermap positions occupied by the ``n_live`` live fibres.

    ``dead_fibers`` are the fibermap positions (0-based) with no light; the live
    fibres fill the remaining positions in order. Returns a length-``n_live``
    list of ints (the physical fibre id of each live row).
    """
    dead = sorted(set(int(d) for d in (dead_fibers or [])))
    total = int(n_live) + len(dead)
    deadset = set(dead)
    return [i for i in range(total) if i not in deadset]


def insert_dead_fibre_rows(arr, dead_fibers, fill=0.0):
    """Expand a LIVE-indexed array to FIBERMAP indexing.

    Inserts a ``fill`` row at each dead-fibre position, preserving the live rows
    in order at their fibermap positions. ``arr`` may be 1-D ``(n_live,)`` or
    N-D ``(n_live, ...)``. Exact inverse of removing the dead rows.
    """
    arr = np.asarray(arr)
    dead = sorted(set(int(d) for d in (dead_fibers or [])))
    if not dead:
        return arr
    n_live = arr.shape[0]
    total = n_live + len(dead)
    if any(d < 0 or d >= total for d in dead):
        raise ValueError(
            f"dead-fibre position out of range for {n_live} live + {len(dead)} dead")
    out = np.full((total,) + arr.shape[1:], fill, dtype=arr.dtype)
    out[live_fibre_ids(n_live, dead)] = arr
    return out
