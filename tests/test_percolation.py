"""Percolation = a single cluster spans the box (periodic: wraps around it; walls: touches both walls)."""

import numpy as np
import pandas as pd

from qtft import Params
from qtft.structure import spanning_axes


def _atoms(xyz, types=None):
    xyz = np.asarray(xyz, dtype=float)
    return pd.DataFrame({"id": np.arange(1, len(xyz) + 1), "type": types or [2] * len(xyz),
                         "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})


def _chain(n):
    return np.array([[k, k + 1] for k in range(1, n)])


def test_periodic_wrapping_chain_spans():
    p = Params(box_length=1000.0)
    ring = _atoms([[100 * k + 50, 500, 500] for k in range(10)])             # 10 Ft along x, 100 nm apart
    closed = np.vstack([_chain(10), [[10, 1]]])                              # last bond crosses the boundary
    assert spanning_axes(ring, closed, p) == "x"
    assert spanning_axes(ring, _chain(10), p) == ""                          # open chain: no closed path
    assert spanning_axes(ring, np.empty((0, 2), int), p) == ""


def test_periodic_local_loop_does_not_span():
    p = Params(box_length=1000.0)
    # a small triangle sitting across the boundary closes a loop, but not around the box
    tri = _atoms([[990, 500, 500], [30, 500, 500], [10, 540, 500]])
    assert spanning_axes(tri, np.array([[1, 2], [2, 3], [3, 1]]), p) == ""
    # a second, separate cluster that wraps along z; the triangle is unaffected
    col = [[500, 200, 100 * k + 50] for k in range(10)]
    both = _atoms([[990, 500, 500], [30, 500, 500], [10, 540, 500]] + col)
    bonds = np.vstack([[[1, 2], [2, 3], [3, 1]], _chain(10) + 3, [[13, 4]]])
    assert spanning_axes(both, bonds, p) == "z"


def test_walls_touching_both_sides_spans():
    p = Params(box_length=[1000.0, 800.0, 600.0], boundary="harmonic")
    r = p.radius_ft
    xs = np.linspace(r + 2, 1000 - r - 2, 12)                               # surfaces 2 nm from the walls
    line = _atoms([[x, 400, 300] for x in xs])
    assert spanning_axes(line, _chain(12), p) == "x"
    assert spanning_axes(line, _chain(11), p) == ""                          # last particle not bonded
    short = _atoms([[x, 400, 300] for x in np.linspace(r + 10, 1000 - r - 2, 12)])   # 10 nm gap at x = 0
    assert spanning_axes(short, _chain(12), p) == ""
