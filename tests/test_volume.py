"""Cluster volumes for the confocal comparison: solid volume (union of the spheres) and envelope volume
(closing with an ellipsoidal probe), at the end of the influx."""

import math

import numpy as np
import pandas as pd
import pytest

from qtft import Params
from qtft.structure import _agglomerate_volume, agglomerate_volumes

PROBE = np.array([60.0, 60.0, 150.0])


def _atoms(xyz, types=None):
    xyz = np.asarray(xyz, dtype=float)
    return pd.DataFrame({"id": np.arange(1, len(xyz) + 1), "type": types or [2] * len(xyz),
                         "x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2]})


def _chain(n, first=1):
    return np.array([[k, k + 1] for k in range(first, first + n - 1)])


def test_solid_volume_of_one_sphere():
    rng = np.random.default_rng(0)
    for r in (13.0, 27.5):
        v = [_agglomerate_volume(rng.random((1, 3)) * 1000, np.array([r]), PROBE, 10.0)[0] for _ in range(200)]
        assert np.mean(v) == pytest.approx(4 / 3 * math.pi * r**3, rel=0.03)    # unbiased on average


def _shell(n, radius):
    """n points evenly on a sphere (Fibonacci points)."""
    k = np.arange(n) + 0.5
    z = 1 - 2 * k / n
    phi = math.pi * (1 + 5**0.5) * k
    return radius * np.c_[np.sqrt(1 - z**2) * np.cos(phi), np.sqrt(1 - z**2) * np.sin(phi), z] + 500


def test_envelope_fills_cavities_smaller_than_the_probe():
    # 80 Ft on a sphere of radius 60 nm (touching): the cavity (radius ~47 nm) is smaller than the probe
    solid, envelope = _agglomerate_volume(_shell(80, 60.0), np.full(80, 13.0), PROBE, 5.0)
    assert 4 / 3 * math.pi * 60**3 < envelope < 4 / 3 * math.pi * 73**3    # filled: more than the ball through the centres
    assert envelope > 1.5 * solid
    # the same shell at radius 300 nm (1600 Ft): the cavity is wider than the probe and stays empty,
    # as it would look hollow in the microscope
    solid, envelope = _agglomerate_volume(_shell(1600, 300.0), np.full(1600, 13.0), PROBE, 10.0)
    assert solid == pytest.approx(1600 * 4 / 3 * math.pi * 13**3, rel=0.05)
    assert envelope < 0.3 * 4 / 3 * math.pi * 300**3


def test_envelope_bridges_axial_gaps_only():
    # two 4 x 4 layers of touching Ft, 200 nm apart: bridged along z (probe 150 nm), not along x (60 nm)
    grid = np.array([[a, b] for a in range(4) for b in range(4)]) * 26.0
    z = np.vstack([np.c_[grid, np.zeros(16)], np.c_[grid, np.full(16, 252.0)]])
    x = z[:, [2, 0, 1]]
    radius = np.full(32, 13.0)
    solid_z, env_z = _agglomerate_volume(z, radius, PROBE, 5.0)
    solid_x, env_x = _agglomerate_volume(x, radius, PROBE, 5.0)
    assert solid_z == pytest.approx(solid_x, rel=0.02)
    assert env_z > 3 * solid_z                                       # the gap between the layers is filled
    assert env_x < 1.5 * solid_x


def test_agglomerate_volumes_table_and_periodic_box():
    p = Params(box_length=1000.0)
    # a 5-particle Qt-Ft chain in the middle, the same chain across the x boundary, a pair (too small)
    # and a ring of 10 Ft that wraps around the box along y
    chain = [[400, 500, 500], [440, 500, 500], [480, 500, 500], [520, 500, 500], [560, 500, 500]]
    across = [[(x + 500) % 1000, 200, 200] for x, _, _ in chain]
    pair = [[100, 100, 800], [126, 100, 800]]
    ring = [[800, 100 * k + 50, 300] for k in range(10)]
    types = [1, 2, 2, 2, 2] * 2 + [2, 2] + [2] * 10
    atoms = _atoms(chain + across + pair + ring, types)
    bonds = np.vstack([_chain(5), _chain(5, 6), [[11, 12]], _chain(10, 13), [[22, 13]]])
    table = agglomerate_volumes(atoms, bonds, p)

    assert list(table["size"]) == [10, 5, 5]                         # >= 3 particles, largest first
    ring_row, a, b = table.iloc[0], table.iloc[1], table.iloc[2]
    assert ring_row["spans"] == "y" and np.isnan(ring_row["v_solid_um3"]) and np.isnan(ring_row["v_envelope_um3"])
    assert (a["n_qt"], a["n_ft"], a["n_bonds"], a["spans"]) == (1, 4, 4, "")
    assert a["v_solid_um3"] == pytest.approx(b["v_solid_um3"], rel=0.03)       # unwrapped across the box
    assert a["v_envelope_um3"] == pytest.approx(b["v_envelope_um3"], rel=0.03)
    assert a["v_envelope_um3"] >= a["v_solid_um3"]
