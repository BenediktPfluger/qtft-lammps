"""Short end-to-end run of a tiny system. Skipped if `lmp` is not on the PATH."""

import shutil

import numpy as np
import pytest

from qtft import Params, build_run, load_thermo, read_bonds, read_dump, start

pytestmark = pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")


def _clusters_from_bonds(ids, bonds):
    """Union-find over the bond list; returns {atom id: smallest atom id of its cluster}."""
    parent = {i: i for i in ids}

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in bonds:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    return {i: find(i) for i in ids}


def test_tiny_system(tmp_path):
    # same concentration as the reference, faster binding so that bonds form within the test
    p = Params(n_qt=50, n_ft=200, box_length=464.2, equil_steps=200, prod_steps=3000,
               output_every=500, traj_every=500, k_on=1e-4, data_dir=str(tmp_path), run_name="smoke")
    run_dir = build_run(p)
    start(run_dir, wait=True)
    rep = run_dir / "rep_01"
    assert (rep / "status").read_text().strip() == "done"
    for name in ("traj.dump", "bonds.dump", "final.data", "rdf.dat", "coord_qt.hist", "coord_ft.hist"):
        assert (rep / name).exists(), name

    df = load_thermo(rep)
    assert df["Bonds"].iloc[-1] > 0
    assert df["bond_max"].max() < 1.5 * p.r0                 # no runaway bonds

    step, atoms = list(read_dump(rep / "traj.dump"))[-1]
    bstep, bonds = list(read_bonds(rep / "bonds.dump"))[-1]
    assert step == bstep == p.prod_steps
    assert len(bonds) == df["Bonds"].iloc[-1]
    types = dict(zip(atoms["id"], atoms["type"]))
    assert all(types[a] != types[b] for a, b in bonds)       # only Qt-Ft bonds

    # cluster IDs from LAMMPS match the connected components of the bond list
    ref = _clusters_from_bonds(atoms["id"].tolist(), bonds.tolist())
    assert all(ref[i] == c for i, c in zip(atoms["id"], atoms["cluster"]))
    # bond counts per atom match
    counts = np.bincount(bonds.ravel(), minlength=atoms["id"].max() + 1)
    assert np.array_equal(counts[atoms["id"]], atoms["nbonds"].to_numpy())
