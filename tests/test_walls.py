"""Walls instead of the periodic box: boundary = "reflect" (fix wall/reflect) or "harmonic" (fix wall/harmonic)."""

import json
import shutil

import numpy as np
import pandas as pd
import pytest

from qtft import Params, build_run, read_dump, start
from qtft.tuning import constants_for_alpha
from conftest import influx


def test_wall_params():
    periodic = Params()
    assert periodic.periodic and "reachable volume Qt" not in periodic.derived()
    assert periodic.auto_name().endswith("_ca10-30-90s_periodic")
    h = Params(box_length=[400.0, 500.0, 600.0], boundary="harmonic")
    d = h.derived()
    assert d["alpha wall Qt"] == pytest.approx(0.70, abs=0.01) and d["alpha wall Ft"] == pytest.approx(0.70, abs=0.01)
    assert d["reachable volume Qt"] == pytest.approx(345 * 445 * 545 / (400 * 500 * 600))
    assert h.auto_name().endswith("_harmonic")
    assert "alpha wall Qt" not in Params(boundary="reflect").derived()
    assert Params(boundary="walls").check()[0][0] == "ERROR"
    assert Params(boundary="reflect", box_length=50.0).check()[0][0] == "ERROR"
    assert any(lvl == "ERROR" and "alpha wall Qt" in m for lvl, m in Params(boundary="harmonic", wall_k_qt=50.0).check())
    k = constants_for_alpha(Params(), 0.5)
    assert Params(boundary="harmonic", **k).derived()["alpha wall Ft"] == pytest.approx(0.5)


def test_wall_inputs(tmp_path):
    text = {b: (build_run(Params(box_length=[400.0, 500.0, 600.0], boundary=b, data_dir=str(tmp_path), run_name=b))
                / "rep_01" / "in.lammps").read_text() for b in ("periodic", "reflect", "harmonic")}
    assert "boundary        p p p" in text["periodic"] and "random 400 " in text["periodic"]
    assert " NULL overlap" in text["periodic"]
    for b in ("reflect", "harmonic"):
        assert "boundary        f f f" in text[b] and " ins_qt overlap" in text[b] and " ins_ft overlap" in text[b]
        assert f'if "{b} == {b}" then' in text[b]
    assert "wall/reflect xlo 27.5 xhi 372.5 ylo 27.5 yhi 472.5 zlo 27.5 zhi 572.5 units box" in text["reflect"]
    assert "wall/harmonic xlo EDGE 6 0.0 13 xhi EDGE 6 0.0 13" in text["harmonic"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
@pytest.mark.parametrize("boundary", ["reflect", "harmonic"])
def test_wall_run(tmp_path, boundary):
    from qtft.fibsem import fibsem_export
    from qtft.overlap import overlap_frame
    from qtft.structure import structure

    p = Params(n_qt=50, n_ft=200, box_length=[400.0, 500.0, 600.0], boundary=boundary, equil_steps=200,
               **influx(1000), output_every=500, traj_every=500, k_on=1e-4, data_dir=str(tmp_path), run_name="w")
    run = build_run(p)
    start(run, wait=True)
    rep = run / "rep_01"
    assert (rep / "status").read_text().strip() == "done"
    assert "ITEM: BOX BOUNDS ff ff ff" in (rep / "traj.dump").read_text()

    box = np.array(p.box)
    for _, atoms in read_dump(rep / "traj.dump"):
        pos, types = atoms[["x", "y", "z"]].to_numpy(), atoms["type"].to_numpy()
        assert (atoms[["ix", "iy", "iz"]].to_numpy() == 0).all()
        r = np.where(types == 1, p.radius_qt, p.radius_ft)
        depth = r - np.minimum(pos, box - pos).min(axis=1)                # sphere beyond a face
        assert depth.max() <= (1e-6 if boundary == "reflect" else 3.0)

    # pair overlaps without minimum image, plus the wall rows
    d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=2)
    pair_depth = np.maximum(r[:, None] + r[None, :] - d, 0)
    iu = np.triu_indices(len(pos), 1)
    result = overlap_frame(atoms, p).set_index("family")
    assert result.loc[["Qt-Qt", "Ft-Ft", "Qt-Ft"], "n overlapping"].sum() == (pair_depth[iu] > 0).sum()
    assert result.loc[["Qt-wall", "Ft-wall"], "n overlapping"].sum() == (depth > 0).sum()

    metrics, _ = structure(rep, every=1)
    assert metrics["n_clusters_ge2"].iloc[-1] > 0

    import matplotlib
    matplotlib.use("Agg")
    from qtft import make_report
    assert (make_report(run, every=1) / "figures" / "overlap.pdf").exists()

    out = fibsem_export(rep, plot=False)
    meta = json.loads((out / "sim_metadata.json").read_text())
    assert meta["boundary"] == boundary and meta["offset_applied_xyz_nm"] == [0.0, 0.0, 0.0]
    assert (meta["wall_stiffness"] is None) == (boundary == "reflect")
    assert meta["wall_leaks"]["n_encapsulins_beyond_wall"] == int((depth[types == 1] > 0).sum())
    table = pd.read_csv(out / "sim_encapsulin_centroids_clustered.csv")
    qt = atoms[atoms["type"] == 1].sort_values("id")
    assert np.allclose(table[["x_nm", "y_nm", "z_nm"]].to_numpy(), qt[["x", "y", "z"]].to_numpy())
