"""Structure (OVITO), overlap and figure/report tests on tiny LAMMPS runs."""

import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from qtft import Params, build_run, read_bonds, read_dump, start

pytestmark = pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
ovito = pytest.importorskip("ovito")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    common = dict(n_qt=50, n_ft=200, box_length=464.2, equil_steps=200, prod_steps=3000,
                  output_every=500, traj_every=500, n_replicas=2, data_dir=str(data))
    dirs = [build_run(Params(**common, k_on=k, run_name=f"kon{k:g}")) for k in (2e-4, 4e-4)]
    for d in dirs:
        start(d, wait=True)
    return dirs


def _reference_clusters(atoms, bonds, box):
    """Clusters by breadth-first search over bonds, unwrapped with the minimum image convention."""
    ids = atoms["id"].to_numpy()
    pos = dict(zip(ids, atoms[["x", "y", "z"]].to_numpy()))
    nbrs = defaultdict(list)
    for a, b in bonds:
        nbrs[a].append(b)
        nbrs[b].append(a)
    seen, clusters = set(), []
    for i in ids:
        if i in seen:
            continue
        unwrapped, queue = {i: pos[i]}, [i]
        seen.add(i)
        while queue:
            a = queue.pop()
            for b in nbrs[a]:
                if b not in seen:
                    d = pos[b] - pos[a]
                    unwrapped[b] = unwrapped[a] + d - box * np.round(d / box)
                    seen.add(b)
                    queue.append(b)
        clusters.append(np.array(list(unwrapped.values())))
    return clusters


def test_ovito_clusters_match_reference(runs):
    from qtft.structure import cluster_table, load_frames

    rep = runs[0] / "rep_01"
    p = Params.load(runs[0] / "params.json")
    step, data = list(load_frames(rep, every=1))[-1]
    table = cluster_table(data, p)
    atoms = dict(read_dump(rep / "traj.dump"))[step]
    bonds = dict(read_bonds(rep / "bonds.dump"))[step]
    ref = _reference_clusters(atoms, bonds, p.box_length)

    assert sorted(table["size"]) == sorted(len(c) for c in ref)
    assert sorted(table["size"]) == sorted(atoms["cluster"].value_counts())       # same as LAMMPS
    rg_ref = sorted(np.sqrt(((c - c.mean(axis=0)) ** 2).sum(axis=1).mean()) for c in ref if len(c) >= 2)
    rg = sorted(table.loc[table["size"] >= 2, "rg"])
    assert rg == pytest.approx(rg_ref, abs=1e-3)
    assert table["n_bonds"].sum() == len(bonds)
    assert (table["n_qt"] + table["n_ft"] == table["size"]).all()


def test_overlap_frame_matches_brute_force(runs):
    from qtft.overlap import overlap_frame

    p = Params.load(runs[0] / "params.json")
    _, atoms = list(read_dump(runs[0] / "rep_01" / "traj.dump"))[-1]
    result = overlap_frame(atoms, p).set_index("family")
    pos, types = atoms[["x", "y", "z"]].to_numpy(), atoms["type"].to_numpy()
    r = np.where(types == 1, p.radius_qt, p.radius_ft)
    d = pos[:, None, :] - pos[None, :, :]
    d -= p.box_length * np.round(d / p.box_length)
    dist = np.linalg.norm(d, axis=2)
    depth = np.maximum(r[:, None] + r[None, :] - dist, 0)
    iu = np.triu_indices(len(pos), 1)
    for family, (a, b) in {"Qt-Qt": (1, 1), "Ft-Ft": (2, 2), "Qt-Ft": (1, 2)}.items():
        ti, tj = types[iu[0]], types[iu[1]]
        sel = ((ti == a) & (tj == b)) | ((ti == b) & (tj == a))
        assert result.loc[family, "n overlapping"] == (depth[iu][sel] > 0).sum()
        assert result.loc[family, "closest [nm]"] == pytest.approx(dist[iu][sel].min())


def test_report_and_comparison(runs, tmp_path):
    from qtft.figures import compare_report, make_report

    out = make_report(runs[0], every=2)
    for name in ("final_table.csv", "final_table.tex", "bonds_timeseries.csv", "structure_timeseries.csv",
                 "clusters_final.csv", "overlap_final.csv"):
        assert (out / name).exists(), name
    for name in ("overview", "kinetics", "large_agglomerates", "overlap", "journal/panel_g"):
        for ext in ("pdf", "svg", "png"):
            assert (out / "figures" / f"{name}.{ext}").exists(), f"{name}.{ext}"
    assert len(list((out / "figures" / "panels").glob("*.pdf"))) == 12
    table = pd.read_csv(out / "final_table.csv", index_col=0)
    assert "mean Rg (nm)" in table.index

    cmp_dir = compare_report(runs, tmp_path / "cmp")
    assert (cmp_dir / "comparison.pdf").exists() and (cmp_dir / "comparison_table.tex").exists()
