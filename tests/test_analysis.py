"""Structure (OVITO), overlap and figure/report tests on tiny LAMMPS runs."""

import shutil
from collections import defaultdict

import numpy as np
import pandas as pd
import pytest

from qtft import Params, build_run, read_bonds, read_dump, start

from conftest import influx

pytestmark = [pytest.mark.slow,
              pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")]
ovito = pytest.importorskip("ovito")

import matplotlib  # noqa: E402

matplotlib.use("Agg")


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    data = tmp_path_factory.mktemp("data")
    common = dict(n_qt=50, n_ft=200, box_length=464.2, equil_steps=200, **influx(3000),
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
    # a single Qt: solid sphere of volume V_Qt / 0.64
    one_qt = table[(table["size"] == 1) & (table["n_qt"] == 1)].iloc[0]
    assert one_qt["rg_ideal"] == pytest.approx(np.sqrt(3 / 5) * p.radius_qt / 0.64 ** (1 / 3))


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
        closest = dist[iu][sel].min()
        if closest <= 2 * max(p.radius_qt, p.radius_ft):          # search range of overlap_frame
            assert result.loc[family, "closest [nm]"] == pytest.approx(closest)
        else:                                                      # no pair within range -> NaN
            assert np.isnan(result.loc[family, "closest [nm]"])


def test_report_and_comparison(runs, tmp_path):
    from qtft.figures import compare_report, make_report

    out = make_report(runs[0], every=2, n_min_qt=[2, 5], n_qt_ft=(3, 4))
    for name in ("summary_table.csv", "summary_table.tex", "bonds_timeseries.csv", "structure_timeseries.csv",
                 "clusters_final.csv", "overlap_final.csv", "binding_rate.csv", "agglomerates_influx_end.csv",
                 "agglomerates_above_size.csv", "agglomerates_above_size_qt.csv", "agglomerates_above_size_ft.csv",
                 "qt_nn_distances.csv", "qt_pair_correlation.csv", "qt_clark_evans.csv",
                 "qt_ripley_l.csv"):
        assert (out / name).exists(), name
    for name in ("calcium", "overview", "agglomerate_distributions", "kinetics", "agglomerates_above_size", "overlap",
                 "binding_rate", "fibsem_comparison"):
        for ext in ("pdf", "svg", "png"):
            assert (out / "figures" / f"{name}.{ext}").exists(), f"{name}.{ext}"
    panels = {"overview": 12, "calcium": 6, "kinetics": 3, "agglomerate_distributions": 3,   # every panel alone
              "agglomerates_above_size": 8, "fibsem_comparison": 4}
    assert (out / "figures" / "overview" / "coordination_at_end_of_influx.pdf").exists()
    assert {d.name: len(list(d.glob("*.pdf"))) for d in (out / "figures").iterdir() if d.is_dir()} == panels
    assert not (out / "figures" / "journal").exists() and not (out / "figures" / "bond_rates.pdf").exists()
    assert not any((out / "figures" / f"{old}.pdf").exists() for old in ("large_agglomerates", "clusters"))
    qt_counts = pd.read_csv(out / "agglomerates_above_size_qt.csv")
    assert sorted(qt_counts["n_min"].unique()) == [2, 3, 5]                    # thresholds passed on
    reps = sorted(d.name for d in runs[0].glob("rep_*"))
    assert all((out / f"percolation_{rep}.csv").exists() for rep in reps)        # one analysis/ folder per run
    assert not any((runs[0] / rep / "analysis").exists() for rep in reps)
    table = pd.read_csv(out / "summary_table.csv", index_col=0)
    assert "mean Rg (nm)" in table.index

    cmp_dir = compare_report(runs, tmp_path / "cmp")
    assert (cmp_dir / "comparison.pdf").exists() and (cmp_dir / "comparison_table.tex").exists()
    assert len(list((cmp_dir / "comparison").glob("*.pdf"))) == 12
    assert len(list((cmp_dir / "kinetics").glob("*.pdf"))) == 3


def test_fibsem_export(runs):
    import json

    from qtft.fibsem import fibsem_export

    rep = runs[0] / "rep_01"
    p = Params.load(runs[0] / "params.json")
    out = fibsem_export(rep, voxel_nm=4.0)
    table = pd.read_csv(out / "sim_encapsulin_centroids_clustered.csv")
    meta = json.loads((out / "sim_metadata.json").read_text())
    assert (out / "check_xy.png").exists()

    assert list(table.columns) == ["label", "z_nm", "y_nm", "x_nm", "z_vox", "y_vox", "x_vox",
                                   "radius_nm", "volume_nm3", "cluster", "is_clustered"]
    assert len(table) == p.n_qt and list(table["label"]) == list(range(1, p.n_qt + 1))
    assert (table[["x_nm", "y_nm", "z_nm"]] >= p.radius_qt - 1e-9).all().all()   # spheres start at 0
    assert np.allclose(table["x_vox"] * 4.0, table["x_nm"])
    assert meta["bond_check"]["failed"] == 0 and meta["n_encapsulins"] == p.n_qt

    # same grouping of encapsulins as the LAMMPS clusters of the final frame; a Qt is clustered
    # exactly when it has at least one bond
    _, atoms = list(read_dump(rep / "traj.dump"))[-1]
    qt = atoms[atoms["type"] == 1].sort_values("id").reset_index(drop=True)
    pairs = pd.DataFrame({"ours": table["cluster"], "lammps": qt["cluster"]})
    assert pairs.groupby("ours")["lammps"].nunique().max() == 1
    assert pairs.groupby("lammps")["ours"].nunique().max() == 1
    assert (table["is_clustered"].to_numpy() == (qt["nbonds"] > 0).to_numpy()).all()


def test_quicklook(runs):
    from qtft import quicklook

    fig = quicklook(runs[0], overlap_every=1)
    axes = fig.axes
    assert len(axes) == 12
    labels = [ax.get_ylabel() + ax.get_xlabel() for ax in axes]
    assert not any("[" in label for label in labels) and "time (ms)" in labels[-2]
    for ax in axes[8:11]:                                          # overlap panels, one line per replica
        assert len(ax.get_lines()) == 2
    assert len(quicklook(runs[0], overlap=False).axes[8].get_lines()) == 0
