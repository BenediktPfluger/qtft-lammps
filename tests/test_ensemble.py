import shutil

import numpy as np
import pandas as pd
import pytest

from qtft import Params, build_run, read_dump, start
from qtft.ensemble import (binding_rate, common_grid, compare, contact_pairs, ensemble_stats, ensemble_stats_cached,
                           half_time, load_stats, update_index)
from conftest import influx


def test_common_grid_identical_and_overlap():
    a = pd.DataFrame({"time_ms": [0.0, 1.0, 2.0, 3.0]})
    assert np.array_equal(common_grid({"a": a, "b": a.copy()}), [0, 1, 2, 3])
    b = pd.DataFrame({"time_ms": [0.5, 1.5, 2.5]})
    assert np.array_equal(common_grid({"a": a, "b": b}), [1.0, 2.0])     # inside the overlap only


def test_half_time():
    t = np.array([0.0, 1.0, 2.0, 3.0])
    assert half_time(t, np.array([0.0, 0.2, 0.6, 1.0]), 0.5) == pytest.approx(1.75)
    assert half_time(t, np.array([0.6, 0.7, 0.8, 0.9]), 0.5) == 0.0
    assert np.isnan(half_time(t, np.array([0.0, 0.1, 0.2, 0.3]), 0.5))


def test_update_index_lists_built_runs(tmp_path):
    build_run(Params(data_dir=str(tmp_path), run_name="a"))
    build_run(Params(data_dir=str(tmp_path), run_name="b", k_qf=2.0))
    df = pd.read_csv(tmp_path / "index.csv")
    assert list(df["run"]) == ["a", "b"]
    assert df.loc[df["run"] == "b", "changed vs defaults"].item() == "k_qf=2.0"
    assert update_index(tmp_path)["status"].tolist() == ["built:1", "built:1"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_ensemble_of_tiny_runs(tmp_path):
    common = dict(qt_per_um3=500.0, ft_per_um3=2000.0, box_length=464.2, equil_steps=200, **influx(2000),
                  output_every=500, traj_every=500, n_replicas=3, data_dir=str(tmp_path))
    runs = [build_run(Params(**common, k_on=k, run_name=f"kon{k:g}")) for k in (1e-4, 2e-4)]
    for r in runs:
        start(r, wait=True)

    stats = ensemble_stats(runs[0])
    assert stats["traces"]["Bonds"].shape == (3, 5)                        # 3 replicas, 5 frames
    finals = stats["per_replica"]["final_Bonds"]
    assert stats["final"].loc["final_Bonds", "mean"] == pytest.approx(finals.mean())
    assert stats["final"].loc["final_Bonds", "sd"] == pytest.approx(finals.std(ddof=1))
    assert stats["mean"]["Bonds"][-1] == pytest.approx(finals.mean())
    for name in ("ensemble_stats.json", "ensemble_timeseries.csv", "final_state.csv"):
        assert (runs[0] / "analysis" / name).exists()
    assert np.allclose(load_stats(runs[0])["mean"]["Bonds"], stats["mean"]["Bonds"])

    table = compare(runs)
    assert table.attrs["differing parameters"] == ["k_on"]
    assert list(table.columns) == ["k_on=0.0001", "k_on=0.0002"]

    index = pd.read_csv(tmp_path / "index.csv")
    assert set(index["status"]) == {"done:3"}

    # association rate: measured per replica; prediction from the counted contact pairs at t = 0
    rate = binding_rate(runs[0], save=False)
    per_rep, summary = rate["per_replica"], rate["summary"]
    assert len(per_rep) == 3 and (per_rep["k_a measured"] > 0).all()
    assert per_rep["frames"].dtype.kind == "i" and per_rep["contact pairs"].dtype.kind == "i"
    v = summary["value"]
    assert v["k_a measured"] == pytest.approx(per_rep["k_a measured"].mean())
    assert v["k_a predicted"] < min(v["kappa"], v["k_D"])
    assert summary.loc["k_a measured", "1/(M s)"] == pytest.approx(v["k_a measured"] * 6.02214076e8)
    assert summary.loc["contact volume counted", "unit"] == "nm^3"

    # contact pairs at t = 0 against a brute-force count with the minimum image
    p = Params.load(runs[0] / "params.json")
    step, atoms = next(read_dump(runs[0] / "rep_01" / "traj.dump"))
    pos, types, box = atoms[["x", "y", "z"]].to_numpy(), atoms["type"].to_numpy(), np.array(p.box)
    d = pos[types == 1][:, None, :] - pos[types == 2][None, :, :]
    d -= box * np.round(d / box)
    assert contact_pairs(atoms, np.empty((0, 2), int), p) == (np.linalg.norm(d, axis=2) < p.r_bind).sum()
    assert per_rep.loc["rep_01", "contact pairs"] == contact_pairs(atoms, np.empty((0, 2), int), p)

    # the saved stats are reused only while they cover exactly the finished replicas
    assert ensemble_stats_cached(runs[0])["replicas"] == ["rep_01", "rep_02", "rep_03"]
    (runs[0] / "rep_03" / "status").write_text("failed\n")
    assert ensemble_stats_cached(runs[0])["replicas"] == ["rep_01", "rep_02"]
    assert load_stats(runs[0])["replicas"] == ["rep_01", "rep_02"]          # and saved again


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_binding_rate_needs_frames(tmp_path):
    p = Params(qt_per_um3=500.0, ft_per_um3=2000.0, box_length=464.2, equil_steps=200, **influx(500), output_every=500,
               traj_every=500, data_dir=str(tmp_path), run_name="short")
    run = build_run(p)
    start(run, wait=True)
    with pytest.raises(ValueError, match="fewer than 3 frames in the influx"):
        binding_rate(run, save=False)
