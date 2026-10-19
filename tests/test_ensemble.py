import shutil

import numpy as np
import pandas as pd
import pytest

from qtft import Params, build_run, start
from qtft.ensemble import common_grid, compare, ensemble_stats, half_time, load_stats, update_index


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


@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_ensemble_of_tiny_runs(tmp_path):
    common = dict(n_qt=50, n_ft=200, box_length=464.2, equil_steps=200, prod_steps=2000,
                  output_every=500, traj_every=500, n_replicas=3, data_dir=str(tmp_path))
    runs = [build_run(Params(**common, k_on=k, run_name=f"kon{k:g}")) for k in (1e-4, 2e-4)]
    for r in runs:
        start(r, wait=True)

    stats = ensemble_stats(runs[0])
    assert stats["traces"]["Bonds"].shape == (3, 5)                        # 3 replicas, 5 frames
    finals = stats["per_replica"]["final_Bonds"]
    assert stats["final"].loc["final_Bonds", "mean"] == pytest.approx(finals.mean())
    assert stats["final"].loc["final_Bonds", "sd"] == pytest.approx(finals.std(ddof=0))
    assert stats["mean"]["Bonds"][-1] == pytest.approx(finals.mean())
    for name in ("ensemble_stats.json", "ensemble_timeseries.csv", "final_state.csv"):
        assert (runs[0] / "analysis" / name).exists()
    assert np.allclose(load_stats(runs[0])["mean"]["Bonds"], stats["mean"]["Bonds"])

    table = compare(runs)
    assert table.attrs["differing parameters"] == ["k_on"]
    assert list(table.columns) == ["k_on=0.0001", "k_on=0.0002"]

    index = pd.read_csv(tmp_path / "index.csv")
    assert set(index["status"]) == {"done:3"}
