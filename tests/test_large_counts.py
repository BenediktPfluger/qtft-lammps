"""Large-agglomerate counts from traj.dump (several thresholds, count and density) and their figure, on a
small synthetic run folder (no LAMMPS)."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qtft import Params, large_counts
from qtft.figures import RunData, fig_large_count

STEPS = (0, 1000, 2000)
# cluster sizes per replica and frame (a free particle is a cluster of 1)
SIZES = {"rep_01": [[1] * 60, [12, 6, 5, 3] + [1] * 34, [25, 12, 5] + [1] * 18],
         "rep_02": [[1] * 60, [20, 8, 4] + [1] * 28, [30, 10, 10] + [1] * 10]}


def _run(root, box=(600, 600, 600), sizes=SIZES):
    """Run folder with params.json and one finished replica per entry of `sizes` (traj.dump frames with the
    given cluster sizes)."""
    p = Params(n_qt=20, n_ft=40, box_length=list(box))
    run = root / "synthetic"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({"params": p.to_dict()}))
    for rep, frames in sizes.items():
        (run / rep).mkdir()
        (run / rep / "status").write_text("done\n")
        text = ""
        for step, frame in zip(STEPS, frames):
            cluster = np.repeat(np.arange(1, len(frame) + 1) * 7, frame)              # ids need not be dense
            lines = "".join(f"{i + 1} {'Qt' if i < 20 else 'Ft'} 0 0 0 {c} {int(s > 1)}\n"
                            for i, (c, s) in enumerate(zip(cluster, np.repeat(frame, frame))))
            text += (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{len(cluster)}\nITEM: BOX BOUNDS pp pp pp\n"
                     f"0 {box[0]}\n0 {box[1]}\n0 {box[2]}\nITEM: ATOMS id type x y z cluster nbonds\n" + lines)
        (run / rep / "traj.dump").write_text(text)
    return run, p


def _hand(rep, n):
    return [sum(s >= n for s in frame) for frame in SIZES[rep]]


def test_counts_for_several_thresholds(tmp_path):
    run, p = _run(tmp_path)
    t = large_counts(run, [10, 5, 20])
    assert sorted(t["n_min"].unique()) == [5, 10, 20]
    for n in (5, 10, 20):
        d = t[t["n_min"] == n]
        a, b = np.array(_hand("rep_01", n)), np.array(_hand("rep_02", n))
        np.testing.assert_array_equal(d["count_rep_01"], a)
        np.testing.assert_array_equal(d["count_rep_02"], b)
        np.testing.assert_allclose(d["count_mean"], (a + b) / 2)
        np.testing.assert_allclose(d["count_sd"], np.std([a, b], axis=0, ddof=1))
        np.testing.assert_allclose(d["per_1000um3_mean"], (a + b) / 2 * 1000 / 0.216)    # box 0.216 um^3
    np.testing.assert_allclose(t["time_s"].unique(), np.array(STEPS) * p.timestep * 1e-9)
    assert (t["replicas"] == 2).all()


def test_single_threshold_and_validation(tmp_path):
    run, _ = _run(tmp_path)
    t = large_counts(run, 5, save=False)
    assert list(t["n_min"].unique()) == [5]
    assert list(t["count_rep_01"]) == _hand("rep_01", 5)
    for bad in (0, 2.5, [], [5, -1]):
        with pytest.raises(ValueError, match="n_min"):
            large_counts(run, bad, save=False)


def test_density_of_a_rectangular_box_and_one_replica(tmp_path):
    run, _ = _run(tmp_path, box=(400, 500, 1000), sizes={"rep_01": SIZES["rep_01"]})
    d = large_counts(run, 5)
    np.testing.assert_allclose(d["per_1000um3_mean"], np.array(_hand("rep_01", 5)) * 1000 / 0.2)
    assert d["count_sd"].isna().all()


def test_csv_is_reused_and_extended(tmp_path):
    run, _ = _run(tmp_path)
    large_counts(run, [5, 10])
    path = run / "analysis" / "large_agglomerates.csv"
    for rep in ("rep_01", "rep_02"):                                    # make the cache visibly newer
        os.utime(run / rep / "traj.dump", (0, 0))
    first = path.stat().st_mtime_ns
    assert list(large_counts(run, 10)["n_min"].unique()) == [10]
    assert path.stat().st_mtime_ns == first                             # read from the CSV, not rewritten
    t = large_counts(run, 20)                                           # new threshold: recomputed, saved n kept
    assert list(t["n_min"].unique()) == [20]
    assert sorted(pd.read_csv(path)["n_min"].unique()) == [5, 10, 20]


def _rundata(run, p, label="run", color="#1f4e79"):
    return RunData(run, label, color, p, {"time_ms": np.array(STEPS) * p.timestep * 1e-6, "replicas": []})


def test_figure_count_and_density(tmp_path):
    run, p = _run(tmp_path)
    r = _rundata(run, p)
    fig = fig_large_count([r], tmp_path / "figs")
    ax = fig.axes[0]
    assert [l.get_text() for l in ax.get_legend().get_texts()] == ["≥ 5", "≥ 20", "≥ 60", "≥ 80", "≥ 100", "≥ 150"]
    assert ax.get_ylabel() == "agglomerates in the box"
    fig = fig_large_count([r], tmp_path / "figs", n_min=5, density=True, traces=True)
    ax = fig.axes[0]
    assert ax.get_ylabel() == "agglomerates ≥ 5 per 1000 µm³ (×10³)"            # up to 13900: in thousands
    assert ax.get_legend() is None
    traces = [l for l in ax.lines if l.get_lw() == 0.3]
    mean = [l for l in ax.lines if l.get_lw() == 1.0 and len(l.get_xdata()) == len(STEPS)]
    assert len(traces) == 2 and len(mean) == 1
    np.testing.assert_allclose(mean[0].get_ydata(), np.mean([_hand("rep_01", 5), _hand("rep_02", 5)], axis=0)
                               * 1000 / 0.216)
    assert sorted(f.name for f in (tmp_path / "figs").glob("*.png")) == ["large_agglomerates.png",
                                                                        "large_agglomerates_density.png"]
    plt.close("all")


def test_figure_several_runs(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    ra, pa = _run(tmp_path / "a")
    rb, pb = _run(tmp_path / "b", box=(400, 500, 1000))
    runs = [_rundata(ra, pa, "A", "#1f4e79"), _rundata(rb, pb, "B", "#c0392b")]
    fig = fig_large_count(runs, n_min=[5, 20], density=True)
    ax = fig.axes[0]
    assert [l.get_text() for l in fig.legends[0].get_texts()] == ["A", "B"]          # runs below the axes
    assert [l.get_text() for l in ax.get_legend().get_texts()] == ["≥ 5", "≥ 20"]     # thresholds in the axes
    styles = {(l.get_color(), l.get_linestyle()) for l in ax.lines}
    assert {("#1f4e79", "-"), ("#1f4e79", "--"), ("#c0392b", "-"), ("#c0392b", "--")} <= styles
    with pytest.raises(ValueError, match="at most 6"):
        fig_large_count(runs, n_min=[2, 5, 10, 20, 50, 80, 100])
    plt.close("all")
