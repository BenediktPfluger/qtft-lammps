"""Time axes of the figures: ms for short runs, s for long runs (data always in ms)."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

import pandas as pd  # noqa: E402

from qtft import Params  # noqa: E402
from qtft.figures import RunData, _time_axis, p_agglomerate_size, time_unit  # noqa: E402


def _ticks(ax):
    ax.figure.canvas.draw()
    return [t.get_text().replace("−", "-") for t in ax.get_xticklabels()]


def test_time_unit():
    assert time_unit(500.0) == ("ms", 1.0) and time_unit(8000.0) == ("ms", 1.0)
    assert time_unit(10_000.0) == ("s", 1e-3) and time_unit(130_000.0) == ("s", 1e-3)


def test_time_axis_short_and_long():
    fig, (short, long) = plt.subplots(2)
    short.plot([0, 500], [0, 1])
    _time_axis(short, 500.0)
    assert short.get_xlabel() == "time (ms)" and "250" in _ticks(short)
    long.plot([0, 130_000], [0, 1])
    _time_axis(long, 130_000.0)
    assert long.get_xlabel() == "time (s)" and {"20", "40", "120"} <= set(_ticks(long))
    plt.close(fig)


def test_time_axis_shared():
    # the formatter set on the labelled bottom axis also applies to the axes sharing x
    fig, (top, bottom) = plt.subplots(2, sharex=True)
    top.plot([0, 130_000], [0, 1])
    _time_axis(bottom, 130_000.0)
    top.tick_params(labelbottom=True)
    assert top.get_xlabel() == "" and "40" in _ticks(top)
    plt.close(fig)


def test_agglomerate_histogram_per_finished_replica():
    # rep_01 has 4 agglomerates at the end of the influx, rep_02 none: 2 agglomerates per replica in all
    agglomerates = pd.DataFrame({"replica": ["rep_01"] * 4, "size": [10, 10, 10, 10]})
    run = RunData(Path("."), "run", "C0", Params(), {"replicas": ["rep_01", "rep_02"]}, agglomerates=agglomerates)
    fig, ax = plt.subplots()
    p_agglomerate_size(ax, [run])
    assert sum(patch.get_height() for patch in ax.patches) == 2.0
    plt.close(fig)
