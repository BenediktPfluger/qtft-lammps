"""Overlap statistics: depths only of overlapping pairs, pooled over frames weighted by the pair count."""

import numpy as np
import pandas as pd
import pytest

from qtft import Params
from qtft.overlap import _pool, overlap_frame


def _atoms(rows):
    return pd.DataFrame(rows, columns=["id", "type", "x", "y", "z"])


def test_no_overlap_gives_nan_depth():
    p = Params(qt_per_um3=1.0, ft_per_um3=1.0, box_length=1000.0)
    far = overlap_frame(_atoms([(1, 1, 100, 100, 100), (2, 1, 300, 100, 100)]), p).set_index("family")
    assert far.loc["Qt-Qt", "n overlapping"] == 0
    assert far.loc["Qt-Qt", ["mean depth overlapping [nm]", "p95 depth [nm]", "max depth [nm]"]].isna().all()
    near = overlap_frame(_atoms([(1, 1, 100, 100, 100), (2, 1, 153, 100, 100)]), p).set_index("family")
    assert near.loc["Qt-Qt", "mean depth overlapping [nm]"] == pytest.approx(2.0)   # contact 55 nm


def test_pool_weights_by_overlapping_pairs():
    # frame 1: no overlap; frame 2: 1 pair at 1 nm; frame 3: 3 pairs at mean 2 nm (max 3) -> mean over the
    # 4 overlapping pairs (1 + 6) / 4 = 1.75 nm. The old mean of per-frame means with 0 for frame 1 gave 1.0
    frames = pd.DataFrame({
        "family": ["Qt-Qt"] * 3, "Step": [0, 1, 2], "time_ms": [0.0, 0.5, 1.0], "replica": ["rep_01"] * 3,
        "closest [nm]": [60.0, 54.0, 52.0], "n overlapping": [0, 1, 3], "frac overlapping": [0.0, 0.1, 0.3],
        "mean depth overlapping [nm]": [np.nan, 1.0, 2.0], "mean depth all pairs [% contact]": [0.0, 1.0, 2.0],
        "p95 depth [nm]": [np.nan, 1.0, 2.9], "max depth [nm]": [np.nan, 1.0, 3.0]})
    pooled = _pool(frames, ["family"]).loc["Qt-Qt"]
    assert pooled["mean depth overlapping [nm]"] == pytest.approx(1.75)
    assert pooled["p95 depth [nm]"] == pytest.approx(1.95) and pooled["max depth [nm]"] == 3.0
    assert pooled["n overlapping"] == pytest.approx(4 / 3)
    assert np.isnan(_pool(frames.iloc[:1], ["family"]).loc["Qt-Qt", "mean depth overlapping [nm]"])
