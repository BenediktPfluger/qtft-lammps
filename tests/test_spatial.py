"""Qt spatial statistics for the FIB-SEM comparison (qtft.spatial) and their figure, on hand-made frames."""

import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from qtft import Params
from qtft.figures import RunData, fig_fibsem_comparison
from qtft.spatial import clark_evans, peak, qt_nn_intra, qt_pair_correlation, qt_ripley_l, qt_spatial_run

L = 1000.0
RNG = np.random.default_rng(7)


def _atoms(rows):
    """rows of (type, x, y, z, cluster) -> a traj.dump frame as read_dump returns it."""
    return pd.DataFrame(rows, columns=["type", "x", "y", "z", "cluster"])


def _params(boundary="periodic"):
    return Params(n_qt=10, n_ft=10, box_length=L, boundary=boundary)


# agglomerate 1: two Qt across the periodic boundary (20 nm apart through it); agglomerate 5: three Qt
# (nearest neighbours 60, 60, 100 nm); agglomerate 3: one Qt with an Ft (left out); Qt 9: free (left out)
FRAME = _atoms([(1, 10, 500, 500, 1), (1, 990, 500, 500, 1),
                (1, 300, 300, 300, 5), (1, 360, 300, 300, 5), (1, 300, 400, 300, 5),
                (1, 700, 700, 700, 3), (2, 740, 700, 700, 3), (1, 100, 900, 100, 9)])


def test_intra_agglomerate_nearest_neighbour():
    np.testing.assert_allclose(np.sort(qt_nn_intra(FRAME, _params())), [20, 20, 60, 60, 100])
    np.testing.assert_allclose(np.sort(qt_nn_intra(FRAME, _params("harmonic"))), [60, 60, 100, 980, 980])
    np.testing.assert_allclose(np.sort(qt_nn_intra(FRAME, _params(), min_qt=3)), [60, 60, 100])


def _uniform(n, cluster=1):
    pos = RNG.uniform(0, L, (n, 3))
    return _atoms([(1, *x, cluster) for x in pos])


def test_pair_correlation_of_random_points_is_one():
    g = qt_pair_correlation(_uniform(3000), _params(), bin_nm=10.0)
    assert g["r_nm"].iloc[0] == 5.0 and g["r_nm"].iloc[-1] == 495.0                  # up to L / 2
    assert abs(g.loc[g["r_nm"] > 100, "g"].mean() - 1) < 0.03
    assert qt_pair_correlation(FRAME.iloc[5:7], _params())["g"].isna().all()          # a single Qt


def test_clark_evans_random_and_clustered():
    ce = clark_evans(_uniform(3000), _params())
    rho = 3000 / L**3
    assert ce["n_qt"] == 3000 and ce["density_per_um3"] == pytest.approx(rho * 1e9)
    assert ce["expected_nn_nm"] == pytest.approx(0.55396 * rho ** (-1 / 3), rel=1e-4)
    assert ce["R"] == pytest.approx(1, abs=0.04)
    centres = RNG.uniform(0, L, (100, 3))                                           # 30 Qt around each centre
    pos = (centres[:, None, :] + RNG.normal(0, 15, (100, 30, 3))).reshape(-1, 3)
    assert clark_evans(_atoms([(1, *x, 1) for x in pos]), _params())["R"] < 0.3


def test_ripley_l_random_clustered_and_walls():
    random = qt_ripley_l(_uniform(3000), _params(), bin_nm=10.0)
    assert random["r_nm"].iloc[0] == 10.0 and random["r_nm"].iloc[-1] == 500.0
    assert random.loc[random["r_nm"] >= 50, "L_minus_r"].abs().max() < 5                   # random: L(r) = r
    walls = qt_ripley_l(_uniform(3000), _params("harmonic"), bin_nm=10.0)
    assert walls.loc[walls["r_nm"] >= 50, "L_minus_r"].abs().max() < 10                    # border-corrected
    centres = RNG.uniform(0, L, (100, 3))                                                 # 30 Qt around each centre
    pos = (centres[:, None, :] + RNG.normal(0, 15, (100, 30, 3))).reshape(-1, 3)
    clustered = qt_ripley_l(_atoms([(1, *x, 1) for x in pos]), _params(), bin_nm=10.0).set_index("r_nm")
    assert clustered.loc[50.0, "L_minus_r"] > 50 and clustered["L_minus_r"].idxmax() < 150
    assert qt_ripley_l(FRAME.iloc[5:7], _params())["L_minus_r"].isna().all()


def test_peak():
    assert peak(RNG.normal(60, 3, 2000)) == pytest.approx(60, abs=1)
    r = np.arange(0, 200, 5.0)
    g = np.where(r < 50, 5.0, 1.0)
    g[r == 70] = 3.0
    assert peak(r, g, r_min=55) == 70 and peak(r, g) == 0
    assert peak(np.full(5, 7.0)) == 7.0 and peak(np.array([7.0, 7.0 + 1e-13])) == pytest.approx(7.0)
    assert math.isnan(peak(np.empty(0)))


def _run(root):
    """Run folder with two finished replicas: random Qt in the first frame, Qt in pairs 20 nm apart (one
    agglomerate per pair) at the end of the influx (the last frame, the default protocol is longer)."""
    p = Params(n_qt=400, n_ft=0, box_length=L)
    run = root / "spatial"
    run.mkdir()
    (run / "params.json").write_text(json.dumps({"params": p.to_dict()}))
    for rep in ("rep_01", "rep_02"):
        (run / rep).mkdir()
        (run / rep / "status").write_text("done\n")
        first = RNG.uniform(0, L, (400, 3))
        centres = RNG.uniform(100, L - 100, (200, 3))
        last = np.concatenate([centres, centres + [20, 0, 0]])
        text = ""
        for step, pos, cluster in ((0, first, np.arange(1, 401)), (1000, last, np.tile(np.arange(1, 201), 2))):
            lines = "".join(f"{i + 1} Qt {x:.3f} {y:.3f} {z:.3f} {c}\n"
                            for i, ((x, y, z), c) in enumerate(zip(pos, cluster)))
            text += (f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n400\nITEM: BOX BOUNDS pp pp pp\n"
                     f"0 {L}\n0 {L}\n0 {L}\nITEM: ATOMS id type x y z cluster\n" + lines)
        (run / rep / "traj.dump").write_text(text)
    return run, p


def test_spatial_run_files_cache_and_figure(tmp_path):
    run, p = _run(tmp_path)
    res = qt_spatial_run(run)
    ce = res["clark_evans"].set_index(["replica", "frame"])
    assert list(ce["step"]) == [0, 1000, 0, 1000]
    assert ce.loc[("rep_01", "first"), "R"] == pytest.approx(1, abs=0.15)
    assert ce.loc[("rep_01", "influx_end"), "R"] < 0.5
    nn = res["nn"]
    assert set(nn.loc[nn["frame"] == "first", "set"]) == {"all"}                  # intra only at the influx end
    np.testing.assert_allclose(nn.loc[nn["set"] == "intra", "distance_nm"], 20, atol=1e-3)
    assert list(res["rdf"].columns) == ["r_nm", "g_rep_01", "g_rep_02", "g_mean"]
    assert list(res["ripley"].columns) == ["replica", "frame", "r_nm", "L_minus_r"]
    ripley = res["ripley"].groupby(["frame", "r_nm"])["L_minus_r"].mean()
    assert ripley["influx_end"].max() > ripley["first"].abs().max() + 10                  # pairs: clustered
    for name in ("qt_nn_distances.csv", "qt_pair_correlation.csv", "qt_clark_evans.csv", "qt_ripley_l.csv"):
        assert (run / "analysis" / name).exists(), name

    csvs = list((run / "analysis").glob("qt_*.csv"))
    for path in csvs:
        os.utime(path, (2e9, 2e9))                                                   # newer than the dumps
    pd.testing.assert_frame_equal(qt_spatial_run(run)["clark_evans"], res["clark_evans"])
    assert all(path.stat().st_mtime == 2e9 for path in csvs)                         # cached: not rewritten
    qt_spatial_run(run, bin_nm=10.0)                                                 # other r grid: recomputed
    assert all(path.stat().st_mtime != 2e9 for path in csvs)

    rd = RunData(run, "spatial", "C0", p, {"time_ms": np.array([0.0, 1.0])})
    fig = fig_fibsem_comparison(rd, tmp_path / "figs", spatial=res)
    a, b, c, d = fig.axes
    assert "r = 20 nm" in a.texts[0].get_text() and "r = " in b.texts[0].get_text()
    labels = [t.get_text() for t in c.get_legend().get_texts()]
    assert labels[1].startswith("end of influx: R = ") and labels[2] == "random points"
    labels = [t.get_text() for t in d.get_legend().get_texts()]
    assert labels[0] == "random points" and labels[2].startswith("end of influx: max ")
    assert len(list((tmp_path / "figs" / "fibsem_comparison").glob("*.png"))) == 4
    plt.close("all")
