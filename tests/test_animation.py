"""Animation: frame statistics, phases, projections, sphere renderer, stills and video of a small
synthetic run folder (no LAMMPS)."""

import json

import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import pytest
from matplotlib.animation import FFMpegWriter

from qtft import Params, animate
from qtft import animation as an

NAME = "synthetic_run"


def _atoms(rows):
    """rows: (type, cluster, nbonds) per particle."""
    df = pd.DataFrame(rows, columns=["type", "cluster", "nbonds"])
    df.insert(0, "id", np.arange(1, len(df) + 1))
    return df


def test_frame_stats():
    atoms = _atoms([(1, 7, 2), (1, 7, 1), (1, 7, 1), (2, 7, 2), (2, 7, 2),   # 3 Qt + 2 Ft, 4 bonds
                    (1, 9, 1), (2, 9, 1),                                    # 1 Qt + 1 Ft, 1 bond
                    (1, 11, 0), (1, 12, 0), (2, 13, 0)])                     # free
    assert an.frame_stats(atoms) == {"bonds": 5, "largest_qt": 3, "multi_qt_pct": pytest.approx(50.0)}
    assert an.frame_stats(_atoms([(2, 1, 0)])) == {"bonds": 0, "largest_qt": 0, "multi_qt_pct": 0.0}


def test_phases_from_the_protocol():
    p = Params(t_baseline_s=10, t_influx_s=30, t_pumpout_s=90)
    assert [(a, b) for a, b, _, _ in an.phases(p)] == [(0, 10), (10, 40), (40, 130)]
    assert an.status_line(p, 5).endswith("INITIAL DISPERSED STATE")
    assert an.status_line(p, 10) == "CALCIUM ON  |  ASSEMBLY"
    assert an.status_line(p, 129) == "CALCIUM OFF  |  DISASSEMBLY"
    p = Params(t_baseline_s=0, t_influx_s=2, t_pumpout_s=3)
    assert [(a, b) for a, b, _, _ in an.phases(p)] == [(0, 2), (2, 5)]
    assert an.status_line(p, 0) == "CALCIUM ON  |  ASSEMBLY"


def test_scale_bar():
    assert [an.scale_bar_nm(v) for v in (3000, 1000, 600, 7000)] == [500, 200, 100, 1000]


def test_front_view():
    xy, scale, depth = an.front_view(np.array([[0, 0, 0], [300, 600, 300], [600, 300, 600.0]]), [0, 0, 0],
                                     [600, 600, 600], 380)
    np.testing.assert_allclose(xy, [[0, 0], [190, 190], [380, 380]])
    assert scale == pytest.approx(380 / 600)
    np.testing.assert_allclose(depth, [0, 1, 0.5])
    xy, _, _ = an.front_view(np.array([[0, 0, 0.0]]), [0, 0, 0], [800, 500, 400], 400)     # wide box: centred
    np.testing.assert_allclose(xy, [[0, 100]])


def test_perspective(monkeypatch):
    monkeypatch.setattr(an, "AZIMUTH", 0.0)
    monkeypatch.setattr(an, "ELEVATION", 0.0)
    cam = an.Perspective([600, 600, 600], 400)
    xy, r, depth = cam.project(np.array([[300, 0, 300], [300, 600, 300.0]]), np.array([20.0, 20.0]))
    np.testing.assert_allclose(xy, [[200, 200], [200, 200]], atol=1e-9)      # on the axis of view
    assert r[0] > r[1] and depth[0] < depth[1]                                  # nearer: larger, in front
    monkeypatch.setattr(an, "AZIMUTH", 30.0)
    monkeypatch.setattr(an, "ELEVATION", 20.0)
    hidden, front = an.Perspective([600, 600, 600], 400).edges()
    assert len(hidden) == 3 and len(front) == 9
    shared = {tuple(np.round(p, 6)) for p in hidden[0]}
    for edge in hidden[1:]:
        shared &= {tuple(np.round(p, 6)) for p in edge}
    assert len(shared) == 1                                                     # all at the farthest corner


def test_render_spheres():
    xy = np.array([[50.0, 50.0], [55.0, 50.0]])
    colors = ["#1E3CFF", "#E82020"]
    for order in ([0, 1], [1, 0]):                    # the input order does not matter, the depth does
        img = an.render_spheres(100, xy[order], np.array([10.0, 10.0]), [colors[i] for i in order],
                                np.array([0.0, 1.0])[order])
        assert img[50, 50, 3] == pytest.approx(1.0)
        assert img[50, 50, 2] > img[50, 50, 0]        # the nearer, blue sphere covers the red one
    assert img[0, 0, 3] == 0.0                        # background transparent
    rgb, alpha = an._sprite("#1F8A1F", 10.0)
    c = alpha.shape[0] // 2
    assert alpha[c, c] == 1.0 and alpha[0, 0] == 0.0
    assert rgb[c - 4, c - 4].sum() > rgb[c + 4, c + 4].sum()   # light from the top left
    far = an.render_spheres(100, xy[:1], np.array([10.0]), colors[:1], np.array([1.0]))
    near = an.render_spheres(100, xy[:1], np.array([10.0]), colors[:1], np.array([0.0]))
    assert far[50, 50, :3].sum() == pytest.approx((1 - an.DEPTH_CUE) * near[50, 50, :3].sum(), rel=1e-5)


def _synthetic_run(root, n_qt=6, n_ft=12):
    """Run folder below root/single with params.json, a log.lammps and a traj.dump of 6 frames (0-5 ms)."""
    p = Params(qt_per_um3=n_qt / 0.216, ft_per_um3=n_ft / 0.216, box_length=600, t_baseline_s=0.001, t_influx_s=0.002, t_pumpout_s=0.002)
    run = root / "single" / NAME
    rep = run / "rep_01"
    rep.mkdir(parents=True)
    (run / "params.json").write_text(json.dumps({"params": p.to_dict()}))
    keys = ["Step", "v_mean_size", "v_mean_size_qt", "v_mean_size_ft"]
    rows = [[s, 1 + s / 5000, 1 + s / 10000, 1 + s / 8000] for s in range(0, 10001, 500)]
    block = lambda data: "---\nkeywords: " + json.dumps(keys) + "\ndata:\n" + "".join(
        f"  - {json.dumps(r)}\n" for r in data) + "...\n"
    (rep / "log.lammps").write_text(block([[0, 1, 1, 1]]) + block(rows))
    rng = np.random.default_rng(1)
    frames = []
    for step in range(0, 10001, 2000):
        lines = []
        for i in range(n_qt + n_ft):
            qt = i < n_qt
            bonded = step > 0 and i % 3 == 0
            x, y, z = rng.random(3) * 600
            lines.append(f"{i + 1} {'Qt' if qt else 'Ft'} {x} {y} {z} 0 0 0 {27.5 if qt else 13.0} "
                         f"{1 if bonded else i + 1} {1 if bonded else 0}")
        frames.append(f"ITEM: TIMESTEP\n{step}\nITEM: NUMBER OF ATOMS\n{n_qt + n_ft}\nITEM: BOX BOUNDS pp pp pp\n"
                      "0 600\n0 600\n0 600\nITEM: ATOMS id type x y z ix iy iz radius cluster nbonds\n"
                      + "\n".join(lines) + "\n")
    (rep / "traj.dump").write_text("".join(frames))
    return run


def test_find_run(tmp_path):
    run = _synthetic_run(tmp_path)
    assert an.find_run(NAME, tmp_path) == run
    assert an.find_run(run, tmp_path) == run
    with pytest.raises(FileNotFoundError):
        an.find_run("no_such_run", tmp_path)


def test_stills(tmp_path):
    _synthetic_run(tmp_path)
    out = animate(NAME, stills=(0.001, 0.0041), video=False, data_dir=tmp_path)
    assert sorted(f.name for f in out.iterdir()) == ["still_0.001s.png", "still_0.0041s.png"]
    assert mpimg.imread(out / "still_0.001s.png").shape[:2] == (2160, 3840)          # default scale 2: 4K
    animate(NAME, stills=(0.001,), video=False, scale=1, data_dir=tmp_path)
    assert mpimg.imread(out / "still_0.001s.png").shape[:2] == (1080, 1920)


def test_scale_must_give_even_frame_size(tmp_path):
    _synthetic_run(tmp_path)
    with pytest.raises(ValueError, match="even"):
        animate(NAME, stills=(0.001,), video=False, scale=1.001, data_dir=tmp_path)   # 1921.92 x 1081.08


def test_video_needs_ffmpeg(tmp_path, monkeypatch):
    _synthetic_run(tmp_path)
    monkeypatch.setattr(FFMpegWriter, "isAvailable", classmethod(lambda cls: False))
    with pytest.raises(RuntimeError, match="ffmpeg"):
        animate(NAME, data_dir=tmp_path)


@pytest.mark.skipif(not FFMpegWriter.isAvailable(), reason="ffmpeg not installed")
def test_video(tmp_path):
    _synthetic_run(tmp_path)
    out = animate(NAME, fps=30, data_dir=tmp_path)
    assert sorted(f.name for f in out.iterdir()) == ["animation.mp4"]
    assert (out / "animation.mp4").stat().st_size > 10_000
