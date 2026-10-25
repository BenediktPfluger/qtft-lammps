import json
import os
import re
import shutil
import time

import pandas as pd
import pytest

from qtft import Params, build_sweep, progress, start, summary, sweep_results, sweep_runs, update_index
from qtft.run import summary_lines
from conftest import influx


def test_build_sweep_grid(tmp_path):
    p = Params(n_replicas=2, data_dir=str(tmp_path))
    sweep = build_sweep(p, {"temperature": [300.0, 310.0, 320.0], "k_on": [1e-6, 2e-6]}, "T_kon")
    runs = sweep_runs(sweep)
    assert sweep == tmp_path / "sweeps" / "T_kon"
    assert [r.name for r in runs][:2] == ["temperature300_k_on1e-06", "temperature300_k_on2e-06"]
    assert len(runs) == 6 and all((r / "rep_02" / "in.lammps").exists() for r in runs)
    record = json.loads((sweep / "sweep.json").read_text())
    assert record["mode"] == "grid" and record["variants"]["k_on"] == [1e-6, 2e-6]
    q = Params.load(runs[-1] / "params.json")
    assert (q.temperature, q.k_on, q.group) == (320.0, 2e-6, "sweeps/T_kon")
    seeds = [json.loads((r / "params.json").read_text())["seeds"] for r in runs]
    assert all(s == seeds[0] for s in seeds)                          # common random numbers
    index = update_index(tmp_path)
    assert (index["group"] == "sweeps/T_kon").sum() == 6


def test_build_sweep_rejects_before_building(tmp_path):
    p = Params(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="execution settings"):
        build_sweep(p, {"n_replicas": [1, 2]}, "bad")
    with pytest.raises(ValueError, match="unknown"):
        build_sweep(p, {"tempreature": [300.0]}, "bad")
    with pytest.raises(ValueError, match="bind_radius45"):           # R_bind > pair cutoff
        build_sweep(p, {"bind_radius": [40.5, 45.0]}, "bad")
    assert not (tmp_path / "sweeps").exists()


def _touch(path, text, t):
    path.write_text(text)
    os.utime(path, (t, t))


def test_summary_and_progress_of_sweep(tmp_path):
    p = Params(n_replicas=2, max_parallel=2, data_dir=str(tmp_path))
    sweep = build_sweep(p, {"k_on": [1e-6, 2e-6]}, "s")
    a, b = sweep_runs(sweep)
    now = time.time()
    _touch(a / "rep_01" / "pid", "1", now - 100)
    _touch(a / "rep_01" / "status", "done\n", now - 10)
    _touch(b / "rep_01" / "pid", "2", now - 20)
    _touch(b / "rep_01" / "status", "running\n", now - 20)
    for rep in (a / "rep_02", b / "rep_02"):
        rep.joinpath("status").write_text("queued\n")
    s = summary(sweep)
    assert (s["runs"], s["replicas"], s["done"], s["running"], s["queued"]) == (2, 4, 1, 1, 2)
    assert summary_lines(s)[0].startswith("Sweep s  (2 runs, 4 replica(s), 2 at once")
    table = progress(sweep)
    assert table.index.names == ["run", "replica"] and len(table) == 4
    assert table.loc[(b.name, "rep_01"), "status"] == "running"


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_sweep_end_to_end(tmp_path):
    p = Params(n_qt=50, n_ft=200, box_length=464.2, equil_steps=200, **influx(2000),
               output_every=500, traj_every=500, n_replicas=2, max_parallel=2, data_dir=str(tmp_path))
    sweep = build_sweep(p, {"k_on": [1e-4, 2e-4]}, "kon")
    start(sweep, wait=True)
    assert all((rep / "status").read_text().strip() == "done" for r in sweep_runs(sweep) for rep in r.glob("rep_*"))

    # one queue: never more than max_parallel replicas at once, replica-major order
    log = (sweep / "launcher.log").read_text().splitlines()
    running, peak, order = 0, 0, []
    for line in log:
        if m := re.search(r" start (\S+)$", line):
            running += 1
            order.append(m.group(1))
        elif re.search(r" \S+/rep_\d+ (done|FAILED)", line):
            running -= 1
        peak = max(peak, running)
    assert peak == 2
    assert [o.split("/")[1] for o in order] == ["rep_01", "rep_01", "rep_02", "rep_02"]

    results = sweep_results(sweep)
    assert list(results.index) == ["k_on0.0001", "k_on0.0002"]
    assert (results["replicas done"] == 2).all() and results.attrs["swept"] == ["k_on"]
    assert pd.read_csv(sweep / "sweep_results.csv", index_col=0).shape[0] == 2

    import matplotlib
    matplotlib.use("Agg")
    from qtft import sweep_plot
    assert len(sweep_plot(results, y="final_Bonds").axes[0].lines) >= 1
