import os
import time

import pytest

from qtft import Params, build_run, progress, summary
from qtft.run import summary_lines
from conftest import influx

KEYS = "keywords: ['Step', 'Bonds', 'v_frac_bound', 'S/CPU', 'CPULeft']\n"


def _thermo(phase_steps):
    """Fake log.lammps with one YAML thermo block per phase, last line at the given step."""
    return "".join(f"---\n{KEYS}data:\n  - [{step}, 12, 0.1, 100.0, 30.0]\n...\n" for step in phase_steps)


def _touch(path, text, t):
    path.write_text(text)
    os.utime(path, (t, t))


def test_summary_counts_and_estimate(tmp_path):
    run = build_run(Params(n_replicas=4, max_parallel=2, equil_steps=1000, **influx(9000, segment=1000),
                           data_dir=str(tmp_path), run_name="r"))
    now = time.time()
    r1, r2, r3, _ = sorted(run.glob("rep_*"))
    _touch(r1 / "pid", "1", now - 100)                  # done after 90 s
    _touch(r1 / "status", "done\n", now - 10)
    _touch(r2 / "pid", "2", now - 50)                   # running, half of all steps done
    _touch(r2 / "status", "running\n", now - 50)
    (r2 / "log.lammps").write_text(_thermo([1000, 4000]))
    _touch(r3 / "pid", "3", now - 5)                    # running, no thermo output yet
    _touch(r3 / "status", "running\n", now - 5)
    (run / "rep_04" / "status").write_text("queued\n")

    s = summary(run)
    assert (s["done"], s["running"], s["queued"], s["failed"]) == (1, 2, 1, 0)
    assert s["started"] == pytest.approx(now - 100, abs=1)
    assert s["launcher_alive"] is None and not s["finished"]
    # two slots: rep_02 needs 50 s more, rep_03 (no output yet) the 90 s of a finished replica;
    # rep_04 then starts in the slot of rep_02 and takes 90 s
    assert s["eta"] - now == pytest.approx(50 + 90, abs=2)
    assert "1/4 done" in summary_lines(s)[1]

    table = progress(run)
    assert table.loc["rep_02", "phase"] == "production" and table.loc["rep_02", "step"] == 4000
    assert table.loc["rep_03", "phase"] == "starting"


def test_summary_finished(tmp_path):
    run = build_run(Params(n_replicas=2, data_dir=str(tmp_path), run_name="r"))
    now = time.time()
    for i, rep in enumerate(sorted(run.glob("rep_*"))):
        _touch(rep / "pid", "1", now - 60)
        _touch(rep / "status", "done\n" if i == 0 else "failed\n", now - 20 + i)
    s = summary(run)
    assert s["finished"] and s["eta"] is None
    assert s["elapsed"] == pytest.approx(41, abs=1)
    assert summary_lines(s)[-1].startswith("finished") and "1 done, 1 failed" in summary_lines(s)[-1]
