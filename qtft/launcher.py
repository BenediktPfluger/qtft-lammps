"""Background launcher: runs all queued replicas of a run folder or sweep (started by qtft.run.start)."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from .ensemble import update_index
from .run import (POLL_SECONDS, _all_replicas, _command, _hms, _n_parallel, _record, _rep_name, _runs,
                  _set_status, _status, progress, summary, summary_lines)

LOG_EVERY = 60.0     # seconds between progress reports in launcher.log


def _log(text: str) -> None:
    for line in text.splitlines():
        print(f"{time.strftime('%H:%M:%S')} {line}", flush=True)


def _report(run_dir: Path) -> None:
    """Summary plus one line per running replica."""
    _log("\n".join(summary_lines(summary(run_dir))[1:]))
    table = progress(run_dir)
    for name, row in table[table["status"] == "running"].iterrows():
        name = "/".join(name) if isinstance(name, tuple) else name
        if row.get("phase") == "starting" or "step" not in row or row.isna()["step"]:
            _log(f"  {name}: starting (setup, particle insertion)")
        else:
            _log(f"  {name}: {row['phase']} step {int(row['step'])} ({row['done [%]']:.1f} %), "
                 f"{row['steps/s']:.1f} steps/s, {int(row['bonds'])} bonds, "
                 f"{row['left [min]']:.1f} min left in this phase")


def _launcher(run_dir: Path) -> None:
    runs = _runs(run_dir)
    params = _record(runs[0])["params"]          # execution settings are the same for all runs of a sweep
    n_par = _n_parallel(params)
    cmd = _command(params)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    queue = [r for r in _all_replicas(run_dir) if _status(r) == "queued"]
    running: dict[Path, subprocess.Popen] = {}
    _log(f"=== launcher started: {run_dir}" + (f" ({len(runs)} runs)" if len(runs) > 1 else ""))
    _log(f"{len(queue)} replica(s) queued, up to {n_par} at once, {params['mpi_procs']} MPI proc(s) each")
    _log(f"command: {' '.join(cmd)}")
    next_report = time.time() + LOG_EVERY

    while queue or running:
        while queue and len(running) < n_par:
            rep = queue.pop(0)
            if _status(rep) != "queued":                 # stopped while waiting
                continue
            with open(rep / "lammps.out", "w") as out:
                proc = subprocess.Popen(cmd, cwd=rep, env=env, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            (rep / "pid").write_text(str(proc.pid))
            _set_status(rep, "running")
            _log(f"start {_rep_name(run_dir, rep)}")
            running[rep] = proc
        time.sleep(POLL_SECONDS)
        for rep, proc in list(running.items()):
            rc = proc.poll()
            if rc is None:
                continue
            del running[rep]
            if _status(rep) == "stopped":
                continue
            log = rep / "log.lammps"
            ok = rc == 0 and log.exists() and "QTFT_DONE" in log.read_text()
            _set_status(rep, "done" if ok else "failed")
            s = summary(run_dir)
            wall = _hms((rep / "status").stat().st_mtime - (rep / "pid").stat().st_mtime)
            _log(f"{_rep_name(run_dir, rep)} {'done' if ok else f'FAILED (exit code {rc}, see lammps.out)'} "
                 f"in {wall} ({s['done']}/{s['replicas']} done, {s['failed']} failed)")
            update_index(params["data_dir"])
        if (queue or running) and time.time() >= next_report:
            _report(run_dir)
            next_report = time.time() + LOG_EVERY
    _log("=== " + summary_lines(summary(run_dir))[-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run all queued replicas of a run folder or sweep")
    parser.add_argument("run_dir")
    _launcher(Path(parser.parse_args().run_dir).expanduser())
