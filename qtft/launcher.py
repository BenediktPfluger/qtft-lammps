"""Background launcher: runs all queued replicas of a run folder (started by qtft.run.start)."""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from .run import POLL_SECONDS, _command, _record, _replicas, _set_status, _status


def _launcher(run_dir: Path) -> None:
    params = _record(run_dir)["params"]
    n_par = params["max_parallel"] or max(1, (os.cpu_count() or 1) // params["mpi_procs"])
    cmd = _command(params)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    queue = [r for r in _replicas(run_dir) if _status(r) == "queued"]
    running: dict[Path, subprocess.Popen] = {}

    while queue or running:
        while queue and len(running) < n_par:
            rep = queue.pop(0)
            with open(rep / "lammps.out", "w") as out:
                proc = subprocess.Popen(cmd, cwd=rep, env=env, stdout=out, stderr=subprocess.STDOUT,
                                        stdin=subprocess.DEVNULL, start_new_session=True)
            (rep / "pid").write_text(str(proc.pid))
            _set_status(rep, "running")
            print(f"{time.strftime('%H:%M:%S')} start {rep.name}: {' '.join(cmd)}", flush=True)
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
            print(f"{time.strftime('%H:%M:%S')} {rep.name} {'done' if ok else f'FAILED (exit code {rc})'}",
                  flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run all queued replicas of a run folder")
    parser.add_argument("run_dir")
    _launcher(Path(parser.parse_args().run_dir).expanduser())
