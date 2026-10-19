"""Start, monitor and stop LAMMPS runs in the background.

`start(run_dir)` launches a detached launcher process (`python -m qtft.launcher <run_dir>`) that
runs all replicas, at most `max_parallel` at a time. It keeps running when the notebook or
VS Code is closed. Every replica folder holds a `status` file:
built -> queued -> running -> done | failed | stopped.
"""

from __future__ import annotations

import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path

import pandas as pd

from .read import last_thermo

POLL_SECONDS = 2.0


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "params.json").read_text())


def _replicas(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("rep_*") if p.is_dir())


def _status(rep: Path) -> str:
    f = rep / "status"
    return f.read_text().strip() if f.exists() else "unknown"


def _set_status(rep: Path, status: str) -> None:
    (rep / "status").write_text(status + "\n")


def _command(params: dict) -> list[str]:
    lmp = shlex.split(params["lmp_command"])
    args = ["-in", "in.lammps", "-log", "log.lammps", "-screen", "none"]
    if params["mpi_procs"] > 1:
        return shlex.split(params["mpirun_command"]) + ["-np", str(params["mpi_procs"])] + lmp + args
    return lmp + args


def start(run_dir: str | Path, wait: bool = False) -> None:
    """Run all replicas of a built run folder in the background (or in the foreground with wait=True)."""
    run_dir = Path(run_dir).expanduser()
    params = _record(run_dir)["params"]
    reps = _replicas(run_dir)
    busy = [r.name for r in reps if _status(r) in ("queued", "running")]
    if busy:
        raise RuntimeError(f"replicas already queued/running: {busy}")
    for rep in reps:
        _set_status(rep, "queued")

    cmd = [sys.executable, "-m", "qtft.launcher", str(run_dir)]
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    if wait:
        subprocess.run(cmd, env=env, check=True)
        return
    with open(run_dir / "launcher.log", "a") as log:
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    (run_dir / "launcher.pid").write_text(str(proc.pid))
    n_par = params["max_parallel"] or max(1, (os.cpu_count() or 1) // params["mpi_procs"])
    print(f"started {len(reps)} replica(s), up to {n_par} in parallel, "
          f"{params['mpi_procs']} MPI process(es) each: {run_dir}")


def stop(run_dir: str | Path) -> None:
    """Stop the launcher and all running replicas of a run."""
    run_dir = Path(run_dir).expanduser()
    _kill(run_dir / "launcher.pid")
    for rep in _replicas(run_dir):
        if _status(rep) in ("queued", "running"):
            _kill(rep / "pid")
            _set_status(rep, "stopped")
    print(f"stopped: {run_dir}")


def _kill(pid_file: Path) -> None:
    if not pid_file.exists():
        return
    try:
        os.killpg(int(pid_file.read_text()), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, ValueError):
        pass


def progress(run_dir: str | Path) -> pd.DataFrame:
    """One row per replica: status, phase, step, bonds, fraction bound, speed, time left."""
    run_dir = Path(run_dir).expanduser()
    params = _record(run_dir)["params"]
    rows = []
    for rep in _replicas(run_dir):
        row = {"replica": rep.name, "status": _status(rep)}
        last = last_thermo(rep / "log.lammps")
        if last is not None:
            phase, values = last
            total = params["equil_steps"] if phase == "equilibration" else params["prod_steps"]
            row.update({
                "phase": phase,
                "step": int(values["Step"]),
                "done [%]": round(100 * values["Step"] / total, 1),
                "bonds": int(values["Bonds"]),
                "frac_bound": round(values["v_frac_bound"], 3),
                "steps/s": round(values["S/CPU"], 1),
                "left [min]": round(values["CPULeft"] / 60, 1),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("replica")

