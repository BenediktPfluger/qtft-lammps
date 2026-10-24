"""Start, monitor and stop LAMMPS runs in the background.

`start(run_dir)` launches a detached launcher process (`python -m qtft.launcher <run_dir>`) that
runs all replicas, at most `max_parallel` at a time. It keeps running when the notebook or
VS Code is closed. Every replica folder holds a `status` file:
built -> queued -> running -> done | failed | stopped.

`watch(run_dir)` follows a run live in the notebook; the launcher writes the same summary to
`launcher.log` (follow it in a terminal with `tail -f launcher.log`).

All functions also take a sweep folder (with sweep.json, see sweep.py): its runs share one launcher
and one queue, so at most `max_parallel` replicas of the whole sweep run at once.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

from .params import Params
from .read import last_thermo
from .sweep import is_sweep, sweep_runs

POLL_SECONDS = 2.0
PLUGIN_LOAD = re.compile(r'^if "1 == 1" then "plugin load (.+)"$', re.MULTILINE)   # see template.lmp


def _record(run_dir: Path) -> dict:
    return json.loads((run_dir / "params.json").read_text())


def _replicas(run_dir: Path) -> list[Path]:
    return sorted(p for p in run_dir.glob("rep_*") if p.is_dir())


def _runs(path: Path) -> list[Path]:
    """Run folders behind a target: the runs of a sweep folder, or the run folder itself."""
    return sweep_runs(path) if is_sweep(path) else [path]


def _all_replicas(path: Path) -> list[Path]:
    """Replicas of all runs of a target, replica-major (rep_01 of every run first, then rep_02, ...)."""
    per_run = [_replicas(r) for r in _runs(path)]
    return [reps[i] for i in range(max(map(len, per_run), default=0)) for reps in per_run if i < len(reps)]


def _launcher_pid(path: Path) -> Path:
    """launcher.pid of a target; a run inside a sweep is run by the sweep's launcher."""
    if not (path / "launcher.pid").exists() and is_sweep(path.parent):
        return path.parent / "launcher.pid"
    return path / "launcher.pid"


def _rep_name(path: Path, rep: Path) -> str:
    return rep.name if rep.parent == path else f"{rep.parent.name}/{rep.name}"


def _status(rep: Path) -> str:
    f = rep / "status"
    return f.read_text().strip() if f.exists() else "unknown"


def _set_status(rep: Path, status: str) -> None:
    (rep / "status").write_text(status + "\n")


def _n_parallel(params: dict) -> int:
    return params["max_parallel"] or max(1, (os.cpu_count() or 1) // params["mpi_procs"])


def _command(params: dict) -> list[str]:
    lmp = shlex.split(params["lmp_command"])
    args = ["-in", "in.lammps", "-log", "log.lammps", "-screen", "none"]
    if params["mpi_procs"] > 1:
        return shlex.split(params["mpirun_command"]) + ["-np", str(params["mpi_procs"])] + lmp + args
    return lmp + args


def start(run_dir: str | Path, wait: bool = False) -> None:
    """Run all replicas of a built run folder or sweep in the background (or in the foreground with
    wait=True)."""
    run_dir = Path(run_dir).expanduser()
    params = _record(_runs(run_dir)[0])["params"]
    reps = _all_replicas(run_dir)
    busy = [_rep_name(run_dir, r) for r in reps if _status(r) in ("queued", "running")]
    if busy:
        raise RuntimeError(f"replicas already queued/running: {busy}")
    missing = sorted({so for rep in reps for so in PLUGIN_LOAD.findall((rep / "in.lammps").read_text())
                      if not Path(so).exists()})
    if missing:
        raise FileNotFoundError(f"LAMMPS plugin not found: {missing}; build it with "
                                "lammps_plugins/bond_break_random/build.sh (see README)")
    for rep in reps:
        _set_status(rep, "queued")
        (rep / "pid").unlink(missing_ok=True)          # pid file = replica has started
    (run_dir / "launcher.pid").unlink(missing_ok=True)

    cmd = [sys.executable, "-m", "qtft.launcher", str(run_dir)]
    pkg_root = str(Path(__file__).resolve().parent.parent)
    env = {**os.environ, "PYTHONPATH": pkg_root + os.pathsep + os.environ.get("PYTHONPATH", "")}
    with open(run_dir / "launcher.log", "a") as log:
        if wait:
            subprocess.run(cmd, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
            return
        proc = subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)
    (run_dir / "launcher.pid").write_text(str(proc.pid))
    n_par = _n_parallel(params)
    print(f"started {len(reps)} replica(s), up to {n_par} in parallel, "
          f"{params['mpi_procs']} MPI process(es) each: {run_dir}\n"
          f"follow it with watch(run_dir), or in a terminal: tail -f {run_dir / 'launcher.log'}")


def stop(run_dir: str | Path) -> None:
    """Stop the launcher and all running replicas of a run or sweep. A single run inside a running
    sweep stops only its own replicas; the sweep continues with the others."""
    run_dir = Path(run_dir).expanduser()
    _kill(run_dir / "launcher.pid")
    for rep in _all_replicas(run_dir):
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
    """One row per replica: status, phase, step, bonds, fraction bound, speed, time left.

    phase "starting": LAMMPS is running but has not written its first thermo line yet
    (setup, particle insertion). For a sweep: one table with a (run, replica) index.
    """
    run_dir = Path(run_dir).expanduser()
    if is_sweep(run_dir):
        return pd.concat({r.name: progress(r) for r in _runs(run_dir)}, names=["run", "replica"])
    params = _record(run_dir)["params"]
    rows = []
    for rep in _replicas(run_dir):
        row = {"replica": rep.name, "status": _status(rep)}
        last = last_thermo(rep / "log.lammps")
        if last is None and row["status"] == "running":
            row["phase"] = "starting"
        elif last is not None:
            phase, values = last
            total = params["equil_steps"] if phase == "equilibration" else Params.from_dict(params).prod_steps
            row.update({
                "phase": phase,
                "step": int(values["Step"]),
                "done [%]": round(100 * values["Step"] / total, 1),
                "bonds": int(values["Bonds"]),
                "frac_bound": round(values["v_frac_bound"], 3),
                "steps/s": round(values["S/CPU"], 1),
                # remaining steps of the phase at the current speed (LAMMPS's CPULeft covers only the
                # current run command, i.e. one production segment)
                "left [min]": round((total - values["Step"]) / values["S/CPU"] / 60, 1)
                              if values["S/CPU"] > 0 else float("nan"),
            })
        rows.append(row)
    return pd.DataFrame(rows).set_index("replica")



# ------------------------------------------------------------------ run summary and live view
def _alive(pid_file: Path) -> bool | None:
    """True/False if the process in pid_file exists; None without pid file."""
    if not pid_file.exists():
        return None
    try:
        pid = int(pid_file.read_text())
        os.kill(pid, 0)
    except (ProcessLookupError, ValueError):
        return False
    except PermissionError:
        pass
    try:   # an ended child of the notebook kernel stays a zombie until the kernel exits
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _fraction(params: dict, rep: Path) -> float:
    """Fraction of all steps (equilibration + production) a replica has completed."""
    last = last_thermo(rep / "log.lammps")
    if last is None:
        return 0.0
    phase, values = last
    done = values["Step"] + (params["equil_steps"] if phase == "production" else 0)
    return done / (params["equil_steps"] + Params.from_dict(params).prod_steps)


def _hms(seconds: float) -> str:
    seconds = int(round(seconds))
    return f"{seconds // 3600}:{seconds // 60 % 60:02d}:{seconds % 60:02d}"


def summary(run_dir: str | Path) -> dict:
    """Overall state of a run or sweep: counts per status, start, elapsed time, estimated end.

    Replica start = time its pid file was written, end = last change of its status file.
    The estimate uses the speed of the running replicas (or the duration of finished ones)
    and fills the parallel slots with the queued replicas.
    """
    run_dir = Path(run_dir).expanduser()
    runs = _runs(run_dir)
    params = _record(runs[0])["params"]
    run_params = {r: _record(r)["params"] for r in runs}
    reps = _all_replicas(run_dir)
    status = {rep: _status(rep) for rep in reps}
    counts = {s: list(status.values()).count(s) for s in ("done", "failed", "stopped", "running", "queued")}
    now = time.time()
    starts = [(rep / "pid").stat().st_mtime for rep in reps if (rep / "pid").exists()]
    launcher_pid = _launcher_pid(run_dir)
    started = launcher_pid.stat().st_mtime if launcher_pid.exists() else min(starts, default=None)

    finished = [(rep / "status").stat().st_mtime - (rep / "pid").stat().st_mtime
                for rep in reps if status[rep] == "done" and (rep / "pid").exists()]
    left_running, per_replica = [], []
    for rep in reps:
        if status[rep] != "running" or not (rep / "pid").exists():
            continue
        elapsed, f = now - (rep / "pid").stat().st_mtime, _fraction(run_params[rep.parent], rep)
        if f > 0:
            per_replica.append(elapsed / f)
            left_running.append(elapsed * (1 - f) / f)
        else:
            left_running.append(None)
    duration = (sum(finished) / len(finished) if finished
                else sum(per_replica) / len(per_replica) if per_replica else None)
    eta = None
    if duration is not None:
        slots = [duration if t is None else t for t in left_running]
        slots += [0.0] * max(0, _n_parallel(params) - len(slots))
        for _ in range(counts["queued"]):
            i = slots.index(min(slots))
            slots[i] += duration
        eta = now + max(slots, default=0.0)

    active = counts["running"] + counts["queued"]
    end = max(((rep / "status").stat().st_mtime for rep in reps), default=now) if not active else None
    return {
        "run": run_dir.name, "group": run_dir.parent.name, "sweep": is_sweep(run_dir),
        "runs": len(runs), "replicas": len(reps),
        "parallel": _n_parallel(params), "mpi_procs": params["mpi_procs"], **counts,
        "started": started, "elapsed": ((end or now) - started) if started else None,
        "eta": eta if active else None, "finished": not active, "end": end,
        "launcher_alive": _alive(launcher_pid),
    }


def summary_lines(s: dict) -> list[str]:
    """Human-readable summary (used by watch() and the launcher log)."""
    clock = lambda t: datetime.fromtimestamp(t).strftime("%H:%M:%S")
    what = f"Sweep {s['run']}  ({s['runs']} runs, " if s.get("sweep") else f"Run {s['run']}  ({s['group']}, "
    lines = [what + f"{s['replicas']} replica(s), {s['parallel']} at once, {s['mpi_procs']} MPI proc(s) each)"]
    if s["started"] is None:
        return lines + ["not started"]
    state = (f"started {clock(s['started'])} · elapsed {_hms(s['elapsed'])} · "
             f"{s['done']}/{s['replicas']} done · {s['failed']} failed · "
             f"{s['running']} running · {s['queued']} queued")
    if s["stopped"]:
        state += f" · {s['stopped']} stopped"
    lines.append(state)
    if s["finished"]:
        lines.append(f"finished {clock(s['end'])}: {s['done']} done, {s['failed']} failed"
                     + (f", {s['stopped']} stopped" if s["stopped"] else "")
                     + f" (wall {_hms(s['elapsed'])})")
    elif s["launcher_alive"] is False:
        lines.append("WARNING: the launcher is not running any more (WSL restart, stop() or crash?): "
                     "queued replicas will not start and status files are no longer updated; "
                     "check the rep_XX/log.lammps files, then stop() and start() again")
    elif s["eta"] is not None:
        lines.append(f"estimated end ≈ {clock(s['eta'])} (≈ {_hms(s['eta'] - time.time())} left)")
    else:
        lines.append("estimated end: after the first thermo output")
    return lines


def watch(run_dir: str | Path, every: float = 10.0) -> None:
    """Live view of a run in the notebook, refreshed every `every` seconds until it has finished.

    Interrupting (stop button) only stops the view; the run continues in the background.
    """
    from IPython.display import clear_output, display

    run_dir = Path(run_dir).expanduser()
    try:
        while True:
            s = summary(run_dir)
            clear_output(wait=True)
            print("\n".join(summary_lines(s)) + "\n")
            display(progress(run_dir))
            if s["finished"] or s["launcher_alive"] is False:
                return
            time.sleep(every)
    except KeyboardInterrupt:
        print("stopped watching; the run continues in the background")
