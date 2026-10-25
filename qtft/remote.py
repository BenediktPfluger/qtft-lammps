"""Runs on the LRZ Linux cluster (CoolMUC-4, Slurm), for Params(executor="lrz").

The run folder is built locally (build_run) and mirrored on LRZ below LRZ["root"] in the LRZ home:
runs/<group>/<run name>/ there corresponds to <data_dir>/<group>/<run name>/ here. job.slurm (from
job_template.slurm) runs every replica as one task of a Slurm job array and writes the same status
files as the local launcher. LAMMPS on LRZ has fix bond/break/random built in, so the input loads no
plugin. Setup (ssh, LAMMPS build): docs/INSTALL.md, section "LRZ Linux cluster".

All commands go through ssh/rsync to the host LRZ["host"] in non-interactive mode (BatchMode): they
use the connection opened once with `ssh lrz` in a terminal (ControlMaster, 2FA) and fail at once,
with a hint, when it is closed. qtft.run hands runs with executor "lrz" to submit (start), sync
(progress, summary, watch), cancel (stop) and fetch.
"""

from __future__ import annotations

import datetime
import json
import subprocess
import time
from pathlib import Path

from .params import Params, data_path

LRZ = {
    "host": "lrz",                                 # Host entry in ~/.ssh/config (ControlMaster: one 2FA login)
    "root": "qtft-lammps",                         # in the LRZ home; holds runs/ like the repository here
    "cluster": "serial",                           # Slurm cluster ...
    "partition": "serial_long",                    # ... partition (max. 7 days, 1 node per job) ...
    "qos": "cm4_serial_long",                      # ... and the QOS it requires
    "modules": "gcc/15.2.0 intel-mpi/2021.12.0",   # the Spack gcc modules cannot compile <mutex>
    "lmp": "$HOME/src/lammps/build/lmp",           # built with the fix compiled in
    "mail_user": "benedikt.pfluger@tum.de",
    "mail_type": "BEGIN,END,FAIL",                 # for a job array: once for the whole array
}
SSH_OPTIONS = ["-x", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]   # no X11, never ask for a password
SMALL_FILES = ("status", "pid", "log.lammps", "lammps.out")              # what progress/watch fetch
SYNC_SECONDS = 30.0      # progress/summary fetch the small files at most this often
JOB_FILE = "slurm_job.json"
_last_sync: dict[Path, float] = {}
_states: dict[Path, dict[int, tuple[str, str]]] = {}     # Slurm state per array task at the last sync


def remote_dir(local_dir: str | Path, p: Params) -> str:
    """Path on LRZ (relative to the home) of a local run or sweep folder below the data folder."""
    rel = Path(local_dir).expanduser().resolve().relative_to(data_path(p.data_dir).resolve())
    return f"{LRZ['root']}/runs/{rel.as_posix()}"


# ------------------------------------------------------------------ ssh and rsync
def _not_connected(detail: str) -> ConnectionError:
    return ConnectionError(f"no connection to LRZ ({detail.strip() or 'ssh failed'}). Log in once in a "
                           f"terminal with:  ssh {LRZ['host']}  (password + 2FA); the connection then stays "
                           "open for 8 hours and qtft uses it")


def ssh(command: str) -> str:
    """Run a shell command in the LRZ home; returns its output."""
    r = subprocess.run(["ssh", *SSH_OPTIONS, LRZ["host"], command], capture_output=True, text=True)
    if r.returncode == 255:
        raise _not_connected(r.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"command on LRZ failed (exit {r.returncode}): {command}\n{r.stderr.strip()}")
    return r.stdout


def _rsync(*args: str) -> None:
    r = subprocess.run(["rsync", "-a", "-z", "-e", " ".join(["ssh", *SSH_OPTIONS]), *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        ssh("true")                                  # raises ConnectionError if the connection is the cause
        raise RuntimeError(f"rsync failed (exit {r.returncode}): {r.stderr.strip()}")


# ------------------------------------------------------------------ job bookkeeping
def _params(target: Path) -> Params:
    """Params of a run folder, or of the first run of a sweep folder (execution settings are shared)."""
    from .sweep import is_sweep, sweep_runs
    return Params.load((sweep_runs(target)[0] if is_sweep(target) else target) / "params.json")


def _job(target: Path) -> tuple[dict, Path] | None:
    """The Slurm job that runs the replicas of target (submitted for target itself or, for a run
    inside a sweep, for the sweep) and the folder it was submitted from."""
    for folder in (target, target.parent):
        if (folder / JOB_FILE).exists():
            return json.loads((folder / JOB_FILE).read_text()), folder
    return None


def job_info(target: Path) -> dict | None:
    """Job ID, Slurm cluster, folder on LRZ and submission time of target's job (None if not submitted)."""
    found = _job(Path(target))
    return found[0] if found else None


def _tasks(job_dir: Path, target: Path) -> dict[int, Path]:
    """Array task index -> local replica folder, for the replicas of target."""
    lines = (job_dir / "replicas.txt").read_text().split()
    reps = {i: job_dir / line for i, line in enumerate(lines, start=1)}
    return {i: r for i, r in reps.items() if r == target or target in r.parents}


def slurm_states(target: Path) -> dict[int, tuple[str, str]] | None:
    """State and reason per array task of the job of target while it is in the Slurm queue ({} when it
    has left the queue); None if target was never submitted."""
    found = _job(target)
    if found is None:
        return None
    info, job_dir = found
    out = ssh(f"squeue -M {info['cluster']} -h -r -j {info['job']} -o '%i|%T|%r' 2>/dev/null || true")
    states = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) == 3 and "_" in parts[0]:
            task = parts[0].split("_", 1)[1]
            if task.isdigit():
                states[int(task)] = (parts[1], parts[2])
    tasks = _tasks(job_dir, target)
    return {i: s for i, s in states.items() if i in tasks}


def _write_status(job_dir: Path, info: dict, changes: dict[Path, str]) -> None:
    """Set status files locally and on LRZ."""
    if not changes:
        return
    remote = info["remote"]
    ssh(" && ".join(f"echo {s} > {remote}/{r.relative_to(job_dir).as_posix()}/status" for r, s in changes.items()))
    for rep, status in changes.items():
        (rep / "status").write_text(status + "\n")


def _close_lost_tasks(target: Path, fetch_small) -> None:
    """Replicas still queued/running whose array task has left Slurm without finishing (time limit, node
    failure, cancelled outside qtft) get status failed (cancelled: stopped), as sacct reports."""
    found = _job(target)
    if found is None:
        return
    info, job_dir = found
    tasks = _tasks(job_dir, target)

    def active():
        return {i: r for i, r in tasks.items()
                if (r / "status").exists() and (r / "status").read_text().strip() in ("queued", "running")}

    if not active():
        _states[target] = {}
        return
    queued_or_running = _states[target] = slurm_states(target)
    if all(i in queued_or_running for i in active()):
        return
    fetch_small()                    # a task may have finished between fetching its status and squeue
    if all(i in queued_or_running for i in active()):
        return
    out = ssh(f"sacct -M {info['cluster']} -j {info['job']} -X -n -P -o JobID,State 2>/dev/null || true")
    final = {}
    for line in out.splitlines():
        job_task, _, state = line.partition("|")
        task = job_task.split("_", 1)[1] if "_" in job_task else ""
        if task.isdigit():
            final[int(task)] = state.split()[0] if state else ""
    changes = {}
    for i, rep in active().items():
        if i in queued_or_running or final.get(i, "") in ("PENDING", "RUNNING", "REQUEUED", "COMPLETED", ""):
            continue                                 # COMPLETED: the job script itself wrote the status
        changes[rep] = "stopped" if final[i].startswith("CANCELLED") else "failed"
    _write_status(job_dir, info, changes)


# ------------------------------------------------------------------ public: submit, sync, cancel, fetch
def submit(target: Path) -> str:
    """Copy the run or sweep folder to LRZ and submit its job.slurm; returns the Slurm job ID."""
    target = Path(target)
    if not (target / "job.slurm").exists():
        raise FileNotFoundError(f"{target / 'job.slurm'} missing: build the run with executor='lrz'")
    p = _params(target)
    remote = remote_dir(target, p)
    ssh(f"test -x {LRZ['lmp']} || {{ echo 'LAMMPS not found on LRZ: {LRZ['lmp']} (docs/INSTALL.md, section 9)' >&2; exit 1; }}"
        f" && mkdir -p {remote}")
    _rsync("--exclude=analysis/", "--exclude=launcher.*", f"--exclude={JOB_FILE}", f"{target}/",
           f"{LRZ['host']}:{remote}/")
    # pid files of an earlier submission would date the start of the new one
    out = ssh(f"cd {remote} && sed 's|$|/pid|' replicas.txt | xargs rm -f && sbatch --parsable job.slurm")
    job = out.strip().splitlines()[-1].split(";")[0]
    info = {"job": job, "cluster": LRZ["cluster"], "remote": remote,
            "submitted": datetime.datetime.now().isoformat(timespec="seconds")}
    (target / JOB_FILE).write_text(json.dumps(info, indent=2))
    _last_sync.pop(target, None)
    return job


def sync(target: Path, force: bool = False, everything: bool = False) -> None:
    """Fetch status, pid, log.lammps and lammps.out of every replica (everything=True: all files) from
    LRZ; at most every SYNC_SECONDS unless force."""
    target = Path(target)
    found = _job(target)
    if found is None:
        return
    if not force and time.time() - _last_sync.get(target, 0.0) < SYNC_SECONDS:
        return
    info, job_dir = found
    rel = target.relative_to(job_dir).as_posix()
    remote = info["remote"] if rel == "." else f"{info['remote']}/{rel}"
    small = ["--include=*/", *[f"--include={f}" for f in SMALL_FILES], "--include=slurm-*.out", "--exclude=*"]
    _rsync(*([] if everything else small), f"{LRZ['host']}:{remote}/", f"{target}/")
    _close_lost_tasks(target, lambda: _rsync(*small, f"{LRZ['host']}:{remote}/", f"{target}/"))
    _last_sync[target] = time.time()


def last_states(target: Path) -> dict[int, tuple[str, str]]:
    """Slurm state and reason per array task of target's replicas at the last sync ({} if none queued)."""
    return _states.get(Path(target), {})


def cancel(target: Path) -> None:
    """Cancel the array tasks of target's replicas on LRZ; queued/running replicas get status stopped."""
    target = Path(target)
    found = _job(target)
    if found is None:
        return
    info, job_dir = found
    tasks = _tasks(job_dir, target)
    ids = [info["job"]] if job_dir == target else [f"{info['job']}_{i}" for i in tasks]
    ssh(f"scancel -M {info['cluster']} {' '.join(ids)}")
    _write_status(job_dir, info, {r: "stopped" for r in tasks.values()
                                  if (r / "status").read_text().strip() in ("queued", "running")})
    _last_sync.pop(target, None)


def fetch(target: Path) -> None:
    """Download all files of target from LRZ (dumps, logs, ...); the folder then looks like a local run."""
    sync(Path(target), force=True, everything=True)
