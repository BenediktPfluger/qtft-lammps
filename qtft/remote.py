"""Runs on the LRZ Linux cluster (CoolMUC-4, Slurm), for Params(executor="lrz").

The run folder is built locally (build_run) and mirrored on LRZ below LRZ["root"] in the LRZ home:
runs/<group>/<run name>/ there corresponds to <data_dir>/<group>/<run name>/ here. job.slurm (from
job_template.slurm) runs every replica as one task of a Slurm job array and writes the same status
files as the local launcher. LAMMPS on LRZ has fix bond/break/random built in, so the input loads no
plugin. Setup (ssh, LAMMPS build): docs/INSTALL.md, section "LRZ Linux cluster".

All commands go through ssh/rsync to the host LRZ["host"] in non-interactive mode (BatchMode): they
use the connection opened once with `ssh lrz` in a terminal (ControlMaster, 2FA) and fail at once,
with a hint, when it is closed. qtft.run hands runs with executor "lrz" to submit (start), sync
(progress, summary, watch; fetch with everything=True) and cancel (stop). Nothing is copied back by
itself: outputs stay on LRZ until fetch. require_fetched (local files only) stops the analysis of a
run whose finished replicas are not downloaded yet.
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
    "mail_user": "",                               # email for Slurm; set in lrz_local.json, never in git
    "mail_type": "BEGIN,END,FAIL",                 # for a job array: once for the whole array
}
LOCAL_SETTINGS = Path(__file__).resolve().parent.parent / "lrz_local.json"   # personal settings, gitignored


def load_local_settings(path: Path = LOCAL_SETTINGS) -> None:
    """Personal LRZ settings (e.g. {"mail_user": "name@example.org"}) from a gitignored JSON file into LRZ,
    so that they never appear in the repository. A missing file keeps the defaults (no email)."""
    if not path.exists():
        return
    values = json.loads(path.read_text())
    unknown = set(values) - set(LRZ)
    if unknown:
        raise ValueError(f"{path.name}: unknown LRZ settings {sorted(unknown)}; known: {list(LRZ)}")
    LRZ.update(values)


load_local_settings()
SSH_OPTIONS = ["-x", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20"]   # no X11, never ask for a password
STATUS_FILES = ("status", "pid")                                         # progress/watch fetch these ...
LOG_FILES = ("log.lammps", "lammps.out", "slurm-*.out")                  # ... and the new lines of these
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
def job(target: Path) -> tuple[dict, Path] | None:
    """The Slurm job that runs the replicas of target (submitted for target itself or, for a run
    inside a sweep, for the sweep): its slurm_job.json (job ID, cluster, folder on LRZ, submission
    time) and the folder it was submitted from; None if never submitted."""
    target = Path(target)
    for folder in (target, target.parent):
        if (folder / JOB_FILE).exists():
            return json.loads((folder / JOB_FILE).read_text()), folder
    return None


def _tasks(job_dir: Path, target: Path) -> dict[int, Path]:
    """Array task index -> local replica folder, for the replicas of target."""
    lines = (job_dir / "replicas.txt").read_text().split()
    reps = {i: job_dir / line for i, line in enumerate(lines, start=1)}
    return {i: r for i, r in reps.items() if r == target or target in r.parents}


def slurm_states(target: Path) -> dict[int, tuple[str, str]]:
    """State and reason per array task of target's job while it is in the Slurm queue ({} when it has
    left the queue or was never submitted)."""
    found = job(target)
    if found is None:
        return {}
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
    found = job(target)
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
def submit(target: Path, p: Params) -> str:
    """Copy the run or sweep folder to LRZ and submit its job.slurm; returns the Slurm job ID. p: the
    parameters of the run (of a sweep: of any of its runs; the execution settings are shared)."""
    target = Path(target)
    if not (target / "job.slurm").exists():
        raise FileNotFoundError(f"{target / 'job.slurm'} missing: build the run with executor='lrz'")
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
    found = job(target)
    if found is None:
        return
    if not force and time.time() - _last_sync.get(target, 0.0) < SYNC_SECONDS:
        return
    info, job_dir = found
    rel = target.relative_to(job_dir).as_posix()
    source = f"{LRZ['host']}:{info['remote'] if rel == '.' else info['remote'] + '/' + rel}/"
    only = lambda names: ["--include=*/", *[f"--include={n}" for n in names], "--exclude=*"]

    def small():
        _rsync(*only(STATUS_FILES), source, f"{target}/")
        # logs only grow: transfer the new lines (start() removes local logs before a new submission)
        _rsync("--append-verify", *only(LOG_FILES), source, f"{target}/")

    if everything:
        _rsync(source, f"{target}/")
    else:
        small()
    _close_lost_tasks(target, small)
    _last_sync[target] = time.time()


def last_states(target: Path) -> dict[int, tuple[str, str]]:
    """Slurm state and reason per array task of target's replicas at the last sync ({} if none queued)."""
    return _states.get(Path(target), {})


def cancel(target: Path) -> None:
    """Cancel the array tasks of target's replicas on LRZ; queued/running replicas get status stopped."""
    target = Path(target)
    found = job(target)
    if found is None:
        return
    info, job_dir = found
    tasks = _tasks(job_dir, target)
    ids = [info["job"]] if job_dir == target else [f"{info['job']}_{i}" for i in tasks]
    ssh(f"scancel -M {info['cluster']} {' '.join(ids)}")
    _write_status(job_dir, info, {r: "stopped" for r in tasks.values()
                                  if (r / "status").read_text().strip() in ("queued", "running")})
    _last_sync.pop(target, None)


def unfetched(target: str | Path) -> list[Path]:
    """Finished replicas of an LRZ run or sweep whose results are not downloaded yet (no final.data
    here, which LAMMPS writes at the end and only fetch copies). Local files only, no ssh."""
    from .sweep import is_sweep, sweep_runs
    target = Path(target).expanduser()
    if job(target) is None:
        return []
    runs = sweep_runs(target) if is_sweep(target) else [target]
    return [rep for run in runs for rep in sorted(run.glob("rep_*"))
            if (rep / "status").exists() and (rep / "status").read_text().strip() == "done"
            and not (rep / "final.data").exists()]


def require_fetched(target: str | Path) -> None:
    """Stop the analysis of an LRZ run whose finished replicas are not downloaded yet."""
    missing = unfetched(target)
    if missing:
        raise RuntimeError(f"LRZ run: {len(missing)} finished replica(s) not downloaded yet "
                           f"({', '.join(r.name for r in missing[:4])}{', ...' if len(missing) > 4 else ''}); "
                           "download the results first with fetch(run_dir)")
