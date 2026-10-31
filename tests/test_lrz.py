"""Runs on the LRZ Linux cluster (executor="lrz"): parameters, job.slurm and replicas.txt, and the job
script itself, run locally with stand-ins for module and srun."""

import json
import os
import shutil
import subprocess

import pytest

import pandas as pd

from qtft import (Params, build_run, build_sweep, ensemble_stats, fetch, load_thermo, progress, start, stop,
                  summary, watch)
from qtft import remote
from qtft.remote import LRZ, remote_dir

from conftest import influx


def test_lrz_params():
    p = Params(executor="lrz")
    assert p.executor == "lrz" and p.slurm_time == "3-00:00:00" and p.slurm_mem == "4G"
    assert not [m for lvl, m in p.check() if lvl == "ERROR"]
    assert p.auto_name() == Params().auto_name()                       # not part of the run name
    errors = lambda **kw: [m for lvl, m in Params(**kw).check() if lvl == "ERROR"]
    assert errors(executor="slurm")
    assert errors(executor="lrz", slurm_time="8-00:00:00") and errors(executor="lrz", slurm_time="3d")
    assert not errors(executor="lrz", slurm_time="12:00:00")
    assert errors(executor="lrz", slurm_mem="4GB")
    assert not errors(executor="local", slurm_time="3d")                 # only checked for lrz
    warnings = lambda **kw: [m for lvl, m in Params(**kw).check() if lvl == "WARNING" and "LRZ" in m]
    assert warnings(executor="lrz", n_replicas=10, max_parallel=4) == [
        "on LRZ only 4 of 10 replicas run at once (max_parallel); max_parallel=None runs all at once"]
    assert not warnings(executor="lrz", n_replicas=10) and not warnings(executor="local", n_replicas=10, max_parallel=4)
    old = {k: v for k, v in Params().to_dict().items() if k not in ("executor", "slurm_time", "slurm_mem")}
    assert Params.from_dict(old).executor == "local"                     # params.json of older runs
    with pytest.raises(ValueError, match="execution settings"):
        build_sweep(Params(), {"executor": ["local", "lrz"]}, "x")


def test_lrz_mail_from_local_settings(tmp_path, monkeypatch):
    monkeypatch.setitem(LRZ, "mail_user", "")
    common = dict(n_replicas=1, data_dir=str(tmp_path), executor="lrz")
    job = (build_run(Params(run_name="none", **common)) / "job.slurm").read_text()
    assert "#SBATCH --mail-type=NONE\n" in job and "--mail-user" not in job          # no address: no email
    settings = tmp_path / "lrz_local.json"
    settings.write_text(json.dumps({"mail_user": "someone@example.org"}))
    remote.load_local_settings(settings)
    job = (build_run(Params(run_name="mail", **common)) / "job.slurm").read_text()
    assert "#SBATCH --mail-type=BEGIN,END,FAIL\n#SBATCH --mail-user=someone@example.org\n" in job
    settings.write_text(json.dumps({"mail": "x"}))
    with pytest.raises(ValueError, match="unknown LRZ settings"):
        remote.load_local_settings(settings)
    remote.load_local_settings(tmp_path / "missing.json")                     # no file: nothing changes
    assert LRZ["mail_user"] == "someone@example.org"


def test_lrz_job_files(tmp_path):
    common = dict(n_replicas=3, mpi_procs=8, data_dir=str(tmp_path), run_name="r")
    run = build_run(Params(executor="lrz", **common))
    job = (run / "job.slurm").read_text()
    for line in ("#SBATCH --clusters=serial", "#SBATCH --partition=serial_long", "#SBATCH --qos=cm4_serial_long",
                 "#SBATCH --ntasks=8", "#SBATCH --mem=4G", "#SBATCH --time=3-00:00:00", "#SBATCH --array=1-3\n",
                 "#SBATCH --export=NONE", "export SLURM_EXPORT_ENV=ALL", f"module load {LRZ['modules']}",
                 f"srun --mpi=pmi2 {LRZ['lmp']} -in in.lammps"):
        assert line in job, line
    assert job.startswith("#!/bin/bash -l\n") and "{{" not in job
    assert (run / "replicas.txt").read_text() == "rep_01\nrep_02\nrep_03\n"
    assert remote_dir(run, Params(**common)) == "qtft-lammps/runs/ensemble/r"

    # the same input as a local run, except that no plugin is loaded (built into LAMMPS on LRZ)
    local = build_run(Params(**{**common, "data_dir": str(tmp_path / "local")}))
    no_plugin = lambda text: [l for l in text.splitlines() if "plugin load" not in l]
    assert no_plugin((run / "rep_01/in.lammps").read_text()) == no_plugin((local / "rep_01/in.lammps").read_text())
    assert 'if "0 == 1" then "plugin load none"' in (run / "rep_01/in.lammps").read_text()
    assert not (local / "job.slurm").exists()
    assert json.loads((run / "params.json").read_text())["params"]["executor"] == "lrz"

    # a sweep: one job array over the replicas of all runs, replica-major, at most max_parallel at once
    sweep = build_sweep(Params(executor="lrz", n_replicas=2, mpi_procs=4, max_parallel=3,
                               data_dir=str(tmp_path)), {"k_on": [1e-6, 2e-6]}, "s")
    assert "#SBATCH --array=1-4%3\n" in (sweep / "job.slurm").read_text()
    assert (sweep / "replicas.txt").read_text().split() == ["k_on1e-06/rep_01", "k_on2e-06/rep_01",
                                                            "k_on1e-06/rep_02", "k_on2e-06/rep_02"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_job_script_runs_replicas(tmp_path):
    # job.slurm as Slurm would run it, for array tasks 1 and 2; module and srun are stand-ins, and
    # $HOME/src/lammps/build/lmp points to the local lmp. The only test of a replica that fails inside
    # LAMMPS (the fake-LRZ tests below run the same script, all replicas succeed there)
    p = Params(executor="lrz", qt_per_um3=500.0, ft_per_um3=2000.0, box_length=464.2, equil_steps=200, **influx(500),
               output_every=250, traj_every=500, bond_break="longest", n_replicas=2,
               data_dir=str(tmp_path / "data"), run_name="job")
    run = build_run(p)
    text = (run / "rep_02/in.lammps").read_text()                       # rep_02 stops before QTFT_DONE
    assert 'print           "QTFT_DONE"' in text
    (run / "rep_02/in.lammps").write_text(text.replace('print           "QTFT_DONE"', "this_is_not_a_command"))

    bin_dir, home = tmp_path / "bin", tmp_path / "home"
    bin_dir.mkdir()
    (home / "src/lammps/build").mkdir(parents=True)
    (home / "src/lammps/build/lmp").symlink_to(shutil.which("lmp"))
    (bin_dir / "module").write_text("#!/bin/sh\nexit 0\n")
    (bin_dir / "srun").write_text('#!/bin/sh\n[ "$1" = "--mpi=pmi2" ] || exit 9\nshift\nexec "$@"\n')
    for f in ("module", "srun"):
        (bin_dir / f).chmod(0o755)
    env = {**os.environ, "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "HOME": str(home),
           "SLURM_SUBMIT_DIR": str(run), "SLURM_JOB_ID": "4242"}
    for task, (rc, status) in {1: (0, "done"), 2: (1, "failed")}.items():
        result = subprocess.run(["bash", str(run / "job.slurm")], env={**env, "SLURM_ARRAY_TASK_ID": str(task)},
                                capture_output=True, text=True, timeout=300)
        rep = run / f"rep_{task:02d}"
        assert result.returncode == rc, result.stderr
        assert (rep / "status").read_text() == f"{status}\n"
        assert (rep / "pid").read_text() == "4242\n"
    assert "QTFT_DONE" in (run / "rep_01/log.lammps").read_text()


# ------------------------------------------------------------------ start / watch / stop / fetch with a fake LRZ
FAKES = {
    # ssh [options] host command: run the command in the fake LRZ home; FAKE_LRZ_DOWN: no connection
    "ssh": """#!/bin/bash
[ -n "$FAKE_LRZ_DOWN" ] && { echo "Permission denied (keyboard-interactive)." >&2; exit 255; }
while [ $# -gt 0 ]; do case "$1" in -x) shift;; -o) shift 2;; *) break;; esac; done
shift
cd "$FAKE_LRZ" && HOME="$FAKE_LRZ" exec bash -c "$*"
""",
    # rsync -e "ssh ..." lrz:path -> the real rsync on $FAKE_LRZ/path
    "rsync": """#!/bin/bash
args=()
while [ $# -gt 0 ]; do
  case "$1" in -e) shift 2; continue;; lrz:*) args+=("$FAKE_LRZ/${1#lrz:}");; *) args+=("$1");; esac
  shift
done
exec "$REAL_RSYNC" "${args[@]}"
""",
    # sbatch --parsable job.slurm: runs every array task at once (FAKE_SBATCH_HOLD: none, they stay queued)
    "sbatch": """#!/bin/bash
job="${@: -1}"
n=$(sed -n 's/^#SBATCH --array=1-\\([0-9]*\\).*/\\1/p' "$job")
if [ -z "$FAKE_SBATCH_HOLD" ]; then
  for i in $(seq "$n"); do
    SLURM_ARRAY_TASK_ID=$i SLURM_SUBMIT_DIR="$PWD" SLURM_JOB_ID=777 bash "$job" > "slurm-777_$i.out" 2>&1
  done
fi
echo "777;serial"
""",
    "squeue": '#!/bin/bash\ncat "$FAKE_LRZ/.squeue" 2>/dev/null\nexit 0\n',
    "sacct": '#!/bin/bash\ncat "$FAKE_LRZ/.sacct" 2>/dev/null\nexit 0\n',
    "scancel": '#!/bin/bash\necho "$@" >> "$FAKE_LRZ/.scancel"\nrm -f "$FAKE_LRZ/.squeue"\n',
    "module": "#!/bin/sh\nexit 0\n",
    "srun": '#!/bin/sh\n[ "$1" = "--mpi=pmi2" ] || exit 9\nshift\nexec "$@"\n',
}


@pytest.fixture
def fake_lrz(tmp_path, monkeypatch):
    """A fake LRZ home in tmp_path/lrz, reached through stand-ins for ssh, rsync and the Slurm commands;
    its LAMMPS is the local lmp (or a dummy)."""
    home, bin_dir = tmp_path / "lrz", tmp_path / "bin"
    (home / "src/lammps/build").mkdir(parents=True)
    bin_dir.mkdir()
    lmp = home / "src/lammps/build/lmp"
    if shutil.which("lmp"):
        lmp.symlink_to(shutil.which("lmp"))
    else:
        lmp.write_text("#!/bin/sh\nexit 1\n")
        lmp.chmod(0o755)
    for name, text in FAKES.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    real_path = os.pathsep.join(d for d in os.environ["PATH"].split(os.pathsep) if not d.endswith("no_lrz_bin"))
    monkeypatch.setenv("REAL_RSYNC", shutil.which("rsync", path=real_path))
    monkeypatch.setenv("FAKE_LRZ", str(home))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(remote, "_last_sync", {})
    monkeypatch.setattr(remote, "_states", {})
    return home


def _small(tmp_path, **kw):
    return Params(executor="lrz", qt_per_um3=500.0, ft_per_um3=2000.0, box_length=464.2, equil_steps=200, **influx(500),
                  output_every=250, traj_every=500, bond_break="longest", data_dir=str(tmp_path / "data"), **kw)


def test_tests_cannot_reach_lrz(tmp_path):
    # conftest.no_lrz: without the fake LRZ, ssh fails, so start() can never submit a real job
    run = build_run(_small(tmp_path, run_name="guard"))
    with pytest.raises(ConnectionError, match="tests must not reach LRZ"):
        start(run)
    assert (run / "rep_01/status").read_text() == "built\n" and not (run / "slurm_job.json").exists()


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_lrz_start_watch_fetch(tmp_path, fake_lrz):
    run = build_run(_small(tmp_path, n_replicas=2, run_name="e2e"))
    start(run)                                         # copies the folder, sbatch runs both tasks
    there = fake_lrz / "qtft-lammps/runs/ensemble/e2e"
    assert json.loads((run / "slurm_job.json").read_text())["job"] == "777"
    assert (there / "rep_02/traj.dump").exists() and (there / "slurm-777_1.out").exists()
    assert not (run / "rep_01/traj.dump").exists()

    s = summary(run)                                   # fetches status, pid and logs
    assert (s["done"], s["finished"], s["slurm"]["job"]) == (2, True, "777")
    table = progress(run)
    assert list(table["status"]) == ["done", "done"] and (table["step"] == 500).all()
    assert (run / "rep_01/pid").read_text() == "777\n" and not (run / "rep_01/traj.dump").exists()
    for name in ("log.lammps", "lammps.out"):         # logs are fetched by appending: same bytes
        assert (run / "rep_01" / name).read_bytes() == (there / "rep_01" / name).read_bytes()

    # before fetch: analysis stops with a hint; quicklook works from the logs, without overlap
    assert remote.unfetched(run) == [run / "rep_01", run / "rep_02"]
    with pytest.raises(RuntimeError, match=r"fetch\(run_dir\)"):
        ensemble_stats(run)
    import matplotlib
    matplotlib.use("Agg")
    from qtft import quicklook
    texts = [t.get_text() for ax in quicklook(run).axes for t in ax.texts]
    assert texts.count("LRZ run: fetch(run_dir)\nfor the overlap") == 3

    fetch(run)                                         # everything
    assert (run / "rep_01/traj.dump").read_bytes() == (there / "rep_01/traj.dump").read_bytes()
    assert load_thermo(run / "rep_02")["Step"].iloc[-1] == 500
    assert remote.unfetched(run) == [] and ensemble_stats(run)["replicas"] == ["rep_01", "rep_02"]
    index = pd.read_csv(tmp_path / "data/index.csv")
    assert index.loc[index["run"] == "e2e", "executor"].tolist() == ["lrz"]

    # submitted again: the local logs of the first attempt are replaced, not appended to
    before = (there / "rep_01/log.lammps").read_bytes()
    start(run)
    remote._last_sync.clear()
    summary(run)
    assert (run / "rep_01/log.lammps").read_bytes() == (there / "rep_01/log.lammps").read_bytes()
    assert len((run / "rep_01/log.lammps").read_bytes()) == pytest.approx(len(before), rel=0.01)


def test_lrz_watch_connection_lost(tmp_path, fake_lrz, monkeypatch, capsys):
    monkeypatch.setenv("FAKE_SBATCH_HOLD", "1")
    run = build_run(_small(tmp_path, run_name="w"))
    start(run)
    monkeypatch.setenv("FAKE_LRZ_DOWN", "1")
    remote._last_sync.clear()
    watch(run, every=0.1)                              # ends with a notice, no exception
    assert "connection to LRZ lost; the run continues there" in capsys.readouterr().out


def test_lrz_queue_lost_tasks_and_cancel(tmp_path, fake_lrz, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_HOLD", "1")       # submitted, nothing runs
    run = build_run(_small(tmp_path, n_replicas=3, run_name="q"))
    there = fake_lrz / "qtft-lammps/runs/ensemble/q"
    start(run)
    with pytest.raises(RuntimeError, match="already queued"):
        start(run)
    (fake_lrz / ".squeue").write_text("777_1|RUNNING|None\n777_2|PENDING|Resources\n777_3|PENDING|Resources\n")
    s = summary(run)
    assert s["slurm"] == {"job": "777", "running": 1, "pending": 2, "reasons": ["Resources"]} and s["queued"] == 3
    from qtft.run import summary_lines
    assert "LRZ Slurm job 777: 1 running, 2 pending (Resources)" in summary_lines(s)

    # tasks 1 and 2 leave the queue without finishing (time limit, cancelled by hand); task 3 still pending
    (fake_lrz / ".squeue").write_text("777_3|PENDING|Resources\n")
    (fake_lrz / ".sacct").write_text("777_1|TIMEOUT\n777_2|CANCELLED by 1234\n777_3|PENDING\n")
    remote._last_sync.clear()
    assert list(progress(run)["status"]) == ["failed", "stopped", "queued"]
    assert (there / "rep_01/status").read_text() == "failed\n"           # also on LRZ

    stop(run)                                          # cancels the whole job
    assert (fake_lrz / ".scancel").read_text() == "-M serial 777\n"
    assert [(r / "status").read_text() for r in (run / "rep_03", there / "rep_03")] == ["stopped\n"] * 2


def test_lrz_sweep_and_no_connection(tmp_path, fake_lrz, monkeypatch):
    monkeypatch.setenv("FAKE_SBATCH_HOLD", "1")
    sweep = build_sweep(_small(tmp_path, n_replicas=2), {"k_on": [1e-6, 2e-6]}, "s")
    start(sweep)                                       # one job for the whole sweep
    assert (fake_lrz / "qtft-lammps/runs/sweeps/s/job.slurm").exists()
    assert json.loads((sweep / "slurm_job.json").read_text())["remote"] == "qtft-lammps/runs/sweeps/s"
    stop(sweep / "k_on2e-06")                          # one run: only its array tasks (replica-major: 2, 4)
    assert (fake_lrz / ".scancel").read_text() == "-M serial 777_2 777_4\n"
    assert progress(sweep)["status"].tolist() == ["queued", "queued", "stopped", "stopped"]

    monkeypatch.setenv("FAKE_LRZ_DOWN", "1")
    run = build_run(_small(tmp_path, run_name="down"))
    with pytest.raises(ConnectionError, match="ssh lrz"):
        start(run)
    assert (run / "rep_01/status").read_text() == "built\n"            # nothing submitted
