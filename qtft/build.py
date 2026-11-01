"""Fill template.lmp with the parameters and create the run folder."""

from __future__ import annotations

import datetime
import functools
import json
import math
import re
import shlex
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .ensemble import update_index
from .params import EXECUTION_SETTINGS, REPO_DIR, Params
from .remote import LRZ

TEMPLATE = REPO_DIR / "template.lmp"
JOB_TEMPLATE = REPO_DIR / "job_template.slurm"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
PLUGIN = REPO_DIR / "lammps_plugins" / "bond_break_random" / "bondbreakrandom.so"
SEED_NAMES = ("seed_place_qt", "seed_place_ft", "seed_bd_qt", "seed_bd_ft", "seed_bind", "seed_break")


def replica_seeds(master_seed: int, n_replicas: int) -> list[dict]:
    """Independent positive seeds for every random stream of every replica."""
    children = np.random.SeedSequence(master_seed).spawn(n_replicas)
    seeds = []
    for child in children:
        values = child.generate_state(len(SEED_NAMES), dtype=np.uint32) % 900_000_000 + 1   # RanMars accepts 1..900000000
        seeds.append(dict(zip(SEED_NAMES, (int(v) for v in values))))
    return seeds


@functools.lru_cache
def _lmp_help(lmp_command: str) -> str:
    try:
        return subprocess.run(shlex.split(lmp_command) + ["-h"], capture_output=True, text=True,
                              timeout=120).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def plugin_to_load(p: Params) -> Path | None:
    """The plugin with fix bond/break/random if the run needs it and lmp does not have the style built in
    (build it with lammps_plugins/bond_break_random/build.sh). LAMMPS on LRZ has the style built in."""
    if p.bond_break != "random" or p.executor == "lrz" or "bond/break/random" in _lmp_help(p.lmp_command):
        return None
    return PLUGIN


def template_values(p: Params, replica: int, seeds: dict) -> dict:
    small, medium, large = p.size_bounds()
    plugin = plugin_to_load(p)
    return {
        "run_name": p.name,
        "replica": replica,
        "box_x": p.box[0], "box_y": p.box[1], "box_z": p.box[2],
        "boundary": p.boundary, "boundary_flags": "p p p" if p.periodic else "f f f",
        "qt_lo": p.radius_qt, "ft_lo": p.radius_ft,
        **{f"qt_hi_{a}": L - p.radius_qt for a, L in zip("xyz", p.box)},
        **{f"ft_hi_{a}": L - p.radius_ft for a, L in zip("xyz", p.box)},
        "insert_region_qt": "NULL" if p.periodic else "ins_qt",
        "insert_region_ft": "NULL" if p.periodic else "ins_ft",
        "wall_k_qt": p.wall_k_qt, "wall_k_ft": p.wall_k_ft,
        "extra_bond": max(p.maxbond_qt, p.maxbond_ft),
        "extra_special": p.extra_special,
        "k_qq": p.k_qq, "cut_qq": p.cut_qq,
        "k_ff": p.k_ff, "cut_ff": p.cut_ff,
        "k_qf": p.k_qf, "cut_qf": p.cut_qf,
        "bond_k": p.bond_k, "bond_r0": p.r0,
        "neighbor_skin": p.neighbor_skin,
        "comm_cutoff": p.comm_cut,
        "n_qt": p.n_qt, "n_ft": p.n_ft,
        "insert_maxtry": p.insert_maxtry,
        "timestep": p.timestep,
        "temperature": p.temperature,
        "gamma_qt": p.gamma_qt, "gamma_ft": p.gamma_ft,
        "radius_qt": p.radius_qt, "radius_ft": p.radius_ft,
        "large_cluster_min": p.large_cluster_min,
        "size_small": small, "size_medium": medium, "size_large": large,
        "output_every": p.output_every,
        "traj_every": p.traj_every,
        "equil_steps": p.equil_steps,
        "dt_s": p.timestep * 1e-9,
        "on_step": p.phase_steps[0], "off_step": p.phase_steps[0] + p.phase_steps[1],
        "ca_rest": p.ca_rest_uM, "ca_peak": p.ca_peak_uM, "ca_start": p.ca_start_uM, "ca_tau": p.ca_tau_s,
        # rise during the influx: ca_peak - amplitude * exp(-rate * (t - t_on)); a step has amplitude and rate 0
        "ca_rise_amp": p.ca_peak_uM - p.ca_rest_uM if p.ca_rise_s > 0 else 0.0,
        "ca_rate_on": p.ca_rate_on_per_s if p.ca_rise_s > 0 else 0.0,
        "ca_K": p.ca_K_uM, "ca_hill": p.ca_hill,
        "k_on_per_s": p.k_on * 1e9, "k_off_ca": p.k_off_ca_per_s, "k_off_free": p.k_off_free_per_s,
        "break_every": p.break_every, "bond_break": p.bond_break,
        "load_plugin": int(plugin is not None), "plugin_path": plugin or "none",
        "n_segments": p.prod_steps // p.rate_update_every, "segment_steps": p.rate_update_every,
        "rdf_bins": p.rdf_bins, "rdf_cutoff": p.rdf_cut,
        "hist_qt_max": p.maxbond_qt + 0.5, "hist_qt_bins": p.maxbond_qt + 1,
        "hist_ft_max": p.maxbond_ft + 0.5, "hist_ft_bins": p.maxbond_ft + 1,
        "bind_every": p.bind_every,
        "r_bind": p.r_bind,
        "maxbond_qt": p.maxbond_qt, "maxbond_ft": p.maxbond_ft,
        **seeds,
    }


def render(values: dict, template: str | None = None) -> str:
    text = template if template is not None else TEMPLATE.read_text()

    def substitute(match: re.Match) -> str:
        key = match.group(1)
        if key not in values:
            raise KeyError(f"template placeholder '{{{{{key}}}}}' has no value")
        return _fmt(values[key])

    return PLACEHOLDER.sub(substitute, text)


def git_info() -> dict:
    def git(*args):
        try:
            return subprocess.run(["git", *args], cwd=REPO_DIR, capture_output=True,
                                  text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    return {"commit": git("rev-parse", "HEAD"),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}


def _same(a, b) -> bool:
    """Equal parameter values: numbers within 1e-12 relative (20 == 20.0; numpy values as Python numbers),
    lists elementwise, anything else exactly."""
    a, b = (v.tolist() if isinstance(v, (np.generic, np.ndarray)) else v for v in (a, b))
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    numbers = (int, float)
    if isinstance(a, numbers) and isinstance(b, numbers) and not isinstance(a, bool) and not isinstance(b, bool):
        return math.isclose(a, b, rel_tol=1e-12, abs_tol=0.0)
    return a == b


def _check_replace(run_dir: Path, p: Params) -> None:
    """overwrite=True replaces a run folder only if none of its replicas is queued or running and it holds the
    same physical parameters (execution settings such as n_replicas or the seed may differ): the run name
    leaves out some parameters, so another run with different physics can have the same folder."""
    busy = [rep.name for rep in sorted(run_dir.glob("rep_*"))
            if (rep / "status").exists() and (rep / "status").read_text().strip() in ("queued", "running")]
    if busy:
        hint = ("the local status files of an LRZ run may be out of date: update them with summary(run_dir) or "
                "fetch(run_dir), or stop(run_dir) if it still runs" if (run_dir / "job.slurm").exists()
                else "stop the run first")
        raise RuntimeError(f"{run_dir}: replicas {busy} are queued or running; {hint}")
    record = run_dir / "params.json"
    old = json.loads(record.read_text()).get("params", {}) if record.exists() else {}
    new = p.to_dict()
    differing = [k for k in new if k not in EXECUTION_SETTINGS and not (k in old and _same(old[k], new[k]))]
    if differing:
        raise FileExistsError(f"{run_dir} holds a run with other parameters ({', '.join(differing)}); "
                              "to replace it, delete the folder by hand, or give the new run its own run_name")


def build_run(p: Params, overwrite: bool = False) -> Path:
    """Create <data_dir>/<single|ensemble>/<run name>/ with params.json and rep_XX/in.lammps for every replica.
    overwrite=True replaces an existing folder of the same parameters that is not running (_check_replace)."""
    p.raise_on_error()
    run_dir = p.run_dir
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; set a different run_name or use overwrite=True")
        _check_replace(run_dir, p)
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    seeds = replica_seeds(p.master_seed, p.n_replicas)
    for i, s in enumerate(seeds, start=1):
        rep = run_dir / f"rep_{i:02d}"
        rep.mkdir()
        (rep / "in.lammps").write_text(render(template_values(p, i, s)))
        (rep / "status").write_text("built\n")
    if p.executor == "lrz":
        write_job(run_dir, p, sorted(run_dir.glob("rep_*")))

    record = {
        "params": p.to_dict(),
        "derived": p.derived(),
        "checks": p.check(),
        "seeds": seeds,
        "git": git_info(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (run_dir / "params.json").write_text(json.dumps(record, indent=2, default=str))
    update_index(p.data_dir)
    return run_dir


def write_job(target: Path, p: Params, reps: list[Path]) -> Path:
    """job.slurm (from job_template.slurm) and replicas.txt for a run or sweep folder on LRZ: one task
    of a Slurm job array per replica, at most max_parallel at once."""
    (target / "replicas.txt").write_text("".join(f"{r.relative_to(target).as_posix()}\n" for r in reps))
    values = {**LRZ, "job_name": target.name, "mpi_procs": p.mpi_procs, "slurm_mem": p.slurm_mem,
              "slurm_time": p.slurm_time, "n_tasks": len(reps),
              "array_limit": f"%{p.max_parallel}" if p.max_parallel else "",
              "mail": (f"#SBATCH --mail-type={LRZ['mail_type']}\n#SBATCH --mail-user={LRZ['mail_user']}"
                       if LRZ["mail_user"] else "#SBATCH --mail-type=NONE")}       # address from lrz_local.json
    job = target / "job.slurm"
    job.write_text(render(values, JOB_TEMPLATE.read_text()))
    return job


def _fmt(val) -> str:
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.10g}"
    return str(val)
