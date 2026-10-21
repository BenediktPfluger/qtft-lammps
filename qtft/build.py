"""Fill template.lmp with the parameters and create the run folder."""

from __future__ import annotations

import datetime
import json
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np

from .ensemble import update_index
from .params import REPO_DIR, Params

TEMPLATE = REPO_DIR / "template.lmp"
PLACEHOLDER = re.compile(r"\{\{\s*(\w+)\s*\}\}")
SEED_NAMES = ("seed_place_qt", "seed_place_ft", "seed_bd_qt", "seed_bd_ft", "seed_bind", "seed_break")


def replica_seeds(master_seed: int, n_replicas: int) -> list[dict]:
    """Independent positive seeds for every random stream of every replica."""
    children = np.random.SeedSequence(master_seed).spawn(n_replicas)
    seeds = []
    for child in children:
        values = child.generate_state(len(SEED_NAMES), dtype=np.uint32) % 900_000_000 + 1   # RanMars accepts 1..900000000
        seeds.append(dict(zip(SEED_NAMES, (int(v) for v in values))))
    return seeds


def template_values(p: Params, replica: int, seeds: dict) -> dict:
    small, medium, large = p.size_bounds()
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
        "t_on": p.t_on_s, "t_off": p.t_off_s,
        "ca_rest": p.ca_rest_uM, "ca_peak": p.ca_peak_uM, "ca_start": p.ca_start_uM, "ca_tau": p.ca_tau_s,
        "ca_K": p.ca_K_uM, "ca_hill": p.ca_hill,
        "k_on_per_s": p.k_on * 1e9, "k_off_ca": p.k_off_ca_per_s, "k_off_free": p.k_off_free_per_s,
        "break_every": p.break_every,
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


def build_run(p: Params, overwrite: bool = False) -> Path:
    """Create <data_dir>/<single|ensemble>/<run name>/ with params.json and rep_XX/in.lammps for every replica."""
    p.raise_on_error()
    run_dir = p.run_dir
    if run_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{run_dir} exists; set a different run_name or use overwrite=True")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    seeds = replica_seeds(p.master_seed, p.n_replicas)
    for i, s in enumerate(seeds, start=1):
        rep = run_dir / f"rep_{i:02d}"
        rep.mkdir()
        (rep / "in.lammps").write_text(render(template_values(p, i, s)))
        (rep / "status").write_text("built\n")

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


def _fmt(val) -> str:
    if isinstance(val, bool):
        return str(int(val))
    if isinstance(val, float):
        return f"{val:.10g}"
    return str(val)
