"""Export the frame at the end of the calcium influx in the schema of the FIB-SEM segmentation pipeline.

Frame, checks and clusters come from export.py (shared with the SEM export, sem.py; the two exports are
independent of each other): the last traj.dump frame at or before Params.influx_end_step, clusters from
the bond graph numbered by size within a replica, every replica checked before anything is written.

Only encapsulins (Qt) are exported; ferritin is below FIB-SEM resolution but still defines the cluster
connectivity (Qt-Ft-Qt bridges). A Qt in a component with more than one particle is clustered (a Qt-Ft
dimer already counts); a free Qt is a singleton. Clusters spanning the box (periodic: wrapping around it,
walls: touching both opposite walls) are flagged `percolating`; the analysis excludes them from the
geometry metrics.

Periodic box: clusters unwrapped along the bonds (export.check_replica), and every replica shifted so that
the lower corner of the bounding box of its encapsulin spheres (non-percolating clusters) is at the origin.
This breaks periodicity *between* clusters, which is irrelevant for the within-cluster analysis (the SEM
export, sem.py, keeps the box positions instead).

Walls (boundary "reflect" or "harmonic"): positions in the box frame, not shifted (walls at 0 and L).
Both wall types act on the particle surface (reflect: the plane at r_i from the face; harmonic: cutoff
r_i), so a cluster touches a wall when a particle surface (Qt or Ft) is within structure.WALL_TOUCH of a
face. Harmonic walls let a sphere stick out by a fraction of a nm (`wall_leaks`).

fibsem_export_run exports every replica of a run and pools them in <run>/fibsem/ (replica column,
run-wide cluster IDs); see export.py for the checks.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .export import ExportError, check_replica, check_run, counts as _counts, frame_text as _frame_text, pool
from .params import Params
from .structure import WALL_TOUCH

CSV_NAME = "sim_encapsulin_centroids_clustered.csv"
JSON_NAME = "sim_metadata.json"
OUT_SUBDIR = "fibsem"

__all__ = ["ExportError", "fibsem_export", "fibsem_export_run"]


# ------------------------------------------------------------------------------------------------ one replica
def _replica(checked: dict, p: Params, voxel_nm: float) -> dict:
    """FIB-SEM table of one replica from export.check_replica: unwrapped positions, shifted (periodic) or
    in the box frame (walls), voxel coordinates, wall leaks."""
    q = checked["qt"]
    xyz = q[["xu", "yu", "zu"]].to_numpy(dtype=float)
    perc = q["percolating"].to_numpy()
    box = np.array(p.box)
    if p.periodic:
        ref = xyz[~perc] if (~perc).any() else xyz
        lower = (ref - p.radius_qt).min(axis=0)                     # bounding box of the spheres
    else:
        lower = np.zeros(3)                                         # box frame: walls at 0 and L
    xyz = xyz - lower
    volume = 4 / 3 * math.pi * p.radius_qt ** 3
    table = pd.DataFrame({
        "label": np.arange(1, len(q) + 1),
        "particle_id": q["particle_id"].to_numpy(),
        "z_nm": xyz[:, 2], "y_nm": xyz[:, 1], "x_nm": xyz[:, 0],
        "z_vox": xyz[:, 2] / voxel_nm, "y_vox": xyz[:, 1] / voxel_nm, "x_vox": xyz[:, 0] / voxel_nm,
        "radius_nm": p.radius_qt, "volume_nm3": volume,
        **{c: q[c].to_numpy() for c in ("cluster", "is_clustered", "cluster_size_particles", "spans",
                                        "percolating", "touches_wall", "wall_gap_nm")},
    })

    leaks = None
    if not p.periodic:
        depth = p.radius_qt - np.minimum(xyz, box - xyz).min(axis=1)        # sphere beyond a face
        leaks = {"n_encapsulins_beyond_wall": int((depth > 0).sum()),
                 "max_depth_nm": float(max(depth.max(), 0.0)) if len(depth) else 0.0}
    return {**{k: checked[k] for k in ("name", "step", "bond_check", "unwrap_check", "fragment_check", "source")},
            "table": table, "lower": lower, "leaks": leaks}



# ------------------------------------------------------------------------------------------------ metadata
def _metadata(p: Params, reps: list[dict], table: pd.DataFrame, run_dir: Path, voxel_nm: float) -> dict:
    """Metadata of one replica (reps of length 1) or of the pooled run."""
    step = reps[0]["step"]
    box = [float(v) for v in p.box]
    walls = None if p.periodic else {"min": [0.0, 0.0, 0.0], "max": box[::-1]}
    if p.periodic:
        description = ("periodic box: clusters unwrapped along the bonds (minimum image), every replica "
                       "shifted so that the lower corner of the bounding box of its encapsulin spheres "
                       "(non-percolating clusters) is at the origin; periodicity between clusters is broken")
    else:
        wall = {"reflect": "reflecting walls (fix wall/reflect)", "harmonic": "soft harmonic walls (fix wall/harmonic)"}
        description = (f"box with {wall[p.boundary]} on all faces, acting on the particle surface: positions "
                       "in the box frame, not shifted (walls at 0 and the box size); particles are depleted "
                       "next to the walls")
    if len(reps) == 1 or p.periodic:
        leaks = reps[0]["leaks"] if len(reps) == 1 else None
    else:
        leaks = {"n_encapsulins_beyond_wall": sum(r["leaks"]["n_encapsulins_beyond_wall"] for r in reps),
                 "max_depth_nm": max(r["leaks"]["max_depth_nm"] for r in reps)}
    sum_check = lambda key, fields: {f: sum(r[key][f] for r in reps) for f in fields}
    volume = 4 / 3 * math.pi * p.radius_qt ** 3
    meta = {
        "source_run": f"{run_dir.parent.name}/{run_dir.name}",
        "run_name": run_dir.name,
        "n_replicas": len(reps),
        "replicas": [r["name"] for r in reps],
        "source_trajectories": [r["source"] for r in reps],
        "frame": _frame_text(p),
        "step": step, "time_s": step * p.timestep * 1e-9,           # production step and time since its start
        "influx_end_step": int(p.influx_end_step),
        "influx_end_time_s": p.influx_end_step * p.timestep * 1e-9,
        "frame_gap_steps": int(p.influx_end_step - step),
        "frame_gap_s": (p.influx_end_step - step) * p.timestep * 1e-9,
        **_counts(table),
        "qt_radius_nm": p.radius_qt, "qt_volume_nm3": volume,
        "boundary": p.boundary,
        "periodic": p.periodic,
        "wall_touch_nm": WALL_TOUCH,
        "wall_stiffness": ({"Qt": p.wall_k_qt, "Ft": p.wall_k_ft, "unit": "zJ/nm^2",
                            "energy": "k (r - d)^2, d = centre-face distance < r"}
                           if p.boundary == "harmonic" else None),
        "box_size_xyz_nm": box, "box_size_zyx_nm": box[::-1],
        "box_bounds_zyx_nm": walls,
        "frame_description": description,
        "wall_leaks": leaks,
        "voxel_size_nm": voxel_nm,
        "unwrap_method": "structure.unwrap_clusters: bond graph (traj.dump id <-> bonds.dump atom IDs), "
                         "minimum image along a breadth-first spanning tree",
        "bond_check": {**sum_check("bond_check", ("n_bonds", "n_too_long")),
                       "tolerance": reps[0]["bond_check"]["tolerance"],
                       "max_length_ratio": max((r["bond_check"]["max_length_ratio"] or 0.0) for r in reps)},
        "unwrap_check": sum_check("unwrap_check", ("n_bonds_checked", "n_bonds_skipped_percolating", "n_mismatch")),
        "fragment_check": {"match": all(r["fragment_check"]["match"] for r in reps),
                           **sum_check("fragment_check", ("n_components", "n_lammps_fragments"))},
        "per_replica": {r["name"]: {"offset_applied_xyz_nm": (0.0 - r["lower"]).tolist(),   # 0.0 - x avoids -0.0
                                    **_counts(r["table"]), "wall_leaks": r["leaks"],
                                    "bond_check": r["bond_check"], "unwrap_check": r["unwrap_check"],
                                    "fragment_check": r["fragment_check"]} for r in reps},
        "parameters": p.to_dict(),
    }
    return meta


# ------------------------------------------------------------------------------------------------ writing
def _write(out: Path, table: pd.DataFrame, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / CSV_NAME, index=False)
    (out / JSON_NAME).write_text(json.dumps(meta, indent=2))


def _check_plot(table: pd.DataFrame, lower: np.ndarray, box, out: Path, periodic: bool = True) -> None:
    """x-y scatter of the encapsulins coloured by cluster (free ones grey, percolating ones black), with
    the simulation box (periodic) or the walls."""
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    free = ~table["is_clustered"]
    perc = table["percolating"]
    ax.scatter(table.loc[free, "x_nm"], table.loc[free, "y_nm"], s=2, c="0.75", lw=0)
    clustered = table[~free & ~perc]
    ax.scatter(clustered["x_nm"], clustered["y_nm"], s=3, c=clustered["cluster"] % 20, cmap="tab20", lw=0)
    if perc.any():
        ax.scatter(table.loc[perc, "x_nm"], table.loc[perc, "y_nm"], s=3, c="k", lw=0, label="percolating")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
    ax.add_patch(plt.Rectangle((-lower[0], -lower[1]), box[0], box[1], fill=False, lw=0.8, ls="--", color="k"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title(f"encapsulins by cluster (dashed: {'periodic box' if periodic else 'walls'})", fontsize=8)
    fig.savefig(out / "check_xy.png", dpi=200)
    plt.close(fig)


def _write_replica(rep_dir: Path, rep: dict, p: Params, voxel_nm: float, plot: bool) -> Path:
    out = rep_dir / OUT_SUBDIR
    _write(out, rep["table"], _metadata(p, [rep], rep["table"], rep_dir.parent, voxel_nm))
    if plot:
        _check_plot(rep["table"], rep["lower"], p.box, out, p.periodic)
    return out


# ------------------------------------------------------------------------------------------------ public
def fibsem_export(rep_dir: str | Path, voxel_nm: float = 4.0, plot: bool = True) -> Path:
    """Export one replica to <rep>/fibsem/ (CSV, sim_metadata.json, check_xy.png). For checking a single
    replica; does not require it to be finished. The analysis uses the pooled run export
    (fibsem_export_run)."""
    rep_dir = Path(rep_dir).expanduser()
    p = Params.load(rep_dir.parent / "params.json")
    rep = _replica(check_replica(rep_dir, p), p, voxel_nm)
    return _write_replica(rep_dir, rep, p, voxel_nm, plot)


def fibsem_export_run(run_dir: str | Path, voxel_nm: float = 4.0, plot: bool = True) -> Path:
    """Export every replica of a run (<rep>/fibsem/) and pool them in <run>/fibsem/: the encapsulin table
    with a `replica` column, `cluster_in_replica` and run-wide `cluster` / `label` IDs, and the run
    metadata (pooled counts and checks, per-replica summaries). Returns <run>/fibsem/.

    All replicas must be finished and pass all checks (export.py); otherwise ExportError is raised and
    nothing is written. Independent of the SEM export (sem.py)."""
    run_dir = Path(run_dir).expanduser()
    p, rep_dirs, checked = check_run(run_dir)                      # all checks before anything is written
    reps = [_replica(c, p, voxel_nm) for c in checked]

    pooled = pool([(r["name"], r["table"].rename(columns={"label": "label_in_replica"})) for r in reps])
    cols = ["label", "replica", "label_in_replica", "particle_id", "z_nm", "y_nm", "x_nm",
            "z_vox", "y_vox", "x_vox", "radius_nm", "volume_nm3", "cluster", "cluster_in_replica",
            "is_clustered", "cluster_size_particles", "spans", "percolating", "touches_wall", "wall_gap_nm"]
    pooled = pooled[cols]

    for rep_dir, r in zip(rep_dirs, reps):
        _write_replica(rep_dir, r, p, voxel_nm, plot)
    out = run_dir / OUT_SUBDIR
    _write(out, pooled, _metadata(p, reps, pooled, run_dir, voxel_nm))
    return out
