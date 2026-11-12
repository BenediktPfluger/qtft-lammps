"""Export the frame at the end of the calcium influx for the comparison with 2D SEM sections (virtual slicing).

Frame, checks and clusters come from export.py (shared with the FIB-SEM export, fibsem.py; the two exports
are independent of each other). Unlike the FIB-SEM export, the encapsulins (Qt) keep their position in the
simulation box: nothing is unwrapped or shifted, so the arrangement of the clusters relative to each other
is intact, which virtual sections need.
- Periodic box: centres wrapped into [0, L) along every axis; a cluster crossing a periodic face appears
  split into pieces at opposite faces, as in a section through a periodic sample.
- Walls (boundary "reflect" or "harmonic"): centres in the box frame (walls at 0 and L), as dumped.

Nothing SEM-specific is decided here: section thickness, visible depth, Qt diameter for detection and the
field of view are parameters of the slicing in Colab. The export only writes the checked raw state.

Only Qt are exported (ferritin is below SEM resolution). The clusters are the true 3D clusters: connected
components of the bond graph incl. ferritin (Qt-Ft-Qt bridges), numbered by size within a replica
(1 = largest), the same as in the FIB-SEM export. They are not used for the SEM-like 2D clustering, but
allow checking how 2D linking in a section splits or merges real 3D clusters.

Every replica must be finished and pass the checks of export.py (counts, fragments, bond lengths, unwrap,
same frame); in addition the box in traj.dump must equal Params.box with its origin at 0, and every
exported centre must lie in the box. Otherwise ExportError is raised and nothing is written.

Output: <run>/sem/sem_qt_positions.csv, sem_metadata.json (and check_sem.png with plot=True).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .export import ExportError, check_run, counts, frame_text, pool
from .params import Params
from .structure import WALL_TOUCH

CSV_NAME = "sem_qt_positions.csv"
JSON_NAME = "sem_metadata.json"
OUT_SUBDIR = "sem"
BOX_TOLERANCE_NM = 1e-6       # traj.dump box bounds vs Params.box
COLUMNS = ["label", "replica", "particle_id", "x_nm", "y_nm", "z_nm", "radius_nm",
           "cluster", "cluster_in_replica", "cluster_size_qt", "cluster_size_particles",
           "is_clustered", "spans", "percolating", "touches_wall", "wall_gap_nm"]


def _column_doc(periodic: bool) -> dict:
    """Meaning of the CSV columns (for the metadata), worded for a periodic box or for walls."""
    if periodic:
        xyz = "centre in the simulation box, wrapped into [0, L) along every axis (periodic box)"
        spans = "axes along which the cluster wraps around the periodic box ('' = none)"
        perc = "cluster wraps around the periodic box along at least one axis"
    else:
        xyz = "centre in the box frame, walls at 0 and L (as dumped, not shifted)"
        spans = f"axes along which the cluster touches both opposite walls (within {WALL_TOUCH} nm; '' = none)"
        perc = "cluster touches both opposite walls along at least one axis"
    return {
        "label": "running number of the encapsulin in the run (1..n)",
        "replica": "replica folder (rep_01, ...)",
        "particle_id": "LAMMPS atom ID",
        "x_nm, y_nm, z_nm": xyz,
        "radius_nm": "effective Qt radius of the simulation (Params.radius_qt)",
        "cluster": "run-wide ID of the true 3D cluster (bond graph incl. ferritin)",
        "cluster_in_replica": "cluster ID within the replica, numbered by size (1 = largest)",
        "cluster_size_qt": "number of Qt in the cluster",
        "cluster_size_particles": "number of particles in the cluster incl. ferritin",
        "is_clustered": "cluster has more than one particle (a Qt-Ft dimer counts)",
        "spans": spans,
        "percolating": perc,
        "touches_wall": (f"a particle surface (Qt or Ft) of the cluster is within {WALL_TOUCH} nm of a wall "
                         "(always False in a periodic box)"),
        "wall_gap_nm": "smallest surface-wall gap over all particles of the cluster (empty in a periodic box)",
    }


# ------------------------------------------------------------------------------------------------ helpers
def _dump_box(path: Path) -> np.ndarray:
    """Box bounds of the first frame of traj.dump: array (3, 2) of lo, hi for x, y, z."""
    with open(path) as f:
        for line in f:
            if line.startswith("ITEM: BOX BOUNDS"):
                return np.array([[float(v) for v in next(f).split()[:2]] for _ in range(3)])
    raise ExportError(f"{path}: no 'ITEM: BOX BOUNDS' header")


def _replica_table(rep_dir: Path, checked: dict, p: Params) -> tuple[pd.DataFrame, dict]:
    """SEM table of one replica (positions in the box) from export.check_replica, and the wrap summary."""
    name = checked["name"]
    box = np.array(p.box, dtype=float)
    bounds = _dump_box(rep_dir / "traj.dump")
    if np.abs(bounds[:, 0]).max() > BOX_TOLERANCE_NM or np.abs(bounds[:, 1] - box).max() > BOX_TOLERANCE_NM:
        raise ExportError(f"{name}: traj.dump box {bounds.tolist()} differs from Params.box {box.tolist()} "
                          "with origin 0")

    q = checked["qt"]
    raw = q[["x", "y", "z"]].to_numpy(dtype=float)
    if p.periodic:
        pos = np.mod(raw, box)
        pos[pos >= box] -= box[np.nonzero(pos >= box)[1]]         # float edge case: mod can return exactly L
        inside = (pos >= 0) & (pos < box)
    else:
        pos = raw.copy()
        inside = (pos >= 0) & (pos <= box)
    if not inside.all():
        raise ExportError(f"{name}: {int((~inside.all(axis=1)).sum())} encapsulin centres outside the box")
    wrap = {"n_wrapped": int((pos != raw).any(axis=1).sum()),
            "max_outside_nm": float(max(0.0, (-raw).max(), (raw - box).max())) if len(raw) else 0.0}

    n_qt_cluster = q.groupby("cluster").size()
    table = pd.DataFrame({
        "particle_id": q["particle_id"].to_numpy(),
        "x_nm": pos[:, 0], "y_nm": pos[:, 1], "z_nm": pos[:, 2],
        "radius_nm": p.radius_qt,
        "cluster": q["cluster"].to_numpy(),
        "cluster_size_qt": q["cluster"].map(n_qt_cluster).to_numpy(),
        **{c: q[c].to_numpy() for c in ("cluster_size_particles", "is_clustered", "spans", "percolating",
                                        "touches_wall", "wall_gap_nm")},
    })
    return table, wrap


def _check_plot(pooled: pd.DataFrame, box, out: Path, replica: str, periodic: bool) -> None:
    """x-y, x-z and y-z projections of one replica (free Qt grey, clusters coloured, percolating black),
    box or walls outlined."""
    t = pooled[pooled["replica"] == replica]
    free = ~t["is_clustered"]
    perc = t["percolating"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for ax, (a, b) in zip(axs, (("x", "y"), ("x", "z"), ("y", "z"))):
        ax.scatter(t.loc[free, f"{a}_nm"], t.loc[free, f"{b}_nm"], s=1, c="0.75", lw=0)
        c = t[~free & ~perc]
        ax.scatter(c[f"{a}_nm"], c[f"{b}_nm"], s=2, c=c["cluster_in_replica"] % 20, cmap="tab20", lw=0)
        if perc.any():
            ax.scatter(t.loc[perc, f"{a}_nm"], t.loc[perc, f"{b}_nm"], s=2, c="k", lw=0, label="percolating")
        la, lb = box["xyz".index(a)], box["xyz".index(b)]
        ax.add_patch(plt.Rectangle((0, 0), la, lb, fill=False, lw=0.8, ls="--", color="k"))
        ax.set_aspect("equal")
        ax.set_xlabel(f"{a} (nm)")
        ax.set_ylabel(f"{b} (nm)")
    if perc.any():
        axs[0].legend(loc="upper right", fontsize=7, frameon=False)
    frame = "wrapped into the periodic box" if periodic else "box frame, dashed = walls"
    fig.suptitle(f"{replica}: Qt in the box ({frame}; full projections; grey = free)", fontsize=9)
    fig.savefig(out / "check_sem.png", dpi=200)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ public
def sem_export_run(run_dir: str | Path, plot: bool = True) -> Path:
    """Export every replica of a run to <run>/sem/ (Qt positions in the box + metadata). Returns the folder.

    All replicas must be finished and pass all checks; otherwise ExportError is raised and nothing is
    written. Works with a single replica. Independent of the FIB-SEM export (fibsem.py)."""
    run_dir = Path(run_dir).expanduser()
    p, rep_dirs, checked = check_run(run_dir)                      # all checks before anything is written
    tables, wraps = [], {}
    for rep_dir, c in zip(rep_dirs, checked):
        table, wraps[c["name"]] = _replica_table(rep_dir, c, p)
        tables.append((c["name"], table))
    pooled = pool(tables)[COLUMNS]

    box = [float(v) for v in p.box]
    volume_um3 = math.prod(box) * 1e-9
    step = checked[0]["step"]
    if p.periodic:
        coordinates = ("Qt centres in the simulation box, wrapped into [0, L) along every axis; not unwrapped, "
                       "not shifted: clusters crossing a periodic face appear split at opposite faces")
    else:
        coordinates = "Qt centres in the box frame (walls at 0 and L), as dumped, not shifted"
    meta = {
        "source_run": f"{run_dir.parent.name}/{run_dir.name}",
        "run_name": run_dir.name,
        "n_replicas": len(checked),
        "replicas": [c["name"] for c in checked],
        "source_trajectories": [c["source"] for c in checked],
        "frame": frame_text(p),
        "step": step, "time_s": step * p.timestep * 1e-9,           # production step and time since its start
        "influx_end_step": int(p.influx_end_step),
        "influx_end_time_s": p.influx_end_step * p.timestep * 1e-9,
        "frame_gap_steps": int(p.influx_end_step - step),
        "frame_gap_s": (p.influx_end_step - step) * p.timestep * 1e-9,
        "boundary": p.boundary,
        "periodic": p.periodic,
        "wall_touch_nm": None if p.periodic else WALL_TOUCH,
        "box_size_xyz_nm": box,
        "box_origin_xyz_nm": [0.0, 0.0, 0.0],
        "box_volume_um3": volume_um3,
        "coordinates": coordinates,
        "qt_radius_nm": p.radius_qt,
        "qt_diameter_nm": 2 * p.radius_qt,
        "n_qt_per_replica": int(p.n_qt),
        "qt_per_um3": p.n_qt / volume_um3,
        **counts(pooled),
        "columns": _column_doc(p.periodic),
        "checks": {
            "export_checks": "counts, fragment, bond and unwrap checks of export.check_replica passed for every replica",
            "box": "traj.dump box bounds equal Params.box with origin 0",
            "wrap": wraps,
            "bond_check": {c["name"]: c["bond_check"] for c in checked},
            "unwrap_check": {c["name"]: c["unwrap_check"] for c in checked},
            "fragment_check": {c["name"]: c["fragment_check"] for c in checked},
        },
        "per_replica": {name: counts(pooled[pooled["replica"] == name]) for name in wraps},
        "parameters": p.to_dict(),
    }

    out = run_dir / OUT_SUBDIR
    out.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(out / CSV_NAME, index=False)
    (out / JSON_NAME).write_text(json.dumps(meta, indent=2))
    if plot:
        _check_plot(pooled, box, out, checked[0]["name"], p.periodic)
    return out
