"""Export the frame at the end of the calcium influx for the comparison with 2D SEM sections (virtual slicing).

Frame, checks and clusters come from export.py (shared with the FIB-SEM export, fibsem.py; the two exports
are independent of each other). Unlike the FIB-SEM export, the encapsulins (Qt) keep their position in the
simulation box: nothing is unwrapped or shifted, so the arrangement of the clusters relative to each other
is intact, which virtual sections need.
- Periodic box: centres wrapped into [0, L) along every axis; a cluster crossing a periodic face appears
  split into pieces at opposite faces, as in a section through a periodic sample.
- Walls (boundary "reflect" or "harmonic"): centres in the box frame (walls at 0 and L), as dumped.
Positions are rounded to POSITION_DECIMALS (1e-6 nm) and then wrapped, so no written value reads back as L.

Nothing SEM-specific is decided here: section thickness, visible depth, Qt diameter for detection and the
field of view are parameters of the slicing in Colab. The export only writes the checked raw state.

Only Qt are exported (ferritin is below SEM resolution). The clusters are the true 3D clusters: connected
components of the bond graph incl. ferritin (Qt-Ft-Qt bridges), numbered by size within a replica
(1 = largest), the same as in the FIB-SEM export. They are not used for the SEM-like 2D clustering, but
allow checking how 2D linking in a section splits or merges real 3D clusters.

Every replica must pass the checks of export.py (counts, fragments, bond lengths, unwrap); in addition the
box in traj.dump must equal Params.box with its origin at 0, and every exported centre must lie in the box.
sem_export_run additionally needs every replica finished and all at the same frame. Otherwise ExportError
is raised and nothing is written.

Output (like the FIB-SEM export): sem_qt_positions.csv, sem_metadata.json and check_sem.png (plot=True)
in every <run>/rep_XX/sem/; all replicas pooled in <run>/sem/ (CSV and metadata).
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .export import ExportError, check_replica, check_run, counts, frame_text, pool
from .params import Params
from .structure import WALL_TOUCH

CSV_NAME = "sem_qt_positions.csv"
JSON_NAME = "sem_metadata.json"
OUT_SUBDIR = "sem"
BOX_TOLERANCE_NM = 1e-6       # traj.dump box bounds vs Params.box
POSITION_DECIMALS = 6         # positions rounded to 1e-6 nm before the final wrap
REPLICA_COLUMNS = ["label", "particle_id", "x_nm", "y_nm", "z_nm", "radius_nm", "cluster", "cluster_size_qt",
                   "cluster_size_particles", "is_clustered", "spans", "percolating", "touches_wall", "wall_gap_nm"]
POOLED_COLUMNS = ["label", "replica", "label_in_replica", "particle_id", "x_nm", "y_nm", "z_nm", "radius_nm",
                  "cluster", "cluster_in_replica", "cluster_size_qt", "cluster_size_particles", "is_clustered",
                  "spans", "percolating", "touches_wall", "wall_gap_nm"]


def _column_doc(periodic: bool, pooled: bool) -> dict:
    """Meaning of the CSV columns (for the metadata), worded for a periodic box or for walls."""
    if periodic:
        xyz = "centre in the simulation box, wrapped into [0, L) along every axis (periodic box)"
        spans = "axes along which the cluster wraps around the periodic box ('' = none)"
        perc = "cluster wraps around the periodic box along at least one axis"
    else:
        xyz = "centre in the box frame, walls at 0 and L (as dumped, not shifted)"
        spans = f"axes along which the cluster touches both opposite walls (within {WALL_TOUCH} nm; '' = none)"
        perc = "cluster touches both opposite walls along at least one axis"
    doc = {
        "label": "running number of the encapsulin (1..n)" + (" in the run" if pooled else " in the replica"),
        "particle_id": "LAMMPS atom ID",
        "x_nm, y_nm, z_nm": xyz + f", rounded to 1e-{POSITION_DECIMALS} nm",
        "radius_nm": "effective Qt radius of the simulation (Params.radius_qt)",
        "cluster": ("run-wide ID of the true 3D cluster (bond graph incl. ferritin)" if pooled else
                    "ID of the true 3D cluster (bond graph incl. ferritin), numbered by size (1 = largest)"),
        "cluster_size_qt": "number of Qt in the cluster",
        "cluster_size_particles": "number of particles in the cluster incl. ferritin",
        "is_clustered": "cluster has more than one particle (a Qt-Ft dimer counts)",
        "spans": spans,
        "percolating": perc,
        "touches_wall": (f"a particle surface (Qt or Ft) of the cluster is within {WALL_TOUCH} nm of a wall "
                         "(always False in a periodic box)"),
        "wall_gap_nm": "smallest surface-wall gap over all particles of the cluster (empty in a periodic box)",
    }
    if pooled:
        doc |= {"replica": "replica folder (rep_01, ...)",
                "label_in_replica": "label in the replica's own export (rep_XX/sem/)",
                "cluster_in_replica": "cluster ID within the replica, numbered by size (1 = largest)"}
    return doc


# ------------------------------------------------------------------------------------------------ one replica
def _dump_box(path: Path) -> np.ndarray:
    """Box bounds of the first frame of traj.dump: array (3, 2) of lo, hi for x, y, z."""
    with open(path) as f:
        for line in f:
            if line.startswith("ITEM: BOX BOUNDS"):
                return np.array([[float(v) for v in next(f).split()[:2]] for _ in range(3)])
    raise ExportError(f"{path}: no 'ITEM: BOX BOUNDS' header")


def _box_positions(raw: np.ndarray, box: np.ndarray, periodic: bool) -> np.ndarray:
    """Centres in the box: rounded to POSITION_DECIMALS, wrapped into [0, L) (periodic) or as dumped (walls).
    Rounding comes before the final wrap, so a centre a hair below L ends at 0, never at L."""
    pos = np.round(np.mod(raw, box) if periodic else raw, POSITION_DECIMALS)
    if periodic:
        at_l = pos >= box
        pos[at_l] -= box[np.nonzero(at_l)[1]]
    return pos + 0.0                                                # -0.0 -> 0.0


def _replica(rep_dir: Path, checked: dict, p: Params) -> dict:
    """SEM table of one replica (positions in the box) from export.check_replica, with the box check and
    the wrap summary. Raises ExportError if the dump box differs from Params.box or a centre is outside."""
    name = checked["name"]
    box = np.array(p.box, dtype=float)
    bounds = _dump_box(rep_dir / "traj.dump")
    if np.abs(bounds[:, 0]).max() > BOX_TOLERANCE_NM or np.abs(bounds[:, 1] - box).max() > BOX_TOLERANCE_NM:
        raise ExportError(f"{name}: traj.dump box {bounds.tolist()} differs from Params.box {box.tolist()} "
                          "with origin 0")

    q = checked["qt"]
    raw = q[["x", "y", "z"]].to_numpy(dtype=float)
    pos = _box_positions(raw, box, p.periodic)
    inside = (pos >= 0) & ((pos < box) if p.periodic else (pos <= box))
    if not inside.all():
        raise ExportError(f"{name}: {int((~inside.all(axis=1)).sum())} encapsulin centres outside the box")
    outside = (raw < 0) | (raw >= box)
    wrap = {"n_wrapped": int(outside.any(axis=1).sum()),
            "max_outside_nm": float(max(0.0, (-raw).max(), (raw - box).max())) if len(raw) else 0.0}

    n_qt_cluster = q.groupby("cluster").size()
    table = pd.DataFrame({
        "label": np.arange(1, len(q) + 1),
        "particle_id": q["particle_id"].to_numpy(),
        "x_nm": pos[:, 0], "y_nm": pos[:, 1], "z_nm": pos[:, 2],
        "radius_nm": p.radius_qt,
        "cluster": q["cluster"].to_numpy(),
        "cluster_size_qt": q["cluster"].map(n_qt_cluster).to_numpy(),
        **{c: q[c].to_numpy() for c in ("cluster_size_particles", "is_clustered", "spans", "percolating",
                                        "touches_wall", "wall_gap_nm")},
    })[REPLICA_COLUMNS]
    return {**{k: checked[k] for k in ("name", "step", "bond_check", "unwrap_check", "fragment_check", "source")},
            "table": table, "wrap": wrap}


# ------------------------------------------------------------------------------------------------ metadata
def _metadata(p: Params, reps: list[dict], table: pd.DataFrame, run_dir: Path) -> dict:
    """Metadata of one replica (reps of length 1, table of the replica) or of the pooled run."""
    pooled = "replica" in table.columns
    box = [float(v) for v in p.box]
    volume_um3 = math.prod(box) * 1e-9
    step = reps[0]["step"]
    if p.periodic:
        coordinates = ("Qt centres in the simulation box, wrapped into [0, L) along every axis; not unwrapped, "
                       "not shifted: clusters crossing a periodic face appear split at opposite faces")
    else:
        coordinates = "Qt centres in the box frame (walls at 0 and L), as dumped, not shifted"
    return {
        "source_run": f"{run_dir.parent.name}/{run_dir.name}",
        "run_name": run_dir.name,
        "n_replicas": len(reps),
        "replicas": [r["name"] for r in reps],
        "source_trajectories": [r["source"] for r in reps],
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
        "position_resolution_nm": 10.0 ** -POSITION_DECIMALS,
        "qt_radius_nm": p.radius_qt,
        "qt_diameter_nm": 2 * p.radius_qt,
        "n_qt_per_replica": int(p.n_qt),
        "qt_per_um3": p.n_qt / volume_um3,
        **counts(table),
        "columns": _column_doc(p.periodic, pooled),
        "checks": {
            "export_checks": "counts, fragment, bond and unwrap checks of export.check_replica passed",
            "box": "traj.dump box bounds equal Params.box with origin 0",
            "wrap": {r["name"]: r["wrap"] for r in reps},
            "bond_check": {r["name"]: r["bond_check"] for r in reps},
            "unwrap_check": {r["name"]: r["unwrap_check"] for r in reps},
            "fragment_check": {r["name"]: r["fragment_check"] for r in reps},
        },
        "per_replica": ({r["name"]: counts(table[table["replica"] == r["name"]]) for r in reps} if pooled
                        else {reps[0]["name"]: counts(table)}),
        "parameters": p.to_dict(),
    }


# ------------------------------------------------------------------------------------------------ writing
def _write(out: Path, table: pd.DataFrame, meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / CSV_NAME, index=False)
    (out / JSON_NAME).write_text(json.dumps(meta, indent=2))


def _check_plot(table: pd.DataFrame, box, out: Path, name: str, periodic: bool) -> None:
    """x-y, x-z and y-z projections of one replica (free Qt grey, clusters coloured, percolating black),
    box or walls outlined."""
    free = ~table["is_clustered"]
    perc = table["percolating"]
    fig, axs = plt.subplots(1, 3, figsize=(12, 4.2), layout="constrained")
    for ax, (a, b) in zip(axs, (("x", "y"), ("x", "z"), ("y", "z"))):
        ax.scatter(table.loc[free, f"{a}_nm"], table.loc[free, f"{b}_nm"], s=1, c="0.75", lw=0)
        c = table[~free & ~perc]
        ax.scatter(c[f"{a}_nm"], c[f"{b}_nm"], s=2, c=c["cluster"] % 20, cmap="tab20", lw=0)
        if perc.any():
            ax.scatter(table.loc[perc, f"{a}_nm"], table.loc[perc, f"{b}_nm"], s=2, c="k", lw=0, label="percolating")
        la, lb = box["xyz".index(a)], box["xyz".index(b)]
        ax.add_patch(plt.Rectangle((0, 0), la, lb, fill=False, lw=0.8, ls="--", color="k"))
        ax.set_aspect("equal")
        ax.set_xlabel(f"{a} (nm)")
        ax.set_ylabel(f"{b} (nm)")
    if perc.any():
        axs[0].legend(loc="upper right", fontsize=7, frameon=False)
    frame = "wrapped into the periodic box" if periodic else "box frame, dashed = walls"
    fig.suptitle(f"{name}: Qt in the box ({frame}; full projections; grey = free)", fontsize=9)
    fig.savefig(out / "check_sem.png", dpi=200)
    plt.close(fig)


def _write_replica(rep_dir: Path, rep: dict, p: Params, plot: bool) -> Path:
    out = rep_dir / OUT_SUBDIR
    _write(out, rep["table"], _metadata(p, [rep], rep["table"], rep_dir.parent))
    if plot:
        _check_plot(rep["table"], [float(v) for v in p.box], out, rep["name"], p.periodic)
    return out


# ------------------------------------------------------------------------------------------------ public
def sem_export(rep_dir: str | Path, plot: bool = True) -> Path:
    """Export one replica to <rep>/sem/ (CSV, sem_metadata.json, check_sem.png). For checking a single
    replica; does not require it to be finished. The analysis uses the pooled run export (sem_export_run)."""
    rep_dir = Path(rep_dir).expanduser()
    p = Params.load(rep_dir.parent / "params.json")
    rep = _replica(rep_dir, check_replica(rep_dir, p), p)
    return _write_replica(rep_dir, rep, p, plot)


def sem_export_run(run_dir: str | Path, plot: bool = True) -> Path:
    """Export every replica of a run (<rep>/sem/) and pool them in <run>/sem/: the Qt table with a `replica`
    column, `label_in_replica`, `cluster_in_replica` and run-wide `cluster` / `label` IDs, and the run
    metadata (pooled counts and checks, per-replica summaries). Returns <run>/sem/.

    All replicas must be finished and pass all checks; otherwise ExportError is raised and nothing is
    written. Works with a single replica. Independent of the FIB-SEM export (fibsem.py)."""
    run_dir = Path(run_dir).expanduser()
    p, rep_dirs, checked = check_run(run_dir)                      # all checks before anything is written
    reps = [_replica(rep_dir, c, p) for rep_dir, c in zip(rep_dirs, checked)]
    pooled = pool([(r["name"], r["table"].rename(columns={"label": "label_in_replica"})) for r in reps])
    pooled = pooled[POOLED_COLUMNS]

    for rep_dir, r in zip(rep_dirs, reps):
        _write_replica(rep_dir, r, p, plot)
    out = run_dir / OUT_SUBDIR
    _write(out, pooled, _metadata(p, reps, pooled, run_dir))
    return out
