"""Export the frame at the end of the calcium influx in the schema of the FIB-SEM segmentation pipeline.

The frame is the last traj.dump frame at or before Params.influx_end_step (the run end without an
influx): the end of the agglomeration phase, before the pump-out drives the deagglomeration. It is the
same frame as for the agglomerate distributions (structure.agglomerate_volumes_run).

Only encapsulins (Qt) are exported; ferritin is below FIB-SEM resolution but still defines the cluster
connectivity (Qt-Ft-Qt bridges). Clusters are the connected components of the bond graph (ground truth,
no distance-based clustering), numbered by size within a replica (1 = largest). A Qt in a component with
more than one particle is clustered (a Qt-Ft dimer already counts); a free Qt is a singleton.

Positions are read with qtft.read and unwrapped with structure.unwrap_clusters (minimum image along a
spanning tree of the bonds; the image flags of traj.dump are not used, fix bond/create makes them
inconsistent). The same function marks the axes along which a cluster spans the box: in a periodic box
it wraps around the box (no defined unwrapped shape), with walls it touches both opposite walls. Such
clusters are flagged `percolating`; the analysis excludes them from the geometry metrics.

Periodic box: every replica is shifted so that the lower corner of the bounding box of its encapsulin
spheres (non-percolating clusters) is at the origin. This breaks periodicity *between* clusters, which is
irrelevant for the within-cluster analysis.

Walls (boundary "reflect" or "harmonic"): positions in the box frame, not shifted (walls at 0 and L).
Both wall types act on the particle surface (reflect: the plane at r_i from the face; harmonic: cutoff
r_i), so a cluster touches a wall when a particle surface (Qt or Ft) is within structure.WALL_TOUCH of a
face. Harmonic walls let a sphere stick out by a fraction of a nm (`wall_leaks`).

fibsem_export_run exports every replica of a run and pools them in <run>/fibsem/ (replica column,
run-wide cluster IDs). Every replica must be finished and pass all checks, otherwise the export stops and
nothing is written:
- particle counts (n_qt, n_ft, unique IDs),
- fragment check: the clusters equal the LAMMPS fragments (cluster column of traj.dump),
- bond check: minimum-image bond length <= BOND_TOLERANCE x contact for all but MAX_FAILED_FRACTION,
- unwrap check: every bond of a non-percolating cluster keeps its minimum-image vector after unwrapping,
- all replicas at the same frame.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .params import Params
from .read import dump_steps, read_bonds, read_dump
from .remote import require_fetched
from .structure import WALL_TOUCH, unwrap_clusters

BOND_TOLERANCE = 1.5          # a bond may be at most 1.5 x contact distance (r_i + r_j) long
MAX_FAILED_FRACTION = 0.05    # stop the export if a larger fraction of the bonds is too long
UNWRAP_TOLERANCE_NM = 1e-6    # unwrapped bond vector vs its minimum image
CSV_NAME = "sim_encapsulin_centroids_clustered.csv"
JSON_NAME = "sim_metadata.json"
OUT_SUBDIR = "fibsem"


class ExportError(RuntimeError):
    """A replica failed a check: the export stops and nothing is written."""


# ------------------------------------------------------------------------------------------------ helpers
def _done(rep: Path) -> bool:
    return (rep / "status").exists() and (rep / "status").read_text().strip() == "done"


def _export_step(rep_dir: Path, p: Params) -> int:
    before = [s for s in dump_steps(rep_dir / "traj.dump") if s <= p.influx_end_step]
    if not before:
        raise ExportError(f"{rep_dir.name}: no traj.dump frame at or before step {p.influx_end_step}")
    return max(before)


def _frame_text(p: Params) -> str:
    if p.t_influx_s > 0:
        return (f"end of the influx: last traj.dump frame at or before production step {p.influx_end_step} "
                f"(t = {p.influx_end_step * p.timestep * 1e-9:g} s)")
    return "run end (no influx)"


def _counts(table: pd.DataFrame) -> dict:
    """Counts of one table (one replica or pooled): encapsulins, clusters, flags, size histogram."""
    cl = table.groupby("cluster")
    per = cl.agg(n=("label", "size"), size_all=("cluster_size_particles", "first"),
                 perc=("percolating", "first"), wall=("touches_wall", "first"))
    return {
        "n_encapsulins": int(len(table)),
        "n_clustered": int(table["is_clustered"].sum()),
        "n_free": int((~table["is_clustered"]).sum()),
        "n_clusters": int(len(per)),
        "n_multi_particle_clusters": int((per["size_all"] > 1).sum()),
        "n_percolating_clusters": int(per["perc"].sum()),
        "n_percolating_encapsulins": int(per.loc[per["perc"], "n"].sum()),
        "n_clusters_touching_wall": int(per["wall"].sum()),
        "largest_cluster_encapsulins": int(per["n"].max()),
        "largest_cluster_particles_incl_ferritin": int(per["size_all"].max()),
        "cluster_size_histogram_encapsulins": {int(k): int(v) for k, v in
                                               per["n"].value_counts().sort_index().items()},
    }


# ------------------------------------------------------------------------------------------------ one replica
def _replica(rep_dir: Path, p: Params, voxel_nm: float) -> dict:
    """Read, check and tabulate one replica (nothing is written). Raises ExportError on a failed check."""
    name = rep_dir.name
    step = _export_step(rep_dir, p)
    a_step, atoms = next(read_dump(rep_dir / "traj.dump", start=step))
    if a_step != step:
        raise ExportError(f"{name}: traj.dump frame {step} could not be read (got {a_step})")
    try:
        b_step, bonds = next(read_bonds(rep_dir / "bonds.dump", start=step))
    except StopIteration:
        b_step, bonds = None, None
    if b_step != step:
        raise ExportError(f"{name}: step {step} missing in bonds.dump")

    missing = [c for c in ("id", "type", "x", "y", "z", "cluster") if c not in atoms.columns]
    if missing:
        raise ExportError(f"{name}: traj.dump lacks the columns {missing} (expected the dump of template.lmp: "
                          "id type x y z ix iy iz radius cluster nbonds)")

    # --- particle counts
    types = atoms["type"].to_numpy()
    ids = atoms["id"].to_numpy()
    n_qt, n_ft = int((types == 1).sum()), int((types == 2).sum())
    if (n_qt, n_ft) != (p.n_qt, p.n_ft) or len(np.unique(ids)) != len(ids):
        raise ExportError(f"{name}: expected {p.n_qt} Qt + {p.n_ft} Ft with unique IDs, "
                          f"found {n_qt} Qt + {n_ft} Ft ({len(np.unique(ids))} unique IDs)")

    # --- clusters, unwrapped positions, spanning axes
    label, pos_u, spans = unwrap_clusters(atoms, bonds, p)
    pos = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    box = np.array(p.box)
    radius = np.where(types == 1, p.radius_qt, p.radius_ft)
    n_comp = int(label.max()) + 1
    percolating = spans.any(axis=1)

    # --- fragment check: our components = LAMMPS fragments (1:1)
    frag = atoms["cluster"].to_numpy()
    n_frag = len(np.unique(frag))
    n_pairs = len(np.unique(np.c_[label, frag], axis=0))
    if not (n_pairs == n_comp == n_frag):
        raise ExportError(f"{name}: bond-graph clusters ({n_comp}) do not match the LAMMPS fragments "
                          f"({n_frag}; {n_pairs} distinct pairs) - traj.dump and bonds.dump disagree")
    fragment_check = {"n_components": n_comp, "n_lammps_fragments": n_frag, "match": True}

    # --- bond check (physical length, minimum image) and unwrap check
    index = np.empty(ids.max() + 1, dtype=np.int64)
    index[ids] = np.arange(len(ids))
    if len(bonds) and (bonds.max() > ids.max() or not np.isin(bonds, ids).all()):
        raise ExportError(f"{name}: bonds.dump refers to atom IDs that are not in traj.dump")
    i, j = (index[bonds[:, 0]], index[bonds[:, 1]]) if len(bonds) else (np.empty(0, int), np.empty(0, int))
    d_raw = pos[j] - pos[i]
    d_mic = d_raw - box * np.round(d_raw / box) if p.periodic else d_raw
    length = np.linalg.norm(d_mic, axis=1)
    ratio = length / (radius[i] + radius[j])
    failed = int((ratio > BOND_TOLERANCE).sum())
    bond_check = {"n_bonds": int(len(bonds)), "n_too_long": failed,
                  "tolerance": f"<= {BOND_TOLERANCE} x contact (r_i + r_j), minimum image",
                  "max_length_nm": float(length.max()) if len(length) else None,
                  "max_length_ratio": float(ratio.max()) if len(ratio) else None}
    if len(bonds) and failed / len(bonds) > MAX_FAILED_FRACTION:
        raise ExportError(f"{name}: {failed} of {len(bonds)} bonds longer than {BOND_TOLERANCE} x contact")
    keep = ~percolating[label[i]]
    dev = np.abs((pos_u[j] - pos_u[i]) - d_mic)[keep].max(axis=1) if len(bonds) else np.empty(0)
    n_bad = int((dev > UNWRAP_TOLERANCE_NM).sum())
    unwrap_check = {"n_bonds_checked": int(keep.sum()), "n_bonds_skipped_percolating": int((~keep).sum()),
                    "n_mismatch": n_bad, "max_deviation_nm": float(dev.max()) if len(dev) else 0.0}
    if n_bad:
        raise ExportError(f"{name}: {n_bad} bonds change their minimum-image vector when unwrapped")

    # --- per-cluster size and flags (from all particles, Qt and Ft)
    size_all = np.bincount(label, minlength=n_comp)
    if p.periodic:
        gap = np.full(n_comp, np.nan)
        at_wall = np.zeros(n_comp, dtype=bool)
    else:
        g = np.minimum(pos - radius[:, None], box - pos - radius[:, None]).min(axis=1)   # surface-face gap
        gap = np.full(n_comp, np.inf)
        np.minimum.at(gap, label, g)
        at_wall = gap <= WALL_TOUCH

    # --- encapsulins: number the Qt clusters by size (1 = largest; ties by smallest particle ID)
    qt = np.nonzero(types == 1)[0]
    qt = qt[np.argsort(ids[qt])]
    comps = np.unique(label[qt])
    first_id = np.full(n_comp, np.iinfo(np.int64).max)
    np.minimum.at(first_id, label, ids)
    order = comps[np.lexsort((first_id[comps], -size_all[comps]))]
    number = np.zeros(n_comp, dtype=int)
    number[order] = np.arange(1, len(order) + 1)
    lq = label[qt]

    xyz = pos_u[qt]
    if p.periodic:
        ref = xyz[~percolating[lq]] if (~percolating[lq]).any() else xyz
        lower = (ref - p.radius_qt).min(axis=0)                     # bounding box of the spheres
    else:
        lower = np.zeros(3)                                         # box frame: walls at 0 and L
    xyz = xyz - lower
    volume = 4 / 3 * math.pi * p.radius_qt ** 3
    table = pd.DataFrame({
        "label": np.arange(1, len(qt) + 1),
        "particle_id": ids[qt],
        "z_nm": xyz[:, 2], "y_nm": xyz[:, 1], "x_nm": xyz[:, 0],
        "z_vox": xyz[:, 2] / voxel_nm, "y_vox": xyz[:, 1] / voxel_nm, "x_vox": xyz[:, 0] / voxel_nm,
        "radius_nm": p.radius_qt, "volume_nm3": volume,
        "cluster": number[lq],
        "is_clustered": size_all[lq] > 1,
        "cluster_size_particles": size_all[lq],
        "spans": ["".join(a for a, s in zip("xyz", spans[c]) if s) for c in lq],
        "percolating": percolating[lq],
        "touches_wall": at_wall[lq],
        "wall_gap_nm": gap[lq],
    })

    leaks = None
    if not p.periodic:
        depth = p.radius_qt - np.minimum(xyz, box - xyz).min(axis=1)        # sphere beyond a face
        leaks = {"n_encapsulins_beyond_wall": int((depth > 0).sum()),
                 "max_depth_nm": float(max(depth.max(), 0.0)) if len(depth) else 0.0}
    return {"name": name, "step": int(step), "table": table, "lower": lower, "leaks": leaks,
            "bond_check": bond_check, "unwrap_check": unwrap_check, "fragment_check": fragment_check,
            "source": f"{rep_dir.parent.parent.name}/{rep_dir.parent.name}/{name}/traj.dump"}


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
    rep = _replica(rep_dir, p, voxel_nm)
    return _write_replica(rep_dir, rep, p, voxel_nm, plot)


def fibsem_export_run(run_dir: str | Path, voxel_nm: float = 4.0, plot: bool = True) -> Path:
    """Export every replica of a run (<rep>/fibsem/) and pool them in <run>/fibsem/: the encapsulin table
    with a `replica` column, `cluster_in_replica` and run-wide `cluster` / `label` IDs, and the run
    metadata (pooled counts and checks, per-replica summaries). Returns <run>/fibsem/.

    All replicas must be finished and pass all checks (see the module docstring); otherwise ExportError
    is raised and nothing is written."""
    run_dir = Path(run_dir).expanduser()
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
    rep_dirs = sorted(run_dir.glob("rep_*"))
    if not rep_dirs:
        raise ExportError(f"no replicas (rep_*) in {run_dir}")
    unfinished = [r.name for r in rep_dirs if not _done(r)]
    if unfinished:
        raise ExportError(f"replicas not finished: {', '.join(unfinished)} - export stopped, nothing written")

    reps = [_replica(r, p, voxel_nm) for r in rep_dirs]            # all checks before anything is written
    steps = {r["name"]: r["step"] for r in reps}
    if len(set(steps.values())) != 1:
        raise ExportError(f"replicas exported at different frames: {steps}")

    parts, offset = [], 0
    for r in reps:
        t = r["table"]
        parts.append(t.assign(cluster=t["cluster"] + offset).rename(columns={"label": "label_in_replica"})
                      .assign(cluster_in_replica=t["cluster"], replica=r["name"]))
        offset += int(t["cluster"].max())
    pooled = pd.concat(parts, ignore_index=True)
    pooled.insert(0, "label", np.arange(1, len(pooled) + 1))
    cols = ["label", "replica", "label_in_replica", "particle_id", "z_nm", "y_nm", "x_nm",
            "z_vox", "y_vox", "x_vox", "radius_nm", "volume_nm3", "cluster", "cluster_in_replica",
            "is_clustered", "cluster_size_particles", "spans", "percolating", "touches_wall", "wall_gap_nm"]
    pooled = pooled[cols]

    for rep_dir, r in zip(rep_dirs, reps):
        _write_replica(rep_dir, r, p, voxel_nm, plot)
    out = run_dir / OUT_SUBDIR
    _write(out, pooled, _metadata(p, reps, pooled, run_dir, voxel_nm))
    return out
