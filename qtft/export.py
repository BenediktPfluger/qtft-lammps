"""Shared part of the FIB-SEM export (fibsem.py) and the SEM export (sem.py): frame, checks and clusters.

Both exports take the frame at the end of the calcium influx: the last traj.dump frame at or before
Params.influx_end_step (the run end without an influx), the end of the agglomeration phase before the
pump-out drives the deagglomeration. It is the same frame as for the agglomerate distributions
(structure.agglomerate_volumes_run).

Clusters are the connected components of the bond graph (ground truth, no distance-based clustering),
including ferritin (Qt-Ft-Qt bridges), numbered by size within a replica (1 = largest; ties by the smallest
particle ID). A Qt in a component with more than one particle is clustered (a Qt-Ft dimer already counts).
Positions are read with qtft.read and unwrapped with structure.unwrap_clusters (minimum image along a
spanning tree of the bonds; the image flags of traj.dump are not used, fix bond/create makes them
inconsistent). The same function marks the axes along which a cluster spans the box: in a periodic box it
wraps around the box, with walls it touches both opposite walls. Such clusters are `percolating`.

Every replica must be finished and pass all checks, otherwise ExportError is raised and the calling export
writes nothing:
- particle counts (n_qt, n_ft, unique IDs),
- fragment check: the clusters equal the LAMMPS fragments (cluster column of traj.dump),
- bond check: minimum-image bond length <= BOND_TOLERANCE x contact for all but MAX_FAILED_FRACTION,
- unwrap check: every bond of a non-percolating cluster keeps its minimum-image vector after unwrapping,
- all replicas at the same frame.

No matplotlib here; the exports write their own files and check plots.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .params import Params
from .read import dump_steps, read_bonds, read_dump
from .remote import require_fetched
from .structure import WALL_TOUCH, unwrap_clusters

BOND_TOLERANCE = 1.5          # a bond may be at most 1.5 x contact distance (r_i + r_j) long
MAX_FAILED_FRACTION = 0.05    # stop the export if a larger fraction of the bonds is too long
UNWRAP_TOLERANCE_NM = 1e-6    # unwrapped bond vector vs its minimum image


class ExportError(RuntimeError):
    """A replica failed a check: the export stops and nothing is written."""


# ------------------------------------------------------------------------------------------------ helpers
def done(rep: Path) -> bool:
    """The replica finished (status file says done)."""
    return (rep / "status").exists() and (rep / "status").read_text().strip() == "done"


def export_step(rep_dir: Path, p: Params) -> int:
    """Last traj.dump step at or before the end of the influx."""
    before = [s for s in dump_steps(rep_dir / "traj.dump") if s <= p.influx_end_step]
    if not before:
        raise ExportError(f"{rep_dir.name}: no traj.dump frame at or before step {p.influx_end_step}")
    return max(before)


def frame_text(p: Params) -> str:
    """The exported frame in words (for the metadata)."""
    if p.t_influx_s > 0:
        return (f"end of the influx: last traj.dump frame at or before production step {p.influx_end_step} "
                f"(t = {p.influx_end_step * p.timestep * 1e-9:g} s)")
    return "run end (no influx)"


def counts(table: pd.DataFrame) -> dict:
    """Counts of one encapsulin table (one replica or pooled): encapsulins, clusters, flags, size histogram.
    Needs the columns label, cluster, is_clustered, cluster_size_particles, percolating, touches_wall."""
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
def check_replica(rep_dir: Path, p: Params) -> dict:
    """Read, check and tabulate the export frame of one replica (nothing is written).

    Returns a dict with name, step, source (traj.dump relative to runs/), the checks (bond_check,
    unwrap_check, fragment_check) and `qt`: one row per encapsulin, ordered by particle ID, with
    particle_id, x/y/z (as dumped), xu/yu/zu (unwrapped along the bonds; with walls = as dumped), cluster
    (numbered by size), is_clustered, cluster_size_particles (incl. Ft), spans (axes, '' = none),
    percolating, touches_wall and wall_gap_nm (with walls: smallest surface-face gap over all particles of
    the cluster; periodic: False and NaN). Raises ExportError on a failed check."""
    name = rep_dir.name
    step = export_step(rep_dir, p)
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

    table = pd.DataFrame({
        "particle_id": ids[qt],
        "x": pos[qt, 0], "y": pos[qt, 1], "z": pos[qt, 2],
        "xu": pos_u[qt, 0], "yu": pos_u[qt, 1], "zu": pos_u[qt, 2],
        "cluster": number[lq],
        "is_clustered": size_all[lq] > 1,
        "cluster_size_particles": size_all[lq],
        "spans": ["".join(a for a, s in zip("xyz", spans[c]) if s) for c in lq],
        "percolating": percolating[lq],
        "touches_wall": at_wall[lq],
        "wall_gap_nm": gap[lq],
    })
    return {"name": name, "step": int(step), "qt": table,
            "bond_check": bond_check, "unwrap_check": unwrap_check, "fragment_check": fragment_check,
            "source": f"{rep_dir.parent.parent.name}/{rep_dir.parent.name}/{name}/traj.dump"}


# ------------------------------------------------------------------------------------------------ whole run
def check_run(run_dir: Path) -> tuple[Params, list[Path], list[dict]]:
    """Params, replica folders and check_replica() of every replica of a run, all at the same frame.

    The run must be fetched (LRZ) and every replica finished; any failed check raises ExportError before
    the calling export writes anything."""
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
    rep_dirs = sorted(run_dir.glob("rep_*"))
    if not rep_dirs:
        raise ExportError(f"no replicas (rep_*) in {run_dir}")
    unfinished = [r.name for r in rep_dirs if not done(r)]
    if unfinished:
        raise ExportError(f"replicas not finished: {', '.join(unfinished)} - export stopped, nothing written")
    reps = [check_replica(r, p) for r in rep_dirs]
    steps = {r["name"]: r["step"] for r in reps}
    if len(set(steps.values())) != 1:
        raise ExportError(f"replicas exported at different frames: {steps}")
    return p, rep_dirs, reps


def pool(tables: list[tuple[str, pd.DataFrame]]) -> pd.DataFrame:
    """Concatenate per-replica tables [(replica name, table with a `cluster` column), ...]: adds the columns
    replica and cluster_in_replica, makes `cluster` run-wide (offset by the clusters of the replicas before)
    and inserts a run-wide `label` (1..n) as the first column."""
    parts, offset = [], 0
    for name, t in tables:
        parts.append(t.assign(cluster=t["cluster"] + offset, cluster_in_replica=t["cluster"], replica=name))
        offset += int(t["cluster"].max())
    pooled = pd.concat(parts, ignore_index=True)
    pooled.insert(0, "label", np.arange(1, len(pooled) + 1))
    return pooled
