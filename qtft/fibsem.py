"""Export the final frame of a run in the schema of the FIB-SEM segmentation pipeline.

Only encapsulins (Qt) are exported; ferritin is below FIB-SEM resolution but still defines the
cluster connectivity (Qt-Ft-Qt bridges). Cluster IDs are the connected components of the bond
graph (ground truth, no distance-based clustering). A Qt in a component with more than one
particle is clustered (a Qt-Ft dimer already counts); a free Qt is a singleton with its own ID.

Periodic box: every cluster is unwrapped across the box boundaries (OVITO), then all positions are
shifted so that the lower corner of the bounding box of all encapsulin spheres is at the origin.
Caveat: this breaks periodicity *between* clusters (they may overlap, the bounding volume may
exceed the box), which biases neighbourhood metrics across clusters (RDF, <M(R)>); per-cluster
statistics are unaffected.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .params import Params
from .structure import load_frames, particle_types

BOND_TOLERANCE = 1.5          # a bond may be at most 1.5 x contact distance long
MAX_FAILED_FRACTION = 0.05    # abort the export if more bonds fail


def fibsem_export(rep_dir: str | Path, voxel_nm: float = 4.0, plot: bool = True) -> Path:
    """Write <rep>/fibsem/sim_encapsulin_centroids_clustered.csv, sim_metadata.json and a check plot."""
    rep_dir = Path(rep_dir).expanduser()
    p = Params.load(rep_dir.parent / "params.json")
    step, data = next(load_frames(rep_dir, last_only=True))

    ids = np.asarray(data.particles["Particle Identifier"])
    types = particle_types(data)
    pos = np.asarray(data.particles.positions, dtype=float)               # unwrapped per cluster
    cluster = np.asarray(data.particles["Cluster"])
    sizes = np.asarray(data.tables["clusters"]["Cluster Size"])
    size_of = sizes[cluster - 1]

    # --- bond check on unwrapped positions (also verifies the unwrapping)
    bonds = data.particles.bonds
    topo = np.asarray(bonds.topology) if bonds is not None else np.empty((0, 2), int)
    radius = np.where(types == 1, p.radius_qt, p.radius_ft)
    length = np.linalg.norm(pos[topo[:, 1]] - pos[topo[:, 0]], axis=1)
    limit = BOND_TOLERANCE * (radius[topo[:, 0]] + radius[topo[:, 1]])
    failed = int((length > limit).sum())
    bond_check = {"n_bonds": int(len(topo)), "failed": failed, "tolerance": f"<= {BOND_TOLERANCE} x contact",
                  "max_length_nm": float(length.max()) if len(length) else None}
    if len(topo) and failed / len(topo) > MAX_FAILED_FRACTION:
        raise RuntimeError(f"{failed} of {len(topo)} bonds longer than {BOND_TOLERANCE} x contact: "
                           "export aborted, nothing written")
    if failed:
        print(f"WARNING: {failed} bonds longer than {BOND_TOLERANCE} x contact")

    # --- encapsulins only, sorted by particle ID
    qt = np.nonzero(types == 1)[0]
    qt = qt[np.argsort(ids[qt])]
    if len(qt) != p.n_qt or len(np.unique(ids[qt])) != len(qt):
        raise RuntimeError(f"expected {p.n_qt} unique encapsulins, found {len(qt)}")
    xyz = pos[qt]
    lower = (xyz - p.radius_qt).min(axis=0)                              # bounding box of the spheres
    xyz = xyz - lower
    renumber = {c: i for i, c in enumerate(np.unique(cluster[qt]), start=1)}
    volume = 4 / 3 * math.pi * p.radius_qt**3
    table = pd.DataFrame({
        "label": np.arange(1, len(qt) + 1),
        "z_nm": xyz[:, 2], "y_nm": xyz[:, 1], "x_nm": xyz[:, 0],
        "z_vox": xyz[:, 2] / voxel_nm, "y_vox": xyz[:, 1] / voxel_nm, "x_vox": xyz[:, 0] / voxel_nm,
        "radius_nm": p.radius_qt, "volume_nm3": volume,
        "cluster": [renumber[c] for c in cluster[qt]],
        "is_clustered": size_of[qt] > 1,
    })

    # --- metadata
    qt_per_cluster = table.groupby("cluster").size()
    multi = sorted({c for c, s in zip(cluster[qt], size_of[qt]) if s > 1})
    box = [p.box_length] * 3
    upper = (xyz + p.radius_qt).max(axis=0)
    meta = {
        "source_trajectory": str(rep_dir / "traj.dump"),
        "final_step": int(step), "final_time_us": step * p.timestep * 1e-3,
        "n_encapsulins": int(len(table)),
        "n_clustered": int(table["is_clustered"].sum()),
        "n_free": int((~table["is_clustered"]).sum()),
        "n_clusters": int(table["cluster"].nunique()),
        "n_multi_particle_clusters": len(multi),
        "largest_cluster_encapsulins": int(qt_per_cluster.max()),
        "largest_cluster_particles_incl_ferritin": int(sizes.max()),
        "cluster_size_histogram_encapsulins": {int(k): int(v) for k, v in
                                               qt_per_cluster.value_counts().sort_index().items()},
        "qt_radius_nm": p.radius_qt, "qt_volume_nm3": volume,
        "boundary": "periodic", "wall_stiffness": None,
        "box_size_xyz_nm": box, "box_size_zyx_nm": box[::-1],
        "offset_applied_xyz_nm": (-lower).tolist(),
        "bounds_zyx_nm": {"min": [0.0, 0.0, 0.0], "max": upper[::-1].tolist()},
        "frame_description": ("periodic box: clusters unwrapped across the boundaries (bond graph), then "
                              "shifted so that the lower corner of the bounding box of all encapsulin "
                              "spheres is at the origin; periodicity between clusters is broken "
                              "(clusters may overlap, the bounding volume may exceed the box)"),
        "wall_leaks": None,
        "voxel_size_nm": voxel_nm,
        "topology_mapping": "by particle ID (traj.dump id <-> bonds.dump atom IDs)",
        "bond_check": bond_check,
        "parameters": p.to_dict(),
    }

    out = rep_dir / "fibsem"
    out.mkdir(exist_ok=True)
    table.to_csv(out / "sim_encapsulin_centroids_clustered.csv", index=False)
    (out / "sim_metadata.json").write_text(json.dumps(meta, indent=2))
    if plot:
        _check_plot(table, lower, p.box_length, out)
    return out


def _check_plot(table: pd.DataFrame, lower: np.ndarray, box: float, out: Path) -> None:
    """x-y scatter of the encapsulins coloured by cluster (free ones grey), with the original box."""
    fig, ax = plt.subplots(figsize=(5, 5), layout="constrained")
    free = ~table["is_clustered"]
    ax.scatter(table.loc[free, "x_nm"], table.loc[free, "y_nm"], s=2, c="0.75", lw=0, label="free")
    clustered = table[~free]
    ax.scatter(clustered["x_nm"], clustered["y_nm"], s=3, c=clustered["cluster"] % 20, cmap="tab20", lw=0)
    ax.add_patch(plt.Rectangle((-lower[0], -lower[1]), box, box, fill=False, lw=0.8, ls="--", color="k"))
    ax.set_aspect("equal")
    ax.set_xlabel("x (nm)")
    ax.set_ylabel("y (nm)")
    ax.set_title("encapsulins by cluster (dashed: periodic box)", fontsize=8)
    fig.savefig(out / "check_xy.png", dpi=200)
    plt.close(fig)


def fibsem_export_run(run_dir: str | Path, voxel_nm: float = 4.0) -> list[Path]:
    """Export every finished replica of a run."""
    run_dir = Path(run_dir).expanduser()
    return [fibsem_export(rep, voxel_nm) for rep in sorted(run_dir.glob("rep_*"))
            if (rep / "status").exists() and (rep / "status").read_text().strip() == "done"]
