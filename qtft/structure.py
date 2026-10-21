"""Cluster geometry with OVITO: radius of gyration, compactness, cluster centres, nearest-neighbour
distances, composition. Clusters are the connected components of the bond graph; in a periodic box
OVITO unwraps each cluster across the boundaries before measuring it (with walls the box is not
periodic and positions are used as they are).

Frames are read from traj.dump (positions) and bonds.dump (bond list of the same step).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .params import Params
from .read import TYPE_IDS, read_bonds

PACKING = 0.64          # random close packing, for the ideal compact sphere


def load_frames(rep_dir: str | Path, every: int = 1, last: bool = True, last_only: bool = False):
    """Yield (step, OVITO DataCollection) for every `every`-th frame of traj.dump (always including
    the last one if `last`; only the last one if `last_only`), with bonds from bonds.dump and
    bond-based cluster analysis applied.

    Afterwards `data.particles.positions` are unwrapped per cluster (periodic box; with walls they
    are unchanged), `data.particles['Cluster']`
    is the cluster index (largest first) and `data.tables['clusters']` holds size, centre of mass
    and radius of gyration of every cluster.
    """
    from ovito.io import import_file
    from ovito.modifiers import ClusterAnalysisModifier

    rep_dir = Path(rep_dir)
    pipeline = import_file(str(rep_dir / "traj.dump"))
    n = pipeline.source.num_frames
    frames = [n - 1] if last_only else list(range(0, n, every))
    if last and frames[-1] != n - 1:
        frames.append(n - 1)
    steps = {pipeline.source.compute(f).attributes["Timestep"]: f for f in frames}
    bonds = {s: b for s, b in read_bonds(rep_dir / "bonds.dump") if s in steps}

    modifier = ClusterAnalysisModifier(
        neighbor_mode=ClusterAnalysisModifier.NeighborMode.Bonding, sort_by_size=True,
        unwrap_particles=True, compute_com=True, compute_gyration=True)
    for step, frame in steps.items():
        data = pipeline.compute(frame)
        ids = np.asarray(data.particles["Particle Identifier"])
        index = np.empty(ids.max() + 1, dtype=np.int64)
        index[ids] = np.arange(len(ids))
        pairs = index[bonds[step]] if len(bonds.get(step, [])) else np.empty((0, 2), dtype=np.int64)
        pos = np.asarray(data.particles.positions)
        box = np.diag(data.cell[...][:3, :3])
        new = data.particles_.create_bonds(count=len(pairs))
        new.create_property("Topology", data=pairs)
        # minimum-image shift of every bond (bond vector = x_b - x_a + box * shift); none without
        # periodicity (walls: OVITO reads the ff flags of BOX BOUNDS)
        shift = -np.round((pos[pairs[:, 1]] - pos[pairs[:, 0]]) / box).astype(np.int32)
        new.create_property("Periodic Image", data=shift * np.asarray(data.cell.pbc, dtype=np.int32))
        # equal weights for centre of mass and Rg; OVITO would otherwise take per-type masses from
        # the type names (it reads "Ft" as fluorine)
        data.particles_.create_property("Mass", data=np.ones(len(ids)))
        data.apply(modifier)
        yield step, data


def particle_types(data) -> np.ndarray:
    """Numeric type of every particle (1 = Qt, 2 = Ft), for named (Qt, Ft) and numeric dumps alike."""
    ptypes = data.particles.particle_types
    raw = np.asarray(ptypes)
    lut = np.arange(max([t.id for t in ptypes.types] + [int(raw.max()) if len(raw) else 0]) + 1)
    for t in ptypes.types:
        lut[t.id] = TYPE_IDS.get(t.name, t.id)
    return lut[raw]


def cluster_table(data, p: Params) -> pd.DataFrame:
    """One row per cluster: size, composition, bonds, Rg, ideal Rg, centre of mass."""
    t = data.tables["clusters"]
    size = np.asarray(t["Cluster Size"])
    cl = np.asarray(data.particles["Cluster"]) - 1                  # 0-based cluster index
    is_qt = particle_types(data) == 1
    n_qt = np.bincount(cl, weights=is_qt, minlength=len(size)).astype(int)
    topo = np.asarray(data.particles.bonds.topology) if data.particles.bonds is not None else np.empty((0, 2), int)
    n_bonds = np.bincount(cl[topo[:, 0]], minlength=len(size)) if len(topo) else np.zeros(len(size), int)
    com = np.asarray(t["Center of Mass"])
    r_mean = (p.radius_qt + p.radius_ft) / 2
    volume = size * 4 / 3 * math.pi * r_mean**3 / PACKING
    rg_ideal = math.sqrt(3 / 5) * (3 * volume / (4 * math.pi)) ** (1 / 3)
    return pd.DataFrame({
        "cluster": np.asarray(t["Cluster Identifier"]),
        "size": size, "n_qt": n_qt, "n_ft": size - n_qt, "n_bonds": n_bonds,
        "rg": np.asarray(t["Radius of Gyration"]), "rg_ideal": rg_ideal,
        "com_x": com[:, 0], "com_y": com[:, 1], "com_z": com[:, 2],
    })


def frame_metrics(data, table: pd.DataFrame, box, periodic: bool = True) -> dict:
    """Per-frame structure metrics (handover definitions: Rg on clusters >= 3, spatial and
    composition metrics on clusters >= 2). `periodic=False` (walls): plain distances between
    cluster centres."""
    big3 = table[table["size"] >= 3]
    big2 = table[table["size"] >= 2]
    out = {
        "n_clusters_ge2": len(big2),
        "rg_mean": big3["rg"].mean() if len(big3) else np.nan,
        "rg_ratio_mean": (big3["rg"] / big3["rg_ideal"]).mean() if len(big3) else np.nan,
        "qt_frac_mean": (big2["n_qt"] / big2["size"]).mean() if len(big2) else np.nan,
        "inter_nn_mean": np.nan, "inter_nn_sd": np.nan, "intra_nn_mean": np.nan,
    }
    if len(big2) >= 2:
        box = np.array(box)                                             # (Lx, Ly, Lz)
        centres = big2[["com_x", "com_y", "com_z"]].to_numpy()
        if periodic:
            centres = np.mod(centres, box)
        dist, _ = cKDTree(centres, boxsize=box if periodic else None).query(centres, k=2)
        out["inter_nn_mean"], out["inter_nn_sd"] = dist[:, 1].mean(), dist[:, 1].std()
    if len(big3):
        cl = np.asarray(data.particles["Cluster"]) - 1
        pos = np.asarray(data.particles.positions)                      # unwrapped per cluster
        order = np.argsort(cl, kind="stable")
        starts = np.searchsorted(cl[order], big3["cluster"].to_numpy() - 1)
        means = []
        for start, size in zip(starts, big3["size"].to_numpy()):
            members = pos[order[start:start + size]]
            dist, _ = cKDTree(members).query(members, k=2)
            means.append(dist[:, 1].mean())
        out["intra_nn_mean"] = float(np.mean(means))
    return out


def structure(rep_dir: str | Path, every: int = 10) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Structure metrics of one replica per analysed frame, and the cluster table of the last frame."""
    rep_dir = Path(rep_dir)
    p = Params.load(rep_dir.parent / "params.json")
    rows, table = [], None
    for step, data in load_frames(rep_dir, every=every):
        table = cluster_table(data, p)
        rows.append({"Step": step, "time_ms": step * p.timestep * 1e-6,
                     **frame_metrics(data, table, p.box, p.periodic)})
    return pd.DataFrame(rows), table


def structure_ensemble(run_dir: str | Path, every: int = 10, save: bool = True) -> dict:
    """Structure metrics of all finished replicas: mean/SD per frame, per-replica traces and the
    pooled final-frame cluster table. Saved to <run>/analysis/structure_*.csv."""
    run_dir = Path(run_dir).expanduser()
    per_rep, finals = {}, []
    for rep in sorted(run_dir.glob("rep_*")):
        if (rep / "status").exists() and (rep / "status").read_text().strip() == "done":
            df, table = structure(rep, every=every)
            per_rep[rep.name] = df.set_index("Step")
            finals.append(table.assign(replica=rep.name))
    if not per_rep:
        raise ValueError(f"no finished replicas in {run_dir}")
    stacked = pd.concat(per_rep, names=["replica", "Step"])
    grouped = stacked.groupby(level="Step")
    mean, sd = grouped.mean(), grouped.std(ddof=0)
    result = {"mean": mean, "sd": sd, "per_replica": stacked,
              "clusters_final": pd.concat(finals, ignore_index=True)}
    if save:
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        mean.join(sd, lsuffix="_mean", rsuffix="_sd").to_csv(out / "structure_timeseries.csv")
        stacked.to_csv(out / "structure_per_replica.csv")
        result["clusters_final"].to_csv(out / "clusters_final.csv", index=False)
    return result
