"""Cluster geometry with OVITO: radius of gyration, compactness, cluster centres, nearest-neighbour
distances, composition. Clusters are the connected components of the bond graph; in a periodic box
OVITO unwraps each cluster across the boundaries before measuring it (with walls the box is not
periodic and positions are used as they are).

Frames are read from traj.dump (positions) and bonds.dump (bond list of the same step).

Percolation (`percolation`, numpy/scipy only): a cluster percolates when it spans the box along at
least one axis. Periodic box: the cluster wraps around the box, i.e. a closed bond path adds up to a
whole box length (unwrapping it along a spanning tree leaves a bond that disagrees with its minimum
image by a multiple of L). Walls: the cluster touches both opposite walls (a particle surface within
WALL_TOUCH of the face).

Cluster volumes (`cluster_volumes`, numpy/scipy only), for comparison with confocal (lattice SIM)
volumes: the particle spheres of a cluster (Qt and Ft) are drawn on a voxel grid; the solid volume is
their union, the envelope volume the morphological closing of it with an ellipsoidal probe (radii
probe_xy, probe_z, about half the lateral and axial resolution), which fills the gaps the microscope
cannot resolve. Clusters of >= CLUSTER_MIN particles, in the frame at the end of the influx.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage, sparse
from scipy.sparse.csgraph import breadth_first_order, connected_components
from scipy.spatial import cKDTree

from .params import Params
from .read import TYPE_IDS, dump_steps, read_bonds, read_dump

PACKING = 0.64          # random close packing, for the ideal compact sphere (Rg_ideal)
WALL_TOUCH = 5.0        # [nm] a particle surface this close to a wall touches it (percolation with walls)
CLUSTER_MIN = 3         # particles of a cluster for the size and volume distributions
PROBE_XY_NM, PROBE_Z_NM = 60.0, 150.0      # envelope probe: half the lattice-SIM resolution (~120 / 300 nm)
VOXEL_NM = 10.0         # voxel edge of the volume grid [nm]


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
    all_steps = dump_steps(rep_dir / "traj.dump")
    if len(all_steps) != n:
        raise ValueError(f"{rep_dir / 'traj.dump'}: OVITO sees {n} frames, the file has {len(all_steps)}")
    steps = {all_steps[f]: f for f in frames}
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
    """One row per cluster: size, composition, bonds, Rg, ideal Rg, centre of mass.

    Rg_ideal is the Rg of a solid sphere holding the particle volumes of the cluster at random close
    packing: V = (n_Qt V_Qt + n_Ft V_Ft) / 0.64, Rg_ideal = sqrt(3/5) (3V / 4 pi)^(1/3)."""
    t = data.tables["clusters"]
    size = np.asarray(t["Cluster Size"])
    cl = np.asarray(data.particles["Cluster"]) - 1                  # 0-based cluster index
    is_qt = particle_types(data) == 1
    n_qt = np.bincount(cl, weights=is_qt, minlength=len(size)).astype(int)
    topo = np.asarray(data.particles.bonds.topology) if data.particles.bonds is not None else np.empty((0, 2), int)
    n_bonds = np.bincount(cl[topo[:, 0]], minlength=len(size)) if len(topo) else np.zeros(len(size), int)
    com = np.asarray(t["Center of Mass"])
    sphere = lambda r: 4 / 3 * math.pi * r**3
    volume = (n_qt * sphere(p.radius_qt) + (size - n_qt) * sphere(p.radius_ft)) / PACKING
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
    mean, sd = grouped.mean(), grouped.std(ddof=1)                      # NaN with one replica
    result = {"mean": mean, "sd": sd, "per_replica": stacked,
              "clusters_final": pd.concat(finals, ignore_index=True)}
    if save:
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        mean.join(sd, lsuffix="_mean", rsuffix="_sd").to_csv(out / "structure_timeseries.csv")
        stacked.to_csv(out / "structure_per_replica.csv")
        result["clusters_final"].to_csv(out / "clusters_final.csv", index=False)
    return result


# ------------------------------------------------------------------ percolation (box-spanning cluster)
def unwrap_clusters(atoms: pd.DataFrame, bonds: np.ndarray, p: Params):
    """Clusters of the bond graph in one frame: (label, positions, spans). `label` is the cluster of
    every particle (0 ... n_clusters - 1), `positions` are unwrapped per cluster in a periodic box (with
    walls the dumped ones), `spans` (n_clusters x 3, bool) marks the axes along which a cluster spans
    the box. `atoms` from read_dump, `bonds` (atom ID pairs) from read_bonds."""
    ids = atoms["id"].to_numpy()
    pos = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    box = np.array(p.box)
    n = len(ids)
    index = np.empty(ids.max() + 1, dtype=np.int64)
    index[ids] = np.arange(n)
    i, j = (index[bonds[:, 0]], index[bonds[:, 1]]) if len(bonds) else (np.empty(0, int), np.empty(0, int))
    graph = sparse.coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    n_comp, label = connected_components(graph, directed=False)
    spans = np.zeros((n_comp, 3), dtype=bool)
    if not len(bonds):
        return label, pos, spans

    if p.periodic:
        def mic(d):
            return d - box * np.round(d / box)
        # unwrap every cluster along a breadth-first spanning tree, started from a virtual root that
        # is linked to one particle per cluster
        roots = np.unique(label, return_index=True)[1]
        tree = sparse.coo_matrix((np.ones(n_comp), (np.full(n_comp, n), roots)), shape=(n + 1, n + 1))
        full = sparse.bmat([[graph, None], [None, sparse.csr_matrix((1, 1))]]).tocsr() + tree.tocsr()
        order, pred = breadth_first_order(full, n, directed=False, return_predecessors=True)
        depth = np.zeros(n + 1, dtype=np.int64)
        for node in order[1:]:
            depth[node] = depth[pred[node]] + 1
        u = pos.copy()
        nodes = order[1:]
        for d in range(2, depth.max() + 1):                    # depth 1 = cluster roots, kept in place
            level = nodes[depth[nodes] == d]
            u[level] = u[pred[level]] + mic(pos[level] - pos[pred[level]])
        # a bond that disagrees with its minimum image by a box length closes a path around the box
        wraps = np.abs(np.round(((u[j] - u[i]) - mic(pos[j] - pos[i])) / box)) > 0
        np.logical_or.at(spans, label[i], wraps)
        return label, u, spans
    radius = np.where(atoms["type"].to_numpy() == 1, p.radius_qt, p.radius_ft)[:, None]
    lo = np.zeros((n_comp, 3), dtype=bool)
    hi = np.zeros((n_comp, 3), dtype=bool)
    np.logical_or.at(lo, label, pos - radius <= WALL_TOUCH)
    np.logical_or.at(hi, label, box - pos - radius <= WALL_TOUCH)
    return label, pos, lo & hi


def spanning_axes(atoms: pd.DataFrame, bonds: np.ndarray, p: Params) -> str:
    """Axes ("x", "y", "z") along which a single cluster spans the box in one frame, e.g. "" or "xz".
    `atoms` from read_dump, `bonds` (atom ID pairs) from read_bonds."""
    if not len(bonds):
        return ""
    spans = unwrap_clusters(atoms, bonds, p)[2]
    return "".join(axis for axis, s in zip("xyz", spans.any(axis=0)) if s)


def percolation(rep_dir: str | Path) -> pd.DataFrame:
    """Spanning axes of every traj.dump frame of a replica (columns Step, time_ms, axes). Cached in
    rep_XX/analysis/percolation.csv once the replica is done."""
    rep_dir = Path(rep_dir)
    cache = rep_dir / "analysis" / "percolation.csv"
    if cache.exists():
        return pd.read_csv(cache, keep_default_na=False)
    p = Params.load(rep_dir.parent / "params.json")
    bond_frames = read_bonds(rep_dir / "bonds.dump")
    rows = []
    for step, atoms in read_dump(rep_dir / "traj.dump"):
        for bond_step, bonds in bond_frames:                    # bonds.dump has every traj.dump step
            if bond_step == step:
                break
        else:
            raise ValueError(f"step {step} missing in {rep_dir / 'bonds.dump'}")
        rows.append({"Step": step, "time_ms": step * p.timestep * 1e-6, "axes": spanning_axes(atoms, bonds, p)})
    df = pd.DataFrame(rows, columns=["Step", "time_ms", "axes"])
    if (rep_dir / "status").exists() and (rep_dir / "status").read_text().strip() == "done":
        cache.parent.mkdir(exist_ok=True)
        df.to_csv(cache, index=False)
    return df


# ------------------------------------------------------------------ cluster volumes (confocal comparison)
def _cluster_volume(pos: np.ndarray, radius: np.ndarray, probe: np.ndarray, voxel: float) -> tuple[float, float]:
    """Solid and envelope volume [nm^3] of one cluster: union of the spheres (pos, radius) on a voxel
    grid, and its closing with an ellipsoid of radii `probe` (x, y, z)."""
    pad = probe + 2 * voxel                                 # room for the dilation inside the grid
    lo = np.floor(((pos - radius[:, None]).min(axis=0) - pad) / voxel) * voxel    # grid anchored at 0
    shape = np.ceil(((pos + radius[:, None]).max(axis=0) + pad - lo) / voxel).astype(int)
    solid = np.zeros(shape, dtype=bool)
    for x, r in zip(pos, radius):                           # voxel centres inside the sphere
        i0 = np.maximum(np.floor((x - r - lo) / voxel - 0.5).astype(int), 0)
        i1 = np.minimum(np.ceil((x + r - lo) / voxel - 0.5).astype(int) + 1, shape)
        c = [lo[k] + (np.arange(i0[k], i1[k]) + 0.5) * voxel - x[k] for k in range(3)]
        inside = c[0][:, None, None] ** 2 + c[1][None, :, None] ** 2 + c[2][None, None, :] ** 2 <= r * r
        solid[i0[0]:i1[0], i0[1]:i1[1], i0[2]:i1[2]] |= inside
    # distances in units of the probe radii: the ellipsoid becomes the unit sphere
    sampling = voxel / probe
    dilated = ndimage.distance_transform_edt(~solid, sampling=sampling) <= 1
    closed = (ndimage.distance_transform_edt(dilated, sampling=sampling) > 1) | solid
    return solid.sum() * voxel**3, closed.sum() * voxel**3


def cluster_volumes(atoms: pd.DataFrame, bonds: np.ndarray, p: Params, probe_xy_nm: float = PROBE_XY_NM,
                    probe_z_nm: float = PROBE_Z_NM, voxel_nm: float = VOXEL_NM) -> pd.DataFrame:
    """One row per cluster of >= CLUSTER_MIN particles (largest first): size, composition, bonds, the
    axes it spans and its solid and envelope volume [um^3]. A cluster that wraps around the periodic
    box has no defined volume (NaN)."""
    label, pos, spans = unwrap_clusters(atoms, bonds, p)
    is_qt = atoms["type"].to_numpy() == 1
    radius = np.where(is_qt, p.radius_qt, p.radius_ft)
    size = np.bincount(label)
    n_qt = np.bincount(label, weights=is_qt, minlength=len(size)).astype(int)
    index = np.empty(atoms["id"].max() + 1, dtype=np.int64)
    index[atoms["id"].to_numpy()] = np.arange(len(atoms))
    n_bonds = np.bincount(label[index[bonds[:, 0]]], minlength=len(size)) if len(bonds) else np.zeros(len(size), int)
    probe = np.array([probe_xy_nm, probe_xy_nm, probe_z_nm])
    rows = []
    for c in np.argsort(-size, kind="stable"):
        if size[c] < CLUSTER_MIN:
            break
        axes = "".join(a for a, s in zip("xyz", spans[c]) if s)
        v_solid = v_env = np.nan
        if not (axes and p.periodic):
            members = label == c
            v_solid, v_env = _cluster_volume(pos[members], radius[members], probe, voxel_nm)
        rows.append({"size": size[c], "n_qt": n_qt[c], "n_ft": size[c] - n_qt[c], "n_bonds": n_bonds[c],
                     "spans": axes, "v_solid_um3": v_solid * 1e-9, "v_envelope_um3": v_env * 1e-9})
    columns = ["size", "n_qt", "n_ft", "n_bonds", "spans", "v_solid_um3", "v_envelope_um3"]
    return pd.DataFrame(rows, columns=columns)


def cluster_volumes_run(run_dir: str | Path, probe_xy_nm: float = PROBE_XY_NM, probe_z_nm: float = PROBE_Z_NM,
                        voxel_nm: float = VOXEL_NM, save: bool = True) -> pd.DataFrame:
    """Cluster sizes and volumes of all finished replicas in the traj.dump frame at the end of the
    influx (the last frame at or before it; the last frame without an influx). Saved to
    <run>/analysis/clusters_influx_end.csv."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    tables = []
    for rep in sorted(run_dir.glob("rep_*")):
        if not ((rep / "status").exists() and (rep / "status").read_text().strip() == "done"):
            continue
        step = max(s for s in dump_steps(rep / "traj.dump") if s <= p.influx_end_step)
        _, atoms = next(read_dump(rep / "traj.dump", start=step))
        b_step, bonds = next(read_bonds(rep / "bonds.dump", start=step))
        if b_step != step:
            raise ValueError(f"step {step} missing in {rep / 'bonds.dump'}")
        table = cluster_volumes(atoms, bonds, p, probe_xy_nm, probe_z_nm, voxel_nm)
        tables.append(table.assign(replica=rep.name, step=step, time_s=step * p.timestep * 1e-9,
                                   cluster=np.arange(1, len(table) + 1)))
    if not tables:
        raise ValueError(f"no finished replicas in {run_dir}")
    result = pd.concat(tables, ignore_index=True).assign(probe_xy_nm=probe_xy_nm, probe_z_nm=probe_z_nm,
                                                         voxel_nm=voxel_nm)
    result = result[["replica", "step", "time_s", "cluster", *tables[0].columns[:-4], "probe_xy_nm",
                     "probe_z_nm", "voxel_nm"]]
    if save:
        (run_dir / "analysis").mkdir(exist_ok=True)
        result.to_csv(run_dir / "analysis" / "clusters_influx_end.csv", index=False)
    return result
