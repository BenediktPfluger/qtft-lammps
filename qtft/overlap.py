"""Interpenetration (overlap) diagnostics of the soft repulsion, per pair family.

A pair at distance d overlaps by max(r_i + r_j - d, 0). Bonded and non-bonded pairs are treated
alike. "mean depth all pairs" averages over *all* pairs of a family (also the many pairs far apart)
and is given in % of the contact distance; use it to rank parameter sets.

With walls (boundary "reflect" or "harmonic") the box is not periodic, and the rows "Qt-wall" and
"Ft-wall" give how far particle spheres stick out of the box (depth = r_i - distance of the centre
to the nearest face); they are always 0 for reflecting walls.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .params import Params
from .read import read_dump

FAMILIES = {"Qt-Qt": (1, 1), "Ft-Ft": (2, 2), "Qt-Ft": (1, 2)}
WALL_FAMILIES = {"Qt-wall": 1, "Ft-wall": 2}


def overlap_frame(atoms: pd.DataFrame, p: Params) -> pd.DataFrame:
    """Overlap statistics of one frame (atoms from read_dump), one row per pair family. The depth columns
    describe the overlapping pairs only and are NaN when no pair of the family overlaps."""
    box = np.array(p.box)
    pos = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    types = atoms["type"].to_numpy()
    radius = np.where(types == 1, p.radius_qt, p.radius_ft)
    if p.periodic:
        pos = np.mod(pos, box)
    tree = cKDTree(pos, boxsize=box if p.periodic else None)
    pairs = tree.query_pairs(2 * max(p.radius_qt, p.radius_ft), output_type="ndarray")
    delta = pos[pairs[:, 1]] - pos[pairs[:, 0]]
    if p.periodic:
        delta -= box * np.round(delta / box)
    dist = np.linalg.norm(delta, axis=1)
    contact = radius[pairs[:, 0]] + radius[pairs[:, 1]]
    depth = np.maximum(contact - dist, 0.0)
    ti, tj = np.minimum(types[pairs[:, 0]], types[pairs[:, 1]]), np.maximum(types[pairs[:, 0]], types[pairs[:, 1]])
    n = {1: int((types == 1).sum()), 2: int((types == 2).sum())}

    rows = []
    for family, (a, b) in FAMILIES.items():
        sel = (ti == a) & (tj == b)
        d, dep = dist[sel], depth[sel]
        total = n[a] * n[b] if a != b else n[a] * (n[a] - 1) // 2
        c = (p.radius_qt if a == 1 else p.radius_ft) + (p.radius_qt if b == 1 else p.radius_ft)
        over = dep[dep > 0]
        rows.append({
            "family": family,
            "closest [nm]": d.min() if len(d) else np.nan,
            "n overlapping": len(over),
            "frac overlapping": len(over) / total if total else np.nan,
            "mean depth overlapping [nm]": over.mean() if len(over) else np.nan,
            "mean depth all pairs [% contact]": 100 * dep.sum() / total / c if total else np.nan,
            "p95 depth [nm]": np.percentile(over, 95) if len(over) else np.nan,
            "max depth [nm]": over.max() if len(over) else np.nan,
        })
    if not p.periodic:
        wall_depth = radius - np.minimum(pos, box - pos).min(axis=1)       # r_i - distance to nearest face
        for family, t in WALL_FAMILIES.items():
            dep = np.maximum(wall_depth[types == t], 0.0)
            over, r = dep[dep > 0], (p.radius_qt if t == 1 else p.radius_ft)
            rows.append({
                "family": family,
                "closest [nm]": (r - wall_depth[types == t]).min() if n[t] else np.nan,
                "n overlapping": len(over),
                "frac overlapping": len(over) / n[t] if n[t] else np.nan,
                "mean depth overlapping [nm]": over.mean() if len(over) else np.nan,
                "mean depth all pairs [% contact]": 100 * dep.sum() / n[t] / r if n[t] else np.nan,
                "p95 depth [nm]": np.percentile(over, 95) if len(over) else np.nan,
                "max depth [nm]": over.max() if len(over) else np.nan,
            })
    return pd.DataFrame(rows)


def overlap_series(rep_dir: str | Path, every: int = 10) -> pd.DataFrame:
    """Overlap statistics of every `every`-th trajectory frame of one replica (long format)."""
    rep_dir = Path(rep_dir)
    p = Params.load(rep_dir.parent / "params.json")
    out = []
    for i, (step, atoms) in enumerate(read_dump(rep_dir / "traj.dump")):
        if i % every:
            continue
        df = overlap_frame(atoms, p)
        df.insert(0, "time_ms", step * p.timestep * 1e-6)
        df.insert(0, "Step", step)
        out.append(df)
    return pd.concat(out, ignore_index=True)


def _pool(frames: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Overlap statistics of several frames (and replicas) combined per group: means of the counts and
    fractions; over the frames with an overlap, the mean depth weighted by the number of overlapping pairs
    (the mean over all overlapping pairs), the mean of the per-frame 95th percentiles and the largest depth
    (NaN where nothing overlaps)."""
    numeric = [c for c in frames.columns if c not in ("family", "replica", *keys)]
    by = [frames[k] for k in keys]
    out = frames.groupby(by)[numeric].mean()
    n = frames["n overlapping"].groupby(by).sum()
    weighted = (frames["mean depth overlapping [nm]"] * frames["n overlapping"]).groupby(by).sum()
    out["mean depth overlapping [nm]"] = (weighted / n).where(n > 0)
    out["max depth [nm]"] = frames["max depth [nm]"].groupby(by).max()
    return out


def overlap_run(run_dir: str | Path, every: int = 10, last_frames: int = 5, save: bool = True) -> dict:
    """Time series (pooled over replicas) and pooled statistics of the last `last_frames` analysed
    frames of all finished replicas (_pool: depths over the overlapping pairs only). Saved to
    <run>/analysis/overlap_*.csv."""
    run_dir = Path(run_dir).expanduser()
    series = []
    for rep in sorted(run_dir.glob("rep_*")):
        if (rep / "status").exists() and (rep / "status").read_text().strip() == "done":
            series.append(overlap_series(rep, every=every).assign(replica=rep.name))
    if not series:
        raise ValueError(f"no finished replicas in {run_dir}")
    all_frames = pd.concat(series, ignore_index=True)
    timeseries = _pool(all_frames, ["family", "Step"]).reset_index()
    last_steps = sorted(all_frames["Step"].unique())[-last_frames:]
    final = (_pool(all_frames[all_frames["Step"].isin(last_steps)], ["family"]).drop(columns=["Step", "time_ms"])
             .reindex([f for f in (*FAMILIES, *WALL_FAMILIES) if f in set(all_frames["family"])]))
    result = {"timeseries": timeseries, "final": final, "per_frame": all_frames}
    if save:
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        timeseries.to_csv(out / "overlap_timeseries.csv", index=False)
        final.to_csv(out / "overlap_final.csv")
    return result
