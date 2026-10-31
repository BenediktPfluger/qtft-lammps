"""Spatial statistics of the encapsulins (Qt) for the comparison with 3D FIB-SEM, which resolves Qt but not
ferritin (numpy/scipy only, read from traj.dump: positions and the bond-connected cluster column).

Two frames per replica: the first traj.dump frame (production start, binding was off before) and the
frame at the end of the influx (the last frame at or before Params.influx_end_step, as for the agglomerate
distributions). Distances use the minimum image in a periodic box and plain distances with walls.

- Intra-agglomerate nearest neighbour (`qt_nn_intra`): for every Qt in an agglomerate with >= QT_MIN Qt,
  the distance to the nearest Qt of the same agglomerate.
- Pair correlation g(r) of all Qt (`qt_pair_correlation`), normalised to their density in the full box
  volume; with walls there is no edge correction, so g(r) falls below its true value at large r.
- Clark-Evans index (`clark_evans`) of all Qt: R = mean nearest-neighbour distance / its expectation for
  randomly placed points of the same density, Gamma(4/3) (3 / (4 pi rho))^(1/3) = 0.554 rho^(-1/3).
  R = 1 random, R < 1 clustered, R > 1 more regular than random.
- Ripley's K of all Qt (`qt_ripley_l`), as Besag's L(r) - r with L = (3 K / (4 pi))^(1/3): 0 random, > 0
  clustered at scale r, < 0 more regular. K(r) = mean number of other Qt within r of a Qt / rho; exact up
  to L/2 in a periodic box, border correction with walls (only Qt at least r from every wall are centres).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from .ensemble import replica_series
from .params import Params
from .read import dump_steps, read_dump
from .remote import require_fetched

QT_MIN = 2              # Qt per agglomerate for the intra-agglomerate distances (as resolved in SEM)
RDF_BIN_NM = 5.0        # bin width of g(r) [nm]
RDF_MAX_NM = 1000.0     # largest r of g(r) and L(r) [nm] (at most half the shortest box edge)
FRAMES = ("first", "influx_end")
OUTPUTS = {"nn": "qt_nn_distances.csv", "rdf": "qt_pair_correlation.csv", "clark_evans": "qt_clark_evans.csv",
           "ripley": "qt_ripley_l.csv"}


def _qt(atoms: pd.DataFrame, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Positions (wrapped into [0, L) in a periodic box) and cluster IDs of the Qt of one frame."""
    qt = atoms[atoms["type"].to_numpy(dtype=int) == 1]
    pos = qt[["x", "y", "z"]].to_numpy(dtype=float)
    if p.periodic:
        box = np.array(p.box)
        pos = np.mod(pos, box)
        pos[pos >= box] = 0.0                                   # mod can round up to exactly L
    return pos, qt["cluster"].to_numpy(dtype=int)


def _tree(pos: np.ndarray, p: Params) -> cKDTree:
    return cKDTree(pos, boxsize=np.array(p.box) if p.periodic else None)


def _in_agglomerates(cluster: np.ndarray, min_qt: int = QT_MIN) -> np.ndarray:
    """Mask of the Qt whose cluster holds >= min_qt Qt."""
    _, inverse, counts = np.unique(cluster, return_inverse=True, return_counts=True)
    return counts[inverse] >= min_qt


def qt_nn_intra(atoms: pd.DataFrame, p: Params, min_qt: int = QT_MIN) -> np.ndarray:
    """Distance [nm] from every Qt in an agglomerate with >= min_qt Qt to the nearest Qt of the same
    agglomerate."""
    pos, cluster = _qt(atoms, p)
    keep = _in_agglomerates(cluster, min_qt)
    pos, cluster = pos[keep], cluster[keep]
    out = []
    for c in np.unique(cluster):
        members = pos[cluster == c]
        out.append(_tree(members, p).query(members, k=2)[0][:, 1])
    return np.concatenate(out) if out else np.empty(0)


def qt_nn_all(atoms: pd.DataFrame, p: Params) -> np.ndarray:
    """Distance [nm] from every Qt to the nearest other Qt (no cluster information)."""
    pos, _ = _qt(atoms, p)
    return _tree(pos, p).query(pos, k=2)[0][:, 1] if len(pos) > 1 else np.empty(0)


def _r_edges(p: Params, bin_nm: float, r_max_nm: float) -> np.ndarray:
    return np.arange(0.0, min(r_max_nm, min(p.box) / 2) + bin_nm / 2, bin_nm)


def qt_pair_correlation(atoms: pd.DataFrame, p: Params, bin_nm: float = RDF_BIN_NM,
                        r_max_nm: float = RDF_MAX_NM) -> pd.DataFrame:
    """g(r) of all Qt: columns r_nm (bin centre), g. Ordered pairs per shell / (N (N - 1) / V * shell volume),
    V = full box volume."""
    pos, _ = _qt(atoms, p)
    edges = _r_edges(p, bin_nm, r_max_nm)
    r = 0.5 * (edges[1:] + edges[:-1])
    n = len(pos)
    if n < 2:
        return pd.DataFrame({"r_nm": r, "g": np.full(len(r), np.nan)})
    tree = _tree(pos, p)
    pairs = np.diff(tree.count_neighbors(tree, edges).astype(float))          # ordered pairs per shell
    shell = 4 / 3 * math.pi * (edges[1:] ** 3 - edges[:-1] ** 3)
    return pd.DataFrame({"r_nm": r, "g": pairs / (n * (n - 1) / math.prod(p.box) * shell)})


def clark_evans(atoms: pd.DataFrame, p: Params) -> dict:
    """Clark-Evans index of all Qt: n_qt, density_per_um3, mean_nn_nm, expected_nn_nm (random points of the
    same density in the full box volume), R."""
    d = qt_nn_all(atoms, p)
    rho = len(d) / math.prod(p.box)                                              # [1/nm^3]
    expected = math.gamma(4 / 3) * (3 / (4 * math.pi * rho)) ** (1 / 3) if len(d) else np.nan
    mean = float(d.mean()) if len(d) else np.nan
    return {"n_qt": len(d), "density_per_um3": rho * 1e9, "mean_nn_nm": mean, "expected_nn_nm": expected,
            "R": mean / expected}


def qt_ripley_l(atoms: pd.DataFrame, p: Params, bin_nm: float = RDF_BIN_NM,
                r_max_nm: float = RDF_MAX_NM) -> pd.DataFrame:
    """Besag's L(r) - r of all Qt: columns r_nm (bin edges from bin_nm), L_minus_r [nm]. K(r) = V / (N - 1)
    * mean number of other Qt within r of a Qt, V = full box volume; with walls the mean runs over the Qt
    at least r from every wall (border correction, NaN if there are none)."""
    pos, _ = _qt(atoms, p)
    r = _r_edges(p, bin_nm, r_max_nm)[1:]
    n = len(pos)
    if n < 2:
        return pd.DataFrame({"r_nm": r, "L_minus_r": np.full(len(r), np.nan)})
    tree = _tree(pos, p)
    if p.periodic:
        mean_count = (tree.count_neighbors(tree, r).astype(float) - n) / n            # without self pairs
    else:
        border = np.min(np.concatenate([pos, np.array(p.box) - pos], axis=1), axis=1)  # distance to nearest wall
        mean_count = np.array([np.mean(tree.query_ball_point(pos[border >= x], x, return_length=True) - 1)
                               if (border >= x).any() else np.nan for x in r])
    k = math.prod(p.box) / (n - 1) * mean_count
    return pd.DataFrame({"r_nm": r, "L_minus_r": np.cbrt(3 * k / (4 * math.pi)) - r})


def density_estimate(x: np.ndarray):
    """Gaussian kernel density estimate of the sample x, or None if it is undefined (fewer than two
    distinct values, or values too close together for a bandwidth)."""
    x = np.asarray(x, dtype=float)
    if len(np.unique(x)) < 2:
        return None
    try:
        return gaussian_kde(x)
    except np.linalg.LinAlgError:
        return None


def peak(x: np.ndarray, y: np.ndarray | None = None, r_min: float = 0.0) -> float:
    """Position of the maximum: of y(x) for x >= r_min (g(r)); without y, of the kernel density estimate
    of the sample x (nearest-neighbour distances; their median if the estimate is undefined). NaN if
    undefined."""
    x = np.asarray(x, dtype=float)
    if y is None:
        kde = density_estimate(x)
        if kde is None:
            return float(np.median(x)) if len(x) else float("nan")
        grid = np.linspace(x.min(), x.max(), 2000)
        return float(grid[np.argmax(kde(grid))])
    y = np.asarray(y, dtype=float)
    ok = (x >= r_min) & np.isfinite(y)
    return float(x[ok][np.argmax(y[ok])]) if ok.any() else float("nan")


def _frame_steps(dump: Path, p: Params) -> dict[str, int]:
    steps = dump_steps(dump)
    return {"first": steps[0], "influx_end": max(s for s in steps if s <= p.influx_end_step)}


def qt_spatial_run(run_dir: str | Path, bin_nm: float = RDF_BIN_NM, r_max_nm: float = RDF_MAX_NM,
                   save: bool = True) -> dict[str, pd.DataFrame]:
    """Intra-agglomerate and all-Qt nearest-neighbour distances, g(r), the Clark-Evans index and Ripley's
    L(r) - r of all finished replicas in the first frame and at the end of the influx. Returns {"nn", "rdf",
    "clark_evans", "ripley"}; with `save` written to <run>/analysis/qt_nn_distances.csv,
    qt_pair_correlation.csv (g at the end of the influx), qt_clark_evans.csv and qt_ripley_l.csv (long
    format: replica, frame, r_nm, L_minus_r), and reused while they are newer than every traj.dump and hold
    the same replicas and r grid."""
    run_dir = Path(run_dir).expanduser()
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
    reps = list(replica_series(run_dir, names_only=True))
    if not reps:
        raise ValueError(f"no finished replicas in {run_dir}")
    dumps = [run_dir / rep / "traj.dump" for rep in reps]
    paths = {key: run_dir / "analysis" / name for key, name in OUTPUTS.items()}
    if save and all(path.exists() for path in paths.values()):
        newest = max(d.stat().st_mtime for d in dumps)
        if all(path.stat().st_mtime >= newest for path in paths.values()):
            cached = {key: pd.read_csv(path) for key, path in paths.items()}
            r = cached["rdf"]["r_nm"].to_numpy()
            if (sorted(cached["clark_evans"]["replica"].unique()) == reps and len(r) > 1
                    and np.isclose(r[1] - r[0], bin_nm) and r[-1] <= r_max_nm):
                return cached

    nn, rdf, ce, ripley = [], {}, [], []
    for rep, dump in zip(reps, dumps):
        for frame, step in _frame_steps(dump, p).items():
            _, atoms = next(read_dump(dump, start=step))
            info = {"replica": rep, "frame": frame, "step": step, "time_s": step * p.timestep * 1e-9}
            sets = {"all": qt_nn_all(atoms, p)}
            if frame == "influx_end":
                sets["intra"] = qt_nn_intra(atoms, p)
                g = qt_pair_correlation(atoms, p, bin_nm=bin_nm, r_max_nm=r_max_nm)
                rdf["r_nm"], rdf[f"g_{rep}"] = g["r_nm"], g["g"]
            for name, d in sets.items():
                nn.append(pd.DataFrame({**info, "set": name, "distance_nm": d}))
            ce.append({**info, **clark_evans(atoms, p)})
            ripley.append(qt_ripley_l(atoms, p, bin_nm=bin_nm, r_max_nm=r_max_nm).assign(replica=rep, frame=frame))
    rdf = pd.DataFrame(rdf)
    rdf["g_mean"] = rdf[[f"g_{rep}" for rep in reps]].mean(axis=1)
    ripley = pd.concat(ripley, ignore_index=True)[["replica", "frame", "r_nm", "L_minus_r"]]
    result = {"nn": pd.concat(nn, ignore_index=True), "rdf": rdf, "clark_evans": pd.DataFrame(ce),
              "ripley": ripley}
    if save:
        (run_dir / "analysis").mkdir(exist_ok=True)
        for key, path in paths.items():
            result[key].to_csv(path, index=False)
    return result
