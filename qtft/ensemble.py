"""Ensemble statistics over the replicas of a run, and the index of all runs in the data folder.

All time series come from the production thermo data in log.lammps (computed by LAMMPS
during the run). Averages are mean +- sample SD (ddof = 1) over replicas on a common time grid;
with a single replica the SD is NaN.
"""

from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from .params import EXECUTION_SETTINGS, N_A, NM3_TO_L, Params, data_path
from .read import load_thermo, read_bonds, read_dump
from .structure import percolation

SKIP_COLUMNS = {"Step", "time_ms", "S/CPU", "CPULeft"}
TO_PER_M_S = NM3_TO_L * N_A * 1e9          # rate constant: 1 nm^3/ns = 6.02e8 1/(M s)


# ------------------------------------------------------------------ per-replica data
def replica_series(run_dir: str | Path, only_done: bool = True, names_only: bool = False) -> dict:
    """Production thermo data of every finished replica, with derived fraction columns
    (`names_only`: only the replica names as keys, nothing read)."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    n_total = p.n_qt + p.n_ft
    out = {}
    for rep in sorted(run_dir.glob("rep_*")):
        status = (rep / "status").read_text().strip() if (rep / "status").exists() else ""
        if only_done and status != "done":
            continue
        if names_only:
            out[rep.name] = None
            continue
        df = load_thermo(rep)
        df["frac_bound_qt"] = df["n_qt_bound"] / p.n_qt
        df["frac_bound_ft"] = df["n_ft_bound"] / p.n_ft
        df["largest_frac"] = df["largest"] / n_total
        df["mean_size_frac"] = df["mean_size"] / n_total
        out[rep.name] = df
    return out


def common_grid(series: dict[str, pd.DataFrame], column: str = "time_ms") -> np.ndarray:
    """Shared time grid: identical grids pass through; otherwise the first replica's points
    inside the overlap of all replicas (never extrapolated)."""
    grids = [df[column].to_numpy() for df in series.values()]
    if all(len(g) == len(grids[0]) and np.allclose(g, grids[0]) for g in grids):
        return grids[0]
    lo, hi = max(g[0] for g in grids), min(g[-1] for g in grids)
    g0 = grids[0]
    return g0[(g0 >= lo) & (g0 <= hi)]


def half_time(t: np.ndarray, y: np.ndarray, level: float) -> float:
    """First time y reaches `level` (linear interpolation between frames); NaN if never."""
    above = np.nonzero(y >= level)[0]
    if len(above) == 0:
        return float("nan")
    i = above[0]
    if i == 0:
        return float(t[0])
    return float(t[i - 1] + (level - y[i - 1]) * (t[i] - t[i - 1]) / (y[i] - y[i - 1]))


# ------------------------------------------------------------------ ensemble statistics
def ensemble_stats(run_dir: str | Path, save: bool = True) -> dict:
    """Mean/SD time series, per-replica traces and final-state scalars of a run.

    Percolation = a cluster spans the box along at least one axis (structure.percolation, from the
    trajectory, cached per replica): `percolated` in the last frame (the mean over replicas is the
    fraction of percolated replicas) and `t_percolation_ms`, the first frame with a spanning cluster.

    Saved to <run>/analysis/: ensemble_stats.json, ensemble_timeseries.csv, final_state.csv.
    """
    run_dir = Path(run_dir).expanduser()
    series = replica_series(run_dir)
    if not series:
        raise ValueError(f"no finished replicas in {run_dir}")
    t = common_grid(series)
    columns = [c for c in next(iter(series.values())).columns if c not in SKIP_COLUMNS]
    traces = {c: np.array([np.interp(t, df["time_ms"], df[c]) for df in series.values()])
              for c in columns}

    mean = {c: traces[c].mean(axis=0) for c in columns}
    sd = {c: traces[c].std(axis=0, ddof=1) if len(series) > 1 else np.full(len(t), np.nan)
          for c in columns}

    per_rep = []
    for name, df in series.items():
        tt = df["time_ms"].to_numpy()
        bonds = df["Bonds"].to_numpy()
        spanning = percolation(run_dir / name)
        spanned = spanning.loc[spanning["axes"] != "", "time_ms"]
        per_rep.append({
            "replica": name,
            "t_half_bonds_ms": half_time(tt, bonds, 0.5 * bonds[-1]) if bonds[-1] > 0 else float("nan"),
            "t_half_qt_ms": half_time(tt, df["frac_bound_qt"].to_numpy(), 0.5),
            "t_half_ft_ms": half_time(tt, df["frac_bound_ft"].to_numpy(), 0.5),
            "percolated": float(spanning["axes"].iloc[-1] != "") if len(spanning) else float("nan"),
            "t_percolation_ms": float(spanned.iloc[0]) if len(spanned) else float("nan"),
            **{f"final_{c}": float(df[c].iloc[-1]) for c in columns},
        })
    per_rep = pd.DataFrame(per_rep).set_index("replica")

    final = pd.DataFrame({"mean": per_rep.mean(numeric_only=True),
                          "sd": per_rep.std(numeric_only=True, ddof=1),
                          "n": per_rep.count(numeric_only=True)})

    result = {"time_ms": t, "mean": mean, "sd": sd, "traces": traces,
              "replicas": list(series), "per_replica": per_rep, "final": final}
    if save:
        _save(run_dir, result)
    return result


def _save(run_dir: Path, r: dict) -> None:
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    ts = {"time_ms": r["time_ms"]}
    for c in r["mean"]:
        ts[f"{c}_mean"] = r["mean"][c]
        ts[f"{c}_sd"] = r["sd"][c]
    pd.DataFrame(ts).to_csv(out / "ensemble_timeseries.csv", index=False)
    r["final"].to_csv(out / "final_state.csv")
    payload = {
        "replicas": r["replicas"],
        "time_ms": r["time_ms"].tolist(),
        "mean": {c: v.tolist() for c, v in r["mean"].items()},
        "sd": {c: v.tolist() for c, v in r["sd"].items()},
        "traces": {c: v.tolist() for c, v in r["traces"].items()},
        "per_replica": json.loads(r["per_replica"].to_json(orient="index")),
        "final": json.loads(r["final"].to_json(orient="index")),
    }
    (out / "ensemble_stats.json").write_text(json.dumps(payload))


def cached_stats(run_dir: str | Path) -> dict:
    """The saved ensemble_stats.json if it covers exactly the replicas finished now, otherwise
    ensemble_stats() computed afresh (and saved)."""
    run_dir = Path(run_dir).expanduser()
    if (run_dir / "analysis" / "ensemble_stats.json").exists():
        stats = load_stats(run_dir)
        if stats["replicas"] == list(replica_series(run_dir, names_only=True)):
            return stats
    return ensemble_stats(run_dir)


def load_stats(run_dir: str | Path) -> dict:
    """Read a saved ensemble_stats.json back (arrays as numpy, tables as DataFrames)."""
    d = json.loads((Path(run_dir).expanduser() / "analysis" / "ensemble_stats.json").read_text())
    return {
        "replicas": d["replicas"],
        "time_ms": np.array(d["time_ms"]),
        "mean": {c: np.array(v) for c, v in d["mean"].items()},
        "sd": {c: np.array(v) for c, v in d["sd"].items()},
        "traces": {c: np.array(v) for c, v in d["traces"].items()},
        # to_json writes NaN as null: back to float
        "per_replica": pd.DataFrame.from_dict(d["per_replica"], orient="index").astype(float),
        "final": pd.DataFrame.from_dict(d["final"], orient="index").astype(float),
    }


# ------------------------------------------------------------------ binding rate (Collins-Kimball)
def collins_kimball(p: Params) -> dict:
    """Continuum estimate of the contact reaction [nm^3/ns]: binding happens in the overlap layer at
    contact R = r_Qt + r_Ft, Boltzmann-weighted thickness sigma sqrt(pi/2), sigma^2 = kT/(2 k_QF)/(1 - alpha/2).
    kappa = k_on 4 pi R^2 sigma sqrt(pi/2), k_D = 4 pi (D_Qt + D_Ft) R, k_a = k_D kappa / (k_D + kappa).
    Valid only while the relative step per time step is small against sigma; binding_rate() counts the
    contact pairs instead."""
    R = p.cut_qf
    alpha = p.alpha(p.k_qf, p.gamma_qt, p.gamma_ft)
    layer = math.sqrt(p.kT / (2 * p.k_qf) / (1 - alpha / 2)) * math.sqrt(math.pi / 2)
    volume = 4 * math.pi * R**2 * layer
    kappa = p.k_on * volume
    k_d = 4 * math.pi * (p.D_qt + p.D_ft) * R
    return {"layer_nm": layer, "volume": volume, "kappa": kappa, "k_D": k_d, "k_a": k_d * kappa / (k_d + kappa)}


def contact_pairs(atoms: pd.DataFrame, bonds: np.ndarray, p: Params) -> int:
    """Number of Qt-Ft pairs closer than R_bind that are not bonded (binding candidates) in one frame."""
    box = np.array(p.box)
    pos = atoms[["x", "y", "z"]].to_numpy(dtype=float)
    if p.periodic:
        pos = np.mod(pos, box)
    types, ids = atoms["type"].to_numpy(), atoms["id"].to_numpy()
    qt, ft = types == 1, types == 2
    boxsize = box if p.periodic else None
    near = cKDTree(pos[qt], boxsize=boxsize).query_ball_tree(cKDTree(pos[ft], boxsize=boxsize), p.r_bind)
    qt_ids, ft_ids = ids[qt], ids[ft]
    pairs = {(int(qt_ids[i]), int(ft_ids[j])) for i, js in enumerate(near) for j in js}
    bonded = {(int(a), int(b)) for a, b in bonds} | {(int(b), int(a)) for a, b in bonds}
    return len(pairs - bonded)


def binding_rate(run_dir: str | Path, max_frac_bound: float = 0.1, save: bool = True) -> dict:
    """Association rate constant k_a of every finished replica, measured and predicted.

    Measured: k_a = (dN_bonds/dt)|_0 * V / (N_Qt N_Ft), the slope of the bond count (fit through the
    origin) over the production frames up to a fraction bound of `max_frac_bound` (at least the first
    3 frames), where almost all particles are still free.
    Predicted (Collins-Kimball with the counted contact pairs): kappa = rate per pair * pairs * V /
    (N_Qt N_Ft), with pairs = Qt-Ft pairs within R_bind in the first production frame (equilibrium contact
    density: end of the equilibration, no bonds yet) and rate per pair = p_bind / (bind_every dt);
    k_a = k_D kappa / (k_D + kappa), k_D = 4 pi D R (steady state). The measured early rate starts at kappa
    and approaches k_a within ~R^2/D (2.3 ms at the defaults), so it can lie slightly above k_a.
    The continuum estimate (collins_kimball) is listed for comparison.

    Returns {"per_replica", "summary", "window_ms"}. Summary rows (value, unit, 1/(M s)): k_a measured
    (+ SD), k_a predicted (counted contact pairs, + SD), kappa, k_D, k_a continuum estimate, contact volume
    counted and continuum [nm^3]; saved to <run>/analysis/binding_rate.csv.
    """
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    series = replica_series(run_dir)
    if not series:
        raise ValueError(f"no finished replicas in {run_dir}")
    per_volume = math.prod(p.box) / (p.n_qt * p.n_ft)                  # V / (N_Qt N_Ft)
    rate_per_pair = p.p_bind / (p.bind_every * p.timestep)
    continuum = collins_kimball(p)
    k_d = continuum["k_D"]
    rows = []
    for name, df in series.items():
        early = df[df["frac_bound"] <= max_frac_bound]
        if len(early) < 3:
            early = df.iloc[:3]
        if (early["time_ms"] > 0).sum() < 2:
            raise ValueError(f"{run_dir / name}: fewer than 2 production frames, cannot fit the binding rate")
        t, b = early["time_ms"].to_numpy() * 1e6, early["Bonds"].to_numpy()          # ns, bonds
        slope = (t * b).sum() / (t * t).sum()                                         # bonds/ns, N(0) = 0

        step, atoms = next(read_dump(run_dir / name / "traj.dump"))
        bond_step, bonds = next(read_bonds(run_dir / name / "bonds.dump"))
        if step != 0 or bond_step != 0:
            raise ValueError(f"{run_dir / name}: the dumps do not start at production step 0")
        pairs = contact_pairs(atoms, bonds, p)
        kappa = rate_per_pair * pairs * per_volume
        rows.append({"replica": name, "window_ms": t[-1] * 1e-6, "frames": len(t), "contact pairs": pairs,
                     "k_a measured": slope * per_volume,
                     "kappa": kappa, "k_a predicted": k_d * kappa / (k_d + kappa)})
    per_rep = pd.DataFrame(rows).set_index("replica")

    def sd(col):
        return per_rep[col].std(ddof=1) if len(per_rep) > 1 else float("nan")

    rates = {"k_a measured": per_rep["k_a measured"].mean(), "k_a measured SD": sd("k_a measured"),
             "k_a predicted": per_rep["k_a predicted"].mean(), "k_a predicted SD": sd("k_a predicted"),
             "kappa": per_rep["kappa"].mean(), "k_D": k_d, "k_a continuum estimate": continuum["k_a"]}
    summary = pd.DataFrame({"value": rates, "unit": "nm^3/ns"})
    summary["1/(M s)"] = summary["value"] * TO_PER_M_S
    volumes = pd.DataFrame({"value": {"contact volume counted": per_rep["contact pairs"].mean() * per_volume,
                                      "contact volume continuum": continuum["volume"]}, "unit": "nm^3"})
    summary = pd.concat([summary, volumes]).rename_axis("quantity")
    result = {"per_replica": per_rep, "summary": summary, "window_ms": float(per_rep["window_ms"].max())}
    if save:
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        summary.to_csv(out / "binding_rate.csv")
    return result


# ------------------------------------------------------------------ index of all runs
INDEX_PARAMS = ("n_qt", "n_ft", "box_length", "boundary", "k_qq", "k_ff", "k_qf", "bond_k", "k_on",
                "timestep", "prod_steps", "n_replicas")


def update_index(data_dir: str | Path = "runs") -> pd.DataFrame:
    """Rebuild <data_dir>/index.csv: one line per run folder (single, ensemble and sweep runs) with
    status and changed parameters."""
    data_dir = data_path(data_dir)
    defaults = Params().to_dict()
    rows = []
    records = [f for f in sorted(data_dir.glob("*/*/params.json"))
               if f.parent.parent.name in ("single", "ensemble")]
    records += sorted(data_dir.glob("sweeps/*/*/params.json"))
    for record_file in records:
        record = json.loads(record_file.read_text())
        params = record["params"]
        statuses = [(rep / "status").read_text().strip() if (rep / "status").exists() else "?"
                    for rep in sorted(record_file.parent.glob("rep_*"))]
        changed = [f"{k}={v}" for k, v in params.items()
                   if k not in EXECUTION_SETTINGS and v != defaults.get(k)]
        rows.append({
            "group": record_file.parent.parent.relative_to(data_dir).as_posix(),
            "run": record_file.parent.name,
            "created": record.get("created"),
            "status": ", ".join(f"{s}:{statuses.count(s)}" for s in sorted(set(statuses))),
            **{k: params.get(k) for k in INDEX_PARAMS},
            "changed vs defaults": "; ".join(changed),
            "git": (record.get("git") or {}).get("commit"),
        })
    df = pd.DataFrame(rows)
    if rows:
        df.to_csv(data_dir / "index.csv", index=False)
    return df


def differing_labels(run_dirs) -> tuple[list[str], list[str]]:
    """Labels of runs built from the physical parameters that differ between them."""
    run_dirs = [Path(r).expanduser() for r in run_dirs]
    params = [Params.load(r / "params.json") for r in run_dirs]
    differing = [f.name for f in dataclasses.fields(Params)
                 if f.name not in EXECUTION_SETTINGS
                 and len({json.dumps(getattr(q, f.name)) for q in params}) > 1]
    labels = [", ".join(f"{k}={getattr(q, k)}" for k in differing) or r.name
              for r, q in zip(run_dirs, params)]
    return labels, differing


def compare(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Side-by-side final-state table (mean +- SD) of several runs, with the parameters that differ."""
    run_dirs = [Path(r).expanduser() for r in run_dirs]
    labels, differing = differing_labels(run_dirs)
    cols = {}
    for r, label in zip(run_dirs, labels):
        final = cached_stats(r)["final"]
        cols[label] = final.apply(lambda row: f"{row['mean']:.4g} ± {row['sd']:.2g}", axis=1)
    table = pd.DataFrame(cols)
    table.attrs["differing parameters"] = differing
    return table
