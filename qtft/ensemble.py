"""Ensemble statistics over the replicas of a run, and the index of all runs in the data folder.

All time series come from the production thermo data in log.lammps (computed by LAMMPS
during the run). Averages are mean +- population SD over replicas on a common time grid.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .params import Params, data_path
from .read import load_thermo

SKIP_COLUMNS = {"Step", "time_ms", "S/CPU", "CPULeft"}


# ------------------------------------------------------------------ per-replica data
def replica_series(run_dir: str | Path, only_done: bool = True) -> dict[str, pd.DataFrame]:
    """Production thermo data of every finished replica, with derived fraction columns."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    n_total = p.n_qt + p.n_ft
    out = {}
    for rep in sorted(run_dir.glob("rep_*")):
        status = (rep / "status").read_text().strip() if (rep / "status").exists() else ""
        if only_done and status != "done":
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
    sd = {c: traces[c].std(axis=0) for c in columns}

    per_rep = []
    for name, df in series.items():
        tt = df["time_ms"].to_numpy()
        bonds = df["Bonds"].to_numpy()
        largest = df["largest_frac"].to_numpy()
        per_rep.append({
            "replica": name,
            "t_half_bonds_ms": half_time(tt, bonds, 0.5 * bonds[-1]) if bonds[-1] > 0 else float("nan"),
            "t_half_qt_ms": half_time(tt, df["frac_bound_qt"].to_numpy(), 0.5),
            "t_half_ft_ms": half_time(tt, df["frac_bound_ft"].to_numpy(), 0.5),
            "percolated": bool(largest[-1] >= 0.5),
            "t_percolation_ms": half_time(tt, largest, 0.5),
            **{f"final_{c}": float(df[c].iloc[-1]) for c in columns},
        })
    per_rep = pd.DataFrame(per_rep).set_index("replica")

    final = pd.DataFrame({"mean": per_rep.mean(numeric_only=True),
                          "sd": per_rep.std(numeric_only=True, ddof=0),
                          "n": per_rep.count(numeric_only=True)})
    final.loc["percolated_replicas", ["mean", "sd", "n"]] = [per_rep["percolated"].sum(), 0, len(per_rep)]

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


def load_stats(run_dir: str | Path) -> dict:
    """Read a saved ensemble_stats.json back (arrays as numpy, tables as DataFrames)."""
    d = json.loads((Path(run_dir).expanduser() / "analysis" / "ensemble_stats.json").read_text())
    return {
        "replicas": d["replicas"],
        "time_ms": np.array(d["time_ms"]),
        "mean": {c: np.array(v) for c, v in d["mean"].items()},
        "sd": {c: np.array(v) for c, v in d["sd"].items()},
        "traces": {c: np.array(v) for c, v in d["traces"].items()},
        "per_replica": pd.DataFrame.from_dict(d["per_replica"], orient="index"),
        "final": pd.DataFrame.from_dict(d["final"], orient="index"),
    }


# ------------------------------------------------------------------ index of all runs
# settings that do not change the physics; ignored when listing or labelling parameter differences
TECHNICAL = {"data_dir", "group", "run_name", "n_replicas", "mpi_procs", "max_parallel",
             "lmp_command", "mpirun_command", "master_seed"}
INDEX_PARAMS = ("n_qt", "n_ft", "box_length", "k_qq", "k_ff", "k_qf", "bond_k", "k_on",
                "timestep", "prod_steps", "n_replicas")


def update_index(data_dir: str | Path = "runs") -> pd.DataFrame:
    """Rebuild <data_dir>/index.csv: one line per run folder (single and ensemble) with status
    and changed parameters."""
    data_dir = data_path(data_dir)
    defaults = Params().to_dict()
    rows = []
    for record_file in sorted(data_dir.glob("*/*/params.json")):
        if record_file.parent.parent.name not in ("single", "ensemble"):
            continue
        record = json.loads(record_file.read_text())
        params = record["params"]
        statuses = [(rep / "status").read_text().strip() if (rep / "status").exists() else "?"
                    for rep in sorted(record_file.parent.glob("rep_*"))]
        changed = [f"{k}={v}" for k, v in params.items()
                   if k not in TECHNICAL and v != defaults.get(k)]
        rows.append({
            "group": record_file.parent.parent.name,
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
                 if f.name not in TECHNICAL
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
        stats_file = r / "analysis" / "ensemble_stats.json"
        final = load_stats(r)["final"] if stats_file.exists() else ensemble_stats(r)["final"]
        cols[label] = final.apply(lambda row: f"{row['mean']:.4g} ± {row['sd']:.2g}", axis=1)
    table = pd.DataFrame(cols)
    table.attrs["differing parameters"] = differing
    return table
