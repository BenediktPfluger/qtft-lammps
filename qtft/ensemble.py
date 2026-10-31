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
from .remote import require_fetched
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


def deagglomeration(df: pd.DataFrame, p: Params) -> dict:
    """Peaks of bonds and of the largest cluster, and the deagglomeration half-times: time after the end
    of the influx (start of the pump-out) until bonds / the largest cluster fall to half of their value
    at that moment (NaN without influx or pump-out, or if never reached) [ms]."""
    t = df["time_ms"].to_numpy()
    out = {}
    for key, col in (("bonds", "Bonds"), ("largest", "largest")):
        y = df[col].to_numpy()
        out[f"{key}_peak"] = float(y.max())
        out[f"t_{key}_peak_ms"] = float(t[y.argmax()])
        t_off = p.t_off_s * 1e3
        half = float("nan")
        if p.t_influx_s > 0 and p.t_pumpout_s > 0:
            after = t >= t_off
            y0 = np.interp(t_off, t, y)
            if y0 > 0:
                half = half_time(t[after], -y[after], -0.5 * y0) - t_off            # first fall below y0 / 2
        out[f"t_half_unbind_{key}_ms"] = half
    return out


# ------------------------------------------------------------------ ensemble statistics
def ensemble_stats(run_dir: str | Path, save: bool = True) -> dict:
    """Mean/SD time series, per-replica traces and final-state scalars of a run.

    Percolation = a cluster spans the box along at least one axis (structure.percolation, from the
    trajectory, cached per replica): `percolated` in the last frame (the mean over replicas is the
    fraction of percolated replicas) and `t_percolation_ms`, the first frame with a spanning cluster.
    Calcium protocol (deagglomeration): peaks of bonds and largest cluster and the half-times after the
    end of the influx. "final" values are those at the end of the whole protocol.

    Saved to <run>/analysis/: ensemble_stats.json, ensemble_timeseries.csv, final_state.csv.
    """
    run_dir = Path(run_dir).expanduser()
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
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
            **deagglomeration(df, p),
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


def ensemble_stats_cached(run_dir: str | Path) -> dict:
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
    """Continuum estimate of the contact reaction during the influx [nm^3/ns]: binding happens in the
    overlap layer at contact R = r_Qt + r_Ft, Boltzmann-weighted thickness sigma sqrt(pi/2),
    sigma^2 = kT/(2 k_QF)/(1 - alpha/2). kappa = k_on f_peak 4 pi R^2 sigma sqrt(pi/2) (f_peak: calmodulin
    loading at peak calcium), k_D = 4 pi (D_Qt + D_Ft) R, k_a = k_D kappa / (k_D + kappa).
    Valid only while the relative step per time step is small against sigma; binding_rate() counts the
    contact pairs instead."""
    R = p.cut_qf
    alpha = p.alpha(p.k_qf, p.gamma_qt, p.gamma_ft)
    layer = math.sqrt(p.kT / (2 * p.k_qf) / (1 - alpha / 2)) * math.sqrt(math.pi / 2)
    volume = 4 * math.pi * R**2 * layer
    kappa = p.k_on * float(p.loading(p.ca_peak_uM)) * volume
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
    """Association rate constant k_a during the calcium influx of every finished replica, measured and
    predicted (needs an influx phase with a calcium step, ca_rise_s = 0: the fit assumes the peak rate from
    the influx start on).

    Measured: k_a = (dN_bonds/dt) at the influx start * V / (N_Qt N_Ft), the slope of the bond count (fit
    through the value at the influx start) over the influx frames up to an increase of the fraction bound
    by `max_frac_bound` (at least the first 3 frames), where almost all particles are still free.
    Predicted (Collins-Kimball with the counted contact pairs): kappa = rate per pair * pairs * V /
    (N_Qt N_Ft), with pairs = unbonded Qt-Ft pairs within R_bind in the traj.dump frame at the influx start
    (contact density before the influx: after the equilibration and the baseline, where binding is weak)
    and rate per pair = p_on / (bind_every dt) at peak calcium;
    k_a = k_D kappa / (k_D + kappa), k_D = 4 pi D R (steady state). The measured early rate starts at kappa
    and approaches k_a within ~R^2/D (2.3 ms at the defaults), so it can lie slightly above k_a.
    The continuum estimate (collins_kimball) is listed for comparison.

    Returns {"per_replica", "summary", "window_ms"}. Summary rows (value, unit, 1/(M s)): k_a measured
    (+ SD), k_a predicted (counted contact pairs, + SD), kappa, k_D, k_a continuum estimate, contact volume
    counted and continuum [nm^3]; saved to <run>/analysis/binding_rate.csv.
    """
    run_dir = Path(run_dir).expanduser()
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
    if p.t_influx_s <= 0:
        raise ValueError(f"{run_dir}: no influx phase, the binding rate is measured during the influx")
    if p.ca_rise_s > 0:
        raise ValueError(f"{run_dir}: calcium rises over ca_rise_s = {p.ca_rise_s:g} s; the binding rate is "
                         "measured with a calcium step (ca_rise_s = 0)")
    series = replica_series(run_dir)
    if not series:
        raise ValueError(f"no finished replicas in {run_dir}")
    per_volume = math.prod(p.box) / (p.n_qt * p.n_ft)                  # V / (N_Qt N_Ft)
    on_step, off_step = p.phase_steps[0], p.phase_steps[0] + p.phase_steps[1]
    p_on = 1.0 - math.exp(-p.k_on * float(p.loading(p.ca_peak_uM)) * p.timestep * p.bind_every)
    rate_per_pair = p_on / (p.bind_every * p.timestep)
    continuum = collins_kimball(p)
    k_d = continuum["k_D"]
    rows = []
    for name, df in series.items():
        influx = df[(df["Step"] >= on_step) & (df["Step"] <= off_step)].reset_index(drop=True)
        if len(influx) < 3 or influx["Step"].iloc[0] != on_step:
            raise ValueError(f"{run_dir / name}: fewer than 3 frames in the influx, cannot fit the binding rate")
        early = influx[influx["frac_bound"] - influx["frac_bound"].iloc[0] <= max_frac_bound]
        if len(early) < 3:
            early = influx.iloc[:3]
        t = (early["Step"].to_numpy() - on_step) * p.timestep                            # ns since influx start
        b = early["Bonds"].to_numpy() - early["Bonds"].iloc[0]
        slope = (t * b).sum() / (t * t).sum()                                            # bonds/ns

        step, atoms = next(read_dump(run_dir / name / "traj.dump", start=on_step))
        bond_step, bonds = next(read_bonds(run_dir / name / "bonds.dump", start=on_step))
        if step != on_step or bond_step != on_step:
            raise ValueError(f"{run_dir / name}: no traj.dump frame at the influx start (step {on_step}); "
                             "the baseline must be a multiple of traj_every")
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


BOND_RATES = ("formed_per_s", "broken_per_s", "k_a_per_M_s")


def bond_rates(run_dir: str | Path, bin_s: float | None = None, save: bool = True) -> dict:
    """Bond formation and breaking rates over the whole run, for any calcium protocol (rise or step).

    Per bin (default: one calcium segment, rate_update_every steps; `bin_s` must be a multiple of
    output_every): formed_per_s = increase of Bonds + n_broken (bonds formed) per second, broken_per_s =
    increase of n_broken per second, and the apparent association rate constant k_a_per_M_s =
    formed_per_s * V / (N_Qt N_Ft) in 1/(M s), normalized by the total particle numbers like the measured
    k_a of binding_rate. It falls as particles fill their bond slots, even at constant calcium.
    Times are the bin centres.

    Returns {"time_ms", "mean", "sd", "traces", "replicas", "bin_ms"} (as ensemble_stats, keys
    BOND_RATES); saved to <run>/analysis/bond_rates.csv.
    """
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    series = replica_series(run_dir)
    if not series:
        raise ValueError(f"no finished replicas in {run_dir}")
    bin_steps = p.rate_update_every if bin_s is None else round(bin_s / (p.timestep * 1e-9))
    if bin_steps < 1 or bin_steps % p.output_every:
        raise ValueError(f"bin_s = {bin_s} s is not a positive multiple of output_every "
                         f"({p.output_every * p.timestep * 1e-9:g} s)")
    bin_ns = bin_steps * p.timestep
    per_volume = math.prod(p.box) / (p.n_qt * p.n_ft)                  # V / (N_Qt N_Ft)
    traces = {c: [] for c in BOND_RATES}
    for df in series.values():
        edges = df[df["Step"] % bin_steps == 0]
        formed = np.diff((edges["Bonds"] + edges["n_broken"]).to_numpy()) / bin_ns          # bonds/ns
        traces["formed_per_s"].append(formed * 1e9)
        traces["broken_per_s"].append(np.diff(edges["n_broken"].to_numpy()) / bin_ns * 1e9)
        traces["k_a_per_M_s"].append(formed * per_volume * TO_PER_M_S)
    n = min(len(tr) for tr in traces["formed_per_s"])                  # replicas share the grid
    traces = {c: np.array([tr[:n] for tr in v]) for c, v in traces.items()}
    t = (np.arange(n) + 0.5) * bin_ns * 1e-6
    mean = {c: v.mean(axis=0) for c, v in traces.items()}
    sd = {c: v.std(axis=0, ddof=1) if len(series) > 1 else np.full(n, np.nan) for c, v in traces.items()}
    result = {"time_ms": t, "mean": mean, "sd": sd, "traces": traces, "replicas": list(series),
              "bin_ms": bin_ns * 1e-6}
    if save:
        out = run_dir / "analysis"
        out.mkdir(exist_ok=True)
        table = {"time_ms": t}
        for c in BOND_RATES:
            table[f"{c}_mean"] = mean[c]
            table[f"{c}_sd"] = sd[c]
        pd.DataFrame(table).to_csv(out / "bond_rates.csv", index=False)
    return result


SIZE_THRESHOLDS = (5, 20, 60, 80, 100, 150)     # default thresholds [particles] of agglomerates_above_size
SIZE_THRESHOLDS_QT = (5, 10, 20, 50, 70)        # default thresholds [Qt per agglomerate] (about 4x fewer Qt than Ft)
SIZE_THRESHOLDS_QT_FT = (15, 20)                # default pair (>= n Qt, >= n Ft) shown together in one panel
# what the size of an agglomerate counts -> suffix of analysis/agglomerates_above_size<suffix>.csv
SPECIES_SUFFIX = {"all": "", "qt": "_qt", "ft": "_ft"}
SPECIES_TYPE = {"qt": 1, "ft": 2}


def _thresholds(n_min) -> list[int]:
    """One number or a list of agglomerate sizes -> sorted distinct positive integers."""
    values = np.atleast_1d(n_min).ravel()
    if len(values) == 0 or any(v != int(v) or v < 1 for v in values):
        raise ValueError(f"n_min must be one positive integer or a list of them, not {n_min!r}")
    return sorted({int(v) for v in values})


def agglomerates_above_size(run_dir: str | Path, n_min=SIZE_THRESHOLDS, save: bool = True,
                            species: str = "all") -> pd.DataFrame:
    """Number of agglomerates with >= n particles over time, for every n in `n_min` (one number or a
    list), from the cluster column of the traj.dump frames of every finished replica (resolution
    traj_every; a free particle is a cluster of 1). Also as a density per 1000 um^3 of the full box
    volume Lx Ly Lz: the box count scaled up, so one agglomerate in the box is 1000 / V_box[um^3].
    `species`: "all" counts all particles of an agglomerate, "qt" / "ft" only its Qt / Ft (agglomerates
    with >= n Qt; the agglomerates are the same, only their size is measured differently).

    Returns one row per frame and threshold: step, time_s, n_min, replicas, count_mean, count_sd,
    per_1000um3_mean, per_1000um3_sd (mean +- SD over replicas; SD NaN with one replica) and the count
    of every replica (count_<replica>). With `save`, written to <run>/analysis/agglomerates_above_size.csv
    (species "qt" / "ft": agglomerates_above_size_qt.csv / _ft.csv) and reused while it holds every requested
    threshold, the same replicas and is newer than their dumps (otherwise recomputed for the requested
    and the saved thresholds).
    """
    return _agglomerates_above_size_species(run_dir, {species: n_min}, save)[species]


def _agglomerates_above_size_species(run_dir: str | Path, n_min: dict, save: bool = True) -> dict[str, pd.DataFrame]:
    """agglomerates_above_size for several species at once, n_min = {species: thresholds}; the traj.dump frames are
    read once for all species that are not cached. Returns {species: table}."""
    run_dir = Path(run_dir).expanduser()
    require_fetched(run_dir)
    p = Params.load(run_dir / "params.json")
    bad = set(n_min) - set(SPECIES_SUFFIX)
    if bad:
        raise ValueError(f"species must be one of {list(SPECIES_SUFFIX)}, not {sorted(bad)}")
    wanted = {sp: _thresholds(n) for sp, n in n_min.items()}
    reps = list(replica_series(run_dir, names_only=True))
    if not reps:
        raise ValueError(f"no finished replicas in {run_dir}")
    dumps = [run_dir / rep / "traj.dump" for rep in reps]
    paths = {sp: run_dir / "analysis" / f"agglomerates_above_size{SPECIES_SUFFIX[sp]}.csv" for sp in wanted}
    result, todo = {}, {}
    for sp, ns in wanted.items():
        if save and paths[sp].exists():
            cached = pd.read_csv(paths[sp])
            cached_reps = [c[len("count_"):] for c in cached.columns
                           if c.startswith("count_") and c not in ("count_mean", "count_sd")]
            if cached_reps == reps:
                fresh = paths[sp].stat().st_mtime >= max(d.stat().st_mtime for d in dumps)
                if fresh and set(ns) <= set(cached["n_min"]):
                    result[sp] = cached[cached["n_min"].isin(ns)].reset_index(drop=True)
                    continue
                if fresh:
                    ns = sorted(set(ns) | set(cached["n_min"]))
        todo[sp] = ns
    if not todo:
        return result

    per_rep = {sp: {} for sp in todo}                                   # species -> replica -> {step: counts per n}
    for rep, dump in zip(reps, dumps):
        counts = {sp: {} for sp in todo}
        for step, atoms in read_dump(dump):
            cluster = atoms["cluster"].to_numpy(dtype=int)
            types = atoms["type"].to_numpy(dtype=int)
            for sp, ns in todo.items():
                sizes = np.bincount(cluster if sp == "all" else cluster[types == SPECIES_TYPE[sp]])
                sizes = sizes[sizes > 0]
                counts[sp][step] = [(sizes >= n).sum() for n in ns]
        for sp in todo:
            per_rep[sp][rep] = counts[sp]
    to_density = 1000 / (math.prod(p.box) * 1e-9)                       # box count -> per 1000 um^3
    for sp, ns in todo.items():
        steps = sorted(set.intersection(*(set(c) for c in per_rep[sp].values())))   # frames of all replicas
        rows = []
        for i, n in enumerate(ns):
            values = np.array([[per_rep[sp][rep][s][i] for s in steps] for rep in reps], dtype=float)
            sd = values.std(axis=0, ddof=1) if len(reps) > 1 else np.full(len(steps), np.nan)
            table = pd.DataFrame({"step": steps, "time_s": np.array(steps) * p.timestep * 1e-9, "n_min": n,
                                  "replicas": len(reps), "count_mean": values.mean(axis=0), "count_sd": sd,
                                  "per_1000um3_mean": values.mean(axis=0) * to_density,
                                  "per_1000um3_sd": sd * to_density})
            for rep, row in zip(reps, values):
                table[f"count_{rep}"] = row
            rows.append(table)
        table = pd.concat(rows, ignore_index=True)
        if save:
            paths[sp].parent.mkdir(exist_ok=True)
            table.to_csv(paths[sp], index=False)
        result[sp] = table[table["n_min"].isin(wanted[sp])].reset_index(drop=True)
    return {sp: result[sp] for sp in wanted}


# ------------------------------------------------------------------ index of all runs
INDEX_PARAMS = ("n_qt", "n_ft", "box_length", "boundary", "k_qq", "k_ff", "k_qf", "bond_k", "k_on",
                "k_off_ca_per_s", "k_off_free_per_s", "timestep", "t_baseline_s", "t_influx_s", "t_pumpout_s",
                "n_replicas")


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
            "executor": params.get("executor", "local"),             # runs from before LRZ: local
            "changed vs defaults": "; ".join(changed),
            "git": (record.get("git") or {}).get("commit"),
        })
    df = pd.DataFrame(rows)
    if rows:
        df.to_csv(data_dir / "index.csv", index=False)
    return df


SHORT_NAMES = {"n_qt": "Qt", "n_ft": "Ft", "box_length": "L", "k_off_ca_per_s": "k_off,Ca",
               "k_off_free_per_s": "k_off,free", "ca_rest_uM": "Ca rest", "ca_peak_uM": "Ca peak", "ca_K_uM": "K",
               "ca_hill": "n", "ca_rise_s": "rise", "ca_tau_s": "τ", "t_baseline_s": "t_base",
               "t_influx_s": "t_influx", "t_pumpout_s": "t_pump", "timestep": "dt"}   # short labels (figure legends)


def _short_value(v) -> str:
    if isinstance(v, (list, tuple)):
        return "×".join(_short_value(x) for x in v)
    return f"{v:g}" if isinstance(v, float) else str(v)


def differing_labels(run_dirs, short: bool = False) -> tuple[list[str], list[str]]:
    """Labels of runs built from the physical parameters that differ between them: "k=v, ..." (tables), or
    with `short` "Qt 1280, L 4000" (short names from SHORT_NAMES, numbers without trailing .0; legends)."""
    run_dirs = [Path(r).expanduser() for r in run_dirs]
    params = [Params.load(r / "params.json") for r in run_dirs]
    differing = [f.name for f in dataclasses.fields(Params)
                 if f.name not in EXECUTION_SETTINGS
                 and len({json.dumps(getattr(q, f.name)) for q in params}) > 1]
    if short:
        labels = [", ".join(f"{SHORT_NAMES.get(k, k)} {_short_value(getattr(q, k))}" for k in differing) or r.name
                  for r, q in zip(run_dirs, params)]
    else:
        labels = [", ".join(f"{k}={getattr(q, k)}" for k in differing) or r.name
                  for r, q in zip(run_dirs, params)]
    return labels, differing


def compare(run_dirs: list[str | Path]) -> pd.DataFrame:
    """Side-by-side final-state table (mean +- SD) of several runs, with the parameters that differ."""
    run_dirs = [Path(r).expanduser() for r in run_dirs]
    labels, differing = differing_labels(run_dirs)
    cols = {}
    for r, label in zip(run_dirs, labels):
        final = ensemble_stats_cached(r)["final"]
        cols[label] = final.apply(lambda row: f"{row['mean']:.4g} ± {row['sd']:.2g}", axis=1)
    table = pd.DataFrame(cols)
    table.attrs["differing parameters"] = differing
    return table
