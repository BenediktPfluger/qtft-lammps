"""Force-constant sweep: how small can the particle overlap get?

Each pair family sits in a harmonic well E = k x^2 (x = overlap depth) driven by thermal noise.
The continuous dynamics give a depth spread sigma^2 = kT / 2k; the Euler-Maruyama step of
fix brownian widens it by 1 / (1 - alpha/2), with alpha = 2 k dt (1/gamma_i + 1/gamma_j)
(`Params.alpha`). The mean depth of overlapping pairs is sigma * sqrt(2/pi). At a fixed time step
the overlap is smallest at alpha = 1; at a fixed alpha it scales with sqrt(dt).

`alpha_sweep(p, {500.0: alphas})` builds one run per time step and target alpha (all four constants
set to reach it) as a sweep, run with `start(sweep_dir)` through one queue;
`sweep_table(sweep_dir)` compares measured and predicted overlap and the binding kinetics.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd

from .ensemble import ensemble_stats
from .overlap import overlap_ensemble
from .params import Params
from .sweep import build_sweep_from, is_sweep, sweep_runs


def _gammas(p: Params) -> dict:
    gq, gf = p.gamma_qt, p.gamma_ft
    return {"Qt-Qt": (gq, gq), "Ft-Ft": (gf, gf), "Qt-Ft": (gq, gf)}


def constants_for_alpha(p: Params, alpha: float) -> dict:
    """k_qq, k_ff, k_qf, bond_k and the wall constants that give every family the per-step
    overshoot `alpha` (the wall constants only act with boundary="harmonic")."""
    k = {fam: alpha / (2 * p.timestep * (1 / gi + 1 / gj)) for fam, (gi, gj) in _gammas(p).items()}
    return {"k_qq": k["Qt-Qt"], "k_ff": k["Ft-Ft"], "k_qf": k["Qt-Ft"], "bond_k": k["Qt-Ft"],
            "wall_k_qt": alpha * p.gamma_qt / (2 * p.timestep), "wall_k_ft": alpha * p.gamma_ft / (2 * p.timestep)}


def predicted_overlap(p: Params) -> pd.Series:
    """Predicted mean depth of overlapping pairs [nm] per family. Qt-Ft uses the bond constant,
    since most Qt-Ft contacts are bonded once binding has started."""
    k = {"Qt-Qt": p.k_qq, "Ft-Ft": p.k_ff, "Qt-Ft": p.bond_k}
    out = {}
    for fam, (gi, gj) in _gammas(p).items():
        alpha = p.alpha(k[fam], gi, gj)
        out[fam] = math.sqrt(2 / math.pi) * math.sqrt(p.kT / (2 * k[fam]) / (1 - alpha / 2))
    return pd.Series(out, name="predicted depth [nm]")


def alpha_sweep(p: Params, alphas: dict[float, list], name: str = "sweep", overwrite: bool = False) -> Path:
    """Build one run per time step and target alpha from the base parameters `p` (not started),
    e.g. alphas = {500.0: [0.3, 0.7, 0.9], 250.0: [0.9]}. The equilibration and the segment length
    (rate_update_every) are scaled with the time step; the calcium phases are given in seconds, so every
    run covers the same simulated time as `p`.
    Runs go to <data_dir>/tuning/<name>/alpha<a>_dt<dt>ns/; start them with start(sweep_dir).
    """
    overrides, labels = [], []
    for dt, values in alphas.items():
        steps = {"timestep": dt, "equil_steps": round(p.equil_steps * p.timestep / dt),
                 "rate_update_every": round(p.rate_update_every * p.timestep / dt)}
        for a in values:
            overrides.append({**steps, **constants_for_alpha(dataclasses.replace(p, timestep=dt), a)})
            labels.append(f"alpha{a:g}_dt{dt:g}ns")
    return build_sweep_from(p, overrides, labels, group=f"tuning/{name}", overwrite=overwrite,
                            info={"name": name, "mode": "alpha", "alphas": {str(k): v for k, v in alphas.items()}})


def sweep_table(sweep, every: int = 5, last_frames: int = 10) -> pd.DataFrame:
    """One row per run and pair family: alpha, k, predicted and measured overlap (mean over the
    last `last_frames` analysed frames of all replicas), final fraction bound and half-time of
    bonds. `sweep` is a sweep folder (or a list of run folders of one sweep). Saved to
    sweep_summary.csv in the sweep folder."""
    run_dirs = sweep_runs(sweep) if is_sweep(sweep) else [Path(r) for r in sweep]
    rows = []
    for run_dir in run_dirs:
        p = Params.load(run_dir / "params.json")
        measured = overlap_ensemble(run_dir, every=every, last_frames=last_frames)["final"]
        final = ensemble_stats(run_dir)["final"]["mean"]
        predicted = predicted_overlap(p)
        k = {"Qt-Qt": p.k_qq, "Ft-Ft": p.k_ff, "Qt-Ft": p.bond_k}
        for fam, (gi, gj) in _gammas(p).items():
            rows.append({
                "run": run_dir.name, "timestep [ns]": p.timestep, "family": fam,
                "alpha": p.alpha(k[fam], gi, gj), "k [zJ/nm^2]": k[fam],
                "predicted depth [nm]": predicted[fam],
                "mean depth [nm]": measured.loc[fam, "mean depth overlapping [nm]"],
                "p95 depth [nm]": measured.loc[fam, "p95 depth [nm]"],
                "max depth [nm]": measured.loc[fam, "max depth [nm]"],
                "frac overlapping": measured.loc[fam, "frac overlapping"],
                "final frac bound": final["final_frac_bound"],
                "t_half bonds [ms]": final["t_half_bonds_ms"],
            })
    df = pd.DataFrame(rows)
    df.to_csv(run_dirs[0].parent / "sweep_summary.csv", index=False)
    return df
