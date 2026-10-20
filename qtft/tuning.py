"""Force-constant sweep: how small can the particle overlap get?

Each pair family sits in a harmonic well E = k x^2 (x = overlap depth) driven by thermal noise.
The continuous dynamics give a depth spread sigma^2 = kT / 2k; the Euler-Maruyama step of
fix brownian widens it by 1 / (1 - alpha/2), with alpha = 2 k dt (1/gamma_i + 1/gamma_j)
(`Params.alpha`). The mean depth of overlapping pairs is sigma * sqrt(2/pi). At a fixed time step
the overlap is smallest at alpha = 1; at a fixed alpha it scales with sqrt(dt).

`alpha_sweep(p, alphas)` builds one run per target alpha (all four constants set to reach it),
`sweep_table(run_dirs)` compares measured and predicted overlap and the binding kinetics.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

import pandas as pd

from .build import build_run
from .ensemble import ensemble_stats
from .overlap import overlap_ensemble
from .params import Params, data_path


def _gammas(p: Params) -> dict:
    gq, gf = p.gamma_qt, p.gamma_ft
    return {"Qt-Qt": (gq, gq), "Ft-Ft": (gf, gf), "Qt-Ft": (gq, gf)}


def constants_for_alpha(p: Params, alpha: float) -> dict:
    """k_qq, k_ff, k_qf and bond_k that give every family the per-step overshoot `alpha`."""
    k = {fam: alpha / (2 * p.timestep * (1 / gi + 1 / gj)) for fam, (gi, gj) in _gammas(p).items()}
    return {"k_qq": k["Qt-Qt"], "k_ff": k["Ft-Ft"], "k_qf": k["Qt-Ft"], "bond_k": k["Qt-Ft"]}


def predicted_overlap(p: Params) -> pd.Series:
    """Predicted mean depth of overlapping pairs [nm] per family. Qt-Ft uses the bond constant,
    since most Qt-Ft contacts are bonded once binding has started."""
    k = {"Qt-Qt": p.k_qq, "Ft-Ft": p.k_ff, "Qt-Ft": p.bond_k}
    out = {}
    for fam, (gi, gj) in _gammas(p).items():
        alpha = p.alpha(k[fam], gi, gj)
        out[fam] = math.sqrt(2 / math.pi) * math.sqrt(p.kT / (2 * k[fam]) / (1 - alpha / 2))
    return pd.Series(out, name="predicted depth [nm]")


def alpha_sweep(p: Params, alphas, timestep: float | None = None, prod_steps: int | None = None,
                name: str = "sweep", overwrite: bool = False) -> list[Path]:
    """Build one run per target alpha from the base parameters `p` (not started).

    timestep/prod_steps override those of `p` (e.g. a smaller time step with more steps).
    Runs go to <data_dir>/tuning/<name>/alpha<a>_dt<dt>ns/.
    """
    q = dataclasses.replace(p, timestep=timestep or p.timestep, prod_steps=prod_steps or p.prod_steps)
    run_dirs = []
    for a in alphas:
        r = dataclasses.replace(q, **constants_for_alpha(q, a), group="tuning",
                                run_name=f"{name}/alpha{a:g}_dt{q.timestep:g}ns")
        run_dirs.append(build_run(r, overwrite=overwrite))
    return run_dirs


def sweep_table(run_dirs, every: int = 5, last_frames: int = 10, name: str = "sweep") -> pd.DataFrame:
    """One row per run and pair family: alpha, k, predicted and measured overlap (mean over the
    last `last_frames` analysed frames of all replicas), final fraction bound and half-time of
    bonds. Saved to <data_dir>/tuning/<name>/sweep_summary.csv."""
    rows = []
    for run_dir in map(Path, run_dirs):
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
    out = data_path(Params.load(Path(run_dirs[0]) / "params.json").data_dir) / "tuning" / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "sweep_summary.csv", index=False)
    return df
