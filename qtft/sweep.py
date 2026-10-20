"""Parameter sweeps: one run per combination of parameter values (grid).

`build_sweep(p, {"temperature": [300, 310], "k_on": [1e-6, 2e-6]}, "T_kon")` builds 2 x 2 runs from the
base parameters `p` in <data_dir>/sweeps/T_kon/<label>/ and writes sweep.json. All variants keep the
master seed of `p`, so replica i of every variant uses the same random numbers (common random numbers:
differences between variants come from the parameters, not from noise).

`start(sweep_dir)` runs all replicas of all variants through one queue (see run.py);
`sweep_results(sweep_dir)` gives one row per variant with the final-state values over the replicas.
"""

from __future__ import annotations

import dataclasses
import datetime
import itertools
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from .build import build_run, git_info
from .ensemble import ensemble_stats
from .params import Params, box_label, data_path

# execution settings: shared by all variants (one queue, same seeds), not sweepable
FIXED = {"n_replicas", "mpi_procs", "max_parallel", "master_seed", "data_dir", "group", "run_name",
         "lmp_command", "mpirun_command"}
RESULTS = ["final_frac_bound", "final_Bonds", "final_largest_frac", "final_mean_size",
           "t_half_bonds_ms", "t_half_qt_ms", "t_half_ft_ms", "t_percolation_ms"]


def _value_label(value) -> str:
    if isinstance(value, (list, tuple)):
        return box_label(value)
    return f"{value:g}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def build_sweep_from(p: Params, overrides: list[dict], labels: list[str], group: str,
                     info: dict | None = None, overwrite: bool = False) -> Path:
    """Build one run per dict of parameter overrides in <data_dir>/<group>/<label>/ and write
    <data_dir>/<group>/sweep.json. All variants are checked before any folder is created."""
    fields = {f.name for f in dataclasses.fields(Params)}
    keys = {k for o in overrides for k in o}
    if keys - fields:
        raise ValueError(f"unknown parameters: {sorted(keys - fields)}")
    if keys & FIXED:
        raise ValueError(f"cannot sweep execution settings: {sorted(keys & FIXED)}")
    if len(set(labels)) != len(labels):
        raise ValueError(f"run labels are not unique: {labels}")
    variants = [dataclasses.replace(p, **o, group=group, run_name=label) for o, label in zip(overrides, labels)]
    errors = []
    for q in variants:
        errors += [f"{q.run_name}: {msg}" for lvl, msg in q.check() if lvl == "ERROR"]
    if errors:
        raise ValueError("invalid variants:\n  " + "\n  ".join(errors))

    sweep_dir = data_path(p.data_dir) / group
    if sweep_dir.exists():
        if not overwrite:
            raise FileExistsError(f"{sweep_dir} exists; choose a different name or use overwrite=True")
        shutil.rmtree(sweep_dir)
    for q in variants:
        build_run(q)
    record = {
        **(info or {}),
        "base": p.to_dict(),
        "runs": [{"label": label, "overrides": o} for o, label in zip(overrides, labels)],
        "git": git_info(),
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
    }
    (sweep_dir / "sweep.json").write_text(json.dumps(record, indent=2, default=str))
    for q in variants:
        for lvl, msg in q.check():
            print(f"{q.run_name}: {lvl}: {msg}")
    return sweep_dir


def build_sweep(p: Params, variants: dict[str, list], name: str, overwrite: bool = False) -> Path:
    """Build one run per combination of the listed values (grid) in <data_dir>/sweeps/<name>/."""
    variants = {k: list(v) for k, v in variants.items()}
    combos = [dict(zip(variants, values)) for values in itertools.product(*variants.values())]
    labels = ["_".join(f"{k}{_value_label(v)}" for k, v in c.items()) for c in combos]
    sweep_dir = build_sweep_from(p, combos, labels, group=f"sweeps/{name}", overwrite=overwrite,
                                 info={"name": name, "mode": "grid", "variants": variants})
    print(f"sweep {name}: {len(combos)} run(s) x {p.n_replicas} replica(s) = "
          f"{len(combos) * p.n_replicas} simulations in {sweep_dir}")
    print(pd.DataFrame(combos, index=labels).to_string(float_format="{:g}".format))
    return sweep_dir


def is_sweep(path: str | Path) -> bool:
    return (Path(path).expanduser() / "sweep.json").exists()


def load_sweep(sweep_dir: str | Path) -> dict:
    return json.loads((Path(sweep_dir).expanduser() / "sweep.json").read_text())


def sweep_runs(sweep_dir: str | Path) -> list[Path]:
    """Run folders of a sweep, in the order they were built."""
    sweep_dir = Path(sweep_dir).expanduser()
    return [sweep_dir / r["label"] for r in load_sweep(sweep_dir)["runs"]]


def sweep_results(sweep_dir: str | Path, save: bool = True) -> pd.DataFrame:
    """One row per variant: the swept parameters, the number of finished replicas and the mean and SD
    over them of the final-state values (fraction bound, bonds, largest cluster, mean size, half-times)
    and the number of percolated replicas. Saved to <sweep>/sweep_results.csv."""
    sweep_dir = Path(sweep_dir).expanduser()
    record = load_sweep(sweep_dir)
    rows = []
    for run in record["runs"]:
        row = {"run": run["label"], **run["overrides"]}
        run_dir = sweep_dir / run["label"]
        try:
            final = ensemble_stats(run_dir)["final"]
        except ValueError:                       # no finished replica yet
            rows.append({**row, "replicas done": 0})
            continue
        row["replicas done"] = int(final.loc["percolated_replicas", "n"])
        for key in RESULTS:
            row[key] = final.loc[key, "mean"] if key in final.index else np.nan
            row[f"{key} sd"] = final.loc[key, "sd"] if key in final.index else np.nan
        row["percolated replicas"] = int(final.loc["percolated_replicas", "mean"])
        rows.append(row)
    df = pd.DataFrame(rows).set_index("run")
    df.attrs["swept"] = list(record.get("variants") or dict.fromkeys(k for r in record["runs"] for k in r["overrides"]))
    if save:
        df.to_csv(sweep_dir / "sweep_results.csv")
    return df
