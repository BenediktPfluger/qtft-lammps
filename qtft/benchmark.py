"""Performance measurement on the local machine and sizing of the bond bookkeeping.

`benchmark(p)` runs short productions of the system defined by `p` for several numbers of MPI
processes per replica. For each setting it fills the machine (cores // procs replicas at once),
so the measured throughput includes the competition for memory bandwidth of a real ensemble.

`special_need(rep_dir)` reports how many 1-2/1-3/1-4 neighbours the atoms of a finished run
actually have, to size `extra_special`.
"""

from __future__ import annotations

import dataclasses
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from .build import build_run
from .params import Params
from .read import read_bonds, read_dump
from .run import start

LOOP = re.compile(r"Loop time of (\S+) on (\d+) procs for (\d+) steps")
SECTION = re.compile(r"^(Pair|Bond|Neigh|Comm|Output|Modify|Other)\s*\|.*\|\s*([\d.]+)\s*$")


def loop_timing(log_path: str | Path) -> dict:
    """Wall time, steps and timing breakdown [% of total] of the last `run` in a log.lammps."""
    text = Path(log_path).read_text()
    loops = list(LOOP.finditer(text))
    if not loops:
        raise ValueError(f"no completed run in {log_path}")
    last = loops[-1]
    out = {"wall_s": float(last.group(1)), "procs": int(last.group(2)), "steps": int(last.group(3))}
    for line in text[last.end():].splitlines():
        m = SECTION.match(line.strip())
        if m:
            out[f"{m.group(1)} [%]"] = float(m.group(2))
        if line.startswith("Nlocal"):
            break
    return out


def benchmark(p: Params, procs=(1, 2, 4, 8, 16), cores: int | None = None, steps: int = 2000,
              variants: dict | None = None, name: str = "benchmark") -> pd.DataFrame:
    """Measure throughput for each number of MPI processes per replica (and optional variants).

    p:        parameter set to measure (system size, potentials, ...); run settings are overridden
    procs:    MPI processes per replica to try
    cores:    cores to fill (default: all cores of this machine)
    steps:    production steps per measurement (equilibration is kept short)
    variants: optional {label: {param: value}} to compare settings, e.g. {"skin 20": {"neighbor_skin": 20}}
    """
    cores = cores or os.cpu_count() or 1
    variants = variants or {"default": {}}
    rows = []
    for label, overrides in variants.items():
        for n in procs:
            if n > cores:
                continue
            parallel = cores // n
            q = dataclasses.replace(
                p, **overrides, equil_steps=200, prod_steps=steps,
                output_every=min(p.output_every, steps), traj_every=min(p.output_every, steps),
                n_replicas=parallel, mpi_procs=n, max_parallel=parallel,
                run_name=f"{name}/{_slug(label)}_np{n}")
            run_dir = build_run(q, overwrite=True)
            start(run_dir, wait=True)
            timings = [loop_timing(rep / "log.lammps") for rep in sorted(run_dir.glob("rep_*"))]
            wall = max(t["wall_s"] for t in timings)
            rate = steps / np.mean([t["wall_s"] for t in timings])          # steps/s of one replica
            row = {"variant": label, "procs/replica": n, "replicas at once": parallel,
                   "steps/s per replica": rate,
                   "replica-steps/s total": parallel * steps / wall}
            for key in [k for k in timings[0] if k.endswith("[%]")]:
                row[key] = np.mean([t[key] for t in timings])
            rows.append(row)
    df = pd.DataFrame(rows)
    df["h per replica"] = p.prod_steps / df["steps/s per replica"] / 3600
    df[f"h for {p.n_replicas} replicas"] = (
        np.ceil(p.n_replicas / df["replicas at once"]) * df["h per replica"])
    out = p.run_dir.parent / name
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "results.csv", index=False)
    return df


def recommend(df: pd.DataFrame, n_replicas: int) -> pd.Series:
    """Row with the shortest total wall time for n_replicas replicas."""
    total = np.ceil(n_replicas / df["replicas at once"]) * df["h per replica"]
    return df.loc[total.idxmin()]


def special_need(rep_dir: str | Path) -> dict:
    """Max/mean number of 1-2, 1-2+1-3 and 1-2+1-3+1-4 neighbours in the last frame of a run."""
    rep_dir = Path(rep_dir)
    *_, (_, bonds) = read_bonds(rep_dir / "bonds.dump")
    *_, (_, atoms) = read_dump(rep_dir / "traj.dump")
    n = int(atoms["id"].max()) + 1
    a = sparse.coo_matrix((np.ones(len(bonds)), (bonds[:, 0], bonds[:, 1])), shape=(n, n)).tocsr()
    a = ((a + a.T) > 0).astype(np.int8)
    reach, power = a.copy(), a.copy()
    counts = {}
    for order, label in ((1, "1-2"), (2, "1-3"), (3, "1-4")):
        if order > 1:
            power = ((power @ a) > 0).astype(np.int8)
            reach = ((reach + power) > 0).astype(np.int8)
        r = reach.tolil()
        r.setdiag(0)
        per_atom = np.asarray(r.tocsr().sum(axis=1)).ravel()[atoms["id"].to_numpy()]
        counts[f"max up to {label}"] = int(per_atom.max())
        counts[f"mean up to {label}"] = float(per_atom.mean())
    return counts


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9.+-]+", "_", text).strip("_")
