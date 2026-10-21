"""Readers for the LAMMPS output files of one replica folder."""

from __future__ import annotations

import io
from itertools import islice
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .params import Params

PHASES = ("equilibration", "production")
TYPE_IDS = {"Qt": 1, "Ft": 2}   # type labels in traj.dump -> numeric atom types


def read_thermo(log_path: str | Path) -> list[pd.DataFrame]:
    """All YAML thermo blocks of a log.lammps, one DataFrame per `run` command. Other lines inside a
    block (LAMMPS warnings printed during the run) are skipped."""
    blocks, current = [], None
    for line in Path(log_path).read_text().splitlines():
        if line == "---":
            current = []
        elif line == "..." and current is not None:
            doc = yaml.safe_load("\n".join(current))
            blocks.append(pd.DataFrame(doc["data"], columns=doc["keywords"]))
            current = None
        elif current is not None and line.startswith(("keywords:", "data:", "  - [")):
            current.append(line)
    return blocks


def last_thermo(log_path: str | Path) -> tuple[str, dict] | None:
    """(phase, values) of the most recent thermo line, also while a run is in progress."""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    keywords, last, n_blocks = None, None, 0
    for line in log_path.read_text().splitlines():
        if line.startswith("keywords:"):
            keywords = yaml.safe_load(line)["keywords"]
            n_blocks += 1
        elif line.startswith("  - [") and keywords:
            last = yaml.safe_load(line.strip())[0]
    if last is None:
        return None
    phase = PHASES[min(n_blocks, 2) - 1]
    return phase, dict(zip(keywords, last))


def load_thermo(rep_dir: str | Path) -> pd.DataFrame:
    """Production thermo data of a replica with a time column in ms and short column names.
    The production runs in segments (one thermo block each, see template.lmp); they are joined and the
    step repeated at every segment start is dropped."""
    rep_dir = Path(rep_dir)
    p = Params.load(rep_dir.parent / "params.json")
    blocks = read_thermo(rep_dir / "log.lammps")
    if len(blocks) < 2:
        raise ValueError(f"no production data in {rep_dir / 'log.lammps'}")
    df = pd.concat(blocks[1:], ignore_index=True).drop_duplicates("Step", keep="first").reset_index(drop=True)
    df = df.rename(columns=lambda c: c.removeprefix("v_").removeprefix("c_"))
    df.insert(1, "time_ms", df["Step"] * p.timestep * 1e-6)
    return df


def read_rdf(path: str | Path) -> dict[int, pd.DataFrame]:
    """rdf.dat (fix ave/time, mode vector): {step: DataFrame(r, g, coord)}."""
    frames, lines = {}, Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        step, nrows = map(int, lines[i].split()[:2])
        rows = np.loadtxt(lines[i + 1:i + 1 + nrows], ndmin=2)
        frames[step] = pd.DataFrame(rows[:, 1:4], columns=["r", "g", "coord"])
        i += 1 + nrows
    return frames


def read_histo(path: str | Path) -> pd.DataFrame:
    """Histogram file of fix ave/histo: rows = steps, columns = bin centres, values = counts."""
    data, lines = {}, Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("#") or not lines[i].strip():
            i += 1
            continue
        step, nbins = map(int, lines[i].split()[:2])
        rows = np.loadtxt(lines[i + 1:i + 1 + nbins], ndmin=2)
        data[step] = pd.Series(rows[:, 2], index=rows[:, 1])
        i += 1 + nbins
    return pd.DataFrame(data).T.rename_axis("Step")


def read_dump(path: str | Path, start: int = 0):
    """Iterate over frames of traj.dump: yields (step, DataFrame with the dumped columns).
    Frames before step `start` are skipped without parsing them.

    Type labels (Qt, Ft) are converted to the numeric types 1, 2; older numeric dumps read the same.
    """
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.startswith("ITEM: TIMESTEP"):
                step = int(f.readline())
                f.readline()
                n = int(f.readline())
                if step < start:
                    for _ in islice(f, 5 + n):          # box bounds, column header, atom lines
                        pass
                    continue
                for _ in range(4):
                    f.readline()
                columns = f.readline().split()[2:]
                block = "".join(islice(f, n))
                if not block:
                    yield step, pd.DataFrame(columns=columns)
                    continue
                df = pd.read_csv(io.StringIO(block), sep=r"\s+", header=None, names=columns)
                if "type" in df and not pd.api.types.is_numeric_dtype(df["type"]):
                    df["type"] = pd.to_numeric(df["type"].replace(TYPE_IDS))
                yield step, df


def dump_steps(path: str | Path) -> list[int]:
    """Timesteps of all frames of a dump file (headers only, atom lines are skipped unparsed)."""
    steps = []
    with open(path) as f:
        for line in f:
            if line.startswith("ITEM: TIMESTEP"):
                steps.append(int(next(f)))
                next(f)
                n = int(next(f))
                for _ in islice(f, 5 + n):          # box bounds, column header, atom lines
                    pass
    return steps


def read_bonds(path: str | Path, start: int = 0):
    """Iterate over frames of bonds.dump: yields (step, int array of shape (n_bonds, 2)).
    Frames before step `start` are skipped without parsing them."""
    with open(path) as f:
        while True:
            line = f.readline()
            if not line:
                return
            if line.startswith("ITEM: TIMESTEP"):
                step = int(f.readline())
                f.readline()
                n = int(f.readline())
                if step < start:
                    for _ in islice(f, 5 + n):          # box bounds, column header, bond lines
                        pass
                    continue
                for _ in range(5):
                    f.readline()
                rows = [f.readline().split()[1:3] for _ in range(n)]
                yield step, np.array(rows, dtype=int).reshape(n, 2)
