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


ROW_START = "  - ["        # a data row of a YAML thermo block: "  - [2000, 1.338, ..., ]"


def _rows(lines: list[str]) -> np.ndarray:
    """Numeric values of complete thermo data rows (a half-written last row is skipped)."""
    text = "\n".join(line[len(ROW_START):line.rindex("]")].rstrip().rstrip(",")
                     for line in lines if line.rstrip().endswith("]"))
    return np.loadtxt(io.StringIO(text), delimiter=",", ndmin=2) if text else np.empty((0, 0))


def _frame(keywords: list[str], rows: np.ndarray) -> pd.DataFrame:
    """Thermo block as a DataFrame; columns with only whole numbers as int64 (as the YAML reader gave)."""
    rows = rows.reshape(-1, len(keywords))
    whole = np.all(np.isfinite(rows) & (rows == np.round(rows)) & (np.abs(rows) < 2**53), axis=0)
    return pd.DataFrame({c: rows[:, j].astype(np.int64) if whole[j] else rows[:, j]
                         for j, c in enumerate(keywords)})


def read_thermo(log_path: str | Path) -> list[pd.DataFrame]:
    """All YAML thermo blocks of a log.lammps, one DataFrame per `run` command. Other lines inside a
    block (LAMMPS warnings printed during the run) are skipped. The keywords are read with YAML, the data
    rows numerically, all blocks with the same keywords at once (much faster than YAML on long logs)."""
    blocks, keywords, rows = [], None, None              # blocks: (keywords line, rows) per finished block
    for line in Path(log_path).read_text().splitlines():
        if line == "---":
            keywords, rows = None, []
        elif line == "..." and rows is not None:
            if keywords is not None:
                blocks.append((keywords, [r for r in rows if r.rstrip().endswith("]")]))
            keywords, rows = None, None
        elif rows is not None and line.startswith("keywords:"):
            keywords = line
        elif rows is not None and line.startswith(ROW_START):
            rows.append(line)
    out = [None] * len(blocks)
    for kw_line in dict.fromkeys(kw for kw, _ in blocks):
        keywords = yaml.safe_load(kw_line)["keywords"]
        members = [i for i, (kw, _) in enumerate(blocks) if kw == kw_line]
        values = _rows([r for i in members for r in blocks[i][1]]).reshape(-1, len(keywords))
        ends = np.cumsum([len(blocks[i][1]) for i in members])
        for i, part in zip(members, np.split(values, ends[:-1])):
            out[i] = _frame(keywords, part)
    return out


def last_thermo(log_path: str | Path) -> tuple[str, dict] | None:
    """(phase, values) of the most recent complete thermo row, also while a run is in progress. Reads only
    the last keywords line and the last row (fast on long logs)."""
    log_path = Path(log_path)
    if not log_path.exists():
        return None
    text = log_path.read_text()
    n_blocks = text.count("\nkeywords:") + text.startswith("keywords:")
    if not n_blocks:
        return None
    k = text.rfind("keywords:")
    keywords = yaml.safe_load(text[k:text.index("\n", k)] if "\n" in text[k:] else text[k:])["keywords"]
    tail = text[k:].splitlines()
    rows = _rows([line for line in tail if line.startswith(ROW_START)][-2:])
    if not len(rows):
        if n_blocks < 2:
            return None
        # new block without a row yet: the last row of the previous block
        previous = text[:text.rfind("\nkeywords:")]
        k = previous.rfind("keywords:")
        keywords = yaml.safe_load(previous[k:previous.index("\n", k)])["keywords"]
        rows = _rows([line for line in previous[k:].splitlines() if line.startswith(ROW_START)][-2:])
        n_blocks -= 1
        if not len(rows):
            return None
    phase = PHASES[min(n_blocks, 2) - 1]
    whole = lambda v: np.isfinite(v) and v == round(v) and abs(v) < 2**53          # as the YAML reader gave
    return phase, {c: int(v) if whole(v) else float(v) for c, v in zip(keywords, rows[-1])}


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
