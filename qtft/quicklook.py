"""Quick overview plots of a run (all replicas overlaid). Not the publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .overlap import FAMILIES, overlap_frame
from .params import Params
from .read import load_thermo, read_dump

# (thermo column, axis label, scale factor)
PANELS = [
    ("Bonds", "bonds", 1),
    ("frac_bound", "fraction bound", 1),
    ("n_clusters", "clusters (incl. free)", 1),
    ("largest", "largest cluster (particles)", 1),
    ("mean_size", "mean cluster size", 1),
    ("bond_max", "max bond length (nm)", 1),
    ("PotEng", "potential energy (zJ)", 1),
    ("pvir", "virial pressure (Pa)", 1e6),       # units nano: 1 pressure unit = 1 MPa
]
OVERLAP = "mean depth overlapping [nm]"


def _overlap(rep: Path, p: Params, every: int) -> pd.DataFrame:
    """Overlap per family of every `every`-th frame of traj.dump; stops at an incomplete frame
    (replica still writing)."""
    rows = []
    try:
        for i, (step, atoms) in enumerate(read_dump(rep / "traj.dump")):
            if i % every:
                continue
            if len(atoms) < p.n_qt + p.n_ft or atoms[["x", "y", "z"]].isna().any().any():
                break
            rows.append(overlap_frame(atoms, p).assign(time_ms=step * p.timestep * 1e-6))
    except (ValueError, IndexError):          # half-written header at the end of the file
        pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def quicklook(run_dir: str | Path, overlap: bool = True, overlap_every: int = 10):
    """Thermo time series, potential energy, pressure and (with `overlap`) the mean overlap depth
    of each pair family from every `overlap_every`-th trajectory frame, all replicas overlaid."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    fig, axes = plt.subplots(3, 4, figsize=(14, 8.5), sharex=True)
    thermo_axes, overlap_axes, legend_ax = axes.flat[:8], axes.flat[8:11], axes.flat[11]
    for ax, (_, label, _) in zip(thermo_axes, PANELS):
        ax.set_ylabel(label)
    for ax, family in zip(overlap_axes, FAMILIES):
        ax.set_ylabel(f"mean overlap {family.replace('-', '–')} (nm)")
    for rep in sorted(run_dir.glob("rep_*")):
        if not (rep / "log.lammps").exists():
            continue
        try:
            df = load_thermo(rep)
        except ValueError:
            continue
        for ax, (col, _, scale) in zip(thermo_axes, PANELS):
            ax.plot(df["time_ms"], df[col] * scale, lw=1, label=rep.name)
        ov = _overlap(rep, p, overlap_every) if overlap and (rep / "traj.dump").exists() else pd.DataFrame()
        for ax, family in zip(overlap_axes, FAMILIES):
            if len(ov):
                sel = ov[ov["family"] == family]
                ax.plot(sel["time_ms"], sel[OVERLAP], lw=1, label=rep.name)
    axes[1, 3].tick_params(labelbottom=True)       # the legend panel below it has no axis
    for ax in [*axes[-1, :3], axes[1, 3]]:
        ax.set_xlabel("time (ms)")
    handles, labels = thermo_axes[0].get_legend_handles_labels()
    legend_ax.axis("off")
    if handles:
        legend_ax.legend(handles, labels, frameon=False, fontsize=8, loc="center")
    fig.tight_layout()
    return fig
