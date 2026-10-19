"""Quick overview plots of a run (all replicas overlaid). Not the publication figures."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .read import load_thermo

PANELS = [
    ("Bonds", "bonds"),
    ("frac_bound", "fraction bound"),
    ("n_clusters", "clusters (incl. free)"),
    ("largest", "largest cluster [particles]"),
    ("mean_size", "mean cluster size"),
    ("bond_max", "max bond length [nm]"),
]


def quicklook(run_dir: str | Path):
    run_dir = Path(run_dir).expanduser()
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True)
    for rep in sorted(run_dir.glob("rep_*")):
        if not (rep / "log.lammps").exists():
            continue
        try:
            df = load_thermo(rep)
        except ValueError:
            continue
        for ax, (col, label) in zip(axes.flat, PANELS):
            ax.plot(df["time_ms"], df[col], lw=1, label=rep.name)
            ax.set_ylabel(label)
    for ax in axes[-1]:
        ax.set_xlabel("time [ms]")
    axes.flat[0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    return fig
