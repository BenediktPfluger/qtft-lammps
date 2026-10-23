"""Figures and tables in Nature journal style (handover section 7).

Every figure is drawn at its final size (89 mm single column, 183 mm double column, <= 170 mm high),
Arial 7 pt labels / 6 pt ticks, 0.5 pt axes, inward ticks on all sides, 1 pt lines, frameless
legends, no grid. Saved as PDF + SVG (editable text) + PNG (300 dpi). Ensemble data: mean line with
a +-1 SD band, optionally the individual replicas. Time in ms (axes in s for long runs); dotted
vertical lines mark the start and end of the calcium influx. A figure of several panels is saved
as <name>.* and every panel alone in <name>/<panel>.*.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from .ensemble import TO_PER_M_S, binding_rate, bond_rates, cached_stats, ensemble_stats
from .params import Params
from .read import read_histo

MM = 1 / 25.4
SINGLE, DOUBLE, MAX_HEIGHT = 89 * MM, 183 * MM, 170 * MM
SPECIES = {"Qt": "#54aa00", "Ft": "#fea900", "QtC": "#55aaff", "FtC": "#fe0000"}
LINE = {"Qt": "#54aa00", "Ft": "#fe0000"}          # line plots per species
RUN_COLORS = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d35400", "#16a085", "#7f8c8d"]
CATEGORY_COLORS = ["#bdbdbd", "#9ecae1", "#4292c6", "#08519c", "#08306b"]
PRESSURE_TO_PA = 1e6                               # nano pressure unit ag/(nm ns^2) = 1 MPa

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7,
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "lines.linewidth": 1.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "axes.grid": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
}


# ---------------------------------------------------------------------------- data
@dataclass
class RunData:
    run_dir: Path
    label: str
    color: str
    p: Params
    stats: dict
    structure: pd.DataFrame | None = None
    coord_final: dict = field(default_factory=dict)    # {"Qt": Series, "Ft": Series} mean count per replica
    overlap: pd.DataFrame | None = None

    @property
    def n_total(self) -> int:
        return self.p.n_qt + self.p.n_ft


def load_run(run_dir: str | Path, label: str | None = None, color: str | None = None) -> RunData:
    """Collect everything the figures need from a run folder (uses saved analysis files if present)."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    analysis = run_dir / "analysis"
    stats = cached_stats(run_dir)
    structure = None
    if (analysis / "structure_timeseries.csv").exists():
        structure = pd.read_csv(analysis / "structure_timeseries.csv")
    overlap = None
    if (analysis / "overlap_timeseries.csv").exists():
        overlap = pd.read_csv(analysis / "overlap_timeseries.csv")
    coord = {}
    for species, fname in (("Qt", "coord_qt.hist"), ("Ft", "coord_ft.hist")):
        finals = [read_histo(run_dir / rep / fname).iloc[-1] for rep in stats["replicas"]
                  if (run_dir / rep / fname).exists()]
        if finals:
            coord[species] = sum(finals) / len(finals)
    return RunData(run_dir, label or run_dir.name, color or RUN_COLORS[0], p, stats, structure, coord, overlap)


def load_runs(run_dirs, labels=None) -> list[RunData]:
    labels = labels or [None] * len(run_dirs)
    return [load_run(r, lab, RUN_COLORS[i % len(RUN_COLORS)]) for i, (r, lab) in enumerate(zip(run_dirs, labels))]


# ---------------------------------------------------------------------------- helpers
def band(ax, t, mean, sd, color, label=None, ls="-", traces=None):
    if traces is not None:
        for tr in traces:
            ax.plot(t, tr, color=color, lw=0.3, alpha=0.4, ls=ls)
    ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=0.25, lw=0)
    few = np.count_nonzero(np.isfinite(mean)) <= 5                  # show sparse data as points
    ax.plot(t, mean, color=color, ls=ls, label=label, marker="o" if few else None, ms=2)


def _series(run: RunData, key: str, scale: float = 1.0):
    s = run.stats
    return s["time_ms"], s["mean"][key] * scale, s["sd"][key] * scale, s["traces"][key] * scale


def _structure(run: RunData, key: str):
    st = run.structure
    return st["time_ms_mean"].to_numpy(), st[f"{key}_mean"].to_numpy(), st[f"{key}_sd"].to_numpy()


LONG_RUN_MS = 10_000.0          # time axes are labelled in s from this run length on, otherwise in ms


def time_unit(t_end_ms: float) -> tuple[str, float]:
    """Unit of a time axis that ends at t_end_ms, and the factor from ms: s for long runs, otherwise ms."""
    return ("s", 1e-3) if t_end_ms >= LONG_RUN_MS else ("ms", 1.0)


def _time_axis(ax, t_end: float, label: bool = True):
    """x axis of a time series plotted in ms: limits, ticks and the unit (the data stays in ms; for
    long runs the ticks show s)."""
    ax.set_xlim(0, t_end)
    if abs(t_end - 500) < 1e-6:
        ax.xaxis.set_major_locator(MultipleLocator(250))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
    unit, factor = time_unit(t_end)
    if factor != 1.0:
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v * factor:g}"))
    if label:
        ax.set_xlabel(f"time ({unit})")


def _thousands(ax, values) -> bool:
    if np.nanmax(np.abs(values)) >= 2000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v / 1000:g}"))
        return True
    return False


def _no_data(ax, text):
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=6, color="0.4")


# ---------------------------------------------------------------------------- panels
# Each panel draws one quantity for one or several runs into `ax`. One run: species colours;
# several runs: one colour per run (and Qt solid / Ft dashed where both species are shown).

def p_energy(ax, runs, traces=False):
    for r in runs:
        t, m, s, tr = _series(r, "PotEng")
        band(ax, t, m, s, r.color, r.label, traces=tr if traces else None)
    ax.set_ylabel("potential energy (zJ)")


def p_pressure(ax, runs, traces=False):
    for r in runs:
        t, m, s, tr = _series(r, "pvir", PRESSURE_TO_PA)
        band(ax, t, m, s, r.color, r.label, traces=tr if traces else None)
    ax.set_ylabel("virial pressure (Pa)")


def _simple(key, ylabel, scale_by_n=False):
    def panel(ax, runs, traces=False):
        values = []
        for r in runs:
            t, m, s, tr = _series(r, key, 1 / r.n_total if scale_by_n else 1.0)
            band(ax, t, m, s, r.color, r.label, traces=tr if traces else None)
            values.append(m)
        k = _thousands(ax, np.concatenate(values))
        ax.set_ylabel(ylabel + (" (×10³)" if k else ""))
    return panel


def _species_panel(key, ylabel):
    """As _simple; for one run also the Qt-only and Ft-only counts (thermo columns <key>_qt, <key>_ft)."""
    def panel(ax, runs, traces=False):
        single = len(runs) == 1
        curves = [(r, key, r.color, "all" if single else r.label) for r in runs]
        if single:
            curves += [(runs[0], f"{key}_{name.lower()}", LINE[name], name) for name in ("Qt", "Ft")]
        values = []
        for r, column, color, label in curves:
            t, m, s, tr = _series(r, column)
            band(ax, t, m, s, color, label, traces=tr if traces else None)
            values.append(m)
        k = _thousands(ax, np.concatenate(values))
        ax.set_ylabel(ylabel + (" (×10³)" if k else ""))
        if single:
            ax.legend()
    return panel


p_bonds = _simple("Bonds", "bonds")
p_clusters = _simple("n_clusters", "agglomerates + free particles")
p_mean_size = _species_panel("mean_size", "mean agglomerate size")
p_mean_size_n = _simple("mean_size", "mean size / N", scale_by_n=True)
p_largest = _species_panel("largest", "largest agglomerate")
p_largest_n = _simple("largest", "largest / N", scale_by_n=True)
p_large_count = _simple("n_large", "agglomerates ≥ N_min")


def p_counts(ax, runs, traces=False):
    r = runs[0]
    t = r.stats["time_ms"]
    mean, sd = r.stats["mean"], r.stats["sd"]
    curves = {"Qt": (r.p.n_qt - mean["n_qt_bound"], sd["n_qt_bound"]),
              "Ft": (r.p.n_ft - mean["n_ft_bound"], sd["n_ft_bound"]),
              "QtC": (mean["n_qt_bound"], sd["n_qt_bound"]),
              "FtC": (mean["n_ft_bound"], sd["n_ft_bound"])}
    for name, (m, s) in curves.items():
        band(ax, t, m, s, SPECIES[name], name)
    k = _thousands(ax, np.concatenate([c[0] for c in curves.values()]))
    ax.set_ylabel("particles" + (" (×10³)" if k else ""))
    ax.legend(ncol=2, loc="best")


def p_fraction_bound(ax, runs, traces=False):
    for r in runs:
        for species, ls in (("qt", "-"), ("ft", "--")):
            t, m, s, tr = _series(r, f"frac_bound_{species}")
            color = LINE["Qt" if species == "qt" else "Ft"] if len(runs) == 1 else r.color
            label = ("Qt" if species == "qt" else "Ft") if len(runs) == 1 else f"{r.label} {species.capitalize()}"
            band(ax, t, m, s, color, label, ls=ls if len(runs) > 1 else "-", traces=tr if traces else None)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction bound")
    ax.legend(ncol=2 if len(runs) > 1 else 1)


CATEGORIES = [("f_mono", "monomers"), ("f_small", "small"), ("f_medium", "medium"),
              ("f_large", "large"), ("f_vlarge", "very large")]


def p_categories(ax, runs, traces=False):
    r = runs[0]
    t = r.stats["time_ms"]
    small, medium, large = r.p.size_bounds()
    names = {"f_mono": "1", "f_small": f"2–{small}", "f_medium": f"{small + 1}–{medium}",
             "f_large": f"{medium + 1}–{large}", "f_vlarge": f">{large}"}
    ys = [r.stats["mean"][k] for k, _ in CATEGORIES]
    labels = [f"{lab} ({names[k]})" for k, lab in CATEGORIES]
    ax.stackplot(t, ys, colors=CATEGORY_COLORS, labels=labels, lw=0)
    ax.set_ylim(0, 1)
    ax.set_ylabel("fraction of particles")
    ax.legend(loc="center right", reverse=True)


def _structure_panel(key, ylabel, ref=None):
    def panel(ax, runs, traces=False):
        drawn = False
        for r in runs:
            if r.structure is None:
                continue
            t, m, s = _structure(r, key)
            band(ax, t, m, s, r.color, r.label)
            drawn = True
        if ref is not None:
            ax.axhline(ref, color="0.5", lw=0.5, ls="--")
        if not drawn:
            _no_data(ax, "run structure analysis")
        ax.set_ylabel(ylabel)
    return panel


p_rg = _structure_panel("rg_mean", "mean Rg (nm)")
p_rg_ratio = _structure_panel("rg_ratio_mean", "Rg / Rg,ideal")
p_qt_fraction = _structure_panel("qt_frac_mean", "Qt fraction per agglomerate", ref=0.5)


def p_coordination(ax, runs, traces=False):
    for r in runs:
        for species, ls in (("qt", "-"), ("ft", "--")):
            t, m, s, _ = _series(r, f"deg_{species}_mean")
            bound = r.stats["mean"][f"n_{species}_bound"] > 0           # undefined before the first bond
            t, m, s = t[bound], m[bound], s[bound]
            name = "Qt" if species == "qt" else "Ft"
            color = LINE[name] if len(runs) == 1 else r.color
            band(ax, t, m, s, color, name if len(runs) == 1 else f"{r.label} {name}",
                 ls=ls if len(runs) > 1 else "-")
    ax.set_ylabel("mean coordination")
    if len(runs) == 1:
        ax.legend()


def p_coord_dist(ax, runs, traces=False):
    r = runs[0]
    if not r.coord_final:
        _no_data(ax, "no histogram files")
        return
    width, highest = 0.4, 1
    for i, name in enumerate(("Qt", "Ft")):
        h = r.coord_final.get(name)
        if h is None:
            continue
        h = h[h.index >= 1]                                       # bound particles only
        if (h > 0).any():
            highest = max(highest, h.index[h > 0].max())
        ax.bar(h.index + (i - 0.5) * width, h.to_numpy(), width=width,
               color=SPECIES["QtC" if name == "Qt" else "FtC"], label=f"{name}C", lw=0)
    ax.set_xlim(0.3, highest + 0.7)
    ax.set_xlabel("bonds per particle (final)")
    ax.set_ylabel("particles per replica")
    ax.legend()


def p_overlap(ax, runs, traces=False):
    r = runs[0]
    if r.overlap is None:
        _no_data(ax, "run overlap analysis")
        return
    colors = {"Qt-Qt": SPECIES["Qt"], "Ft-Ft": SPECIES["Ft"], "Qt-Ft": "#555555",
              "Qt-wall": SPECIES["Qt"], "Ft-wall": SPECIES["Ft"]}            # wall rows only with walls
    for family, df in r.overlap.groupby("family"):
        ax.plot(df["time_ms"], df["n overlapping"].clip(lower=0.5), color=colors[family], label=family,
                ls="--" if family.endswith("wall") else "-")
    ax.set_yscale("log")
    ax.set_ylabel("overlapping pairs")
    ax.legend()


def p_calcium(ax, runs, traces=False):
    r = runs[0]
    t, mean = r.stats["time_ms"], r.stats["mean"]
    ax.plot(t, mean["ca"], color="k", label="Ca²⁺")
    ax.set_ylabel("calcium (µM)")
    a2 = ax.twinx()
    a2.plot(t, mean["f"], color=LINE["Qt"], ls="--", label="loading f")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("calmodulin loading f")
    ax.legend(handles=ax.get_lines() + a2.get_lines(), loc="upper right")


def p_rates(ax, runs, traces=False):
    r = runs[0]
    t, mean = r.stats["time_ms"], r.stats["mean"]
    ax.plot(t, mean["k_on_s"], color=LINE["Qt"], label="k_on per pair")
    ax.plot(t, mean["k_off_s"], color=LINE["Ft"], label="k_off per bond")
    ax.set_yscale("log")
    ax.set_ylabel("rate (1/s)")
    ax.legend(loc="lower right")


def p_bonds_broken(ax, runs, traces=False):
    r = runs[0]
    t, mean, sd, tr = r.stats["time_ms"], r.stats["mean"], r.stats["sd"], r.stats["traces"]
    band(ax, t, mean["Bonds"], sd["Bonds"], r.color, "bonds", traces=tr["Bonds"] if traces else None)
    band(ax, t, mean["n_broken"], sd["n_broken"], "0.5", "broken (cumulative)", ls="--")
    ax.set_ylabel("bonds")
    ax.legend(loc="upper left")


def _rate_panels(rates: dict) -> list:
    """Panels of fig_bond_rates for a result of bond_rates()."""
    t, mean, sd, tr = rates["time_ms"], rates["mean"], rates["sd"], rates["traces"]

    def formed_broken(ax, runs, traces=False):
        band(ax, t, mean["formed_per_s"], sd["formed_per_s"], runs[0].color, "formed",
             traces=tr["formed_per_s"] if traces else None)
        band(ax, t, mean["broken_per_s"], sd["broken_per_s"], "0.5", "broken", ls="--",
             traces=tr["broken_per_s"] if traces else None)
        k = _thousands(ax, np.concatenate([mean["formed_per_s"], mean["broken_per_s"]]))
        ax.set_ylabel("bonds per second" + (" (×10³)" if k else ""))
        ax.legend(loc="upper right")

    def k_a(ax, runs, traces=False):
        band(ax, t, mean["k_a_per_M_s"], sd["k_a_per_M_s"], runs[0].color,
             traces=tr["k_a_per_M_s"] if traces else None)
        ax.set_yscale("log")
        ax.set_ylabel("apparent k_a (M⁻¹ s⁻¹)")

    return [("bonds formed and broken", formed_broken, True), ("apparent k_a", k_a, True)]


# ---------------------------------------------------------------------------- figures
def _figure(width, height):
    return plt.figure(figsize=(width, height), layout="constrained")


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)


def _t_end(runs) -> float:
    return max(r.stats["time_ms"][-1] for r in runs)


def phase_lines(ax, runs) -> None:
    """Dotted vertical lines at the start and end of the calcium influx (time axis in ms), where they lie
    inside the run; none if the runs have different phase times."""
    bounds = {(r.p.t_on_s, r.p.t_off_s) for r in runs}
    if len(bounds) != 1:
        return
    end = _t_end(runs)
    for t in sorted({b * 1e3 for b in bounds.pop()}):
        if 0 < t < end:
            ax.axvline(t, color="0.6", lw=0.5, ls=":", zorder=0)


def _grid(panels, runs, nrows, ncols, width, height, titles=True, traces=False, legend_below=False,
          sharex=False):
    fig = _figure(width, height)
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex)
    t_end = _t_end(runs)
    for ax, (title, func, is_time) in zip(axes.flat, panels):
        func(ax, runs, traces=traces)
        if is_time:
            phase_lines(ax, runs)
            _time_axis(ax, t_end, label=not sharex or ax.get_subplotspec().is_last_row())
        if titles:
            ax.set_title(title, loc="left", fontweight="bold")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    if legend_below and len(runs) > 1:
        handles, labels = axes.flat[2].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=min(len(runs), 4))
    return fig


def _single_panels(panels, runs, folder: Path, traces=False):
    """Every panel of a figure alone (89 mm), into folder/<panel>.*; several runs get a legend."""
    t_end = _t_end(runs)
    for title, func, is_time in panels:
        fig = _figure(SINGLE, 65 * MM)
        ax = fig.add_subplot()
        func(ax, runs, traces=traces)
        if is_time:
            phase_lines(ax, runs)
            _time_axis(ax, t_end)
        if len(runs) > 1 and ax.get_legend() is None and ax.get_legend_handles_labels()[0]:
            ax.legend()
        save(fig, folder / _slug(title))
        plt.close(fig)


def _panel_figure(name, panels, runs, out_dir, nrows, ncols, width, height, traces=False, **grid):
    """A figure of several panels: saved as out_dir/<name>.* and every panel alone in out_dir/<name>/."""
    with plt.rc_context(STYLE):
        fig = _grid(panels, runs, nrows, ncols, width, height, traces=traces, **grid)
        if out_dir:
            save(fig, out_dir / name)
            _single_panels(panels, runs, out_dir / name, traces=traces)
    return fig


OVERVIEW = [
    ("potential energy", p_energy, True), ("pressure", p_pressure, True), ("bonds", p_bonds, True),
    ("particle counts", p_counts, True), ("agglomerates + free", p_clusters, True),
    ("mean agglomerate size", p_mean_size, True),
    ("largest agglomerate", p_largest, True), ("size categories", p_categories, True),
    ("mean Rg", p_rg, True),
    ("Qt fraction", p_qt_fraction, True), ("coordination", p_coordination, True),
    ("final coordination", p_coord_dist, False),
]

COMPARISON = [
    ("potential energy", p_energy, True), ("pressure", p_pressure, True), ("bonds", p_bonds, True),
    ("agglomerates + free", p_clusters, True),
    ("mean size", p_mean_size, True), ("mean size / N", p_mean_size_n, True),
    ("largest", p_largest, True), ("largest / N", p_largest_n, True),
    ("mean Rg", p_rg, True), ("Rg / Rg,ideal", p_rg_ratio, True),
    ("coordination", p_coordination, True), ("Qt fraction", p_qt_fraction, True),
]


def fig_overview(run: RunData, out_dir: Path | None = None, traces=False):
    """4 x 3 metrics overview of one run (183 mm); panels also alone in overview/."""
    return _panel_figure("overview", OVERVIEW, [run], out_dir, 4, 3, DOUBLE, MAX_HEIGHT, traces=traces)


def fig_comparison(runs: list[RunData], out_dir: Path | None = None):
    """3 x 4 overlay of several runs with a shared legend below (183 mm); panels also alone in comparison/."""
    return _panel_figure("comparison", COMPARISON, runs, out_dir, 3, 4, DOUBLE, 140 * MM, legend_below=True)


KINETICS = [("bonds", p_bonds, True), ("fraction bound", p_fraction_bound, True),
            ("mean agglomerate size", p_mean_size, True)]


def fig_kinetics(runs: list[RunData], out_dir: Path | None = None, traces=False):
    """Bonds, fraction bound (Qt, Ft) and mean agglomerate size on one time axis (89 mm); panels also
    alone in kinetics/."""
    return _panel_figure("kinetics", KINETICS, runs, out_dir, 3, 1, SINGLE, 120 * MM, traces=traces,
                         titles=False, sharex=True, legend_below=True)


def fig_large_count(runs: list[RunData], out_dir: Path | None = None, traces=False):
    """Number of agglomerates with >= N_min particles over time (89 mm)."""
    with plt.rc_context(STYLE):
        fig = _figure(SINGLE, 60 * MM)
        ax = fig.add_subplot()
        p_large_count(ax, runs, traces=traces)
        ax.set_ylabel(f"agglomerates ≥ {runs[0].p.large_cluster_min}")
        phase_lines(ax, runs)
        _time_axis(ax, _t_end(runs))
        if len(runs) > 1:
            ax.legend()
        if out_dir:
            save(fig, out_dir / "large_agglomerates")
    return fig


def fig_overlap(run: RunData, out_dir: Path | None = None):
    """Number of overlapping pairs per family over time, log scale (89 mm)."""
    with plt.rc_context(STYLE):
        fig = _figure(SINGLE, 60 * MM)
        ax = fig.add_subplot()
        p_overlap(ax, [run])
        phase_lines(ax, [run])
        _time_axis(ax, run.stats["time_ms"][-1])
        if out_dir:
            save(fig, out_dir / "overlap")
    return fig


def _sci(x: float, sd: float = float("nan")) -> str:
    """1.91e7 -> '1.91·10⁷'; with an SD: '(1.91 ± 0.05)·10⁷'."""
    exponent = math.floor(math.log10(abs(x))) if x else 0
    power = "·10" + str(exponent).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    if sd == sd:                                                        # not NaN
        return f"({x / 10**exponent:.2f} ± {sd / 10**exponent:.2f}){power}"
    return f"{x / 10**exponent:.3g}{power}"


def fig_binding_rate(run: RunData, out_dir: Path | None = None, max_frac_bound: float = 0.1,
                     rate: dict | None = None):
    """Bonds formed since the start of the calcium influx (mean +- SD over replicas) with the fitted
    initial rate (solid) and the Collins-Kimball prediction from the counted contact pairs (dashed); k_a in
    1/(M s) in the legend (89 mm). `rate`: a result of binding_rate() (computed here if not given)."""
    rate = rate or binding_rate(run.run_dir, max_frac_bound=max_frac_bound, save=False)
    s = rate["summary"]["value"]
    pairs_per_volume = run.p.n_qt * run.p.n_ft / math.prod(run.p.box)
    window = rate["window_ms"]
    on_ms, off_ms = run.p.t_on_s * 1e3, run.p.t_off_s * 1e3
    t_all = run.stats["time_ms"]
    show = (t_all >= on_ms) & (t_all <= min(on_ms + 2 * window, off_ms))
    bonds = run.stats["mean"]["Bonds"]
    with plt.rc_context(STYLE):
        fig = _figure(SINGLE, 60 * MM)
        ax = fig.add_subplot()
        t = t_all[show] - on_ms
        band(ax, t, bonds[show] - bonds[show][0], run.stats["sd"]["Bonds"][show], run.color, "simulation")
        def label(name, key):
            return f"{name}: {_sci(s[key] * TO_PER_M_S, s[f'{key} SD'] * TO_PER_M_S)} M⁻¹ s⁻¹"

        ax.plot(t, s["k_a measured"] * pairs_per_volume * t * 1e6, color="k", lw=0.8,
                label=label("fit k_a", "k_a measured"))
        ax.plot(t, s["k_a predicted"] * pairs_per_volume * t * 1e6, color="0.4", lw=0.8, ls="--",
                label=label("Collins–Kimball", "k_a predicted"))
        ax.axvline(window, color="0.6", lw=0.5, ls=":")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time since influx start (ms)")
        ax.set_ylabel("bonds formed")
        ax.legend(loc="upper left")
        if out_dir:
            save(fig, out_dir / "binding_rate")
    return fig


CALCIUM = [("calcium and loading", p_calcium, True), ("rates", p_rates, True),
           ("bonds and broken bonds", p_bonds_broken, True), ("fraction bound", p_fraction_bound, True),
           ("largest agglomerate", p_largest, True), ("mean agglomerate size", p_mean_size, True)]


def fig_calcium(run: RunData, out_dir: Path | None = None, traces=False):
    """Calcium protocol and its effect (183 mm): calcium and calmodulin loading f; binding rate per pair
    and unbinding rate per bond; bonds and broken bonds; fraction bound; largest agglomerate and mean
    agglomerate size, each for all particles, Qt and Ft (mean +- SD over replicas); panels also alone
    in calcium/."""
    return _panel_figure("calcium", CALCIUM, [run], out_dir, 3, 2, DOUBLE, 140 * MM, traces=traces,
                         titles=False, sharex=True)


def fig_bond_rates(run: RunData, out_dir: Path | None = None, traces=False, rates: dict | None = None):
    """Bonds formed and broken per second (top) and the apparent association rate constant k_a in
    1/(M s), normalized by the total particle numbers (bottom, log scale), per calcium segment (mean +- SD
    over replicas; 89 mm); panels also alone in bond_rates/. `rates`: a result of bond_rates() (computed
    here if not given)."""
    rates = rates or bond_rates(run.run_dir, save=False)
    return _panel_figure("bond_rates", _rate_panels(rates), [run], out_dir, 2, 1, SINGLE, 90 * MM,
                         traces=traces, titles=False, sharex=True)


SWEEP_LABELS = {
    "final_frac_bound": "fraction bound (final)", "final_Bonds": "bonds (final)",
    "final_largest_frac": "largest cluster (fraction of particles, final)",
    "final_mean_size": "mean cluster size (final)", "t_half_bonds_ms": "half-time of bonds (ms)",
    "t_half_qt_ms": "half-time Qt bound (ms)", "t_half_ft_ms": "half-time Ft bound (ms)",
    "t_percolation_ms": "percolation time (ms)", "bonds_peak": "bonds (peak)",
    "largest_peak": "largest cluster (peak)", "t_half_unbind_bonds_ms": "deagglomeration half-time, bonds (ms)",
    "t_half_unbind_largest_ms": "deagglomeration half-time, largest cluster (ms)",
}


def sweep_plot(results: pd.DataFrame, y: str = "final_frac_bound", x: str | None = None,
               out_dir: Path | None = None):
    """A result of `sweep_results` (mean +- SD over replicas) against one swept parameter (default: the
    first); one line per combination of the other swept parameters (89 mm)."""
    swept = results.attrs.get("swept") or []
    x = x or swept[0]
    others = [k for k in swept if k != x]
    with plt.rc_context(STYLE):
        fig = _figure(SINGLE, 60 * MM)
        ax = fig.add_subplot()
        groups = results.groupby(others, sort=False) if others else [((), results)]
        for i, (key, df) in enumerate(groups):
            key = key if isinstance(key, tuple) else (key,)
            df = df.sort_values(x)
            label = ", ".join(f"{k} = {v:g}" if isinstance(v, (int, float)) else f"{k} = {v}"
                              for k, v in zip(others, key)) or None
            ax.errorbar(df[x], df[y], yerr=df.get(f"{y} sd"), marker="o", ms=3, capsize=2,
                        color=RUN_COLORS[i % len(RUN_COLORS)], label=label)
        ax.set_xlabel(x)
        ax.set_ylabel(SWEEP_LABELS.get(y, y))
        if others:
            ax.legend()
        if out_dir:
            save(fig, out_dir / f"sweep_{y}")
    return fig


# ---------------------------------------------------------------------------- tables
def final_table(run: RunData) -> pd.DataFrame:
    """Final-state table (handover 6.8): one row per metric, mean +- SD over replicas."""
    f = run.stats["final"]
    n = run.n_total

    def ms(key, scale=1.0):
        return f.loc[key, "mean"] * scale, f.loc[key, "sd"] * scale

    rows = [
        ("number of bonds", *ms("final_Bonds")),
        ("number of clusters (incl. free particles)", *ms("final_n_clusters")),
        ("mean cluster size", *ms("final_mean_size")),
        ("largest cluster (particles)", *ms("final_largest")),
        ("largest cluster (% of all)", *ms("final_largest", 100 / n)),
        ("fraction bound", *ms("final_frac_bound")),
        ("half-time of bonds (ms)", *ms("t_half_bonds_ms")),
    ]
    if run.structure is not None:
        last = run.structure.iloc[-1]
        rows += [("mean Rg (nm)", last["rg_mean_mean"], last["rg_mean_sd"]),
                 ("mean Qt fraction", last["qt_frac_mean_mean"], last["qt_frac_mean_sd"])]
    return pd.DataFrame(rows, columns=["metric", "mean", "sd"]).set_index("metric")


def to_latex(table: pd.DataFrame, caption: str = "") -> str:
    """Minimal LaTeX tabular of a table with 'mean'/'sd' columns (or preformatted strings)."""
    lines = [r"\begin{tabular}{l" + "r" * (1 if "mean" in table.columns else len(table.columns)) + "}",
             r"\hline"]
    if "mean" in table.columns:
        lines.append(r"Metric & Value \\ \hline")
        for name, row in table.iterrows():
            lines.append(f"{_tex(name)} & ${row['mean']:.4g} \\pm {row['sd']:.2g}$ \\\\")
    else:
        lines.append("Metric & " + " & ".join(_tex(c) for c in table.columns) + r" \\ \hline")
        for name, row in table.iterrows():
            lines.append(_tex(name) + " & " + " & ".join(_tex(str(v)).replace("±", r"$\pm$") for v in row) + r" \\")
    lines += [r"\hline", r"\end{tabular}"]
    if caption:
        lines.insert(0, f"% {caption}")
    return "\n".join(lines) + "\n"


def _tex(text: str) -> str:
    return str(text).replace("%", r"\%").replace("_", r"\_").replace("≥", r"$\geq$")


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() else "_" for c in text).strip("_").lower()


# ---------------------------------------------------------------------------- report
def make_report(run_dir: str | Path, every: int = 10, structure_analysis: bool = True,
                overlap_analysis: bool = True, traces: bool = False) -> Path:
    """Full analysis of one run: ensemble statistics, structure (OVITO) and overlap analysis,
    all figures and tables into <run>/analysis/."""
    run_dir = Path(run_dir).expanduser()
    out = run_dir / "analysis"
    stats = ensemble_stats(run_dir)
    if structure_analysis:
        from .structure import structure_ensemble
        structure_ensemble(run_dir, every=every)
    if overlap_analysis:
        from .overlap import overlap_ensemble
        overlap_ensemble(run_dir, every=every)
    run = load_run(run_dir)

    figs = out / "figures"
    rate = [fig_binding_rate(run, figs, rate=binding_rate(run_dir))] if run.p.t_influx_s > 0 and run.p.ca_rise_s == 0 else []  # + csv
    for f in (fig_calcium(run, figs, traces=traces), fig_bond_rates(run, figs, traces, bond_rates(run_dir)),  # + csv
              fig_overview(run, figs, traces=traces), fig_kinetics([run], figs, traces=traces),
              fig_large_count([run], figs, traces=traces), fig_overlap(run, figs), *rate):
        plt.close(f)

    table = final_table(run)
    table.to_csv(out / "final_table.csv")
    (out / "final_table.tex").write_text(to_latex(table, run.label))
    t = stats["time_ms"]
    pd.DataFrame({"Step": np.round(t / (run.p.timestep * 1e-6)).astype(int), "time_us": t * 1e3,
                  "time_ms": t, "bonds_mean": stats["mean"]["Bonds"], "bonds_sd": stats["sd"]["Bonds"]}
                 ).to_csv(out / "bonds_timeseries.csv", index=False)
    (out / "report.json").write_text(json.dumps({"every": every, "replicas": stats["replicas"]}))
    return out


def compare_report(run_dirs, out_dir: str | Path, labels=None) -> Path:
    """Comparison figures and table of several runs into out_dir."""
    from .ensemble import compare, differing_labels
    out_dir = Path(out_dir).expanduser()
    runs = load_runs(run_dirs, labels or differing_labels(run_dirs)[0])
    for f in (fig_comparison(runs, out_dir), fig_kinetics(runs, out_dir), fig_large_count(runs, out_dir)):
        plt.close(f)
    table = compare(run_dirs)
    table.to_csv(out_dir / "comparison_table.csv")
    (out_dir / "comparison_table.tex").write_text(to_latex(table))
    return out_dir
