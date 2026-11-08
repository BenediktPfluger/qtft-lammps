"""Figures and tables in Nature journal style (every figure and panel: docs/repo_structure.md, section 6).

Every figure is drawn at its final size (89 mm single column, 183 mm double column, <= 170 mm high),
Arial 7 pt labels / 6 pt ticks, 0.5 pt axes, inward ticks on all sides, 1 pt lines, frameless
legends, no grid. Saved as PDF + SVG (editable text) + PNG (300 dpi). Ensemble data: mean line with
a +-1 SD band, optionally the individual replicas. Time in ms (axes in s for long runs); dotted
vertical lines mark the start and end of the calcium influx. Every figure is saved in its own folder,
<name>/<name>.*; a figure of several panels also has every panel alone in <name>/<panel>.*.
make_report, compare_report and sweep_report also collect all their figures in one report.pdf, after
cover pages with the run names, the summary table and the simulation parameters.
"""

from __future__ import annotations

import datetime
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.ticker import MultipleLocator, NullFormatter
from matplotlib.transforms import offset_copy

from .ensemble import (SIZE_THRESHOLDS, SIZE_THRESHOLDS_QT, SIZE_THRESHOLDS_QT_FT, TO_PER_M_S, cluster_sizes,
                       _agglomerates_above_size_species, _thresholds, _unique, agglomerates_above_size, binding_rate,
                       bond_rates, ensemble_stats, ensemble_stats_cached, final_table)
from .params import Params
from .read import read_histo
from .spatial import QT_MIN, density_estimate, peak, qt_spatial_run
from .structure import AGGLOMERATE_MIN, agglomerate_volumes_run

MM = 1 / 25.4
SINGLE, ONE_HALF, DOUBLE, MAX_HEIGHT = 89 * MM, 120 * MM, 183 * MM, 170 * MM
SPECIES_COLORS = {"Qt": "#54aa00", "Ft": "#fea900", "Qt bound": "#55aaff", "Ft bound": "#fe0000"}
LINE = {"Qt": "#54aa00", "Ft": "#fe0000"}          # line plots per species
RUN_COLORS = ["#1f4e79", "#c0392b", "#2e8b57", "#8e44ad", "#d35400", "#16a085", "#7f8c8d"]
CATEGORY_COLORS = ["#bdbdbd", "#9ecae1", "#4292c6", "#08519c", "#08306b"]
PRESSURE_TO_PA = 1e6                               # nano pressure unit ag/(nm ns^2) = 1 MPa

STYLE = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
    "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 6, "axes.titleweight": "normal",
    "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
    "axes.linewidth": 0.5, "lines.linewidth": 1.0,
    "xtick.major.width": 0.5, "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4, "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5, "ytick.major.size": 2.5, "xtick.minor.size": 1.5, "ytick.minor.size": 1.5,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "legend.frameon": False, "axes.grid": False,
    "pdf.fonttype": 42, "svg.fonttype": "none", "savefig.dpi": 300,
}
GUIDE_LW = 0.5                                     # thin guide lines: phases, references, peaks, random points
TRACE_LW, TRACE_ALPHA = 0.3, 0.4                   # individual replicas (traces=True)
BAND_ALPHA = 0.25                                  # ±1 SD band


# ---------------------------------------------------------------------------- data
@dataclass
class RunData:
    run_dir: Path
    label: str
    color: str
    p: Params
    stats: dict
    structure: pd.DataFrame | None = None
    coord_influx_end: dict = field(default_factory=dict)  # {"Qt"/"Ft": Series} mean count per replica, end of influx
    overlap: pd.DataFrame | None = None
    agglomerates: pd.DataFrame | None = None           # agglomerates_influx_end.csv (sizes and volumes)
    _sizes: pd.DataFrame | None = field(default=None, repr=False)
    _rates: dict | None = field(default=None, repr=False)

    @property
    def n_total(self) -> int:
        return self.p.n_qt + self.p.n_ft

    @property
    def sizes(self) -> pd.DataFrame:
        """cluster_sizes of the run (computed from traj.dump or read from its cache on first use)."""
        if self._sizes is None:
            self._sizes = cluster_sizes(self.run_dir)
        return self._sizes

    @property
    def rates(self) -> dict:
        """bond_rates of the run (computed from the thermo data on first use, not saved)."""
        if self._rates is None:
            self._rates = bond_rates(self.run_dir, save=False)
        return self._rates


def load_run(run_dir: str | Path, label: str | None = None, color: str | None = None) -> RunData:
    """Collect everything the figures need from a run folder (uses saved analysis files if present)."""
    run_dir = Path(run_dir).expanduser()
    p = Params.load(run_dir / "params.json")
    analysis = run_dir / "analysis"
    stats = ensemble_stats_cached(run_dir)
    structure = None
    if (analysis / "structure_timeseries.csv").exists():
        structure = pd.read_csv(analysis / "structure_timeseries.csv")
    overlap = None
    if (analysis / "overlap_timeseries.csv").exists():
        overlap = pd.read_csv(analysis / "overlap_timeseries.csv")
    coord = {}
    for species, fname in (("Qt", "coord_qt.hist"), ("Ft", "coord_ft.hist")):
        rows = []
        for rep in stats["replicas"]:
            if (run_dir / rep / fname).exists():
                h = read_histo(run_dir / rep / fname)
                rows.append(h[h.index <= p.influx_end_step].iloc[-1])     # last row at or before the influx end
        if rows:
            coord[species] = sum(rows) / len(rows)
    agglomerates = None
    if (analysis / "agglomerates_influx_end.csv").exists():
        agglomerates = pd.read_csv(analysis / "agglomerates_influx_end.csv")
        agglomerates["spans"] = agglomerates["spans"].fillna("")            # "" = spans no axis
    return RunData(run_dir, label or run_dir.name, color or RUN_COLORS[0], p, stats, structure, coord, overlap,
                   agglomerates)


def load_runs(run_dirs, labels=None) -> list[RunData]:
    """load_run of several runs, one colour each; labels (default: the folder names) made unique with (1), (2), ...
    as in final_table, so every run has its own legend entry."""
    labels = _unique([lab or Path(r).expanduser().name for r, lab in zip(run_dirs, labels or [None] * len(run_dirs))])
    return [load_run(r, lab, RUN_COLORS[i % len(RUN_COLORS)]) for i, (r, lab) in enumerate(zip(run_dirs, labels))]


# ---------------------------------------------------------------------------- helpers
def band(ax, t, mean, sd, color, label=None, ls="-", traces=None):
    if traces is not None:
        for tr in traces:
            ax.plot(t, tr, color=color, lw=TRACE_LW, alpha=TRACE_ALPHA, ls=ls)
    ax.fill_between(t, mean - sd, mean + sd, color=color, alpha=BAND_ALPHA, lw=0)
    few = np.count_nonzero(np.isfinite(mean)) <= 5                  # show sparse data as points
    ax.plot(t, mean, color=color, ls=ls, label=label, marker="o" if few else None, ms=2)


def _series(run: RunData, key: str, scale: float = 1.0, traces: bool = False):
    """Time, mean, SD and (only with traces, read on demand) the replica traces of a thermo column."""
    s = run.stats
    return s["time_ms"], s["mean"][key] * scale, s["sd"][key] * scale, s["traces"][key] * scale if traces else None


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
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center", fontsize=plt.rcParams["legend.fontsize"], color="0.4")


# ---------------------------------------------------------------------------- panels
# Each panel draws one quantity for one or several runs into `ax`. One run: species colours;
# several runs: one colour per run (and Qt solid / Ft dashed where both species are shown).

def p_energy(ax, runs, traces=False):
    for r in runs:
        t, m, s, tr = _series(r, "PotEng", traces=traces)
        band(ax, t, m, s, r.color, r.label, traces=tr if traces else None)
    ax.set_ylabel("potential energy (zJ)")


def p_pressure(ax, runs, traces=False):
    for r in runs:
        t, m, s, tr = _series(r, "pvir", PRESSURE_TO_PA, traces=traces)
        band(ax, t, m, s, r.color, r.label, traces=tr if traces else None)
    ax.set_ylabel("virial pressure (Pa)")


def _simple(key, ylabel, scale_by_n=False):
    def panel(ax, runs, traces=False):
        values = []
        for r in runs:
            t, m, s, tr = _series(r, key, 1 / r.n_total if scale_by_n else 1.0, traces=traces)
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
            t, m, s, tr = _series(r, column, traces=traces)
            band(ax, t, m, s, color, label, traces=tr if traces else None)
            values.append(m)
        k = _thousands(ax, np.concatenate(values))
        ax.set_ylabel(ylabel + (" (×10³)" if k else ""))
        if single:
            ax.legend()
    return panel


p_clusters = _simple("n_clusters", "agglomerates + free particles")
p_largest = _species_panel("largest", "largest agglomerate")
p_largest_n = _simple("largest", "largest / N", scale_by_n=True)


COUNT_STYLES = {"Qt bound": "-", "Ft bound": "--", "Qt free": ":", "Ft free": "-."}   # several runs


def p_counts(ax, runs, traces=False):
    """Free and bound Qt and Ft. One run: species colours; several runs: run colours, one line style per
    count (COUNT_STYLES, legend below the axes, under the time label, so it covers no curve)."""
    values = []
    for r in runs:
        t = r.stats["time_ms"]
        mean, sd = r.stats["mean"], r.stats["sd"]
        curves = {"Qt free": (r.p.n_qt - mean["n_qt_bound"], sd["n_qt_bound"]),
                  "Ft free": (r.p.n_ft - mean["n_ft_bound"], sd["n_ft_bound"]),
                  "Qt bound": (mean["n_qt_bound"], sd["n_qt_bound"]),
                  "Ft bound": (mean["n_ft_bound"], sd["n_ft_bound"])}
        for name, (m, s) in curves.items():
            if len(runs) == 1:
                band(ax, t, m, s, SPECIES_COLORS[name.removesuffix(" free")], name)
            else:
                band(ax, t, m, s, r.color, r.label if name == "Qt bound" else None, ls=COUNT_STYLES[name])
            values.append(m)
    k = _thousands(ax, np.concatenate(values))
    ax.set_ylabel("particles" + (" (×10³)" if k else ""))
    if len(runs) == 1:
        ax.legend(ncol=2, loc="best")
    else:                                          # 26 pt below the axes: under the tick labels and time label
        below = offset_copy(ax.transAxes, fig=ax.figure, y=-26, units="points")
        _style_legend(ax, COUNT_STYLES, ncol=2, handlelength=2.5, columnspacing=1.0, loc="upper center",
                      bbox_to_anchor=(0.5, 0), bbox_transform=below)


def p_fraction_bound(ax, runs, traces=False):
    for r in runs:
        for species, ls in (("qt", "-"), ("ft", "--")):
            t, m, s, tr = _series(r, f"frac_bound_{species}", traces=traces)
            color = LINE["Qt" if species == "qt" else "Ft"] if len(runs) == 1 else r.color
            label = ("Qt" if species == "qt" else "Ft") if len(runs) == 1 else None
            band(ax, t, m, s, color, label, ls=ls if len(runs) > 1 else "-", traces=tr if traces else None)
    ax.set_ylim(0, 1.02)
    ax.set_ylabel("fraction bound")
    if len(runs) == 1:
        ax.legend()
    else:
        _species_style_legend(ax)


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
            ax.axhline(ref, color="0.5", lw=GUIDE_LW, ls="--")
        if not drawn:
            _no_data(ax, "no structure analysis (make_report or structure_run)")
        ax.set_ylabel(ylabel)
    return panel


p_rg = _structure_panel("rg_mean", "mean Rg (nm)")
p_rg_ratio = _structure_panel("rg_ratio_mean", "Rg / Rg,ideal")
p_qt_fraction = _structure_panel("qt_frac_mean", "Qt fraction per agglomerate", ref=0.5)


def _style_legend(ax, styles: dict, **kwargs) -> None:
    """Legend of line styles {label: style} in grey (several runs; the run colours are in the legend below)."""
    ax.legend([plt.Line2D([], [], color="0.3", ls=ls) for ls in styles.values()], list(styles), **kwargs)


def _species_style_legend(ax) -> None:
    _style_legend(ax, {"Qt": "-", "Ft": "--"})


def p_coordination(ax, runs, traces=False):
    for r in runs:
        for species, ls in (("qt", "-"), ("ft", "--")):
            t, m, s, _ = _series(r, f"deg_{species}_mean")
            bound = r.stats["mean"][f"n_{species}_bound"] > 0           # undefined before the first bond
            t, m, s = t[bound], m[bound], s[bound]
            name = "Qt" if species == "qt" else "Ft"
            if len(runs) == 1:
                band(ax, t, m, s, LINE[name], name)
            else:
                band(ax, t, m, s, r.color, r.label if species == "qt" else None, ls=ls)
    ax.set_ylabel("mean coordination")
    if len(runs) == 1:
        ax.legend()
    else:
        _species_style_legend(ax)


def p_coordination_hist(ax, runs, traces=False):
    """Bound particles per number of bonds at the end of the influx, per replica. One run: bars per species;
    several runs: steps, run colours, Qt solid and Ft dashed."""
    highest, drawn = 1, False
    for r in runs:
        for i, name in enumerate(("Qt", "Ft")):
            h = r.coord_influx_end.get(name)
            if h is None:
                continue
            h = h[h.index >= 1]                                       # bound particles only
            if (h > 0).any():
                highest = max(highest, h.index[h > 0].max())
            if len(runs) == 1:
                width = 0.4
                ax.bar(h.index + (i - 0.5) * width, h.to_numpy(), width=width,
                       color=SPECIES_COLORS[f"{name} bound"], label=f"{name} bound", lw=0)
            else:
                ax.step(h.index, h.to_numpy(), where="mid", color=r.color, ls="-" if name == "Qt" else "--",
                        label=r.label if name == "Qt" else None)
            drawn = True
    if not drawn:
        _no_data(ax, "no histogram files")
        return
    ax.set_xlim(0.3, highest + 0.7)
    ax.set_xlabel("bonds per particle")
    ax.set_ylabel("particles per replica")
    if len(runs) == 1:
        ax.legend()
    else:
        _species_style_legend(ax)


def _agglomerate_hist(column, xlabel):
    """Histogram (log bins) of one column of the agglomerates at the end of the influx: one run per replica
    (filled), several runs per 1000 um^3 of box volume (outlines, a colour per run, common bins)."""
    def panel(ax, runs, traces=False):
        data = [(r, r.agglomerates[column].dropna().to_numpy(dtype=float)) for r in runs
                if r.agglomerates is not None]
        if not data:
            _no_data(ax, "run make_report")
            return
        values = np.concatenate([v for _, v in data])
        if not len(values):
            _no_data(ax, f"no agglomerates of ≥ {AGGLOMERATE_MIN} particles")
            return
        lo, hi = values.min(), values.max()
        bins = np.geomspace(lo, hi, 21) if hi > lo else np.array([lo / 1.1, hi * 1.1])
        for r, v in data:
            n_rep = len(r.stats["replicas"])          # also the replicas without any agglomerate
            if len(runs) == 1:
                ax.hist(v, bins=bins, weights=np.full(len(v), 1 / n_rep), color=r.color, lw=0)
            else:
                per = 1000 / (n_rep * math.prod(r.p.box) * 1e-9)                 # per 1000 um^3 of box
                ax.hist(v, bins=bins, weights=np.full(len(v), per), histtype="step", color=r.color,
                        label=r.label)
        ax.set_xscale("log")
        ax.xaxis.set_minor_formatter(NullFormatter())                   # labels at the decades only
        ax.set_xlabel(xlabel)
        ax.set_ylabel("agglomerates per replica" if len(runs) == 1 else "agglomerates / 1000 µm³")
    return panel


p_agglomerate_size = _agglomerate_hist("size", "particles per agglomerate")
p_volume_solid = _agglomerate_hist("v_solid_um3", "solid volume (µm³)")
p_volume_envelope = _agglomerate_hist("v_envelope_um3", "envelope volume (µm³)")


# line styles of the pair families in p_overlap (wall rows only with walls)
OVERLAP_STYLES = {"Qt-Qt": "-", "Ft-Ft": "--", "Qt-Ft": ":", "Qt-wall": "-.", "Ft-wall": (0, (5, 1, 1, 1, 1, 1))}


def p_overlap(ax, runs, traces=False):
    """Overlapping pairs per family over time, log scale. One run: a colour per family (walls dashed);
    several runs: a colour per run, a line style per family (style legend in the panel)."""
    colors = {"Qt-Qt": SPECIES_COLORS["Qt"], "Ft-Ft": SPECIES_COLORS["Ft"], "Qt-Ft": "#555555",
              "Qt-wall": SPECIES_COLORS["Qt"], "Ft-wall": SPECIES_COLORS["Ft"]}
    families = []
    for r in runs:
        if r.overlap is None:
            continue
        groups = dict(tuple(r.overlap.groupby("family")))
        for i, family in enumerate(f for f in OVERLAP_STYLES if f in groups):   # Qt-Qt (solid) carries the run
            df = groups[family]
            if len(runs) == 1:
                color, ls, label = colors[family], "--" if family.endswith("wall") else "-", family
            else:
                color, ls, label = r.color, OVERLAP_STYLES[family], r.label if i == 0 else None
            ax.plot(df["time_ms"], df["n overlapping"].clip(lower=0.5), color=color, ls=ls, label=label)
            families += [family] * (family not in families)
    if not families:
        _no_data(ax, "run overlap analysis")
        return
    ax.set_yscale("log")
    ax.set_ylabel("overlapping pairs")
    if len(runs) == 1:
        ax.legend()
    else:
        _style_legend(ax, {f: OVERLAP_STYLES[f] for f in OVERLAP_STYLES if f in families}, ncol=2, handlelength=3)


# Calcium panels. One run: fixed colours per quantity; several runs: one colour per run, the two quantities
# of a panel as solid and dashed lines (legend of the styles in the panel).

def p_calcium(ax, runs, traces=False):
    one = len(runs) == 1
    a2 = ax.twinx()
    for r in runs:
        t, mean = r.stats["time_ms"], r.stats["mean"]
        ax.plot(t, mean["ca"], color="k" if one else r.color, label="Ca²⁺" if one else r.label)
        a2.plot(t, mean["f"], color=LINE["Qt"] if one else r.color, ls="--", label="loading f" if one else None)
    ax.set_ylabel("calcium (µM)")
    a2.set_ylim(0, 1.02)
    a2.set_ylabel("calmodulin loading f")
    if one:
        ax.legend(handles=ax.get_lines() + a2.get_lines(), loc="upper right")
    else:
        _style_legend(ax, {"Ca²⁺": "-", "loading f": "--"}, loc="upper right")


def p_rate_constants(ax, runs, traces=False):
    one = len(runs) == 1
    for r in runs:
        t, mean = r.stats["time_ms"], r.stats["mean"]
        ax.plot(t, mean["k_on_s"], color=LINE["Qt"] if one else r.color,
                label="k_on (per contact pair)" if one else r.label)
        ax.plot(t, mean["k_off_s"], color=LINE["Ft"] if one else r.color, ls="-" if one else "--",
                label="k_off (per bond)" if one else None)
    ax.set_yscale("log")
    ax.set_ylabel("rate constant (1/s)")
    if one:
        ax.legend(loc="best")
    else:
        _style_legend(ax, {"k_on (per contact pair)": "-", "k_off (per bond)": "--"}, loc="best")


def p_formed_broken(ax, runs, traces=False):
    """Bonds formed and broken per second, per calcium segment (bond_rates)."""
    one, values = len(runs) == 1, []
    for r in runs:
        t, mean, sd, tr = (r.rates[k] for k in ("time_ms", "mean", "sd", "traces"))
        band(ax, t, mean["formed_per_s"], sd["formed_per_s"], r.color, "binding (formed)" if one else r.label,
             traces=tr["formed_per_s"] if traces else None)
        band(ax, t, mean["broken_per_s"], sd["broken_per_s"], "0.5" if one else r.color,
             "unbinding (broken)" if one else None, ls="--", traces=tr["broken_per_s"] if traces else None)
        values += [mean["formed_per_s"], mean["broken_per_s"]]
    k = _thousands(ax, np.concatenate(values))
    ax.set_ylabel("rate (bonds/s)" + (" (×10³)" if k else ""))
    if one:
        ax.legend(loc="upper right")
    else:
        _style_legend(ax, {"binding (formed)": "-", "unbinding (broken)": "--"}, loc="upper right")


def p_k_a(ax, runs, traces=False):
    """Apparent association rate constant per calcium segment (bond_rates), log scale."""
    for r in runs:
        t, mean, sd, tr = (r.rates[k] for k in ("time_ms", "mean", "sd", "traces"))
        band(ax, t, mean["k_a_per_M_s"], sd["k_a_per_M_s"], r.color, None if len(runs) == 1 else r.label,
             traces=tr["k_a_per_M_s"] if traces else None)
    ax.set_yscale("log")
    ax.set_ylabel("apparent k_a (M⁻¹ s⁻¹)")


def p_bonds_broken(ax, runs, traces=False):
    one = len(runs) == 1
    for r in runs:
        t, mean, sd, tr = r.stats["time_ms"], r.stats["mean"], r.stats["sd"], r.stats["traces"]
        band(ax, t, mean["Bonds"], sd["Bonds"], r.color, "bonds" if one else r.label,
             traces=tr["Bonds"] if traces else None)
        band(ax, t, mean["n_broken"], sd["n_broken"], "0.5" if one else r.color,
             "broken (cumulative)" if one else None, ls="--")
    ax.set_ylabel("bonds")
    if one:
        ax.legend(loc="upper left")
    else:
        _style_legend(ax, {"bonds": "-", "broken (cumulative)": "--"}, loc="upper left")


RATE_PANELS = [("binding and unbinding rates", p_formed_broken, True), ("apparent k_a", p_k_a, True)]


# ---------------------------------------------------------------------------- figures
def _figure(width, height):
    return plt.figure(figsize=(width, height), layout="constrained")


def save(fig, stem: Path) -> None:
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(stem.with_suffix(f".{ext}"), dpi=300)


def save_figure(fig, out_dir: Path, name: str) -> None:
    """A whole figure into its own folder: out_dir/<name>/<name>.*."""
    save(fig, out_dir / name / name)


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
            ax.axvline(t, color="0.6", lw=GUIDE_LW, ls=":", zorder=0)


def _grid(panels, runs, nrows, ncols, width, height, titles=True, traces=False, legend_below=False,
          sharex=False):
    fig = _figure(width, height)
    axes = fig.subplots(nrows, ncols, squeeze=False, sharex=sharex)
    t_end = _t_end(runs)
    for ax, panel in zip(axes.flat, panels):
        if panel is None:                                       # empty slot (a panel of one run only)
            ax.set_visible(False)
            continue
        title, func, is_time = panel
        func(ax, runs, traces=traces)
        if is_time:
            phase_lines(ax, runs)
            _time_axis(ax, t_end, label=not sharex or ax.get_subplotspec().is_last_row())
        if titles:
            ax.set_title(title, loc="left")
    for ax in axes.flat[len(panels):]:
        ax.set_visible(False)
    if legend_below and len(runs) > 1:
        _run_legend(fig, runs)
    return fig


def _run_handles(runs):
    """Legend handles and labels of the runs, in run order: a solid line in the colour of each run (independent
    of which panels have data)."""
    return [plt.Line2D([], [], color=r.color) for r in runs], [r.label for r in runs]


def _run_legend(fig, runs) -> None:
    """The runs in a legend below the axes of the figure."""
    handles, labels = _run_handles(runs)
    fig.legend(handles, labels, loc="outside lower center", ncol=_legend_columns(fig, labels))


def _legend_columns(fig, labels) -> int:
    """Columns of a run legend below the axes: as many (at most 4) as fit the figure width at 6 pt."""
    per_column = 1.3 * max(map(len, labels)) + 12                    # mm: about 1.3 mm per character + line
    return max(1, min(len(labels), 4, int(fig.get_figwidth() * 25.4 // per_column)))


def _single_panels(panels, runs, folder: Path, traces=False):
    """Every panel of a figure alone (89 mm), into folder/<panel>.*; several runs get a legend."""
    t_end = _t_end(runs)
    for title, func, is_time in filter(None, panels):
        fig = _figure(SINGLE, 65 * MM)
        ax = fig.add_subplot()
        func(ax, runs, traces=traces)
        if is_time:
            phase_lines(ax, runs)
            _time_axis(ax, t_end)
        drawn = {r.label for r in runs} & set(ax.get_legend_handles_labels()[1])     # a line of some run
        if len(runs) > 1 and drawn:                # the runs: in the axes, or below them if the axes have a legend
            if ax.get_legend() is None:
                ax.legend(*_run_handles(runs))
            else:
                _run_legend(fig, runs)
        save(fig, folder / _slug(title))
        plt.close(fig)


def _panel_figure(name, panels, runs, out_dir, nrows, ncols, width, height, traces=False, **grid):
    """A figure of several panels: saved as out_dir/<name>/<name>.* and every panel alone in out_dir/<name>/."""
    with plt.rc_context(STYLE):
        fig = _grid(panels, runs, nrows, ncols, width, height, traces=traces, **grid)
        if out_dir:
            save_figure(fig, out_dir, name)
            _single_panels(panels, runs, out_dir / name, traces=traces)
    return fig


# overview of one or several runs, 4 x 3 by rows. Bonds and fraction bound are in fig_calcium, the mean size in
# fig_cluster_sizes. Panels in ONE_RUN_ONLY stay an empty slot with several runs, so every panel keeps its place.
OVERVIEW = [
    ("potential energy", p_energy, True), ("pressure", p_pressure, True), ("particle counts", p_counts, True),
    ("agglomerates + free", p_clusters, True), ("largest agglomerate", p_largest, True),
    ("largest / N", p_largest_n, True),
    ("size categories", p_categories, True), ("coordination", p_coordination, True),
    ("coordination", p_coordination_hist, False),      # title completed by influx_end_label in fig_overview
    ("mean Rg", p_rg, True), ("Rg / Rg,ideal", p_rg_ratio, True), ("Qt fraction", p_qt_fraction, True),
]
ONE_RUN_ONLY = {p_categories}


def influx_end_label(p: Params) -> str:
    """Name of the frame of the agglomerate distributions (Params.influx_end_step)."""
    return "end of influx" if p.t_influx_s > 0 else "final"


def _as_runs(runs) -> list[RunData]:
    """One RunData or a list of them, as a list."""
    return [runs] if isinstance(runs, RunData) else list(runs)


def fig_overview(runs, out_dir: Path | None = None, traces=False):
    """4 x 3 metrics overview (OVERVIEW) of one run or several (one colour per run, legend below; the panels of
    ONE_RUN_ONLY left empty), 183 mm; panels also alone in overview/."""
    runs = _as_runs(runs)
    frame = _frame_label(runs)
    panels = [None if len(runs) > 1 and f in ONE_RUN_ONLY else
              (f"{t} at {frame}" if f is p_coordination_hist else t, f, is_time) for t, f, is_time in OVERVIEW]
    return _panel_figure("overview", panels, runs, out_dir, 4, 3, DOUBLE, MAX_HEIGHT, traces=traces,
                         legend_below=True)


AGGLOMERATE_DISTRIBUTIONS = [("agglomerate size", p_agglomerate_size), ("solid volume", p_volume_solid),
                             ("envelope volume", p_volume_envelope)]


def _frame_label(runs) -> str:
    """influx_end_label of the runs, joined if they differ."""
    return " / ".join(sorted({influx_end_label(r.p) for r in runs}))


def fig_agglomerate_distributions(runs, out_dir: Path | None = None):
    """Distributions of agglomerate size, solid and envelope volume (agglomerates of >= AGGLOMERATE_MIN
    particles) at the end of the influx, 1 x 3 (183 mm): one run per replica, several runs per 1000 um^3 as
    outlines (one colour per run, legend below); panels also alone in agglomerate_distributions/."""
    runs = _as_runs(runs)
    panels = [(f"{t} at {_frame_label(runs)}", f, False) for t, f in AGGLOMERATE_DISTRIBUTIONS]
    return _panel_figure("agglomerate_distributions", panels, runs, out_dir, 1, 3, DOUBLE, 60 * MM,
                         legend_below=True)


def _peak_label(ax, r, frame, left=False) -> None:
    ax.text(0.03 if left else 0.97, 0.95, frame + (f"\nr = {r:.0f} nm" if np.isfinite(r) else ""),
            transform=ax.transAxes, ha="left" if left else "right", va="top")


def fig_fibsem_comparison(runs, out_dir: Path | None = None, spatial: dict | list | None = None,
                          r_plot_nm: float = 500.0):
    """Qt spatial statistics for the comparison with 3D FIB-SEM (Qt only), 2 x 2 (183 mm): distances to the
    nearest Qt of the same agglomerate (agglomerates with >= QT_MIN Qt, 1 nm bins, kernel density estimate)
    and g(r) of all Qt (log scale, up to `r_plot_nm`) at the end of the influx, each with its peak;
    cumulative distribution of the nearest-neighbour distance of all Qt (fraction of Qt with a Qt within
    d) with the random-point curve 1 - exp(-4/3 pi rho d^3) and the Clark-Evans index R, and Ripley's
    L(r) - r of all Qt (up to `r_plot_nm`, with its maximum), each in the first frame and at the end of the
    influx. Pooled over replicas (L: mean over replicas). Several runs: one line per run (one colour each,
    legend below) at the end of the influx only, the peaks as dotted lines, the nearest-neighbour distances as
    density estimate only. `spatial`: a result of qt_spatial_run() (a list of them for several runs; computed
    or read from its cache if not given). Saved as fibsem_comparison.*; panels also alone in
    fibsem_comparison/."""
    runs = _as_runs(runs)
    if spatial is None:
        spatial = [qt_spatial_run(r.run_dir) for r in runs]
    elif isinstance(spatial, dict):
        spatial = [spatial]
    if len(spatial) != len(runs):
        raise ValueError(f"one qt_spatial_run result per run: {len(spatial)} for {len(runs)} runs")
    data = list(zip(runs, spatial))
    end = _frame_label(runs)
    one = len(runs) == 1
    color = LINE["Qt"]

    def p_nn_intra(ax, runs, traces=False):
        drawn = False
        for r, sp in data:
            d = sp["nn"].loc[sp["nn"]["set"] == "intra", "distance_nm"].to_numpy()
            if len(d) < 2:
                continue
            kde = density_estimate(d)
            x = np.linspace(d.min(), d.max(), 400)
            if one:
                ax.hist(d, bins=np.arange(np.floor(d.min()), d.max() + 1, 1.0), density=True, color=color,
                        alpha=0.5, lw=0)
                if kde is not None:
                    ax.plot(x, kde(x), color=color)
                _peak_label(ax, peak(d), end, left=True)          # the distances pile up on the right
            else:
                if kde is not None:
                    ax.plot(x, kde(x), color=r.color, label=r.label)
                else:
                    ax.hist(d, bins=np.arange(np.floor(d.min()), d.max() + 1, 1.0), density=True,
                            histtype="step", color=r.color, label=r.label)
                ax.axvline(peak(d), color=r.color, lw=GUIDE_LW, ls=":")
            drawn = True
        if not drawn:
            _no_data(ax, f"no agglomerates with ≥ {QT_MIN} Qt")
            return
        ax.set_xlabel("distance to nearest Qt in the agglomerate (nm)")
        ax.set_ylabel("probability density (1/nm)")

    def p_rdf(ax, runs, traces=False):
        drawn = False
        for r, sp in data:
            rr, g = sp["rdf"]["r_nm"].to_numpy(), sp["rdf"]["g_mean"].to_numpy()
            if not np.isfinite(g).any():
                continue
            shown = (rr <= r_plot_nm) & (g > 0)
            ax.plot(rr[shown], g[shown], color=color if one else r.color, label=None if one else r.label)
            r_peak = peak(rr, g, r_min=2 * r.p.radius_qt)
            if one:
                _peak_label(ax, r_peak, end)
            elif np.isfinite(r_peak):
                ax.axvline(r_peak, color=r.color, lw=GUIDE_LW, ls=":")
            drawn = True
        if not drawn:
            _no_data(ax, "fewer than 2 Qt")
            return
        ax.axhline(1, color="0.6", lw=GUIDE_LW, ls=":")
        ax.set_yscale("log")
        ax.set_xlim(0, r_plot_nm)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("g(r), Qt–Qt")

    def p_clark_evans(ax, runs, traces=False):
        frames = (("first", "0.6", "first frame"), ("influx_end", color, end)) if one else (("influx_end", None, end),)
        for i, (r, sp) in enumerate(data):
            nn, ce = sp["nn"], sp["clark_evans"]
            d_all = nn[nn["set"] == "all"]
            for frame, c, name in frames:
                d = np.sort(d_all.loc[d_all["frame"] == frame, "distance_nm"].to_numpy())
                rows = ce[ce["frame"] == frame]
                text = f"R = {rows['R'].mean():.2f}, $\\langle d \\rangle$ = {d.mean():.0f} nm"
                ax.step(d, np.arange(1, len(d) + 1) / len(d), where="post", color=c or r.color,
                        label=f"{name}: {text}" if one else r.label)
                if not one:                                       # R of every run in its colour, top left
                    ax.text(0.03, 0.95 - 0.08 * i, text, color=r.color, transform=ax.transAxes, va="top")
            rho = ce["density_per_um3"].mean() * 1e-9                               # [1/nm^3]
            x = np.linspace(0, d_all["distance_nm"].max(), 400)
            ax.plot(x, 1 - np.exp(-4 / 3 * math.pi * rho * x**3), color="k" if one else r.color, lw=GUIDE_LW,
                    ls="--", label="random points" if one else None)
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("distance to nearest Qt, d (nm)")
        ax.set_ylabel("fraction of Qt with a Qt within d")
        if one:
            ax.legend(loc="lower right")
        else:
            _style_legend(ax, {end: "-", "random points": "--"}, loc="lower right")

    def p_ripley(ax, runs, traces=False):
        zero = ax.axhline(0, color="k", lw=GUIDE_LW, ls="--", label="random points")
        frames = (("first", "0.6", "first frame"), ("influx_end", color, end)) if one else (("influx_end", None, end),)
        for r, sp in data:
            ripley = sp["ripley"]
            for frame, c, name in frames:
                mean = ripley[ripley["frame"] == frame].groupby("r_nm")["L_minus_r"].mean()
                mean = mean[mean.index <= r_plot_nm]
                label = name
                if mean.notna().any():
                    label += f": max {mean.max():.0f} nm at r = {mean.idxmax():.0f} nm"
                ax.plot(mean.index, mean.to_numpy(), color=c or r.color, label=label if one else r.label)
        ax.set_xlim(0, r_plot_nm)
        ax.set_xlabel("r (nm)")
        ax.set_ylabel("L(r) − r (nm)")
        if one:
            ax.legend(loc="center left", bbox_to_anchor=(0.2, 0.3))              # below the curve of the influx end
        else:
            ax.legend([zero], ["random points"], loc="lower right")

    panels = [("Qt nearest neighbour in agglomerates", p_nn_intra, False), ("Qt pair correlation", p_rdf, False),
              ("Qt clustering (Clark–Evans)", p_clark_evans, False),
              ("Qt clustering scale (Ripley's L)", p_ripley, False)]
    return _panel_figure("fibsem_comparison", panels, runs, out_dir, 2, 2, DOUBLE, 120 * MM, legend_below=True)


# rows of fig_cluster_sizes: measure (cluster_sizes column, mean_size = thermo column) and title
SIZE_ROWS = [("mean_size", "mean size"), ("mean_bound", "mean size (bound particles)"),
             ("weight", "weight-average size"), ("weight_bound", "weight-average size (bound particles)")]
SIZE_SPECIES = {"all": "", "Qt": "_qt", "Ft": "_ft"}      # column -> suffix of the measure


def _size_curve(r: RunData, measure: str, species: str, per_n: bool, traces: bool):
    """Time [ms], mean, SD and (with traces) replica traces of a cluster-size measure of one species,
    divided by the particles of that species with per_n."""
    suffix = SIZE_SPECIES[species]
    scale = 1 / {"all": r.n_total, "Qt": r.p.n_qt, "Ft": r.p.n_ft}[species] if per_n else 1.0
    if measure == "mean_size":
        return _series(r, f"mean_size{suffix}", scale, traces=traces)
    d, key = r.sizes, measure + suffix
    tr = d[[c for c in d.columns if c.endswith(f":{key}")]].to_numpy().T * scale if traces else None
    return d["time_s"].to_numpy() * 1e3, d[f"{key}_mean"].to_numpy() * scale, d[f"{key}_sd"].to_numpy() * scale, tr


def _size_panel(measure: str, title: str, species: str | None, per_n: bool):
    """Panel of fig_cluster_sizes: one species for several runs (a colour per run), or (species None, one run)
    all, Qt and Ft as three lines."""
    def draw(ax, runs, traces=False):
        curves = ([(runs[0], name, runs[0].color if name == "all" else LINE[name], name) for name in SIZE_SPECIES]
                  if species is None else [(r, species, r.color, r.label) for r in runs])
        for r, sp, color, label in curves:
            t, m, s, tr = _size_curve(r, measure, sp, per_n, traces)
            band(ax, t, m, s, color, label, traces=tr if traces else None)
        ax.set_ylabel(title + (" / N" if per_n else ""))
        if species is None:
            ax.legend()
    name = title + (" / N" if per_n else "") + (f", {species}" if species else "")
    return name, draw, True


def fig_cluster_sizes(runs, out_dir: Path | None = None, traces=False) -> list:
    """Cluster sizes over time (rows SIZE_ROWS: mean size with the free particles, mean size over the bound
    particles (clusters of >= 2), weight average with the free particles, weight average over the bound
    particles); /N divides by the particles of the species (all: N, Qt: N_Qt, Ft: N_Ft). One run: one figure
    cluster_sizes, 4 x 2 (absolute | / N, each panel with all, Qt and Ft; 120 mm). Several runs: cluster_sizes
    and cluster_sizes_n (/ N), each 4 x 3 (columns all, Qt, Ft; a colour per run; 183 mm). Panels also alone in
    <figure>/."""
    runs = _as_runs(runs)
    if len(runs) == 1:
        panels = [_size_panel(m, title, None, per_n) for m, title in SIZE_ROWS for per_n in (False, True)]
        return [_panel_figure("cluster_sizes", panels, runs, out_dir, 4, 2, ONE_HALF, 150 * MM, traces=traces)]
    return [_panel_figure(name, [_size_panel(m, title, sp, per_n) for m, title in SIZE_ROWS for sp in SIZE_SPECIES],
                          runs, out_dir, 4, 3, DOUBLE, 150 * MM, traces=traces, legend_below=True)
            for name, per_n in (("cluster_sizes", False), ("cluster_sizes_n", True))]


# line styles of the thresholds with several runs (fig_agglomerates_above_size, _total)
THRESHOLD_STYLES = ["-", "--", ":", "-.", (0, (5, 1, 1, 1, 1, 1)), (0, (8, 2))]
SPECIES_UNIT = {"all": "", "qt": " Qt", "ft": " Ft"}     # what the thresholds count, in labels


def _draw_above_size(ax, runs, ns, density, traces, species="all"):
    """Agglomerates with >= n (particles, Qt or Ft: `species`) over time, one line per n, with y label.
    One run: dark (smallest n) to light; several runs: one colour per run, one line style per n."""
    column = "per_1000um3" if density else "count"
    shades = plt.cm.Blues(np.linspace(0.95, 0.45, len(ns))) if len(ns) > 1 else [runs[0].color]
    unit = SPECIES_UNIT[species]
    values = []
    for r in runs:
        table = agglomerates_above_size(r.run_dir, ns, species=species)
        reps = [c for c in table.columns if c.startswith("count_") and c not in ("count_mean", "count_sd")]
        scale = 1000 / (math.prod(r.p.box) * 1e-9) if density else 1.0      # replica traces as plotted
        for i, n in enumerate(ns):
            d = table[table["n_min"] == n]
            tr = d[reps].to_numpy().T * scale
            if len(runs) > 1:
                color, ls = r.color, THRESHOLD_STYLES[i]
                label = r.label if i == 0 else None
            else:
                color, ls, label = shades[i], "-", f"≥ {n}{unit}" if len(ns) > 1 else None
            band(ax, d["time_s"].to_numpy() * 1e3, d[f"{column}_mean"].to_numpy(),
                 d[f"{column}_sd"].to_numpy(), color, label, ls=ls, traces=tr if traces else None)
            values.append(d[f"{column}_mean"].to_numpy())
    k = _thousands(ax, np.concatenate(values))
    size = f" ≥ {ns[0]}{unit}" if len(ns) == 1 else ""
    ax.set_ylabel(f"agglomerates{size} " + ("/ 1000 µm³" if density else "in the box")
                  + (" (×10³)" if k else ""))


def _above_size_curve(r: RunData, species: str, n: int, density: bool):
    """Time [ms], mean, SD and replica traces of the agglomerates with >= n (particles, Qt or Ft)."""
    column = "per_1000um3" if density else "count"
    d = agglomerates_above_size(r.run_dir, n, species=species)
    reps = [c for c in d.columns if c.startswith("count_") and c not in ("count_mean", "count_sd")]
    scale = 1000 / (math.prod(r.p.box) * 1e-9) if density else 1.0          # replica traces as plotted
    return (d["time_s"].to_numpy() * 1e3, d[f"{column}_mean"].to_numpy(), d[f"{column}_sd"].to_numpy(),
            d[reps].to_numpy().T * scale)


def _above_size_thresholds(n_min, n_min_qt, n_min_ft, n_qt_ft) -> tuple[dict, int, int]:
    """Thresholds of fig_agglomerates_above_size: ({species: thresholds}, n Qt, n Ft of the pair of row 4)."""
    if len(n_qt_ft) != 2:
        raise ValueError(f"n_qt_ft must be one pair (>= n Qt, >= n Ft), not {n_qt_ft!r}")
    n_qt, n_ft = (_thresholds(v)[0] for v in n_qt_ft)
    return {"all": _thresholds(n_min), "qt": _thresholds(n_min_qt), "ft": _thresholds(n_min_ft)}, n_qt, n_ft


def _count_above_size(run_dir: Path, ns: dict, n_qt: int, n_ft: int) -> None:
    """Agglomerate counts of every threshold (and the pair) and cluster_sizes of a run in one pass over its
    traj.dump frames, into the caches in <run>/analysis/ (nothing read when they are up to date)."""
    _agglomerates_above_size_species(run_dir, {"all": ns["all"], "qt": ns["qt"] + [n_qt], "ft": ns["ft"] + [n_ft]},
                                     sizes=True)


def fig_agglomerates_above_size(runs, out_dir: Path | None = None, traces=False,
                                n_min=SIZE_THRESHOLDS, n_min_qt=SIZE_THRESHOLDS_QT, n_min_ft=SIZE_THRESHOLDS,
                                n_qt_ft=SIZE_THRESHOLDS_QT_FT):
    """Number of agglomerates with >= n particles (row 1), >= n Qt (row 2) and >= n Ft (row 3) over time,
    in the box (left) and per 1000 um^3 of box volume (right), one line per threshold; row 4: >= n Qt and
    >= n Ft for the pair `n_qt_ft` together, 4 x 2 (120 mm). One run: thresholds dark to light, row 4 Qt green
    and Ft red; several runs: one colour per run (legend below), one line style per threshold (at most 6),
    row 4 Qt solid and Ft dashed. Saved as agglomerates_above_size.*; panels also alone in
    agglomerates_above_size/."""
    runs = _as_runs(runs)
    ns, n_qt, n_ft = _above_size_thresholds(n_min, n_min_qt, n_min_ft, n_qt_ft)
    if len(runs) > 1 and max(map(len, ns.values())) > len(THRESHOLD_STYLES):
        raise ValueError(f"several runs show at most {len(THRESHOLD_STYLES)} thresholds per species (one line "
                         f"style each), not {max(map(len, ns.values()))}")
    for r in runs:
        _count_above_size(r.run_dir, ns, n_qt, n_ft)
    names = {"all": "all particles", "qt": "Qt", "ft": "Ft"}

    def panel(species, density):
        def draw(ax, runs, traces=False):
            _draw_above_size(ax, runs, ns[species], density, traces, species)
            if len(ns[species]) > 1 and len(runs) == 1:
                ax.legend(ncol=2)
            elif len(ns[species]) > 1:
                _style_legend(ax, {f"≥ {n}{SPECIES_UNIT[species]}": THRESHOLD_STYLES[i]
                                   for i, n in enumerate(ns[species])}, ncol=2, handlelength=3)
        return f"{names[species]}, {'density' if density else 'number'}", draw, True

    def pair(density):
        def draw(ax, runs, traces=False):
            values = []
            for r in runs:
                for species, n, ls in (("qt", n_qt, "-"), ("ft", n_ft, "--")):
                    t, m, s, tr = _above_size_curve(r, species, n, density)
                    label = f"≥ {n} {names[species]}"
                    if len(runs) == 1:
                        band(ax, t, m, s, LINE[names[species]], label, traces=tr if traces else None)
                    else:
                        band(ax, t, m, s, r.color, r.label if species == "qt" else None, ls=ls,
                             traces=tr if traces else None)
                    values.append(m)
            k = _thousands(ax, np.concatenate(values))
            ax.set_ylabel("agglomerates " + ("/ 1000 µm³" if density else "in the box") + (" (×10³)" if k else ""))
            if len(runs) == 1:
                ax.legend()
            else:
                _style_legend(ax, {f"≥ {n_qt} Qt": "-", f"≥ {n_ft} Ft": "--"})
        return f"≥ {n_qt} Qt and ≥ {n_ft} Ft, {'density' if density else 'number'}", draw, True

    panels = [panel(sp, density) for sp in ns for density in (False, True)] + [pair(False), pair(True)]
    # 1.5 columns wide: at the full width the 4 rows of the page height would make every panel twice as wide
    # as high; at 120 mm the panels have about the shape of the single panels (89 x 65 mm)
    return _panel_figure("agglomerates_above_size", panels, runs, out_dir, 4, 2, ONE_HALF, MAX_HEIGHT,
                         traces=traces, legend_below=True)


def fig_agglomerates_above_size_total(runs, out_dir: Path | None = None, traces=False,
                                      n_min=SIZE_THRESHOLDS, density: bool = False):
    """Number of agglomerates with >= n particles over time, one line per n in `n_min` (one number or a
    list; counted in the traj.dump frames by agglomerates_above_size), in the box or per 1000 um^3 of box
    volume (`density`) (89 mm). One run: one line per n, dark (smallest n) to light; several runs: one
    colour per run, one line style per n (at most 6). Saved as agglomerates_above_size_total.* (density:
    agglomerates_above_size_total_density.*)."""
    runs = _as_runs(runs)
    ns = _thresholds(n_min)
    if len(runs) > 1 and len(ns) > len(THRESHOLD_STYLES):
        raise ValueError(f"several runs show at most {len(THRESHOLD_STYLES)} thresholds (one line style each), "
                         f"not {len(ns)}")
    with plt.rc_context(STYLE):
        fig = _figure(SINGLE, 60 * MM)
        ax = fig.add_subplot()
        _draw_above_size(ax, runs, ns, density, traces)
        phase_lines(ax, runs)
        _time_axis(ax, _t_end(runs))
        handles, labels = ax.get_legend_handles_labels()
        if len(runs) > 1:                                    # runs below the axes, thresholds in the corner
            _run_legend(fig, runs)
            if len(ns) > 1:
                ax.legend([plt.Line2D([], [], color="0.3", ls=THRESHOLD_STYLES[i]) for i in range(len(ns))],
                          [f"≥ {n}" for n in ns], ncol=2, handlelength=3)
        elif handles:
            ax.legend(handles, labels)
        if out_dir:
            save_figure(fig, out_dir, "agglomerates_above_size_total" + ("_density" if density else ""))
    return fig


def fig_overlap(runs, out_dir: Path | None = None):
    """Number of overlapping pairs per family over time, log scale (89 mm), of one run or several (one colour per
    run, legend below; a line style per family). Saved as overlap/overlap.*."""
    runs = _as_runs(runs)
    with plt.rc_context(STYLE):
        fig = _grid([("overlapping pairs", p_overlap, True)], runs, 1, 1, SINGLE, (60 if len(runs) == 1 else 70) * MM,
                    titles=False, legend_below=True)
        if out_dir:
            save_figure(fig, out_dir, "overlap")
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
        ax.axvline(window, color="0.6", lw=GUIDE_LW, ls=":")
        ax.set_xlim(0, t[-1])
        ax.set_ylim(bottom=0)
        ax.set_xlabel("time since influx start (ms)")
        ax.set_ylabel("bonds formed")
        ax.legend(loc="upper left")
        if out_dir:
            save_figure(fig, out_dir, "binding_rate")
    return fig


CALCIUM = [("calcium and loading", p_calcium, True), ("rate constants", p_rate_constants, True), *RATE_PANELS,
           ("bonds and broken bonds", p_bonds_broken, True), ("fraction bound", p_fraction_bound, True)]


def fig_calcium(runs, out_dir: Path | None = None, traces=False, rates: dict | None = None):
    """Calcium protocol and its effect (CALCIUM, 3 x 2, 183 x 130 mm) of one run or several (one colour per run,
    legend below): calcium and calmodulin loading f; rate constants k_on per contact pair and k_off per bond;
    binding and unbinding rates (bonds formed and broken per second) and the apparent k_a per calcium segment;
    bonds and broken bonds; fraction bound (mean +- SD over replicas); panels also alone in calcium/. Cluster
    sizes are in fig_cluster_sizes. `rates`: a result of bond_rates() of the one run (computed if not given)."""
    runs = _as_runs(runs)
    if rates is not None and len(runs) == 1:
        runs[0]._rates = rates
    return _panel_figure("calcium", CALCIUM, runs, out_dir, 3, 2, DOUBLE, 130 * MM, traces=traces,
                         titles=False, sharex=True, legend_below=True)


def fig_bond_rates(runs, out_dir: Path | None = None, traces=False, rates: dict | None = None):
    """Bonds formed and broken per second (top) and the apparent association rate constant k_a in
    1/(M s), normalized by the total particle numbers (bottom, log scale), per calcium segment (mean +- SD
    over replicas; 89 mm), of one run or several; panels also alone in bond_rates/. `rates`: a result of
    bond_rates() of the one run (computed if not given)."""
    runs = _as_runs(runs)
    if rates is not None and len(runs) == 1:
        runs[0]._rates = rates
    return _panel_figure("bond_rates", RATE_PANELS, runs, out_dir, 2, 1, SINGLE, 90 * MM,
                         traces=traces, titles=False, sharex=True, legend_below=True)


SWEEP_LABELS = {
    "final_frac_bound": "fraction bound (final)", "final_Bonds": "bonds (final)",
    "final_largest_frac": "largest cluster (fraction of particles, final)",
    "final_mean_size": "mean cluster size (final)", "t_half_bonds_ms": "half-time of bonds (ms)",
    "t_half_qt_ms": "half-time Qt bound (ms)", "t_half_ft_ms": "half-time Ft bound (ms)",
    "t_percolation_ms": "percolation time (ms)", "bonds_peak": "bonds (peak)",
    "largest_peak": "largest cluster (peak)", "t_half_unbind_bonds_ms": "deagglomeration half-time, bonds (ms)",
    "t_half_unbind_largest_ms": "deagglomeration half-time, largest cluster (ms)",
    "mean_bound_peak": "mean size (bound particles) (peak)", "weight_peak": "weight-average size (peak)",
    "weight_bound_peak": "weight-average size (bound particles) (peak)",
    "weight_frac_peak": "weight-average size / N (peak)",
}


def fig_sweep(results: pd.DataFrame, y: str = "final_frac_bound", x: str | None = None,
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
            save_figure(fig, out_dir, f"sweep_{y}")
    return fig


# ---------------------------------------------------------------------------- tables
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


# ---------------------------------------------------------------------------- report.pdf
A4 = (210 * MM, 297 * MM)
TEXT_CHARS = 120        # characters per line of a cover page at 7 pt; wider text is set smaller


def _text_pages(title: str, name: str, blocks: list[tuple[str, str]]) -> list:
    """A4 pages of monospace text: a title and a name (run or folder) on its own line, then (heading, text)
    blocks, continued on further pages."""
    lines = []
    for heading, text in blocks:
        lines += [heading, "─" * len(heading), *str(text).splitlines(), ""]
    size = min(7.0, 7.0 * TEXT_CHARS / max(map(len, lines)))
    per_page = int(250 / (size * 1.3 * 25.4 / 72))                   # 250 mm of text at 1.3 line spacing
    pages = []
    for start in range(0, len(lines), per_page):
        fig = plt.figure(figsize=A4)
        fig.text(0.07, 0.965, title + (" (continued)" if start else ""), size=11, weight="bold", va="top")
        fig.text(0.07, 0.945, name, size=min(8.0, 8.0 * 100 / len(name)), family="monospace", va="top")
        fig.text(0.07, 0.915, "\n".join(lines[start:start + per_page]), family="monospace", size=size,
                 va="top", linespacing=1.3)
        pages.append(fig)
    return pages


def _run_record(run_dir: Path) -> str:
    """Run folder, creation date and git commit of a run (from params.json)."""
    record = json.loads((run_dir / "params.json").read_text())
    git = record.get("git") or {}
    return "\n".join([f"folder     {run_dir.parent.name}/{run_dir.name}",
                      f"created    {record.get('created', '?')}",
                      f"git        {git.get('commit', '?')}{' (modified)' if git.get('dirty') else ''}"])


def _write_report(path: Path, cover: list, figs: list) -> None:
    """Cover pages and figures, one per page, into one PDF; closes the cover pages."""
    with PdfPages(path) as pdf:
        for fig in (*cover, *figs):
            pdf.savefig(fig)
    for fig in cover:
        plt.close(fig)


def _now() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M")


# ---------------------------------------------------------------------------- report
def _analyse(run_dir: Path, every: int, structure_analysis: bool, overlap_analysis: bool,
             missing_only: bool) -> None:
    """The analyses the report figures read from <run>/analysis/: structure (OVITO) and overlap analysis (if
    asked for, every `every`-th frame) and the agglomerates at the end of the influx; with `missing_only` only
    those whose file is missing (comparison, sweep). Bond rates, cluster sizes, agglomerate counts and the Qt
    spatial statistics follow in _report_figures (cached where the analysis is slow)."""
    analysis = run_dir / "analysis"

    def due(name):
        return not (missing_only and (analysis / name).exists())

    if structure_analysis and due("structure_timeseries.csv"):
        from .structure import structure_run
        structure_run(run_dir, every=every)
    if overlap_analysis and due("overlap_timeseries.csv"):
        from .overlap import overlap_run
        overlap_run(run_dir, every=every)
    if due("agglomerates_influx_end.csv"):
        agglomerate_volumes_run(run_dir)


def _report_figures(runs: list[RunData], figs: Path, traces: bool, thresholds: tuple) -> list:
    """The report figures of one run or several, in report order, into figs/<figure>/: calcium, overview, overlap,
    cluster sizes, agglomerates above size (thresholds = n_min, n_min_qt, n_min_ft, n_qt_ft), agglomerate
    distributions, FIB-SEM comparison, and for one run with a calcium step the binding rate. Writes
    bond_rates.csv (and binding_rate.csv) of every run. The agglomerate counts and cluster sizes of a run are read
    from its dumps in one pass before the first figure that needs them."""
    ns, n_qt, n_ft = _above_size_thresholds(*thresholds)
    for r in runs:
        r._rates = bond_rates(r.run_dir)                                                         # + csv
        _count_above_size(r.run_dir, ns, n_qt, n_ft)                             # + csv (cached)
    figures = [fig_calcium(runs, figs, traces), fig_overview(runs, figs, traces=traces), fig_overlap(runs, figs),
               *fig_cluster_sizes(runs, figs, traces=traces),
               fig_agglomerates_above_size(runs, figs, traces, *thresholds),
               fig_agglomerate_distributions(runs, figs),
               fig_fibsem_comparison(runs, figs, spatial=[qt_spatial_run(r.run_dir) for r in runs])]
    run = runs[0]
    if len(runs) == 1 and run.p.t_influx_s > 0 and run.p.ca_rise_s == 0:
        figures.append(fig_binding_rate(run, figs, rate=binding_rate(run.run_dir)))              # + csv
    return figures


def make_report(run_dir: str | Path, every: int = 10, structure_analysis: bool = True,
                overlap_analysis: bool = True, traces: bool = False, n_min=SIZE_THRESHOLDS,
                n_min_qt=SIZE_THRESHOLDS_QT, n_min_ft=SIZE_THRESHOLDS, n_qt_ft=SIZE_THRESHOLDS_QT_FT) -> Path:
    """Full analysis of one run: ensemble statistics, structure (OVITO) and overlap analysis,
    all figures and tables into <run>/analysis/. n_min, n_min_qt, n_min_ft: size thresholds of
    fig_agglomerates_above_size (agglomerates with >= n particles, Qt, Ft); n_qt_ft: the pair
    (>= n Qt, >= n Ft) of its last row. Includes the Qt spatial statistics for FIB-SEM (qt_spatial_run,
    fig_fibsem_comparison). The same figures as compare_report, in the same order (_report_figures)."""
    run_dir = Path(run_dir).expanduser()
    out = run_dir / "analysis"
    stats = ensemble_stats(run_dir)
    _analyse(run_dir, every, structure_analysis, overlap_analysis, missing_only=False)
    run = load_run(run_dir)
    figures = _report_figures([run], out / "figures", traces, (n_min, n_min_qt, n_min_ft, n_qt_ft))

    table = final_table([run_dir])[run_dir.name]                  # columns mean, sd
    table.to_csv(out / "summary_table.csv")
    (out / "summary_table.tex").write_text(to_latex(table, run.label))
    t = stats["time_ms"]
    pd.DataFrame({"Step": np.round(t / (run.p.timestep * 1e-6)).astype(int), "time_us": t * 1e3,
                  "time_ms": t, "bonds_mean": stats["mean"]["Bonds"], "bonds_sd": stats["sd"]["Bonds"]}
                 ).to_csv(out / "bonds_timeseries.csv", index=False)
    (out / "report.json").write_text(json.dumps({"every": every, "replicas": stats["replicas"]}))
    cover = _text_pages("Report of one run", run_dir.name, [
        ("Run", f"{_run_record(run_dir)}\nreplicas   {', '.join(stats['replicas'])}\n"
                f"report     {_now()}, every = {every} (frames for the structure and overlap analysis)"),
        ("Final state (mean, SD over replicas)", table.to_string(float_format=lambda x: f"{x:.4g}")),
        ("Simulation parameters", "\n".join(f"{k:<24} {v}" for k, v in run.p.to_dict().items())),
        ("Derived values and checks", run.p.summary())])
    _write_report(out / "report.pdf", cover, figures)
    for f in figures:
        plt.close(f)
    return out


def compare_report(run_dirs, out_dir: str | Path, labels=None, structure_analysis: bool = True,
                   every: int = 10, overlap_analysis: bool = True, traces: bool = False, n_min=SIZE_THRESHOLDS,
                   n_min_qt=SIZE_THRESHOLDS_QT, n_min_ft=SIZE_THRESHOLDS, n_qt_ft=SIZE_THRESHOLDS_QT_FT) -> Path:
    """Comparison figures (out_dir/figures/<figure>/) and table of several runs into out_dir, and all
    figures in out_dir/report.pdf after cover pages with the runs, the table and their parameters. The same
    figures as make_report in the same order, one colour per run (without the binding rate). Every run first
    gets the analyses it is missing (structure and overlap analysis with structure_analysis / overlap_analysis,
    every `every`-th frame; agglomerates at the end of the influx, bond rates, Qt spatial statistics), saved in
    its own analysis/. traces and the size thresholds as in make_report."""
    return _comparison_report(run_dirs, out_dir, labels, structure_analysis, every, overlap_analysis, traces,
                              (n_min, n_min_qt, n_min_ft, n_qt_ft))


def _comparison_report(run_dirs, out_dir, labels=None, structure_analysis=True, every=10, overlap_analysis=True,
                       traces=False, thresholds=(SIZE_THRESHOLDS, SIZE_THRESHOLDS_QT, SIZE_THRESHOLDS,
                                                 SIZE_THRESHOLDS_QT_FT),
                       title="Comparison of runs", blocks=(), first=()) -> Path:
    """compare_report with further cover blocks (heading, text) before the runs and further figures before
    the comparison figures in report.pdf (closed afterwards)."""
    from .ensemble import differing_labels, format_table
    out_dir = Path(out_dir).expanduser()
    run_dirs = [Path(r).expanduser() for r in run_dirs]
    for r in run_dirs:
        _analyse(r, every, structure_analysis, overlap_analysis, missing_only=True)
    runs = load_runs(run_dirs, labels or differing_labels(run_dirs, short=True)[0])
    figures = [*first, *_report_figures(runs, out_dir / "figures", traces, thresholds)]
    numbers = final_table(run_dirs)
    numbers.set_axis([f"{label} {stat}" for label, stat in numbers.columns], axis=1).to_csv(
        out_dir / "comparison_table.csv")
    table = format_table(numbers)
    (out_dir / "comparison_table.tex").write_text(to_latex(table))

    names = [f"run {i}" for i in range(1, len(runs) + 1)]
    params = pd.DataFrame({name: r.p.to_dict() for name, r in zip(names, runs)}).astype(str)
    cover = _text_pages(title, out_dir.name, [
        *blocks,
        ("Runs", "\n\n".join(f"{name}: {r.label}\n{_run_record(r.run_dir)}" for name, r in zip(names, runs))
         + f"\n\nreport     {_now()}"),
        ("Final state (mean ± SD over replicas)", table.set_axis(names, axis=1).to_string()),
        ("Simulation parameters", params.to_string())])
    _write_report(out_dir / "report.pdf", cover, figures)
    for f in figures:
        plt.close(f)
    return out_dir


# results of sweep_results plotted by sweep_report; "final" values are those at the end of the whole
# protocol (after the pump-out), so peaks and half-times describe a calcium cycle better
SWEEP_PLOTS = ["bonds_peak", "largest_peak", "t_half_bonds_ms", "t_half_unbind_bonds_ms", "weight_bound_peak",
               "weight_frac_peak"]


def sweep_report(sweep_dir: str | Path, ys=SWEEP_PLOTS, structure_analysis: bool = True, every: int = 10,
                 overlap_analysis: bool = True, traces: bool = False, n_min=SIZE_THRESHOLDS,
                 n_min_qt=SIZE_THRESHOLDS_QT, n_min_ft=SIZE_THRESHOLDS, n_qt_ft=SIZE_THRESHOLDS_QT_FT) -> Path:
    """Analysis of a parameter sweep into <sweep>/analysis/: sweep_results (<sweep>/sweep_results.csv), one
    fig_sweep plot per result in ys and the comparison of all runs with a finished replica (as compare_report,
    with its options); report.pdf holds cover pages with the sweep results, the sweep plots and the comparison
    figures."""
    from .sweep import sweep_results
    sweep_dir = Path(sweep_dir).expanduser()
    out_dir = sweep_dir / "analysis"
    results = sweep_results(sweep_dir)
    done = results[results["replicas done"] > 0]
    if done.empty:
        raise ValueError(f"{sweep_dir.name}: no run of the sweep has a finished replica yet")
    plots = [fig_sweep(results, y=y, out_dir=out_dir / "figures") for y in ys]
    table = done[[*results.attrs["swept"], "replicas done", *ys]]
    return _comparison_report([sweep_dir / run for run in done.index], out_dir, None, structure_analysis, every,
                              overlap_analysis, traces, (n_min, n_min_qt, n_min_ft, n_qt_ft),
                              title=f"Sweep {sweep_dir.name}",
                              blocks=[("Sweep results (mean over replicas; all values and SD in "
                                       "sweep_results.csv)", table.to_string())], first=plots)
