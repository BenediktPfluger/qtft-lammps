"""Animation of one replica as an MP4 video (1920 x 1080 times `scale`, default 2 = 3840 x 2160; black), for
the animation notebook.

Every saved traj.dump frame becomes one video frame (fps sets the speed: with traj_every = 0.1 s and
30 fps the video runs 3x faster than the simulation). Four panels: the whole box in perspective, the
whole box seen from the front (orthographic along y, x to the right, z up) with the magnified region
outlined, the magnification (a central box of a third of every edge: the particles whose centre lies
inside it in depth and whose sphere reaches into it in x and z), and the mean cluster size (free
particles included) of the analysis figures (thermo columns mean_size, mean_size_qt, mean_size_ft) with
a moving time line.
Header: title, phase (calcium on/off, from the protocol in params.json), production time, calcium
(Params.calcium, the formula of template.lmp). Right column: legend and per-frame statistics of the
dumped frame (bonds, Qt in the cluster with the most Qt, Qt in clusters with >= 2 Qt). Bottom: the
protocol as a timeline bar. Particles are shaded spheres at physical size, coloured by type and by
bonded (nbonds > 0), painted back to front with a slight darkening with depth. Periodic runs show the
wrapped positions: an agglomerate across a box face appears on both sides.

Spheres are drawn by a small numpy renderer (one shaded sprite per colour and pixel radius, pasted
back to front); matplotlib lays out the frame and streams it to ffmpeg (H.264). ffmpeg must be on the
PATH for the video; stills (PNG) work without it.
"""

from __future__ import annotations

import math
import os
import shutil
import time
from contextlib import nullcontext
from functools import lru_cache
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager, patheffects
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon, Rectangle

from .params import Params, data_path
from .read import dump_steps, load_thermo, read_dump
from .remote import require_fetched

TITLE = "Reversible assembly of protein cages"
W, H, DPI = 1920, 1080, 100                    # layout size [px] at scale 1; 1 px = 0.72 pt at this dpi
CRF = 14                                       # H.264 quality (lower = better, larger file)
PT = 72 / DPI

WHITE, GRAY, TEAL, FRAME = "#FFFFFF", "#B0B0B0", "#4FC9B4", "#9A9A9A"
DARK_YELLOW = "#C9A227"
INSET = "#8A9AB0"
PARTICLE = {(1, 0): "#1F8A1F", (1, 1): "#1E3CFF",      # (type, bonded): Qt green / blue ...
            (2, 0): "#F28C38", (2, 1): "#E82020"}      # ... Ft orange / red
PHASE_COLORS = ("#3E4E5A", "#B87A35", "#2A6E66")       # timeline: baseline, influx, pump-out
TRACES = (("mean_size", "all", DARK_YELLOW), ("mean_size_qt", "Qt", PARTICLE[1, 0]), ("mean_size_ft", "Ft", PARTICLE[2, 1]))

ELEVATION, AZIMUTH = 20.0, 30.0                # 3D camera [deg], looking at the box centre ...
CAMERA_DISTANCE = 2.5                          # ... from this many box diagonals
DEPTH_CUE = 0.35                               # the farthest spheres are this much darker
MAGNIFICATION = 1 / 3                          # edge of the magnified central box / box edge

PANEL = 380                                    # image panels [px], square
LEFT_X, RIGHT_X, SIDE_X = 120, 610, 1620       # left column, right column, legend + statistics
TOP_Y, BOTTOM_Y = 520, 95                      # lower edge of the top and bottom panel row
GRAPH = (RIGHT_X, 130, 830, 315)                             # graph axes x, y, width, height [px]
CALCIUM = (1085, TOP_Y, PANEL, PANEL)                        # calcium axes, right of the magnification
BAR_Y, BAR_X = 40, (LEFT_X, W - 60)            # timeline bar

FONT_FILES = ("/mnt/c/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arial.ttf",
              "/Library/Fonts/Arial.ttf", "/usr/share/fonts/truetype/msttcorefonts/Arial.ttf")
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["Arial", "Liberation Sans", "DejaVu Sans"],
         "font.weight": "normal", "mathtext.fontset": "custom", "mathtext.default": "regular",
         **{f"mathtext.{k}": "Arial:normal" for k in ("rm", "it", "bf", "bfit", "sf", "tt", "cal")},
         "text.color": WHITE, "axes.edgecolor": FRAME, "axes.labelcolor": WHITE, "xtick.color": GRAY, "ytick.color": GRAY,
         "figure.facecolor": "black", "axes.facecolor": "black", "savefig.facecolor": "black"}


def _px(size_px: float) -> float:
    """Font size or line width given in pixels -> points."""
    return size_px * PT


def _register_arial() -> None:
    """Arial is not installed in WSL; take it from Windows if available (else the STYLE fallbacks). The
    file is copied once to matplotlib's cache folder: read from /mnt/c, text draws ~10x slower."""
    if any(f.name == "Arial" for f in font_manager.fontManager.ttflist):
        return
    for path in FONT_FILES:
        if os.path.exists(path):
            local = Path(matplotlib.get_cachedir()) / "arial.ttf"
            if not local.exists():
                shutil.copyfile(path, local)
            font_manager.fontManager.addfont(str(local))
            return


# ---------------------------------------------------------------------------- data
def find_run(run: str | Path, data_dir: str | Path = "runs") -> Path:
    """Run folder from a path or from its name alone (looked up in single/, ensemble/, sweeps/*/, tuning/*/)."""
    path = Path(run).expanduser()
    if (path / "params.json").exists():
        return path
    root = data_path(data_dir)
    found = [d for pattern in ("single/{}", "ensemble/{}", "sweeps/*/{}", "tuning/*/{}")
             for d in root.glob(pattern.format(run)) if (d / "params.json").exists()]
    if not found:
        raise FileNotFoundError(f"no run folder '{run}' in {root} (single/, ensemble/, sweeps/, tuning/)")
    if len(found) > 1:
        raise ValueError(f"'{run}' is ambiguous: " + ", ".join(str(d.relative_to(root)) for d in found))
    return found[0]


def frame_stats(atoms: pd.DataFrame) -> dict:
    """Statistics of one traj.dump frame: bonds, Qt in the cluster with the most Qt, % of the Qt in
    clusters with at least 2 Qt."""
    qt_clusters = atoms.loc[atoms["type"] == 1, "cluster"].value_counts()
    n_qt = int(qt_clusters.sum())
    return {"bonds": int(atoms["nbonds"].sum()) // 2,
            "largest_qt": int(qt_clusters.max()) if n_qt else 0,
            "multi_qt_pct": 100.0 * qt_clusters[qt_clusters >= 2].sum() / n_qt if n_qt else 0.0}


def phases(p: Params) -> list[tuple[float, float, str, str]]:
    """(start, end [s], status line, timeline colour) of the protocol phases that last > 0."""
    bounds = (0.0, p.t_on_s, p.t_off_s, p.t_off_s + p.t_pumpout_s)
    status = ("CALCIUM OFF  |  INITIAL DISPERSED STATE", "CALCIUM ON  |  ASSEMBLY", "CALCIUM OFF  |  DISASSEMBLY")
    return [(bounds[i], bounds[i + 1], status[i], PHASE_COLORS[i]) for i in range(3) if bounds[i + 1] > bounds[i]]


def status_line(p: Params, t_s: float) -> str:
    """Status line of the header at production time t_s."""
    current = phases(p)[0][2]
    for start, _, text, _ in phases(p):
        if t_s >= start:
            current = text
    return current


def scale_bar_nm(view_nm: float) -> float:
    """Round scale bar length (1, 2 or 5 x 10^n nm), at most a fifth of the view."""
    target = view_nm / 5
    mag = 10 ** math.floor(math.log10(target))
    return max(f * mag for f in (1, 2, 5) if f * mag <= target)


# ---------------------------------------------------------------------------- sphere renderer
@lru_cache(maxsize=4096)
def _sprite(color: str, radius_q: float) -> tuple[np.ndarray, np.ndarray]:
    """Shaded sphere of radius radius_q px: (rgb, alpha) arrays, odd size, centre in the middle.
    Lambert shading with a highlight towards the top left, antialiased edge."""
    r = max(radius_q, 0.5)
    half = math.ceil(r + 1)
    d = np.arange(-half, half + 1, dtype=np.float32)
    dx, dy = np.meshgrid(d, -d)                              # dy up
    dist = np.hypot(dx, dy)
    alpha = np.clip(r + 0.5 - dist, 0.0, 1.0)
    u, v = dx / r, dy / r
    nz = np.sqrt(np.clip(1.0 - u * u - v * v, 0.0, 1.0))
    light = np.array([-1.0, 1.0, 1.4]) / np.linalg.norm([-1.0, 1.0, 1.4])
    half_vec = (light + [0, 0, 1]) / np.linalg.norm(light + [0, 0, 1])
    lambert = np.clip(u * light[0] + v * light[1] + nz * light[2], 0.0, 1.0)
    spec = np.clip(u * half_vec[0] + v * half_vec[1] + nz * half_vec[2], 0.0, 1.0) ** 40
    base = np.array([int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)], dtype=np.float32)
    rgb = np.clip(base * (0.30 + 0.80 * lambert)[..., None] + 0.40 * spec[..., None], 0.0, 1.0)
    return rgb.astype(np.float32), alpha.astype(np.float32)


def render_spheres(size: int, xy: np.ndarray, radius: np.ndarray, colors, depth: np.ndarray) -> np.ndarray:
    """RGBA image (size x size, row 0 at the top) of spheres at pixel positions xy (x right, y up from the
    lower left corner) with pixel radii, colours (hex) and depth in [0, 1] (1 = farthest; drawn first,
    darkened by DEPTH_CUE). The background stays transparent."""
    img = np.zeros((size, size, 4), np.float32)
    for i in np.argsort(-depth, kind="stable"):
        rgb, alpha = _sprite(colors[i], round(float(radius[i]) * 4) / 4)
        half = alpha.shape[0] // 2
        col, row = int(round(xy[i, 0])), int(round(size - xy[i, 1]))
        r0, r1, c0, c1 = row - half, row + half + 1, col - half, col + half + 1
        if r1 <= 0 or c1 <= 0 or r0 >= size or c0 >= size:
            continue
        s = (slice(max(r0, 0) - r0, alpha.shape[0] - max(r1 - size, 0)),
             slice(max(c0, 0) - c0, alpha.shape[1] - max(c1 - size, 0)))
        a = alpha[s][..., None]
        target = img[max(r0, 0):min(r1, size), max(c0, 0):min(c1, size)]
        target[..., :3] = target[..., :3] * (1 - a) + rgb[s] * (1 - DEPTH_CUE * depth[i]) * a
        target[..., 3:] = target[..., 3:] * (1 - a) + a
    return img


# ---------------------------------------------------------------------------- projections
def front_view(pos: np.ndarray, lo, hi, size: int) -> tuple[np.ndarray, float, np.ndarray]:
    """Orthographic view along y of the region lo..hi (x right, z up), fitted into the panel and centred:
    pixel positions, pixel per nm, depth in [0, 1] (y: 0 at lo = in front)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    scale = size / max(hi[0] - lo[0], hi[2] - lo[2])
    offset = (size - scale * (hi[[0, 2]] - lo[[0, 2]])) / 2
    xy = (pos[:, [0, 2]] - lo[[0, 2]]) * scale + offset
    depth = np.clip((pos[:, 1] - lo[1]) / (hi[1] - lo[1]), 0.0, 1.0)
    return xy, scale, depth


class Perspective:
    """Fixed camera at ELEVATION / AZIMUTH looking at the box centre (azimuth 0, elevation 0 = the front
    view); the projected box fills the panel up to a margin."""

    def __init__(self, box, size: int, margin: float = 0.06):
        box = np.asarray(box, float)
        el, az = math.radians(ELEVATION), math.radians(AZIMUTH)
        self.forward = np.array([-math.sin(az) * math.cos(el), math.cos(az) * math.cos(el), -math.sin(el)])
        self.right = np.array([math.cos(az), math.sin(az), 0.0])
        self.up = np.cross(self.right, self.forward)
        self.camera = box / 2 - CAMERA_DISTANCE * np.linalg.norm(box) * self.forward
        self.corners = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], float) * box
        self.scale, self.offset = 1.0, np.zeros(2)
        xy, _ = self._raw(self.corners)
        lo, hi = xy.min(axis=0), xy.max(axis=0)
        self.scale = size * (1 - 2 * margin) / max(hi - lo)
        self.offset = size / 2 - self.scale * (lo + hi) / 2
        dist = (self.corners - self.camera) @ self.forward
        self.near, self.far = dist.min(), dist.max()

    def _raw(self, pos):
        v = pos - self.camera
        dist = v @ self.forward
        return np.stack([v @ self.right, v @ self.up], axis=1) / dist[:, None], dist

    def project(self, pos: np.ndarray, radius: np.ndarray):
        """Pixel positions, pixel radii and depth in [0, 1] (box nearest to farthest corner)."""
        xy, dist = self._raw(pos)
        return (xy * self.scale + self.offset, radius / dist * self.scale,
                np.clip((dist - self.near) / (self.far - self.near), 0.0, 1.0))

    def edges(self) -> tuple[list, list]:
        """Pixel end points of the 12 box edges: (hidden ones, i.e. at the farthest corner; the others)."""
        xy, dist = self._raw(self.corners)
        xy = xy * self.scale + self.offset
        far = int(np.argmax(dist))
        pairs = [(a, b) for a in range(8) for b in range(a + 1, 8) if bin(a ^ b).count("1") == 1]
        hidden = [(xy[a], xy[b]) for a, b in pairs if far in (a, b)]
        return hidden, [(xy[a], xy[b]) for a, b in pairs if far not in (a, b)]


# ---------------------------------------------------------------------------- the frame
class AnimationFrame:
    """The matplotlib figure of the animation; update(t_s, atoms) draws one traj.dump frame."""

    def __init__(self, p: Params, thermo: pd.DataFrame, title: str = TITLE, scale: float = 2):
        _register_arial()
        self.p, self.box = p, np.array(p.box)
        self.t_end = p.t_off_s + p.t_pumpout_s
        c, h = self.box / 2, self.box * MAGNIFICATION / 2
        self.mag_lo, self.mag_hi = c - h, c + h
        self.camera = Perspective(self.box, PANEL)
        self.size = round(PANEL * scale)               # rendered panel images [px]
        self.fig = plt.figure(figsize=(W / DPI, H / DPI), dpi=DPI * scale)    # layout in px at scale 1
        over = self.fig.add_axes([0, 0, 1, 1], xlim=(0, W), ylim=(0, H))
        over.set_axis_off()
        self.over = over

        # header
        over.text(60, H - 40, title, fontsize=_px(46), color=WHITE, va="top")
        self.status = over.text(60, H - 108, "", fontsize=_px(28), color=TEAL, va="top")
        over.text(W - 60 - 150, H - 40, "Simulation time", fontsize=_px(28), color=WHITE, va="top", ha="right")
        self.time = over.text(W - 60, H - 40, "", fontsize=_px(28), color=WHITE, va="top", ha="right")
        self.calcium = over.text(W - 60, H - 108, "", fontsize=_px(28), color=TEAL, va="top", ha="right")

        # image panels
        self.panels = {}
        for key, x, y, label in (("3d", LEFT_X, TOP_Y, "Whole simulation box"),
                                 ("front", LEFT_X, BOTTOM_Y, "Whole simulation box, front view"),
                                 ("mag", RIGHT_X, TOP_Y, "Fixed central magnification")):
            ax = self.fig.add_axes([x / W, y / H, PANEL / W, PANEL / H], xlim=(0, PANEL), ylim=(0, PANEL))
            ax.set_xticks([]), ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set(color=FRAME, linewidth=_px(1))
            image = ax.imshow(np.zeros((self.size, self.size, 4)), extent=(0, PANEL, 0, PANEL),
                              interpolation="nearest", zorder=1)
            over.text(x, y + PANEL + 10, label, fontsize=_px(22), color=WHITE)
            self.panels[key] = (ax, image)

        ax3d = self.panels["3d"][0]
        hidden, front = self.camera.edges()
        for (a, b), z in [(e, 0.5) for e in hidden] + [(e, 2) for e in front]:
            ax3d.add_line(Line2D([a[0], b[0]], [a[1], b[1]], color=FRAME, linewidth=_px(1), zorder=z))

        axf = self.panels["front"][0]
        zero = np.zeros(3)
        corners, scale, _ = front_view(np.array([zero, self.box]), zero, self.box, PANEL)
        axf.add_patch(Rectangle(corners[0], *(corners[1] - corners[0]), fill=False, edgecolor=FRAME,
                                linewidth=_px(1), zorder=2))
        inset, _, _ = front_view(np.array([self.mag_lo, self.mag_hi]), zero, self.box, PANEL)
        axf.add_patch(Rectangle(inset[0], *(inset[1] - inset[0]), fill=False, edgecolor=INSET,
                                linewidth=_px(1.5), zorder=2))
        self._scale_bar(axf, max(self.box[0], self.box[2]), scale)
        _, mag_scale, _ = front_view(np.zeros((0, 3)), self.mag_lo, self.mag_hi, PANEL)
        self._scale_bar(self.panels["mag"][0], max(self.mag_hi[0] - self.mag_lo[0], self.mag_hi[2] - self.mag_lo[2]),
                        mag_scale)

        self._graph(thermo)
        self._legend_and_stats()
        self._timeline()

    def _scale_bar(self, ax, view_nm, scale):
        length = scale_bar_nm(view_nm)
        x1, y = PANEL - 20, 20
        ax.add_line(Line2D([x1 - length * scale, x1], [y, y], color=WHITE, linewidth=_px(3), zorder=3))
        ax.text(x1 - length * scale / 2, y + 8, f"{length:g} nm", fontsize=_px(18), color=WHITE, ha="center",
                va="bottom", zorder=3, path_effects=[patheffects.withStroke(linewidth=_px(4), foreground="black")])

    def _time_axes(self, rect, title, row_y):
        """Axes over the production time with the influx shaded and a moving time line; title at the height
        of the panel titles of the row whose panels start at row_y."""
        x, y, w, h = rect
        ax = self.fig.add_axes([x / W, y / H, w / W, h / H])
        self.over.text(x, row_y + PANEL + 10, title, fontsize=_px(22), color=WHITE)
        ax.axvspan(self.p.t_on_s, self.p.t_off_s, color=PHASE_COLORS[1], alpha=0.2, linewidth=0, zorder=0)
        self.now.append(ax.axvline(0, color=WHITE, linewidth=_px(1.5), zorder=3))
        ax.set_xlim(0, self.t_end)
        ax.set_xlabel("Time (s)", fontsize=_px(20))
        ax.tick_params(labelsize=_px(18), width=_px(1), length=_px(6))
        for side, spine in ax.spines.items():
            spine.set(color=GRAY, linewidth=_px(1), visible=side in ("left", "bottom"))
        return ax

    def _curve(self, ax, t, values, color, label, **style):
        line, = ax.plot(t, values, color=color, linewidth=_px(2), label=label, zorder=2, **style)
        dot, = ax.plot([], [], "o", color=color, markersize=_px(9), zorder=4)
        self.curves.append((t, values, dot))
        return line

    @staticmethod
    def _legend(ax, lines):
        legend = ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=_px(18),
                           loc="upper right", handlelength=1.5)
        for text in legend.get_texts():
            text.set_color(GRAY)

    def _graph(self, thermo):
        """Average cluster size (bottom right) and calcium + calmodulin loading (top right)."""
        self.now, self.curves = [], []
        ax = self._time_axes(GRAPH, "Average cluster size", BOTTOM_Y)
        t = thermo["time_ms"].to_numpy() / 1000
        lines = [self._curve(ax, t, thermo[column].to_numpy(), color, label) for column, label, color in TRACES]
        top = max(np.nanmax(line.get_ydata()) for line in lines)
        ax.set_ylim(0, top * 1.08 if top > 0 else 1)
        ax.set_ylabel("Average cluster size\n(particles per cluster)", fontsize=_px(20))
        self._legend(ax, lines)

        ax = self._time_axes(CALCIUM, "Calcium", TOP_Y)
        t = np.linspace(0, self.t_end, 2000)
        ca = self.p.calcium(t)
        lines = [self._curve(ax, t, ca, TEAL, "Ca$^{2{+}}$")]
        ax.set_ylim(0, ca.max() * 1.1)
        ax.set_ylabel("Ca$^{2{+}}$ (µM)", fontsize=_px(20))
        twin = ax.twinx()
        lines.append(self._curve(twin, t, self.p.loading(ca), GRAY, "loading f", linestyle=(0, (4, 2))))
        twin.set_ylim(0, 1.1)
        twin.set_ylabel("calmodulin loading f", fontsize=_px(20))
        twin.tick_params(labelsize=_px(18), width=_px(1), length=_px(6))
        for side, spine in twin.spines.items():
            spine.set(color=GRAY, linewidth=_px(1), visible=side == "right")
        self._legend(twin, lines)

    def _legend_and_stats(self):
        o, y = self.over, TOP_Y + PANEL - 20
        for type_id, name in ((1, "Qt"), (2, "Ft")):
            o.text(SIDE_X, y, name, fontsize=_px(22), color=WHITE, va="center")
            for bonded, label in ((0, "unbound"), (1, "bonded")):
                y -= 38
                o.plot([SIDE_X + 10], [y], "o", color=PARTICLE[type_id, bonded], markersize=_px(16))
                o.text(SIDE_X + 34, y, label, fontsize=_px(20), color=GRAY, va="center")
            y -= 60
        self.stats = {}
        for i, (key, caption) in enumerate((("bonds", "bonds"), ("largest_qt", "Qt in largest cluster"),
                                            ("multi_qt_pct", "Qt in multi-Qt clusters"))):
            top = BOTTOM_Y + PANEL - 10 - i * 112
            self.stats[key] = o.text(SIDE_X, top, "", fontsize=_px(48), color=TEAL, va="top")
            o.text(SIDE_X, top - 64, caption, fontsize=_px(18), color=GRAY, va="top")

    def _x(self, t_s):
        return BAR_X[0] + (BAR_X[1] - BAR_X[0]) * t_s / self.t_end

    def _timeline(self):
        o = self.over
        for start, end, _, color in phases(self.p):
            o.add_patch(Rectangle((self._x(start), BAR_Y), self._x(end) - self._x(start), 10, color=color,
                                  linewidth=0))
        labels = [(0.0, "0 s")]
        if self.p.t_influx_s > 0:
            labels += [(self.p.t_on_s, f"{self.p.t_on_s:g} s: ON"), (self.p.t_off_s, f"{self.p.t_off_s:g} s: OFF")]
        labels.append((self.t_end, f"{self.t_end:g} s"))
        for i, (t, text) in enumerate(labels):
            ha = "left" if i == 0 else "right" if i == len(labels) - 1 else "center"
            o.text(self._x(t), BAR_Y - 8, text, fontsize=_px(18), color=GRAY, ha=ha, va="top")
        self.marker = Polygon([[0, 0], [0, 0], [0, 0]], closed=True, color=WHITE, linewidth=0)
        o.add_patch(self.marker)

    def update(self, t_s: float, atoms: pd.DataFrame) -> None:
        """Draw the traj.dump frame `atoms` at production time t_s."""
        pos = atoms[["x", "y", "z"]].to_numpy(float)
        radius = atoms["radius"].to_numpy(float)
        colors = [PARTICLE[t, b] for t, b in zip(atoms["type"], (atoms["nbonds"] > 0).astype(int))]

        n, s = self.size, self.size / PANEL
        xy, r, depth = self.camera.project(pos, radius)
        self.panels["3d"][1].set_data(render_spheres(n, xy * s, r * s, colors, depth))
        zero = np.zeros(3)
        xy, scale, depth = front_view(pos, zero, self.box, n)
        self.panels["front"][1].set_data(render_spheres(n, xy, radius * scale, colors, depth))
        reach = np.abs(pos - (self.mag_lo + self.mag_hi) / 2) - (self.mag_hi - self.mag_lo) / 2
        inside = (reach[:, 1] < 0) & (reach[:, 0] < radius) & (reach[:, 2] < radius)
        xy, scale, depth = front_view(pos[inside], self.mag_lo, self.mag_hi, n)
        self.panels["mag"][1].set_data(render_spheres(n, xy, radius[inside] * scale,
                                                      [c for c, keep in zip(colors, inside) if keep], depth))

        self.status.set_text(status_line(self.p, t_s))
        self.time.set_text(f"{t_s:.2f} s")
        self.calcium.set_text(f"Ca$^{{2{{+}}}}$  {float(self.p.calcium(t_s)):.2f} µM")
        stats = frame_stats(atoms)
        self.stats["bonds"].set_text(f"{stats['bonds']}")
        self.stats["largest_qt"].set_text(f"{stats['largest_qt']}")
        self.stats["multi_qt_pct"].set_text(f"{stats['multi_qt_pct']:.0f} %")
        for line in self.now:
            line.set_xdata([t_s, t_s])
        for t, values, dot in self.curves:
            dot.set_data([t_s], [np.interp(t_s, t, values)])
        x = self._x(min(t_s, self.t_end))
        self.marker.set_xy([[x - 8, BAR_Y + 26], [x + 8, BAR_Y + 26], [x, BAR_Y + 13]])


# ---------------------------------------------------------------------------- public
def animate(run: str | Path, replica: int = 1, fps: int = 30, every: int = 1, stills: tuple = (),
            video: bool = True, title: str = TITLE, scale: float = 2, data_dir: str | Path = "runs") -> Path:
    """Render the animation of one replica: <run>/rep_XX/animation/animation.mp4, one video frame per
    `every`-th traj.dump frame at `fps` frames per second. stills: production times [s] that are also
    saved as still_<t>s.png (the nearest frame). video=False: stills only (no ffmpeg needed).
    scale: frame size 1920 x 1080 times scale (1 = full HD, 2 = 4K; the layout stays the same).
    run: the run folder or only its name. Returns the animation folder."""
    if not all(float(v * scale).is_integer() and v * scale % 2 == 0 for v in (W, H)):
        raise ValueError(f"scale {scale:g} gives {W * scale:g} x {H * scale:g} px; H.264 needs even whole "
                         "numbers (e.g. 1, 1.5, 2, 3)")
    run_dir = find_run(run, data_dir)
    require_fetched(run_dir)
    rep_dir = run_dir / f"rep_{replica:02d}"
    if not (rep_dir / "traj.dump").exists():
        raise FileNotFoundError(f"{rep_dir / 'traj.dump'} missing")
    if video and not FFMpegWriter.isAvailable():
        raise RuntimeError("ffmpeg not found: install it in the lammps env (conda install -c conda-forge ffmpeg); "
                           "stills work without it (video=False)")
    p = Params.load(run_dir / "params.json")
    steps = sorted(set(dump_steps(rep_dir / "traj.dump")))[::every]
    if not steps:
        raise ValueError(f"{rep_dir / 'traj.dump'} has no frames")
    times = np.array(steps) * p.timestep * 1e-9
    still_steps = {steps[int(np.argmin(np.abs(times - t)))]: t for t in stills}
    wanted = set(steps) if video else set(still_steps)
    out = rep_dir / "animation"
    out.mkdir(exist_ok=True)
    if video:
        print(f"{len(steps)} frames at {fps} fps: {len(steps) / fps:.1f} s of video for {times[-1]:.4g} s of "
              f"simulation ({times[-1] / (len(steps) / fps):.3g}x real time)")

    with plt.rc_context(STYLE):
        frame = AnimationFrame(p, load_thermo(rep_dir), title, scale)
        partial = out / "animation.partial.mp4"
        writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-crf", str(CRF)])
        start, done, seen = time.time(), 0, set()
        try:
            with writer.saving(frame.fig, str(partial), DPI * scale) if video else nullcontext():
                for step, atoms in read_dump(rep_dir / "traj.dump"):
                    if step not in wanted or step in seen:
                        continue
                    seen.add(step)
                    frame.update(step * p.timestep * 1e-9, atoms)
                    if video:
                        writer.grab_frame()
                    if step in still_steps:
                        frame.fig.savefig(out / f"still_{still_steps[step]:g}s.png", dpi=DPI * scale)
                    done += 1
                    if video and done % 100 == 0:
                        elapsed = time.time() - start
                        print(f"  {done}/{len(steps)} frames, {elapsed:.0f} s, about "
                              f"{elapsed / done * (len(steps) - done):.0f} s left", flush=True)
            if video:
                os.replace(partial, out / "animation.mp4")
                print(f"done in {time.time() - start:.0f} s: {out / 'animation.mp4'}")
        finally:
            plt.close(frame.fig)
            partial.unlink(missing_ok=True)
    return out
