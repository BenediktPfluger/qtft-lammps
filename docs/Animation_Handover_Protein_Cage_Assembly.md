# Handover: animation of the protein-cage assembly simulation

## Goal
Render an animation (video, e.g. MP4) of a particle simulation of reversible protein-cage assembly. Two particle
types move in a cubic simulation box; adding calcium makes them bond into clusters, and removing it lets the
clusters dissolve. The animation shows the whole box in 3D and as a front projection, a fixed magnified region,
live cluster statistics and the average cluster size over time.

---

## 1. General style
- **Canvas:** 16:9, 1920 × 1080 px, **pure black background** (#000000).
- **Font:** **Arial, regular weight everywhere — no bold text**, including the title and the large numbers.
  Use matplotlib `font.family = 'Arial'`, `font.weight = 'normal'` or the equivalent in the renderer used.
- **Text colors:** white (#FFFFFF) for the title and main labels, light gray (#B0B0B0) for secondary text,
  teal accent (≈ #4FC9B4) for the status line and the statistic values.
- **Panel frames:** thin light-gray outline (≈ 1 px, #9A9A9A) around every image panel.
- **Clean and minimal:** no grid, no unnecessary decorations.

## 2. Layout (four main panels in a 2 × 2 arrangement)

```
+-----------------------------------------------------------------------------------+
| Title                                                     Simulation time  01.25 s |
| STATUS LINE (teal)                                                                 |
|                                                                                    |
|  Whole simulation box (3D)          |       Fixed central magnification     | Legend|
|  [ 3D cube, perspective ]            |  [ magnified central region ]         |       |
|                                      |                                       |       |
|  Whole simulation box (front view)   |       Average cluster size            | Stats |
|  [ front projection + inset frame ]  |  [ full curve + moving time line ]    |       |
|                                                                                    |
|  timeline bar  ▼  0 s ........ 10 s: ON ............ 40 s: OFF .......... 130 s    |
+-----------------------------------------------------------------------------------+
```

- **Left column:**
  - **Top: "Whole simulation box"**, the 3D cube.
  - **Bottom: "Whole simulation box, front view"**, a 2D front projection of the whole box.
- **Right column:**
  - **Top: "Fixed central magnification"**, a magnified view of the central region of the front projection.
  - **Bottom: "Average cluster size"**, a time-course graph.
- **Legend and statistics:** in a narrow column to the right of the right panels (or between the two columns, as
  space allows). The legend sits next to the top row, the statistics next to the bottom row.
- **Header** above all panels, **timeline bar** below all panels, both spanning the full width.
- Panel titles sit above each panel, left-aligned, white, Arial regular, ≈ 20–24 px.
- The image panels are square (equal x and y scale); the graph may be wider than tall.

## 3. Header
- **Top left, title:** "Reversible assembly of protein cages", white, Arial regular, ≈ 40–48 px.
- **Below the title, status line:** teal, upper case, ≈ 26–30 px, changing with the phase:
  - 0–10 s: `CALCIUM OFF  |  INITIAL DISPERSED STATE`
  - 10–40 s: `CALCIUM ON  |  ASSEMBLY`
  - 40–130 s: `CALCIUM OFF  |  DISASSEMBLY`
- **Top right corner of the frame**, right-aligned on the height of the title: "Simulation time  01.25 s", white,
  ≈ 26–30 px; time in seconds with two decimals and a fixed width so the text does not jump (e.g. `  1.25 s`, ` 12.50 s`, `128.50 s`). Updated every frame.

## 4. Particles (all image panels)
- **Two particle types**, rendered as **shaded spheres** (soft 3D look: Lambert shading with a light highlight
  toward the top left, darker edge), drawn back to front so nearer spheres cover farther ones.
- **Qt:** the larger spheres (diameter 55 nm). **Ftn:** the smaller spheres (diameter 26 nm). Sphere sizes to
  physical scale.
- **Colors by state:**

  | Particle | unbound | bonded |
  |---|---|---|
  | Qt | green (≈ #1F8A1F) | blue (≈ #1E3CFF) |
  | Ftn | orange (≈ #F28C38) | red (≈ #E82020) |

- "Bonded" means the particle has at least one bond at that time step (take from the simulation output).
- A subtle depth cue (farther spheres slightly darker) is welcome, but colors must stay recognizable.

## 5. Panel details

### 5.1 Whole simulation box, 3D (left top)
- The full cubic box in **perspective**, viewed obliquely from the front-top (e.g. elevation ≈ 20°, azimuth ≈ 30°),
  with the **12 box edges** drawn as thin light-gray lines.
- All particles of the box as shaded spheres, colored by state.
- Fixed camera (no rotation), so the viewer can follow the clustering.
- The magnified region is **not** marked in the 3D view (only in the front view, 5.2).

### 5.2 Whole simulation box, front view (left bottom)
- Orthographic projection of the **whole box along the depth axis** (view from the front), all particles as
  shaded spheres.
- **Inset frame:** thin blue-gray rectangle (≈ #8A9AB0, 1–2 px) marking the central region shown in the
  magnification, fixed in position (about one third of the box width, centered).
- **Scale bar:** bottom right inside the panel, white line with the label above it (e.g. "500 nm"), Arial regular.

### 5.3 Fixed central magnification (right top)
- The region inside the inset frame, magnified to fill the panel, same projection and rendering as 5.2.
- The region is **fixed in space** (does not follow particles).
- **Scale bar:** bottom right, e.g. "200 nm".
- **Full-depth projection** of that region (all particles along the viewing direction, like 5.2), so the
  magnification shows exactly what lies inside the frame in the front view.

### 5.4 Average cluster size (right bottom)
- Line plot of the **average cluster size over the whole simulation time**. The **complete curve is visible from
  the first frame**.
- A **vertical line** (white, ≈ 1.5 px) marks the current simulation time and moves from left to right; optionally a
  small dot on the curve at the current value.
- **Stimulation period** (10–40 s) shaded behind the curve in the orange of the timeline bar, low opacity (≈ 20 %).
- Curve color: teal accent (≈ #4FC9B4), ≈ 2 px.
- Axes: white/light-gray lines and ticks, Arial regular; x label "Time (s)", y label "Average cluster size" with the
  unit the user defines (e.g. "(particles per cluster)"). No title inside the plot (the panel title is above).
- **Definition of "average cluster size"** must come from the user (e.g. mean number of particles per cluster
  including or excluding single particles, or the weight-averaged size). Compute it from the simulation output
  for every saved time step.

## 6. Legend and statistics
- **Legend** (next to the top row): heading "Qt", then a green dot "unbound" and a blue dot "bonded"; heading
  "Ftn", then an orange dot "unbound" and a red dot "bonded". Headings white, entries light gray, Arial regular,
  ≈ 18–22 px.
- **Statistics** (next to the bottom row), updated every frame, heading "WHOLE SIMULATION BOX" (small, gray, upper
  case), then three entries, each as a large teal number (≈ 44–52 px, regular weight) with a small gray caption
  below:
  - number of `bonds`
  - `Qt in largest cluster`
  - percentage of `Qt in multi-Qt clusters`

## 7. Timeline bar (bottom)
- Full-width horizontal bar (≈ 10 px high) split into three colored segments by phase:
  - 0–10 s, calcium off: slate blue-gray (≈ #3E4E5A)
  - 10–40 s, calcium on: ochre/orange (≈ #B87A35)
  - 40–130 s, calcium off: dark teal (≈ #2A6E66)
- Labels below the bar at the boundaries: `0 s`, `10 s: ON`, `40 s: OFF`, `130 s` (light gray, Arial regular).
- A small white downward triangle (▼) above the bar marks the current time and moves with it.
- Phase times are taken from the simulation settings, not hard-coded.

## 8. Animation and output
- One frame per saved simulation step (or interpolated / subsampled to a fixed frame rate, e.g. 30 fps); the total
  duration and the mapping of simulation time to video time are a setting (real time by default).
- Output: MP4 (H.264), plus optionally a few still frames as PNG (e.g. before, during and after stimulation).
- All panels, the time text, statistics, the time line in the graph and the timeline marker are updated in sync for
  every frame.

## 9. Inputs needed from the simulation
- Per time step: particle positions (x, y, z), particle type (Qt / Ftn), radius, bonded state (or bond list),
  cluster membership.
- Box size, phase times (calcium on/off), time step.
- From these: number of bonds, size of the largest cluster (in Qt), share of Qt in clusters with ≥ 2 Qt, and the
  average cluster size per time step.

## 10. Open point to confirm with the user
- Definition and unit of "average cluster size" (will be provided by the user).
