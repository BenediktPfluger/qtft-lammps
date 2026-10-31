# Repository structure and files

This page lists the folders of the repository and every file a run creates: the simulation input, the
simulation output, the analysis data and the figures. For the Python modules in `qtft/`, see the table in the
[README](../README.md#repository-layout).

Contents:

1. [Repository](#1-repository)
2. [Data folder `runs/`](#2-data-folder-runs)
3. [Simulation input](#3-simulation-input)
4. [Simulation output](#4-simulation-output)
5. [Analysis data](#5-analysis-data)
6. [Figures](#6-figures)
7. [Other outputs](#7-other-outputs)

"Written by" names the function that writes a file, or LAMMPS. The notebook sections refer to
`control.ipynb` (control) and `analysis.ipynb` (analysis).

---

## 1. Repository

```
qtft-lammps/
├── control.ipynb            set parameters, build, run, monitor, sweeps, benchmark
├── analysis.ipynb           ensemble statistics, report, FIB-SEM comparison, comparison of runs, sweeps, FIB-SEM export
├── animation.ipynb          video and stills of one replica
├── qtft/                    the Python package (all code; modules: see the README)
├── template.lmp             LAMMPS input template: model structure with {{placeholders}}, no numbers
├── job_template.slurm       Slurm job script template for runs on LRZ
├── lammps_plugins/
│   └── bond_break_random/   LAMMPS plugin fix bond/break/random: C++ source, build.sh (the built .so is not committed)
├── tests/                   pytest tests; the tests marked slow run short LAMMPS simulations
├── docs/
│   ├── install.md           installation on this PC and on LRZ
│   └── repo_structure.md    this page
├── environment.yml          conda environment "lammps"
├── pytest.ini               pytest settings
├── README.md
├── lrz_local.json           personal LRZ settings (email for Slurm)   gitignored
└── runs/                    all simulation output (section 2)          gitignored
```

---

## 2. Data folder `runs/`

All simulation output goes to `runs/` inside the repository. It is gitignored and never committed. Everything
in `analysis/` can be regenerated from the replica files at any time.

```
runs/
├── index.csv                        one line per single, ensemble and sweep run
├── single/                          runs with n_replicas = 1
│   └── <run name>/                  same structure as below, with rep_01/ only
├── ensemble/                        runs with n_replicas > 1
│   └── <run name>/
│       ├── params.json              the record of the run
│       ├── launcher.log, launcher.pid
│       ├── job.slurm, replicas.txt, slurm_job.json, slurm-*.out    only with executor = "lrz"
│       ├── rep_01/ … rep_NN/        one folder per replica
│       │   ├── in.lammps, status, pid, lammps.out
│       │   ├── log.lammps, traj.dump, bonds.dump, rdf.dat, coord_qt.hist, coord_ft.hist, final.data
│       │   ├── fibsem/              only after the FIB-SEM export
│       │   └── animation/           only after animate()
│       └── analysis/                only after the analysis
│           ├── *.csv, *.json, summary_table.tex, report.pdf
│           └── figures/
│               └── <figure>/        <figure>.pdf/.svg/.png and every panel alone, <panel>.pdf/.svg/.png
├── sweeps/                          only after build_sweep()
│   └── <sweep name>/
│       ├── sweep.json, launcher.log, launcher.pid, sweep_results.csv
│       ├── job.slurm, replicas.txt, slurm_job.json    only with executor = "lrz" (one job array for the whole sweep)
│       └── <label>/                 one run folder per combination
├── benchmark/                       only after benchmark()
└── tuning/                          only after alpha_sweep()
```

`build_run()` creates `runs/` and its `single/` and `ensemble/` subfolders when needed. The subfolder follows
from `n_replicas`; the `group` parameter overrides it.

### Run names

The run folder is named by `Params.auto_name()`, unless `run_name` is set:

```
Qt20_Ft80_L4000_kon1860_koff0.008-0.2_Ca0.1-1uM_K0.5_rise5tau30_t10-30-150s_dt500ns_periodic
```

| Part | Meaning |
|---|---|
| `Qt20_Ft80` | Qt and Ft per µm³ in the full box after rounding the particle numbers (3 significant digits) |
| `L4000` | box edge in nm (`2000x3000x4000` for a rectangular box) |
| `kon1860` | k_on in 1/s at full calmodulin loading |
| `koff0.008-0.2` | k_off with Ca-loaded and with Ca-free calmodulin, in 1/s |
| `Ca0.1-1uM_K0.5` | resting and peak calcium, and the half-loading K, in µM |
| `rise5tau30` | rise time and pump-out time constant, in s |
| `t10-30-150s` | durations of baseline, influx and pump-out |
| `dt500ns` | time step |
| `periodic` | boundary (`periodic`, `reflect` or `harmonic`) |

The force constants, the bond constant and the Hill coefficient are not part of the name. Runs that differ only
in those, or only in execution settings and seeds, need their own `run_name`.

### `index.csv`

One line per single, ensemble and sweep run: group (`single`, `ensemble` or `sweeps/<name>`), name, date,
status counts, key parameters, `executor` (`local` or `lrz`), the parameters that differ from the defaults, and
the git commit. Written by `build_run()`, whenever a replica finishes, and by `fetch()`.

---

## 3. Simulation input

Written by `build_run()` (control, section 2) before LAMMPS starts.

| File | Content | Written by |
|---|---|---|
| `params.json` | **The record of the run**: all parameters (the particle amounts as densities `qt_per_um3`, `ft_per_um3`), derived values (particle numbers, k_BT, γ, α, p, …), check results, seeds of every replica, git commit, creation time | `build_run()` |
| `rep_XX/in.lammps` | the LAMMPS input of this replica, all numbers filled in from `template.lmp`; can be run by hand with `lmp -in in.lammps` | `build_run()` |
| `job.slurm` | only with `executor = "lrz"`: Slurm job script, one array task per replica (cluster, partition, QOS, `mpi_procs` tasks, `slurm_mem`, `slurm_time`, email from `lrz_local.json`) | `build_run()`, `build_sweep()`, from `job_template.slurm` |
| `replicas.txt` | only with `executor = "lrz"`: the replica folders, one per line; array task i runs line i | `build_run()`, `build_sweep()` |

---

## 4. Simulation output

### LAMMPS files in `rep_XX/`

Everything in these files is computed by LAMMPS during the run; no Python or OVITO code runs during a
simulation. All files except `log.lammps` are written at the `traj.dump` steps (every `traj_every` steps).

| File | Content | Written by |
|---|---|---|
| `log.lammps` | full log: input commands, setup, timing, and the **thermo table** every `output_every` steps (YAML; columns below). Main source of all time series | LAMMPS |
| `traj.dump` | trajectory: `id type x y z ix iy iz radius cluster nbonds`. **Open in OVITO** | `template.lmp`: `dump custom` |
| `bonds.dump` | all bonds (`atom1 atom2`) | `template.lmp`: `dump local` |
| `rdf.dat` | Qt–Ft g(r) and coordination number (200 bins up to 121.5 nm by default); snapshots, not time averages | `template.lmp`: `compute rdf` |
| `coord_qt.hist`, `coord_ft.hist` | histogram of bonds per Qt (0 … `maxbond_qt`) and per Ft (0 … `maxbond_ft`) | `template.lmp`: `fix ave/histo` |
| `final.data` | final state: box, positions, type labels (1 Qt, 2 Ft), all bonds, coefficients. Opens in OVITO with bonds; can continue a run with `read_data` | `template.lmp`: `write_data` |
| `log.cite` | references for the LAMMPS features used (only some LAMMPS versions) | LAMMPS |

Columns of `traj.dump`:
- `type`: the type label `Qt` or `Ft`, shown as named types in OVITO.
- `ix iy iz`: periodic image flags; always 0 with walls, where `BOX BOUNDS` reads `ff ff ff`.
- `cluster`: ID of the bond-connected cluster; `nbonds`: bonds of the particle.

`rdf.dat` is a convenience: the g(r) can also be recomputed from `traj.dump` (OVITO coordination analysis,
LAMMPS `rerun`, or Python).

<details>
<summary>Thermo table in <code>log.lammps</code></summary>

One YAML block for the equilibration and one per production segment (`rate_update_every` steps). The step
counter restarts at 0 for the production, and every segment repeats the last step of the previous one.
`qtft.load_thermo(rep_dir)` joins the production blocks and drops the repeated steps.

| Column | Meaning |
|---|---|
| `Step` | time step (production: time = Step × Δt) |
| `PotEng`, `E_vdwl`, `E_bond` | potential energy: total, pair repulsion, bonds (zJ) |
| `c_pvir` | virial pressure from pair and bond forces only (no kinetic term, no wall forces; 1 unit = 1 MPa) |
| `Bonds` | number of bonds |
| `v_n_qt_bound`, `v_n_ft_bound` | Qt and Ft with at least one bond |
| `v_frac_bound` | fraction of all particles that are bound |
| `v_n_clusters` | number of clusters, free particles included |
| `v_mean_size`, `v_largest` | mean and largest cluster size (particles) |
| `v_mean_size_qt`, `v_mean_size_ft` | Qt (Ft) per cluster: number of Qt (Ft) / number of clusters with at least one Qt (Ft); a free particle is a cluster of 1 |
| `v_largest_qt`, `v_largest_ft` | most Qt (Ft) in one cluster (not necessarily the overall largest cluster) |
| `v_n_large` | clusters with ≥ `large_cluster_min` particles |
| `v_ca`, `v_f` | calcium (µM) and calmodulin loading f (production only) |
| `v_k_on_s`, `v_k_off_s` | binding rate per Qt–Ft pair in contact and unbinding rate per bond (1/s; production only) |
| `v_n_broken` | bonds broken since the production start (production only) |
| `v_f_mono` … `v_f_vlarge` | fraction of particles in monomers, small, medium, large and very large clusters |
| `v_deg_qt_mean`, `v_deg_ft_mean`, `v_deg_qt_max`, `v_deg_ft_max` | mean (over bound particles) and maximum bonds per Qt and Ft |
| `v_bond_mean`, `v_bond_max` | mean and maximum bond length (nm) |
| `S/CPU`, `CPULeft` | steps per second, and the time left of the current production segment (s) |

Kinetic energy and temperature are not logged: they have no meaning in overdamped Brownian dynamics.

</details>

### Run control

| File | Content | Written by |
|---|---|---|
| `rep_XX/status` | one word: `built` → `queued` → `running` → `done`, `failed` or `stopped`; its last change is the end time of the replica | `build_run()`, the launcher; on LRZ `job.slurm` |
| `rep_XX/pid` | process ID of LAMMPS, used by `stop()` (LRZ: the Slurm job ID of the array task); its time is the start time | the launcher; on LRZ `job.slurm` |
| `rep_XX/lammps.out` | terminal output of LAMMPS and MPI; usually almost empty. **Look here first when a run fails** | the launcher |
| `launcher.log` | log of the background launcher: start and end of every replica, a summary every minute (done, running, queued, estimated end) and a line per running replica; follow it with `tail -f launcher.log` | `start()` (control, section 3) |
| `launcher.pid` | process ID of the launcher, used by `stop()` | `start()` |
| `slurm_job.json` | only with `executor = "lrz"`: Slurm job ID, cluster, folder on LRZ, submission time; used by `progress`, `watch`, `stop`, `fetch` | `start()` |
| `slurm-<job>_<task>.out` | only with `executor = "lrz"`: output of the job script of one array task | Slurm on LRZ |

On LRZ, a task that left Slurm without an end status (time limit, node failure) is set to `failed`, and a
cancelled one to `stopped`.

---

## 5. Analysis data

Everything here lives in `<run>/analysis/`. It is written by `ensemble_stats()` (analysis, section 2) or
`make_report()` (analysis, section 3). Files marked "cached" are reused while they are newer than every
`traj.dump` and match the requested settings.

### Ensemble statistics

| File | Content | Written by |
|---|---|---|
| `ensemble_stats.json` | mean, SD (ddof = 1) and per-replica traces of all thermo quantities on a common time grid; half-times; percolation; final values per replica | `ensemble_stats()` |
| `ensemble_timeseries.csv` | the same time series as a table (`<quantity>_mean`, `<quantity>_sd`) | `ensemble_stats()` |
| `final_state.csv` | final value of every quantity (end of the calcium protocol): mean, SD, number of replicas; also the peaks and deagglomeration half-times | `ensemble_stats()` |
| `percolation_rep_XX.csv` | per `traj.dump` frame: `Step, time_ms, axes` (the axes along which one cluster spans the box, `""` for none); written once the replica is done (delete it to recompute) | `ensemble_stats()` |

- `percolated` in `ensemble_stats.json`: a cluster spans the box in the last frame. `t_percolation_ms`: the first
  frame with a spanning cluster. In `final_state.csv`, row `percolated` holds the fraction of percolated replicas.
- Extra rows of `final_state.csv`: `bonds_peak`, `t_bonds_peak_ms`, `largest_peak`, `t_largest_peak_ms`,
  `t_half_unbind_bonds_ms`, `t_half_unbind_largest_ms` (half-times of the deagglomeration after the influx).

### Agglomerates and structure

| File | Content | Written by |
|---|---|---|
| `agglomerates_above_size.csv` | number of agglomerates with ≥ n particles over time, for several n (cached) | `make_report()`, `compare_report()` |
| `agglomerates_above_size_qt.csv`, `_ft.csv` | the same with the size counted in Qt or in Ft only (cached) | `make_report()` |
| `agglomerates_influx_end.csv` | every agglomerate of ≥ 3 particles at the end of the influx, with size, composition and volume | `make_report()` |
| `structure_timeseries.csv` | per analysed frame, mean and SD over replicas: Rg, Rg/Rg_ideal, nearest-neighbour distances between and within clusters, Qt fraction per cluster | `make_report()` (OVITO) |
| `structure_per_replica.csv` | the same per replica | `make_report()` (OVITO) |
| `clusters_final.csv` | every cluster of the last frame: size, Qt and Ft count, bonds, Rg, ideal Rg (sphere of the particle volumes at packing 0.64), centre | `make_report()` (OVITO) |

- `agglomerates_above_size*.csv`: one row per `traj.dump` frame and threshold, with columns `step, time_s,
  n_min, replicas, count_mean, count_sd` (in the box), `per_1000um3_mean, per_1000um3_sd` (count · 1000 / V_box
  in µm³), and `count_<replica>`. A free particle is a cluster of 1. New thresholds are added to the file.
  Default thresholds: 5, 20, 60, 80, 100, 150 particles (and Ft); 5, 10, 20, 50, 70 Qt; set with
  `make_report(…, n_min, n_min_qt, n_min_ft, n_qt_ft)`. The `_qt` and `_ft` files also hold the pair `n_qt_ft`.
- `agglomerates_influx_end.csv`: the last `traj.dump` frame at or before the influx end (the run end without
  an influx). Columns `replica, step, time_s, agglomerate` (1 = largest), `size, n_qt, n_ft, n_bonds, spans`
  (axes the agglomerate spans the box along), `v_solid_um3` (union of the particle spheres), `v_envelope_um3`
  (closing with the ellipsoidal probe), `probe_xy_nm, probe_z_nm, voxel_nm`. The volumes are NaN for an
  agglomerate that wraps around the periodic box.

### Rates

| File | Content | Written by |
|---|---|---|
| `bond_rates.csv` | bonds formed and broken per second, and the apparent association rate constant k_a = formed per second · V / (N_Qt N_Ft) in 1/(M s), over time; mean and SD over replicas | `make_report()`, `bond_rates()` |
| `binding_rate.csv` | k_a measured from the start of the bond curve, and predicted by Collins–Kimball from the Qt–Ft contact pairs at the influx start; only with a calcium step (`ca_rise_s = 0`) | `make_report()`, `binding_rate()` |

- `bond_rates.csv`: columns `time_ms` (bin centre), `formed_per_s`, `broken_per_s`, `k_a_per_M_s`, each as
  `_mean` and `_sd`; one row per calcium segment (`rate_update_every`) or per `bin_s`.
- `binding_rate.csv`: columns `value, unit, 1/(M s)`; rows `k_a measured` and `SD` (frames up to 10 % more
  bound), `k_a predicted` and `SD`, `kappa`, `k_D`, `k_a continuum estimate` (analytical; too high when the
  time step is comparable to the overlap layer), `contact volume counted` and `contact volume continuum` (nm³).

### Overlap

| File | Content | Written by |
|---|---|---|
| `overlap_timeseries.csv` | interpenetration statistics per pair family and frame, mean over replicas; with walls also `Qt-wall` and `Ft-wall` (spheres beyond a box face) | `make_report()`, `overlap_run()` |
| `overlap_final.csv` | the same, pooled over the last frames | `make_report()`, `overlap_run()` |

### FIB-SEM comparison (Qt only)

Computed for the first `traj.dump` frame and the frame at the end of the influx (the same frame as
`agglomerates_influx_end.csv`) of every replica. Distances use the minimum image in a periodic box. All four
files are cached.

| File | Content | Written by |
|---|---|---|
| `qt_nn_distances.csv` | nearest-neighbour distances: `replica, frame, step, time_s, set, distance_nm` | `make_report()`, `qt_spatial_run()` |
| `qt_pair_correlation.csv` | Qt–Qt g(r) of all Qt at the end of the influx: `r_nm, g_<replica>, g_mean` | `make_report()`, `qt_spatial_run()` |
| `qt_clark_evans.csv` | Clark–Evans index of all Qt per replica and frame: `replica, frame, step, time_s, n_qt, density_per_um3, mean_nn_nm, expected_nn_nm, R` | `make_report()`, `qt_spatial_run()` |
| `qt_ripley_l.csv` | Ripley's L(r) − r of all Qt per replica and frame: `replica, frame, r_nm, L_minus_r` | `make_report()`, `qt_spatial_run()` |

- `frame`: `first` or `influx_end`. `set` in `qt_nn_distances.csv`: `all` (every Qt to the nearest other Qt,
  both frames) or `intra` (every Qt in an agglomerate with ≥ 2 Qt to the nearest Qt of the same agglomerate,
  end of the influx only).
- g(r): 5 nm bins up to min(L/2, 1000 nm), normalised to the Qt density in the full box; with walls no edge
  correction.
- Clark–Evans: R = mean nearest-neighbour distance / its value for random points of the same density,
  0.554 ρ^(−1/3). R = 1 random, R < 1 clustered.
- Ripley: L = (3K/(4π))^(1/3), K = V/(N − 1) · mean number of other Qt within r (V = full box), 5 nm steps up to
  min(L/2, 1000 nm). L − r = 0 random, > 0 clustered at scale r, < 0 more regular. With walls only Qt at least
  r from every wall are centres (border correction).

### Report

| File | Content | Written by |
|---|---|---|
| `summary_table.csv`, `.tex` | final-state table: bonds, clusters, sizes, fraction bound, half-time, Rg, Qt fraction | `make_report()` |
| `bonds_timeseries.csv` | bonds over time: step, time (µs, ms), mean, SD | `make_report()` |
| `report.json` | settings of the report (`every`, replicas) | `make_report()` |
| `report.pdf` | cover pages and every figure of section 6, one per page (see below) | `make_report()` |

`report.pdf` is rewritten on every `make_report()`. Its cover pages hold the run name and folder, the creation
date and git commit (from `params.json`), the replicas, the report date and `every`, the final-state table, all
simulation parameters, the derived values and the parameter checks. The figures follow as vector pages in the
order of section 6.

---

## 6. Figures

`make_report()` writes every figure to `<run>/analysis/figures/<figure>/`:

- the whole figure as `<figure>.pdf`, `.svg` (editable text) and `.png` (300 dpi);
- for figures with several panels, also every panel alone (89 mm wide) as `<panel>.pdf/.svg/.png`, named after
  the panel title (for example `mean_cluster_size.pdf`).

Common to all figures: mean over replicas as a line with a ±1 SD band; dotted vertical lines at the start and
end of the calcium influx in every time panel; time axes in ms, in s for runs of 10 s or longer. "End of the
influx" means the last `traj.dump` frame at or before it (the run end for a run without an influx).

The figures appear in this order in `report.pdf`.

### `calcium` (3 × 2)

The calcium protocol and the binding it drives. Shown in analysis section 3.

1. **calcium and loading**: calcium (µM) and calmodulin loading f (second axis).
2. **rate constants**: k_on per Qt–Ft contact pair and k_off per bond (1/s, log scale).
3. **binding and unbinding rates**: bonds formed and broken per second (from `bond_rates.csv`).
4. **apparent k_a**: formed per second · V / (N_Qt N_Ft), in 1/(M s), log scale.
5. **bonds and broken bonds**: number of bonds, and bonds broken since the production start.
6. **fraction bound**: fraction of Qt and of Ft with at least one bond.

### `overview` (4 × 3)

All thermo and structure quantities of one run. Shown in analysis section 3.

1. **potential energy** (zJ).
2. **pressure**: virial pressure (Pa).
3. **bonds**.
4. **particle counts**: free and bound Qt and Ft.
5. **agglomerates + free**: number of clusters, free particles included.
6. **mean cluster size**: all particles, Qt only and Ft only (free particles count as clusters of 1).
7. **largest agglomerate**: all particles, Qt only and Ft only.
8. **size categories**: fraction of particles in monomers, small, medium, large and very large clusters
   (stacked; bounds from the particle number).
9. **mean Rg** (nm), from the structure analysis.
10. **Qt fraction**: Qt fraction per agglomerate, from the structure analysis; dashed line at 0.5.
11. **coordination**: mean bonds per bound Qt and per bound Ft.
12. **coordination at end of influx**: histogram of bonds per Qt and per Ft (from `coord_*.hist`).

### `agglomerate_distributions` (1 × 3)

Distributions at the end of the influx, agglomerates of ≥ 3 particles, log bins, counts per replica (from
`agglomerates_influx_end.csv`). Shown in analysis section 3.

1. **agglomerate size at end of influx**: particles per agglomerate.
2. **solid volume at end of influx**: union of the particle spheres (µm³).
3. **envelope volume at end of influx**: closing with the ellipsoidal probe (µm³), comparable to confocal
   volumes.

### `agglomerates_above_size` (4 × 2)

Number of agglomerates above a size threshold over time, one line per threshold. Left column: in the box
(`number`); right column: per 1000 µm³ (`density`). From `agglomerates_above_size*.csv`. Shown in analysis
section 3.

1. **all particles, number** and 2. **all particles, density**: agglomerates with ≥ n particles.
3. **Qt, number** and 4. **Qt, density**: agglomerates with ≥ n Qt.
5. **Ft, number** and 6. **Ft, density**: agglomerates with ≥ n Ft.
7. **≥ 15 Qt and ≥ 20 Ft, number** and 8. **density**: one Qt threshold (green) and one Ft threshold (red)
   together; the pair is set with `n_qt_ft`.

### `fibsem_comparison` (2 × 2, Qt only)

Metrics that FIB-SEM can measure, since it resolves the Qt but not the ferritin (from the `qt_*.csv` files).
Shown in analysis section 4.

1. **Qt nearest neighbour in agglomerates**: distance from every Qt in an agglomerate with ≥ 2 Qt to the nearest
   Qt of the same agglomerate, at the end of the influx; histogram (1 nm bins), kernel density estimate and its
   peak "r = … nm".
2. **Qt pair correlation**: g(r) of all Qt at the end of the influx up to 500 nm, log scale, with its first peak.
3. **Qt clustering (Clark–Evans)**: fraction of Qt with another Qt within d, in the first frame and at the end
   of the influx, with the curve for random points and the Clark–Evans R and mean distance ⟨d⟩ in the legend.
4. **Qt clustering scale (Ripley's L)**: L(r) − r of all Qt in the first frame and at the end of the influx, up
   to 500 nm, with the maximum and its position in the legend.

### `kinetics` (3 × 1)

The main kinetics on one time axis, 89 mm wide. Not shown in the notebook; in `report.pdf`.

1. **bonds**.
2. **fraction bound**: Qt (solid) and Ft (dashed).
3. **mean cluster size**.

### `overlap` (1 panel)

Number of overlapping pairs per pair family (Qt–Qt, Ft–Ft, Qt–Ft; with walls also Qt–wall and Ft–wall) over
time, log scale (from `overlap_timeseries.csv`). Not shown in the notebook; in `report.pdf`.

### `binding_rate` (1 panel)

Only with an influx and a calcium step (`ca_rise_s = 0`). Bonds formed since the influx start, with the fitted
k_a and the Collins–Kimball prediction from the counted contact pairs (from `binding_rate.csv`). Shown in
analysis section 3.

### Figures drawn on request

Not saved by `make_report()`; draw them in the notebook (see the note in analysis section 3):

- `fig_agglomerates_above_size_total(runs, …)`: the total panel of `agglomerates_above_size` alone, also for
  several runs.
- `fig_bond_rates(run, rates=…)`: the two rate panels alone, for example with coarser bins
  (`rates=bond_rates(run_dir, bin_s=…)`).
- `quicklook(run_dir)` (control, section 5): a quick overview of a running or finished run, not saved.

### Comparison of runs: `compare_report(runs, out_dir)`

Analysis section 5. Writes to the chosen `out_dir`:

| File | Content |
|---|---|
| `figures/comparison/` | 3 × 4: potential energy, pressure, bonds, agglomerates + free, mean size, mean size / N, largest, largest / N, mean Rg, Rg / Rg,ideal, coordination, Qt fraction; one colour per run, every panel also alone |
| `figures/kinetics/` | the kinetics figure (above) with one colour per run |
| `figures/agglomerates_above_size_total_density/` | agglomerates with ≥ n particles per 1000 µm³ (so runs with different boxes compare); colour per run, line style per threshold (at most 6) |
| `comparison_table.csv`, `.tex` | final-state values (mean ± SD) of every run, side by side |
| `report.pdf` | cover pages (every run with number, label, folder, creation date and git commit; the comparison table; all simulation parameters, one column per run), then the three figures |

The legends name each run by the parameters that differ, in short form (`Qt 1280, Ft 5120, L 4000`; the short
names are in `SHORT_NAMES` in `qtft/ensemble.py`). `comparison_table.*` uses the full form (`qt_per_um3=20.0, …`).
Cluster sizes are drawn for all particles only; the phase lines only if all runs have the same phase times.

---

## 7. Other outputs

### FIB-SEM export `rep_XX/fibsem/`

Only after `fibsem_export_run(run_dir)` (analysis, section 7). The final frame in the schema of the FIB-SEM
segmentation pipeline.

| File | Content | Written by |
|---|---|---|
| `sim_encapsulin_centroids_clustered.csv` | one row per encapsulin: `label, z_nm, y_nm, x_nm, z_vox, y_vox, x_vox, radius_nm, volume_nm3, cluster, is_clustered` | `fibsem_export()` (OVITO) |
| `sim_metadata.json` | counts, cluster-size histogram, box and offset, frame, bond check, all parameters, `boundary`, `wall_stiffness`, `wall_leaks` | `fibsem_export()` |
| `check_xy.png` | x–y scatter of the encapsulins coloured by cluster, with the box or walls as dashed outline | `fibsem_export()` |

In a periodic box the clusters are unwrapped across the box and shifted to the origin. With walls the
positions stay in the box frame, with the walls at 0 and the box size. `wall_stiffness` holds k of Qt and Ft
for harmonic walls; `wall_leaks` the spheres sticking out of the box and the largest depth (null in a
periodic box).

### Animation `rep_XX/animation/`

Only after `animate(run)` (animation notebook).

| File | Content | Written by |
|---|---|---|
| `animation.mp4` | video, 1920 × 1080 × `scale` (default 2: 3840 × 2160), H.264, one frame per `traj.dump` frame (every `every`-th) at `fps` | `animate()` (needs ffmpeg) |
| `still_<t>s.png` | optional: the frame nearest to production time t (s), for every t in `stills`; `video=False` writes only these | `animate()` |

The video shows a header (title, calcium phase, time, calcium), the whole box in 3D and from the front with the
magnified region outlined, a fixed central magnification, calcium and calmodulin loading f over time, the mean
cluster size (all, Qt, Ft), statistics of the frame (bonds, most Qt in one cluster, % of Qt in clusters with
≥ 2 Qt) and the protocol timeline. It is written as `animation.partial.mp4` and renamed when complete.

### Parameter sweep `runs/sweeps/<sweep name>/`

Only after `build_sweep()` (control, section 4). One run folder per combination of the listed values, named
after the swept values only (for example `temperature300_k_on2e-06/`), with the structure of a run folder; the
launcher files belong to the whole sweep.

| File | Content | Written by |
|---|---|---|
| `sweep.json` | **The record of the sweep**: name, mode (`grid`), the values per parameter, base parameters, label and values of every run, git commit, creation time | `build_sweep()` |
| `launcher.log`, `launcher.pid` | as for a run, for all replicas of the sweep (one queue; replicas named `<label>/rep_XX`) | `start(sweep)` |
| `sweep_results.csv` | one row per run: swept parameters, finished replicas, mean and SD of final fraction bound, bonds, largest cluster fraction, mean cluster size, half-times (bonds, Qt, Ft; ms), percolation time and percolated replicas | `sweep_results()` (analysis, section 6) |

`fig_sweep(results, y, out_dir=…)` saves its plot to `<out_dir>/sweep_<y>/`.

### Benchmark `runs/benchmark/`

Only after `benchmark()` (control, section 6). One folder per measurement, named after the number of hardware
threads filled (`16cores/`, `8cores/`), each with one short run folder per tested setting and `results.csv`
(speed, timing breakdown, projected run times). Benchmark runs are not listed in `index.csv`.

### Force-constant sweep `runs/tuning/`

Only after `alpha_sweep()` (control, section 7). One folder per sweep (default `sweep/`) with `sweep.json`,
`launcher.log` and `launcher.pid` as for a parameter sweep, and one run folder per setting, `alpha<α>_dt<Δt>ns/`
(including `analysis/` after the evaluation). `sweep_table()` writes `sweep_summary.csv`: one row per run and
pair family with α, force constant, predicted and measured overlap depth (mean, 95th percentile, max; nm),
fraction of pairs overlapping, final fraction bound and half-time of bonds. Tuning runs are not listed in
`index.csv`.
