# Output files

All simulation output is written to the data folder (default `runs/` inside the repository;
gitignored, never committed). Runs with one replica go to `single/`, runs with several replicas to
`ensemble/`, parameter sweeps to `sweeps/`. This page lists every file, what it contains and what creates it.

**Created by** means:

- **LAMMPS (automatic)**: LAMMPS always writes it.
- **LAMMPS (`template.lmp`)**: requested explicitly by a command in the input template. Everything
  in these files is computed by LAMMPS during the run; no Python or OVITO code runs during a simulation.
- **Python**: written by the `qtft` helpers (building, starting and monitoring a run, or analysis afterwards).

```
runs/
├── index.csv
├── single/                        runs with n_replicas = 1
│   └── <run name>/                same structure as below, with rep_01/ only
├── ensemble/                      runs with n_replicas > 1
│   └── <run name>/
│       ├── params.json
│       ├── launcher.log, launcher.pid
│       ├── job.slurm, replicas.txt, slurm_job.json, slurm-*.out   only with executor = "lrz"
│       ├── rep_01/ … rep_NN/      one folder per replica
│       │   ├── in.lammps, status, pid, lammps.out
│       │   ├── log.lammps
│       │   ├── traj.dump, bonds.dump, rdf.dat, coord_qt.hist, coord_ft.hist, final.data
│       │   ├── fibsem/            only after the FIB-SEM export
│       │   └── animation/         only after animate()
│       └── analysis/              only after the analysis
├── sweeps/                        only after build_sweep()
│   └── <sweep name>/
│       ├── sweep.json, launcher.log, launcher.pid, sweep_results.csv
│       ├── job.slurm, replicas.txt, slurm_job.json   only with executor = "lrz" (one job array for the whole sweep)
│       └── <label>/               one run folder per combination (structure as above)
├── benchmark/                     only after benchmark()
└── tuning/                        only after alpha_sweep() (same structure as sweeps/)
```

## Data folder `runs/`

| File | Content | Created by |
|---|---|---|
| `index.csv` | One line per single, ensemble and sweep run: group (`single`/`ensemble`/`sweeps/<name>`), name, date, status counts, key parameters, `executor` (`local` or `lrz`; added 2026-10-25, runs from before read as `local`), parameters that differ from the defaults, git commit | Python, automatic: `build_run()`, whenever a replica finishes, and `fetch()` |

The folder and its `single/` / `ensemble/` subfolders are created by `build_run()` if they do not
exist. The subfolder follows from `n_replicas`; the `group` parameter overrides it.

## Run folder `<single|ensemble>/<run name>/`

The generated run name lists the main parameters, the calcium protocol (baseline–influx–pump-out durations in s)
and the boundary (`400Qt_1600Ft_L928.3_kQQ7.5_…_dt500ns_ca10-30-90s_periodic`, `…_reflect`, `…_harmonic`).

| File | Content | Created by |
|---|---|---|
| `params.json` | **The record of the run**: all parameters (including `bond_break`, the unbinding method; a file from before 2026-10-24 has none and is read as `"longest"`), derived values (k_BT, γ, α, p, …), check results, seeds of every replica, git commit, creation time | Python: `build_run()` (control notebook, section 2) |
| `launcher.log` | Log of the background launcher: header (replicas, parallel slots, command); start of every replica; end or failure with wall time and count done; every minute a summary (elapsed, done/running/queued, estimated end) and one line per running replica (phase, step, %, speed, bonds); final summary. Follow it live with `tail -f launcher.log` | Python: `start()` (section 3) |
| `launcher.pid` | Process ID of the launcher, used by `stop()` | Python: `start()` (background runs only) |
| `job.slurm` | Only with `executor = "lrz"`: Slurm job script for LRZ, one array task per replica (cluster/partition/QOS, `mpi_procs` tasks, `slurm_mem`, `slurm_time`, email to the address in the gitignored `lrz_local.json`, none without it; runs LAMMPS with `srun` and writes `status` and `pid` like the local launcher). Submit from the run folder on LRZ: `sbatch job.slurm` | Python: `build_run()` / `build_sweep()`, from `job_template.slurm` |
| `replicas.txt` | Only with `executor = "lrz"`: the replica folders, one per line (relative; array task i runs line i) | Python: `build_run()` / `build_sweep()` |
| `slurm_job.json` | Only with `executor = "lrz"`, after `start()`: Slurm job ID, cluster, folder on LRZ (relative to the LRZ home, `qtft-lammps/runs/<group>/<name>`) and submission time; used by `progress`, `watch`, `stop`, `fetch`. A run of a sweep started as a whole uses the sweep's file | Python: `start()` (`remote.submit`) |
| `slurm-<job>_<task>.out` | Output of the job script of one array task (LAMMPS writes to `rep_XX/lammps.out`); fetched from LRZ with the status files | Slurm on LRZ |

## Replica folder `rep_XX/`

### Control files

| File | Content | Created by |
|---|---|---|
| `in.lammps` | Generated LAMMPS input of this replica, all numbers filled in. Can be rerun by hand: `lmp -in in.lammps` | Python: `build_run()` |
| `status` | One word: `built` → `queued` → `running` → `done` / `failed` / `stopped`. Its last change is the end time of the replica. LRZ: written there by `job.slurm` and fetched by `progress`/`watch`; a task that left Slurm without an end status (time limit, node failure) is set to `failed` (cancelled: `stopped`) | Python: updated at each stage; LRZ: `job.slurm` |
| `pid` | Process ID of the running LAMMPS process, used by `stop()` (LRZ: the Slurm job ID of the replica's array task, not the ID of the array); written when the replica starts, so its time is the start time | Python: when the replica starts; LRZ: `job.slurm` |
| `lammps.out` | Terminal output of LAMMPS/MPI (screen output is switched off, so usually almost empty). Check here first if a run fails | Python: output of the started process |

### LAMMPS output

| File | Content | Created by |
|---|---|---|
| `log.lammps` | Full log: copy of all input commands, setup information, timing summaries and the **thermo table every `output_every` steps** in YAML format (see below). Main source of all time series | LAMMPS (automatic; name set by `-log log.lammps`) |
| `traj.dump` | Trajectory, one frame every `traj_every` steps. Columns: `id type x y z ix iy iz radius cluster nbonds` (`type` = type label `Qt` or `Ft`, shown as named types in OVITO; runs before the type labels have `1`/`2`; `ix iy iz` = periodic image flags, always 0 with walls, where BOX BOUNDS reads `ff ff ff`; `cluster` = bond-connected cluster ID; `nbonds` = bonds of the particle). **Open in OVITO** | LAMMPS (`template.lmp`): `dump traj … custom` with `typelabel` (names from `labelmap`) |
| `bonds.dump` | List of all bonds (`atom1 atom2`, atom IDs) every `traj_every` steps (the `traj.dump` steps) | LAMMPS (`template.lmp`): `compute property/local` + `dump bonds … local` |
| `rdf.dat` | Qt–Ft radial distribution function every `traj_every` steps: per bin r, g(r) and coordination number (default 200 bins up to 121.5 nm); snapshots, not time averages | LAMMPS (`template.lmp`): `compute rdf` + `fix ave/time` |
| `coord_qt.hist` | Histogram of bonds per Qt (0 … `maxbond_qt`) every `traj_every` steps | LAMMPS (`template.lmp`): `fix ave/histo` on `compute nbond/atom` |
| `coord_ft.hist` | Histogram of bonds per Ft (0 … `maxbond_ft`) every `traj_every` steps | LAMMPS (`template.lmp`): `fix ave/histo` on `compute nbond/atom` |
| `final.data` | Final state: box, positions, types with their labels (`Atom Type Labels`: 1 Qt, 2 Ft), **all bonds**, coefficients. Opens in OVITO with bonds; can be used to continue a run (`read_data`) | LAMMPS (`template.lmp`): `write_data` at the end |
| `log.cite` | Literature references for the LAMMPS features used (only written by some LAMMPS versions) | LAMMPS (automatic) |

Notes:

- `bonds.dump`, `rdf.dat` and the histograms are written at the `traj.dump` steps, where the analysis reads
  them (structure, percolation, FIB-SEM, contact pairs at the influx start; final RDF and histograms). Only
  the thermo table in `log.lammps` has the finer `output_every` resolution. `rdf.dat` is a convenience: the
  RDF can also be recomputed from `traj.dump` (OVITO *Coordination analysis*, LAMMPS `rerun`, or Python).
- Windows Explorer may hide the `.dat` extension and show `rdf.dat` as "rdf" with a program icon. It is
  a plain text file.

### Thermo table in `log.lammps`

Written every `output_every` steps: one YAML block for the equilibration and one per production segment
(`rate_update_every` steps; the step counter restarts at 0 for production, and every segment repeats the last step
of the previous one). Read with `qtft.load_thermo(rep_dir)`, which joins the production blocks and drops the
repeated steps.

| Column | Meaning |
|---|---|
| `Step` | time step (production: time = Step × Δt) |
| `PotEng`, `E_vdwl`, `E_bond` | potential energy: total, pair repulsion, bonds (zJ) |
| `c_pvir` | virial pressure from pair and bond forces only (no kinetic term, no wall forces; 1 unit = 1 MPa) |
| `Bonds` | number of bonds |
| `v_n_qt_bound`, `v_n_ft_bound` | Qt / Ft with at least one bond |
| `v_frac_bound` | fraction of all particles that are bound |
| `v_n_clusters` | number of clusters including free particles |
| `v_mean_size`, `v_largest` | mean and largest cluster size (particles) |
| `v_mean_size_qt`, `v_mean_size_ft` | Qt (Ft) per cluster: number of Qt (Ft) / number of clusters that contain at least one Qt (Ft); a free particle is a cluster of 1 |
| `v_largest_qt`, `v_largest_ft` | highest number of Qt (Ft) in one cluster (not necessarily the overall largest cluster) |
| `v_n_large` | number of clusters with ≥ `large_cluster_min` particles |
| `v_ca`, `v_f` | calcium (µM) and calmodulin loading f (production only) |
| `v_k_on_s`, `v_k_off_s` | rate constants: binding per Qt–Ft pair in contact and unbinding per bond (1/s; production only) |
| `v_n_broken` | bonds broken since the start of the production (production only) |
| `v_f_mono` … `v_f_vlarge` | fraction of particles in monomers / small / medium / large / very large clusters |
| `v_deg_qt_mean`, `v_deg_ft_mean`, `v_deg_qt_max`, `v_deg_ft_max` | mean (over bound particles) and maximum bonds per Qt / Ft |
| `v_bond_mean`, `v_bond_max` | mean and maximum bond length (nm) |
| `S/CPU`, `CPULeft` | speed (steps per second) and LAMMPS's estimated time left (s) of the current `run` command, i.e. of one production segment (`progress()` computes the time left of the whole phase from the remaining steps instead) |

Kinetic energy and temperature are not logged: they are meaningless in overdamped Brownian dynamics.

## FIB-SEM export `rep_XX/fibsem/`

Only after `fibsem_export_run(run_dir)` (analysis notebook, section 7).

| File | Content | Created by |
|---|---|---|
| `sim_encapsulin_centroids_clustered.csv` | One row per encapsulin: `label, z_nm, y_nm, x_nm, z_vox, y_vox, x_vox, radius_nm, volume_nm3, cluster, is_clustered` (periodic box: clusters unwrapped across the box, shifted to the origin; walls: positions in the box frame, not shifted, walls at 0 and the box size) | Python + OVITO: `fibsem_export()` |
| `sim_metadata.json` | Counts, cluster-size histogram, box and applied offset, frame description, bond check, all parameters; `boundary` (`periodic`/`reflect`/`harmonic`), `wall_stiffness` (harmonic: k of Qt and Ft in zJ/nm², otherwise null), `wall_leaks` (walls: encapsulin spheres sticking out of the box and the largest depth, otherwise null) | Python: `fibsem_export()` |
| `check_xy.png` | x–y scatter of the encapsulins coloured by cluster, with the periodic box or the walls as dashed outline | Python: `fibsem_export()` |

## Animation `rep_XX/animation/`

Only after `animate(run)` (animation notebook).

| File | Content | Created by |
|---|---|---|
| `animation.mp4` | Video, 1920 × 1080 × `scale` (default 2: 3840 × 2160), H.264 (`yuv420p`, CRF 14), one frame per `traj.dump` frame (every `every`-th) at `fps` frames per second: header (title, calcium phase, production time, calcium in µM), the whole box in 3D and from the front with the magnified region outlined, the fixed central magnification, calcium (µM) and calmodulin loading f over time, the average cluster size (all, Qt, Ft: thermo `mean_size`, `mean_size_qt`, `mean_size_ft`), both graphs with the current time, legend, statistics of the frame (bonds, Qt in the cluster with the most Qt, % of Qt in clusters with ≥ 2 Qt), timeline of the protocol. Written as `animation.partial.mp4` and renamed when complete | Python: `animate()` (needs ffmpeg) |
| `still_<t>s.png` | Optional: the frame nearest to production time t [s] as PNG (same layout and size), for every t in `stills`; `video=False` writes only these | Python: `animate()` |

## Analysis folder `<run name>/analysis/`

Only after `ensemble_stats()` or `make_report()` (analysis notebook). Everything here can be
regenerated from the replica files at any time.

Renamed on 2026-10-29 (old files were deleted; they are regenerated under the new names):
`large_agglomerates*.csv` → `agglomerates_above_size*.csv`, `clusters_influx_end.csv` → `agglomerates_influx_end.csv` (column `cluster` → `agglomerate`), figures `clusters` → `agglomerate_distributions`, `large_agglomerates_by_species` → `agglomerates_above_size`, `large_agglomerates(_density)` → `agglomerates_above_size_total(_density)`. Renamed on 2026-10-30:
`final_table.csv/.tex` → `summary_table.csv/.tex` (same content); single panels
`overview/` and `kinetics/` `mean_agglomerate_size.*` → `mean_cluster_size.*` (thermo `mean_size` counts free
particles as clusters of one).

| File | Content | Created by |
|---|---|---|
| `ensemble_stats.json` | Mean, SD (sample SD, ddof = 1) and per-replica traces of all logged quantities on a common time grid; half-times, percolation (`percolated`: a cluster spans the box in the last frame, `t_percolation_ms`: first frame with a spanning cluster), final values per replica | Python: `ensemble_stats()` |
| `percolation_rep_XX.csv` | One per replica, per `traj.dump` frame: `Step, time_ms, axes` = the axes along which one cluster spans the box (`""` = none, e.g. `xz`); written once the replica is done and reused afterwards (delete it to recompute) | Python: `structure.percolation()`, called by `ensemble_stats()` |
| `ensemble_timeseries.csv` | The same time series as a table (`<quantity>_mean`, `<quantity>_sd`) | Python: `ensemble_stats()` |
| `final_state.csv` | Final values per quantity (at the end of the calcium protocol): mean, SD, number of replicas (row `percolated`: mean = fraction of percolated replicas); rows `bonds_peak`, `t_bonds_peak_ms`, `largest_peak`, `t_largest_peak_ms`, `t_half_unbind_bonds_ms`, `t_half_unbind_largest_ms` (deagglomeration half-times after the end of the influx) | Python: `ensemble_stats()` |
| `structure_timeseries.csv` | Per analysed frame, mean/SD over replicas: Rg, Rg/Rg_ideal, inter- and intra-cluster nearest-neighbour distances, Qt fraction per cluster | Python + OVITO: `structure_run()` |
| `structure_per_replica.csv` | The same per replica | Python + OVITO: `structure_run()` |
| `agglomerates_influx_end.csv` | Every agglomerate (≥ 3 particles) at the end of the influx (last `traj.dump` frame at or before it; run end without an influx) of every replica: `replica, step, time_s, agglomerate` (1 = largest), `size, n_qt, n_ft, n_bonds, spans` (axes the agglomerate spans the box along, `""` = none), `v_solid_um3` (union of the particle spheres), `v_envelope_um3` (closing with the ellipsoidal probe), `probe_xy_nm, probe_z_nm, voxel_nm`; volumes NaN for an agglomerate that wraps around the periodic box | Python: `structure.agglomerate_volumes_run()`, called by `make_report()` |
| `qt_nn_distances.csv` | Qt nearest-neighbour distances (FIB-SEM comparison, Qt only) in the first `traj.dump` frame and at the end of the influx (same frame as `agglomerates_influx_end.csv`) of every replica: `replica, frame` (`first` / `influx_end`), `step, time_s, set, distance_nm`; `set` = `all` (every Qt to the nearest other Qt, both frames) or `intra` (every Qt in an agglomerate with ≥ 2 Qt to the nearest Qt of the same agglomerate, end of the influx only); minimum image in a periodic box | Python: `spatial.qt_spatial_run()`, called by `make_report()`; reused while newer than every `traj.dump` |
| `qt_pair_correlation.csv` | Qt–Qt pair correlation g(r) of all Qt at the end of the influx, normalised to their density in the full box: `r_nm` (bin centre, 5 nm bins up to min(L/2, 1000 nm)), `g_<replica>`, `g_mean`; with walls no edge correction | Python: `spatial.qt_spatial_run()` |
| `qt_clark_evans.csv` | Clark–Evans index of all Qt per replica and frame (`first`, `influx_end`): `replica, frame, step, time_s, n_qt, density_per_um3, mean_nn_nm, expected_nn_nm` (random points of the same density, Γ(4/3)(3/(4πρ))^(1/3) = 0.554 ρ^(−1/3)), `R` = mean / expected (1 random, < 1 clustered) | Python: `spatial.qt_spatial_run()` |
| `qt_ripley_l.csv` | Ripley's K of all Qt as Besag's L(r) − r, L = (3K/(4π))^(1/3), per replica and frame (`first`, `influx_end`), long format: `replica, frame, r_nm` (5 nm steps up to min(L/2, 1000 nm)), `L_minus_r` [nm] (0 random, > 0 clustered at scale r, < 0 more regular); K = V/(N − 1) · mean number of other Qt within r, V = full box; with walls border-corrected (only Qt at least r from every wall as centres) | Python: `spatial.qt_spatial_run()` |
| `clusters_final.csv` | Every cluster of the last frame of every replica: size, Qt/Ft count, bonds, Rg, ideal Rg (solid sphere of the cluster's particle volumes at packing 0.64), centre | Python + OVITO: `structure_run()` |
| `overlap_timeseries.csv` | Interpenetration statistics per pair family and frame (mean over replicas); with walls also the rows `Qt-wall` / `Ft-wall` (spheres beyond a box face) | Python: `overlap_run()` |
| `overlap_final.csv` | The same, pooled over the last frames | Python: `overlap_run()` |
| `binding_rate.csv` | Columns `value, unit, 1/(M s)`. Rows: `k_a measured` (+ `SD`) from the start of the bond curve in the calcium influx (frames up to 10 % more bound, mean and SD over replicas); `k_a predicted` (+ `SD`), Collins–Kimball with the Qt–Ft contact pairs counted in the frame at the influx start; `kappa`, `k_D`; `k_a continuum estimate` (analytical, too high when the step is comparable to the overlap layer); `contact volume counted` and `contact volume continuum` in nm³ | Python: `binding_rate()`, also `make_report()` (only with a calcium step, `ca_rise_s = 0`) |
| `bond_rates.csv` | Columns `time_ms` (bin centre), `formed_per_s`, `broken_per_s` (bonds per second), `k_a_per_M_s` (apparent association rate constant, formed per second · V / (N_Qt N_Ft), 1/(M s)), each as `_mean` and `_sd` over replicas; one row per calcium segment (`rate_update_every`) or per `bin_s` | Python: `bond_rates()`, also `make_report()` (default bins) |
| `agglomerates_above_size.csv` | Number of agglomerates with ≥ n particles over time, counted in every `traj.dump` frame (resolution `traj_every`; a free particle is a cluster of 1), one row per frame and threshold: `step, time_s, n_min, replicas`, `count_mean`, `count_sd` (in the box), `per_1000um3_mean`, `per_1000um3_sd` (density: count · 1000 / V_box in µm³, V_box = Lx·Ly·Lz, also with walls), and `count_<replica>` per replica. Reused while it holds the requested thresholds and the same replicas and is newer than the dumps; new thresholds are added | Python: `agglomerates_above_size()`, also `make_report()` via `fig_agglomerates_above_size()` and `compare_report()` via `fig_agglomerates_above_size_total()` (default n = 5, 20, 60, 80, 100, 150) |
| `agglomerates_above_size_qt.csv`, `agglomerates_above_size_ft.csv` | The same table (same columns, same caching) with the size of an agglomerate counted in Qt or in Ft only: number of agglomerates with ≥ n Qt / ≥ n Ft over time. The agglomerates are the same as in `agglomerates_above_size.csv` (clusters from bonds); computed in the same pass over `traj.dump` as the total; also holds the thresholds of the pair `n_qt_ft` | Python: `agglomerates_above_size(…, species="qt" / "ft")`, `make_report()` via `fig_agglomerates_above_size()` (default n = 5, 10, 20, 50, 70 Qt and 5, 20, 60, 80, 100, 150 Ft; thresholds set with `make_report(…, n_min, n_min_qt, n_min_ft)`) |
| `summary_table.csv`, `summary_table.tex` | Final-state table (bonds, clusters, sizes, fraction bound, half-time, Rg, Qt fraction) as CSV and LaTeX | Python: `make_report()` |
| `bonds_timeseries.csv` | Bond count over time: step, time (µs, ms), mean, SD | Python: `make_report()` |
| `report.json` | Settings used for the report | Python: `make_report()` |
| `figures/` | All figures as PDF + SVG + PNG: `calcium` (3 × 2: calcium and loading f; rate constants k_on per contact pair and k_off per bond; binding and unbinding rates in bonds/s; apparent k_a on a log scale; bonds and broken bonds; fraction bound), `overview` (its coordination histogram at the end of the influx: panel `coordination_at_end_of_influx`), `agglomerate_distributions` (1 × 3 at the end of the influx: agglomerate size, solid volume, envelope volume; log bins, agglomerates per replica), `kinetics`, `agglomerates_above_size` (4 × 2: agglomerates with ≥ n particles, ≥ n Qt and ≥ n Ft (rows 1–3) over time, in the box and per 1000 µm³ (columns), one line per threshold; row 4: one pair ≥ n Qt (green) and ≥ n Ft (red) together, default 15 / 20 (`make_report(…, n_min, n_min_qt, n_min_ft, n_qt_ft)`), from `agglomerates_above_size.csv`, `_qt.csv` and `_ft.csv`; `fig_agglomerates_above_size_total()` draws the total panel alone, also for several runs), `fibsem_comparison` (2 × 2, Qt only: distances to the nearest Qt in the same agglomerate and Qt–Qt g(r) of all Qt at the end of the influx, each with its peak "r = … nm"; cumulative nearest-neighbour distribution of all Qt in the first frame and at the end of the influx with the random-point curve and the Clark–Evans R; Ripley's L(r) − r of all Qt in both frames with its maximum; from the `qt_*.csv` files), `overlap`, `binding_rate` (bonds formed since the influx start with the fitted k_a and the Collins–Kimball prediction from the counted contact pairs; only with an influx and a calcium step, `ca_rise_s = 0`); every figure with several panels (`calcium`, `overview`, `agglomerate_distributions`, `kinetics`, `agglomerates_above_size`, `fibsem_comparison`) also has each panel alone (89 mm) in `figures/<figure>/<panel>.*`. Largest and mean cluster size (in `overview`) are drawn for all particles, Qt only and Ft only (one run). `fig_bond_rates()` draws the two rate panels alone, e.g. with coarser bins (`bond_rates(run_dir, bin_s=…)`); `make_report()` doesn't save it. Dotted vertical lines mark the start and end of the calcium influx in every time panel. Time axes in ms, in s for runs of ≥ 10 s (also in `quicklook`) | Python: `make_report()` |

Comparisons of several runs (`compare_report(runs, out_dir)`) are written to the chosen `out_dir`:
`comparison`, `kinetics`, `agglomerates_above_size_total_density` (per 1000 µm³, so runs with different boxes compare; colour per run, line style per threshold, at most 6) (PDF/SVG/PNG), the single panels in `comparison/` and `kinetics/`
(cluster sizes for all particles only; phase lines only if all runs have the same phase times), and
`comparison_table.csv` / `.tex`.

## Benchmark `runs/benchmark/`

Only after `benchmark()` (control notebook, section 6). One folder per measurement, named after the
number of hardware threads filled (`16cores/`, `8cores/`), each with one short run folder per tested
setting (same structure as above) and `results.csv` with speed, timing breakdown and projected run
times. Benchmark runs are not listed in `index.csv`.

## Parameter sweep `runs/sweeps/<sweep name>/`

Only after `build_sweep()` (control notebook, section 4). One run folder per combination of the listed values,
named after the swept values only (e.g. `temperature300_k_on2e-06/`), with the same structure as a run folder
above, except that the launcher files belong to the whole sweep.

| File | Content | Created by |
|---|---|---|
| `sweep.json` | **The record of the sweep**: name, mode (`grid`), the listed values per parameter, base parameters, label and parameter values of every run, git commit, creation time | Python: `build_sweep()` |
| `launcher.log`, `launcher.pid` | As for a run, for all replicas of the sweep (one queue); replicas are named `<label>/rep_XX` | Python: `start(sweep)` |
| `sweep_results.csv` | One row per run: swept parameters, number of finished replicas, mean and SD over them of final fraction bound, bonds, largest cluster fraction, mean cluster size, half-times (bonds, Qt, Ft bound; ms), percolation time and number of percolated replicas | Python: `sweep_results()` (analysis notebook, section 6) |

## Force-constant sweep `runs/tuning/`

Only after `alpha_sweep()` (control notebook, section 7). One folder per sweep (default `sweep/`) with
`sweep.json`, `launcher.log` and `launcher.pid` as for a parameter sweep, one run folder per setting, named
`alpha<α>_dt<Δt>ns/` (same structure as above, including `analysis/` after the evaluation), and `sweep_summary.csv` from `sweep_table()`: one row per run and pair family with α, force
constant, predicted and measured overlap depth (mean, 95th percentile, max; nm), fraction of pairs
overlapping, final fraction bound and half-time of bonds. Sweep runs are not listed in `index.csv`.
