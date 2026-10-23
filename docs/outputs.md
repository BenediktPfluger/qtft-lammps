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
│       ├── rep_01/ … rep_NN/      one folder per replica
│       │   ├── in.lammps, status, pid, lammps.out
│       │   ├── log.lammps
│       │   ├── traj.dump, bonds.dump, rdf.dat, coord_qt.hist, coord_ft.hist, final.data
│       │   ├── analysis/          percolation.csv, after the first ensemble_stats()
│       │   └── fibsem/            only after the FIB-SEM export
│       └── analysis/              only after the analysis
├── sweeps/                        only after build_sweep()
│   └── <sweep name>/
│       ├── sweep.json, launcher.log, launcher.pid, sweep_results.csv
│       └── <label>/               one run folder per combination (structure as above)
├── benchmark/                     only after benchmark()
└── tuning/                        only after alpha_sweep() (same structure as sweeps/)
```

## Data folder `runs/`

| File | Content | Created by |
|---|---|---|
| `index.csv` | One line per single, ensemble and sweep run: group (`single`/`ensemble`/`sweeps/<name>`), name, date, status counts, key parameters, parameters that differ from the defaults, git commit | Python, automatic: `build_run()` and whenever a replica finishes |

The folder and its `single/` / `ensemble/` subfolders are created by `build_run()` if they do not
exist. The subfolder follows from `n_replicas`; the `group` parameter overrides it.

## Run folder `<single|ensemble>/<run name>/`

The generated run name lists the main parameters, the calcium protocol (baseline–influx–pump-out durations in s)
and the boundary (`400Qt_1600Ft_L928.3_kQQ7.5_…_dt500ns_ca10-30-90s_periodic`, `…_reflect`, `…_harmonic`).

| File | Content | Created by |
|---|---|---|
| `params.json` | **The record of the run**: all parameters, derived values (k_BT, γ, α, p, …), check results, seeds of every replica, git commit, creation time | Python: `build_run()` (control notebook, section 2) |
| `launcher.log` | Log of the background launcher: header (replicas, parallel slots, command); start of every replica; end or failure with wall time and count done; every minute a summary (elapsed, done/running/queued, estimated end) and one line per running replica (phase, step, %, speed, bonds); final summary. Follow it live with `tail -f launcher.log` | Python: `start()` (section 3) |
| `launcher.pid` | Process ID of the launcher, used by `stop()` | Python: `start()` (background runs only) |

## Replica folder `rep_XX/`

### Control files

| File | Content | Created by |
|---|---|---|
| `in.lammps` | Generated LAMMPS input of this replica, all numbers filled in. Can be rerun by hand: `lmp -in in.lammps` | Python: `build_run()` |
| `status` | One word: `built` → `queued` → `running` → `done` / `failed` / `stopped`. Its last change is the end time of the replica | Python: updated at each stage |
| `pid` | Process ID of the running LAMMPS process, used by `stop()`; written when the replica starts, so its time is the start time | Python: when the replica starts |
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
| `v_k_on_s`, `v_k_off_s` | binding rate per Qt–Ft pair in contact and unbinding rate per bond (1/s; production only) |
| `v_n_broken` | bonds broken since the start of the production (production only) |
| `v_f_mono` … `v_f_vlarge` | fraction of particles in monomers / small / medium / large / very large clusters |
| `v_deg_qt_mean`, `v_deg_ft_mean`, `v_deg_qt_max`, `v_deg_ft_max` | mean (over bound particles) and maximum bonds per Qt / Ft |
| `v_bond_mean`, `v_bond_max` | mean and maximum bond length (nm) |
| `S/CPU`, `CPULeft` | speed (steps per second) and LAMMPS's estimated time left (s) of the current `run` command, i.e. of one production segment (`progress()` computes the time left of the whole phase from the remaining steps instead) |

Kinetic energy and temperature are not logged: they are meaningless in overdamped Brownian dynamics.

## FIB-SEM export `rep_XX/fibsem/`

Only after `fibsem_export_run(run_dir)` (analysis notebook, section 5).

| File | Content | Created by |
|---|---|---|
| `sim_encapsulin_centroids_clustered.csv` | One row per encapsulin: `label, z_nm, y_nm, x_nm, z_vox, y_vox, x_vox, radius_nm, volume_nm3, cluster, is_clustered` (periodic box: clusters unwrapped across the box, shifted to the origin; walls: positions in the box frame, not shifted, walls at 0 and the box size) | Python + OVITO: `fibsem_export()` |
| `sim_metadata.json` | Counts, cluster-size histogram, box and applied offset, frame description, bond check, all parameters; `boundary` (`periodic`/`reflect`/`harmonic`), `wall_stiffness` (harmonic: k of Qt and Ft in zJ/nm², otherwise null), `wall_leaks` (walls: encapsulin spheres sticking out of the box and the largest depth, otherwise null) | Python: `fibsem_export()` |
| `check_xy.png` | x–y scatter of the encapsulins coloured by cluster, with the periodic box or the walls as dashed outline | Python: `fibsem_export()` |

## Replica analysis `rep_XX/analysis/`

| File | Content | Created by |
|---|---|---|
| `percolation.csv` | Per `traj.dump` frame: `Step, time_ms, axes` = the axes along which one cluster spans the box (`""` = none, e.g. `xz`); written once the replica is done and reused afterwards (delete it to recompute) | Python: `structure.percolation()`, called by `ensemble_stats()` |

## Analysis folder `<run name>/analysis/`

Only after `ensemble_stats()` or `make_report()` (analysis notebook). Everything here can be
regenerated from the replica files at any time.

| File | Content | Created by |
|---|---|---|
| `ensemble_stats.json` | Mean, SD (sample SD, ddof = 1) and per-replica traces of all logged quantities on a common time grid; half-times, percolation (`percolated`: a cluster spans the box in the last frame, `t_percolation_ms`: first frame with a spanning cluster), final values per replica | Python: `ensemble_stats()` |
| `ensemble_timeseries.csv` | The same time series as a table (`<quantity>_mean`, `<quantity>_sd`) | Python: `ensemble_stats()` |
| `final_state.csv` | Final values per quantity (at the end of the calcium protocol): mean, SD, number of replicas (row `percolated`: mean = fraction of percolated replicas); rows `bonds_peak`, `t_bonds_peak_ms`, `largest_peak`, `t_largest_peak_ms`, `t_half_unbind_bonds_ms`, `t_half_unbind_largest_ms` (deagglomeration half-times after the end of the influx) | Python: `ensemble_stats()` |
| `structure_timeseries.csv` | Per analysed frame, mean/SD over replicas: Rg, Rg/Rg_ideal, inter- and intra-cluster nearest-neighbour distances, Qt fraction per cluster | Python + OVITO: `structure_ensemble()` |
| `structure_per_replica.csv` | The same per replica | Python + OVITO: `structure_ensemble()` |
| `clusters_final.csv` | Every cluster of the last frame of every replica: size, Qt/Ft count, bonds, Rg, ideal Rg (solid sphere of the cluster's particle volumes at packing 0.64), centre | Python + OVITO: `structure_ensemble()` |
| `overlap_timeseries.csv` | Interpenetration statistics per pair family and frame (mean over replicas); with walls also the rows `Qt-wall` / `Ft-wall` (spheres beyond a box face) | Python: `overlap_ensemble()` |
| `overlap_final.csv` | The same, pooled over the last frames | Python: `overlap_ensemble()` |
| `binding_rate.csv` | Columns `value, unit, 1/(M s)`. Rows: `k_a measured` (+ `SD`) from the start of the bond curve in the calcium influx (frames up to 10 % more bound, mean and SD over replicas); `k_a predicted` (+ `SD`), Collins–Kimball with the Qt–Ft contact pairs counted in the frame at the influx start; `kappa`, `k_D`; `k_a continuum estimate` (analytical, too high when the step is comparable to the overlap layer); `contact volume counted` and `contact volume continuum` in nm³ | Python: `binding_rate()`, also `make_report()` (only with a calcium step, `ca_rise_s = 0`) |
| `bond_rates.csv` | Columns `time_ms` (bin centre), `formed_per_s`, `broken_per_s` (bonds per second), `k_a_per_M_s` (apparent association rate constant, formed per second · V / (N_Qt N_Ft), 1/(M s)), each as `_mean` and `_sd` over replicas; one row per calcium segment (`rate_update_every`) or per `bin_s` | Python: `bond_rates()`, also `make_report()` (default bins) |
| `final_table.csv`, `final_table.tex` | Final-state table (bonds, clusters, sizes, fraction bound, half-time, Rg, Qt fraction) as CSV and LaTeX | Python: `make_report()` |
| `bonds_timeseries.csv` | Bond count over time: step, time (µs, ms), mean, SD | Python: `make_report()` |
| `report.json` | Settings used for the report | Python: `make_report()` |
| `figures/` | All figures as PDF + SVG + PNG: `calcium` (calcium and loading f, rates, bonds and broken bonds, fraction bound, largest cluster, mean cluster size), `bond_rates` (bonds formed and broken per second, apparent k_a on a log scale; any calcium protocol), `overview`, `kinetics`, `large_agglomerates`, `overlap`, `binding_rate` (bonds formed since the influx start with the fitted k_a and the Collins–Kimball prediction from the counted contact pairs; only with an influx and a calcium step, `ca_rise_s = 0`); every figure with several panels (`calcium`, `bond_rates`, `overview`, `kinetics`) also has each panel alone (89 mm) in `figures/<figure>/<panel>.*`. Largest and mean cluster size are drawn for all particles, Qt only and Ft only (one run). Dotted vertical lines mark the start and end of the calcium influx in every time panel. Time axes in ms, in s for runs of ≥ 10 s (also in `quicklook`) | Python: `make_report()` |

Comparisons of several runs (`compare_report(runs, out_dir)`) are written to the chosen `out_dir`:
`comparison`, `kinetics`, `large_agglomerates` (PDF/SVG/PNG), the single panels in `comparison/` and `kinetics/`
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
