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

The generated run name lists the main parameters and ends in the boundary
(`400Qt_1600Ft_L928.3_kQQ7.5_…_dt500ns_25ms_periodic`, `…_reflect`, `…_harmonic`).

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
| `bonds.dump` | List of all bonds (`atom1 atom2`, atom IDs) every `output_every` steps | LAMMPS (`template.lmp`): `compute property/local` + `dump bonds … local` |
| `rdf.dat` | Qt–Ft radial distribution function every `output_every` steps: per bin r, g(r) and coordination number (default 200 bins up to 121.5 nm); snapshots, not time averages | LAMMPS (`template.lmp`): `compute rdf` + `fix ave/time` |
| `coord_qt.hist` | Histogram of bonds per Qt (0 … `maxbond_qt`) every `output_every` steps | LAMMPS (`template.lmp`): `fix ave/histo` on `compute nbond/atom` |
| `coord_ft.hist` | Histogram of bonds per Ft (0 … `maxbond_ft`) every `output_every` steps | LAMMPS (`template.lmp`): `fix ave/histo` on `compute nbond/atom` |
| `final.data` | Final state: box, positions, types with their labels (`Atom Type Labels`: 1 Qt, 2 Ft), **all bonds**, coefficients. Opens in OVITO with bonds; can be used to continue a run (`read_data`) | LAMMPS (`template.lmp`): `write_data` at the end |
| `log.cite` | Literature references for the LAMMPS features used (only written by some LAMMPS versions) | LAMMPS (automatic) |

Notes:

- `rdf.dat` and the trajectory are redundant when `traj_every = output_every`: the RDF can be recomputed
  from `traj.dump` (OVITO *Coordination analysis*, LAMMPS `rerun`, or Python). `rdf.dat` keeps the full
  time resolution if the trajectory is written less often.
- Windows Explorer may hide the `.dat` extension and show `rdf.dat` as "rdf" with a program icon. It is
  a plain text file.

### Thermo table in `log.lammps`

Written every `output_every` steps, once for the equilibration run and once for the production run
(two YAML blocks; the step counter restarts at 0 for production). Read with `qtft.load_thermo(rep_dir)`.

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
| `v_n_large` | number of clusters with ≥ `large_cluster_min` particles |
| `v_f_mono` … `v_f_vlarge` | fraction of particles in monomers / small / medium / large / very large clusters |
| `v_deg_qt_mean`, `v_deg_ft_mean`, `v_deg_qt_max`, `v_deg_ft_max` | mean (over bound particles) and maximum bonds per Qt / Ft |
| `v_bond_mean`, `v_bond_max` | mean and maximum bond length (nm) |
| `S/CPU`, `CPULeft` | speed (steps per second) and estimated time left (s) |

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
| `final_state.csv` | Final values per quantity: mean, SD, number of replicas (row `percolated`: mean = fraction of percolated replicas) | Python: `ensemble_stats()` |
| `structure_timeseries.csv` | Per analysed frame, mean/SD over replicas: Rg, Rg/Rg_ideal, inter- and intra-cluster nearest-neighbour distances, Qt fraction per cluster | Python + OVITO: `structure_ensemble()` |
| `structure_per_replica.csv` | The same per replica | Python + OVITO: `structure_ensemble()` |
| `clusters_final.csv` | Every cluster of the last frame of every replica: size, Qt/Ft count, bonds, Rg, ideal Rg (solid sphere of the cluster's particle volumes at packing 0.64), centre | Python + OVITO: `structure_ensemble()` |
| `overlap_timeseries.csv` | Interpenetration statistics per pair family and frame (mean over replicas); with walls also the rows `Qt-wall` / `Ft-wall` (spheres beyond a box face) | Python: `overlap_ensemble()` |
| `overlap_final.csv` | The same, pooled over the last frames | Python: `overlap_ensemble()` |
| `binding_rate.csv` | Association rate constant k_a in nm³/ns and M⁻¹ s⁻¹: measured from the start of the bond curve (mean and SD over replicas, frames up to 10 % bound) and the Collins–Kimball prediction (k_a, κ, k_D) | Python: `binding_rate()`, also `make_report()` |
| `final_table.csv`, `final_table.tex` | Final-state table (bonds, clusters, sizes, fraction bound, half-time, Rg, Qt fraction) as CSV and LaTeX | Python: `make_report()` |
| `bonds_timeseries.csv` | Bond count over time: step, time (µs, ms), mean, SD | Python: `make_report()` |
| `report.json` | Settings used for the report | Python: `make_report()` |
| `figures/` | All figures as PDF + SVG + PNG: `overview`, `kinetics`, `large_agglomerates`, `overlap`, `binding_rate` (start of the bond curve with the fitted k_a and the Collins–Kimball line); `figures/panels/` single panels of the overview; `figures/journal/` panels g–j | Python: `make_report()` |

Comparisons of several runs (`compare_report(runs, out_dir)`) are written to the chosen `out_dir`:
`comparison`, `kinetics`, `large_agglomerates` (PDF/SVG/PNG), `panels/`, and
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
