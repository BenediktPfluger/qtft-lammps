# Model and project decisions

Record of how the Qt–Ft agglomeration model is built in LAMMPS and why, plus the project
conventions. Read this first when continuing the work.

## Physical model

| Topic | Decision | Reason |
|---|---|---|
| Units | `units nano`: nm, ns, energy zJ (= ag nm²/ns²), force pN, mass ag, pressure 1 unit = 1 MPa | parameters appear as physical values without a conversion layer; k_BT(310 K) = 4.28 zJ |
| Boundaries | periodic box (`boundary p p p`), cubic or rectangular: `box_length` is one edge (cube) or `[Lx, Ly, Lz]` | no walls, no lost atoms; clusters crossing the box edge are unwrapped in the analysis (OVITO); one field for both shapes keeps old records and run names valid (run name `L2000` or `L2000x3000x4000`); concentrations use the true volume Lx·Ly·Lz; warning if the shortest edge is below twice the RDF cutoff; in elongated boxes large networks span the shortest edge first, so that edge limits the largest structures to trust |
| Walls | alternative to the periodic box: `boundary = "periodic"` (default), `"reflect"` or `"harmonic"`, walls on all six faces of the same box (`boundary f f f`); **reflect**: `fix wall/reflect` per species at r_i from each face (a centre that crosses the plane is mirrored back, so the sphere surface touches the face; exact no-flux boundary, no parameters); **harmonic**: `fix wall/harmonic` per species at each face with cutoff r_i, E = k (r_i − d)² for centre–face distance d < r_i, `wall_k_qt` = 15.0, `wall_k_ft` = 6.0 zJ/nm² (wall α = 2kΔt/γᵢ ≈ 0.70 like the pair constants; predicted mean depth ≈ 0.4 / 0.6 nm), wall energy included in `pe` (`fix_modify energy yes`), the wall virial not: `c_pvir` is the interaction pressure (pair + bond forces, no kinetic term) in all three boundaries, so it stays comparable; with walls the wall force carries mostly the kinetic part, and a reflecting wall has no force at all; insertion inside the box shrunk by r_i (`region ins_qt/ins_ft`); the generated run name ends in the boundary (`_periodic`, `_reflect`, `_harmonic`) | first step towards confined compartments; reflect is the textbook reflecting boundary for diffusion, harmonic is the same physics as `fix wall/region … harmonic`, which curved shapes (sphere, cylinder, nucleus) will need; concentrations stay defined on the full box Lx·Ly·Lz, the volume a centre can reach (∏(L − 2r_i)/∏L) is listed in the derived values; the LAMMPS RDF is normalised for a homogeneous system and is biased low for particles near a wall; the analysis uses no minimum image (overlap, cluster centres) and OVITO does not unwrap (the dump has `ff` flags) |
| Atom types | 2 types: 1 = Qt (r = 27.5 nm), 2 = Ft (r = 13.0 nm), named by `labelmap` (`traj.dump` writes the names Qt/Ft, `final.data` stores the labels); "bound" = at least one bond | the four ReaDDy reactions collapse to one Qt–Ft rule; no re-typing needed (also later for bond breaking); the names make OVITO show Qt/Ft, the readers convert them back to 1/2. No bound-state types (QtC/FtC): colour by `nbonds` in OVITO instead |
| Dynamics | overdamped Brownian dynamics, `fix brownian` (Euler–Maruyama), Gaussian noise, one fix per species, γ = k_BT/D | particle timescales (µs steps) are ~10⁵ × the momentum relaxation time m/γ, so inertia is irrelevant; velocity Verlet + Langevin would need ~10⁶ × more steps |
| Thermostat | none besides `fix brownian` (friction + noise satisfy fluctuation–dissipation) | `fix brownian` is the only integrator; never add `fix nve/langevin/nvt` |
| Diffusion | D_Qt = 2 × 10⁻⁴, D_Ft = 5 × 10⁻⁴ nm²/ns (handover values, not Stokes–Einstein) | keeps Δt = 500 ns well inside the stable range; Stokes–Einstein would be ~50× faster and need smaller Δt |
| Timestep | Δt = 500 ns (reference) | stability values α = 2kΔt(1/γᵢ + 1/γⱼ): Qt–Qt 0.70, Ft–Ft 0.70, Qt–Ft 0.70, bond 0.70 (force constants chosen for α ≈ 0.7, see Overlap and force constants); 1000 ns would give α ≈ 1.4 (overshooting steps) |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j; k_QQ = 7.5, k_FF = 3.0, k_QF = 4.3 zJ/nm² | soft excluded volume allows the large timestep; LAMMPS k includes the ½; set for α ≈ 0.7 at Δt = 500 ns (least overlap with margin to α = 1, see Overlap and force constants; former handover values 3.32 / 2.0 / 1.245); all three pairs set explicitly (default geometric mixing would give a wrong Qt–Ft cutoff) |
| Bond | `bond_style harmonic`, E = K (r − r₀)², K = 4.3 zJ/nm², r₀ = 40.5 nm | set for α ≈ 0.7 like the repulsion (former handover value 1.66); K = k_QF, so a pair inside contact has the same energy before and after bonding (no jump at bond formation) |
| Bonded pairs | `special_bonds lj 1.0e-50 1.0 1.0` | directly bonded pairs: repulsion off (the bond keeps the distance); pairs 2–3 bonds apart (e.g. two Qt on one Ft) keep the repulsion; 1e-50 instead of 0 keeps bonded pairs in the neighbour list so they appear in the RDF (`fix bond/create` still refuses duplicate bonds) |
| Binding | `fix bond/create`, Qt–Ft only, R_bind = 40.5 nm, prob p_on = 1 − exp(−k_on f Δt) per step with the calmodulin loading f(t) (see Calcium control); k_on = 1.86·10⁻⁶ ns⁻¹ is the rate at full loading (f = 1) (10⁻⁶ × √(4.3/1.245): same kinetics as with the former k_QF, see Overlap and force constants; to be calibrated against experiment); max 60 bonds per Qt, 10 per Ft; loops allowed | contact reaction of the Collins–Kimball type (partially reactive surface at contact, see Binding model); difference to ReaDDy accepted: each atom bonds at most once per step, and only with a partner that also has it as its closest eligible partner |
| Bond bookkeeping | `extra/bond/per/atom 60`, `extra/special/per/atom 1000` (LAMMPS reserves 2001 slots); `comm_modify cutoff` = max(2.5 r₀, RDF cutoff) + skin = 141.5 nm (skin 20) | bond/create keeps 1-2/1-3/1-4 neighbour lists and needs partners 2 bonds away as ghosts; actual need measured with `benchmark.special_need` |
| Start | overlap-free random insertion (`create_atoms random … overlap`): Qt with 55 nm, then Ft with 40.5 nm; 5000 steps (2.5 ms) equilibration without binding | dilute system (φ ≈ 6 %), so the insertion is already close to an equilibrium fluid; only the contact pairs (overlap, needed for binding) are missing at the start: repulsion energy and pressure reach their plateau within ≈ 1000 steps (test 2026-10-20, 4000 Qt / 10 000 Ft, 20 000 steps), 5000 steps give a 5× margin at 0.5 % of a 10⁶-step production; the Ft–Ft start bias (≥ 40.5 instead of 26 nm) is erased (RMS displacement in 2.5 ms: Qt ≈ 55 nm, Ft ≈ 87 nm) |
| Unbinding | `bond_break = "random"` (default): `fix bond/break/random` (own LAMMPS plugin, `lammps_plugins/`) every `break_every` = 10 steps; every bond breaks independently with p_off = 1 − exp(−k_off(t) · break_every · Δt), k_off(t) = f k_off,Ca + (1 − f) k_off,free (see Calcium control), p_off evaluated at every attempt. `"longest"`: the built-in `fix bond/break` with Rmax = 0, p_off fixed per segment | a Poisson process per bond, as the model requires; the built-in `fix bond/break` lets only the longest bond of both atoms break (≈ 0.1 × k_off, coordination dependent; see Unbinding below), kept as an option for comparison with older runs |
| Calcium control | every production follows a calcium protocol: baseline (Ca_rest) → influx (exponential rise to the plateau Ca_peak, 99 % after `ca_rise_s`; optoCRAC one-equation model) → exponential pump-out (τ); durations `t_baseline_s`, `t_influx_s`, `t_pumpout_s` (each may be 0, e.g. 0/30/0 = agglomeration at constant rates); the production runs in segments of `rate_update_every` steps that re-create the binding fix with k_on at the segment start (with `bond_break = "longest"` also the unbinding fix); with `"random"` k_off follows the calcium at every unbinding attempt | the agglomeration–deagglomeration cycle of the experiment (calcium influx, then pumping); all in one LAMMPS input, calcium computed inside LAMMPS from the step |
| Seeds | per replica 6 independent seeds (placement Qt/Ft, Brownian Qt/Ft, binding, unbinding; binding adds the segment number, unbinding too with `bond_break = "longest"`) from `numpy.random.SeedSequence(master_seed)`, range 1…900 000 000 | the LAMMPS Marsaglia generator rejects larger seeds; runs are reproducible |

ReaDDy is not used for validation; the LAMMPS model stands on its own.

## Reference parameters

Only the time step has a reference value: **Δt = 500 ns** (the `Params` default). Particle numbers, box size
and the calcium protocol (and with it the run length) are chosen per run in the control notebook; there is no
reference system. The `Params` default protocol is the experimental cycle 10 s baseline / 30 s influx / 90 s
pump-out (2.6·10⁸ steps); tests use small systems and short protocols.

## What is computed where

| Quantity | Where |
|---|---|
| bonds, free/bound counts, fraction bound, number of clusters, mean/largest size (all particles, Qt only, Ft only: `compute reduce/chunk` per type), clusters ≥ 5, size-category fractions, mean/max coordination, bond lengths, energies, virial pressure | LAMMPS during the run (`compute nbond/atom`, `fragment/atom`, `chunk/atom`, …), YAML thermo in `log.lammps` |
| Qt–Ft RDF, coordination histograms | LAMMPS (`rdf.dat`, `coord_*.hist`) |
| mean ± SD over replicas (sample SD, ddof = 1: replicas are independent samples; NaN with a single replica, figures then show the mean without a band), half-times | `qtft/ensemble.py` |
| percolation: one cluster spans the box along at least one axis; periodic box: the cluster wraps around the box (a closed bond path adds up to a box length, found by unwrapping along a spanning tree; checked against a doubled box), walls: it touches both opposite walls (particle surface within 5 nm of each face); time resolution `traj_every`; not "largest cluster ≥ 50 %" | `qtft/structure.py` (`percolation`, numpy/scipy from `traj.dump` + `bonds.dump`, cached in `rep_XX/analysis/percolation.csv`), used by `ensemble_stats` |
| Rg (centres, equal weights), compactness Rg/Rg,ideal (Rg,ideal = √(3/5) R of a solid sphere holding the cluster's particle volumes n_Qt V_Qt + n_Ft V_Ft at random close packing 0.64), cluster centres, nearest-neighbour distances, composition (need unwrapping across the periodic box; none with walls) | OVITO Python module, `qtft/structure.py` |
| agglomerate size and volume distributions and the coordination histogram at the end of the influx (`Params.influx_end_step`, the last `traj.dump` frame at or before it; the run end without an influx), where the agglomerates are largest, not at the run end where they have dissolved again; agglomerates = clusters of ≥ 3 particles. Volumes for comparison with confocal volumes (lattice SIM, Zeiss Elyra 7; only Ft fluorescent, but the volume counts Qt and Ft): all particle spheres of a cluster on a 10 nm grid (anchored at 0, so the voxel errors average out: single Ft 1.00 ± 0.10 × V); **solid volume** = their union; **envelope volume** = its morphological closing with an ellipsoidal probe, r_xy = 60 nm, r_z = 150 nm (half the lattice-SIM resolution, ~120 / 300 nm): gaps and concavities narrower than the probe, which the microscope does not resolve, are filled; enclosed cavities wider than the probe stay empty (they would look hollow). Probe only, no PSF blur, no volume cutoff. A cluster that wraps around the periodic box has no volume (NaN). Probe and voxel are arguments, not `Params` (they describe the microscope, not the simulation). 3 µm run at t = 40 s: 34 agglomerates, envelope/solid median 1.44, 1 s per frame | `qtft/structure.py` (`cluster_volumes`, `cluster_volumes_run`, numpy/scipy: unwrapping as for percolation, `distance_transform_edt` with anisotropic sampling), called by `make_report` |
| overlap diagnostics | `qtft/overlap.py` (from `traj.dump`) |
| agglomerates with ≥ n particles for several n (default 5, 20, 60, 80, 100, 150; in the 3 µm run n ≈ 60–90 peaks at the end of the influx, smaller n earlier, larger n later): counted in every `traj.dump` frame from the `cluster` column (resolution `traj_every`; equal to the thermo `n_large` for n = `large_cluster_min`), in the box or as a density per 1000 µm³ = count · 1000 / V_box (V_box = Lx·Ly·Lz in µm³, the full box also with walls, like the concentrations). The density is the box count scaled up, not a count in a real 1000 µm³ volume: one agglomerate in a 3 µm box (27 µm³) is already 37 per 1000 µm³, so small counts give a coarse, noisy density. The same also with the size counted in Qt or in Ft only (agglomerates with ≥ n Qt, default n = 5, 10, 20, 50, 70, since an agglomerate holds about 4× fewer Qt than Ft; ≥ n Ft with the total thresholds) | `qtft/ensemble.py` (`large_counts`, `species`), figures `fig_large_by_species` (3 × 2, in the report) and `fig_large_count` (total only, comparisons) |
| figures, tables | `qtft/figures.py` (Nature style as in the handover) |
| FIB-SEM export (periodic: clusters unwrapped and shifted to the origin; walls: positions in the box frame, walls at 0 and L) | `qtft/fibsem.py` |
| animation (video of one replica, see below) | `qtft/animation.py` |

`ke`, `temp` and `etotal` are meaningless in overdamped dynamics and are not logged.

## Project conventions

- All parameters are set in one place: the parameter cell of `control.ipynb` (defaults in
  `qtft/params.py`). The LAMMPS input is generated from `template.lmp`; edit the template only
  when the model changes.
- Simulation output lives in `runs/` inside the repository but gitignored: runs with one replica
  in `runs/single/<run name>/`, runs with several in `runs/ensemble/<run name>/`, benchmarks in
  `runs/benchmark/`, force-constant sweeps in `runs/tuning/`, parameter sweeps in `runs/sweeps/<name>/<label>/`.
  `params.json` is the record of a run, `rep_XX/` holds the replicas, `analysis/` the results;
  `runs/index.csv` lists all single, ensemble and sweep runs.
- Parameter sweeps (`qtft/sweep.py`): a grid of every combination of the listed values, built from the base
  `Params` of the parameter cell (one value per parameter); the label of a run names only the swept values
  (`temperature300_k_on2e-06`). Derived quantities are not adapted (a different temperature or time step
  changes α). Execution settings (`n_replicas`, `mpi_procs`, `max_parallel`, `master_seed`, …) cannot be
  swept. All variants use the same seeds (common random numbers), so differences between variants come from
  the parameters rather than from noise. All replicas of a sweep run through one launcher and one queue, at
  most `max_parallel` at once, replica-major (replica 1 of every variant first), so every variant gives a
  first result early. The force-constant sweep (`alpha_sweep`) uses the same mechanism.
- Runs on the LRZ Linux cluster (`executor = "lrz"`, an execution setting: not in the run name, old runs
  read as `"local"`): the run is built here as usual and runs there as one Slurm job array (`job.slurm` from
  `job_template.slurm`, one task per replica, `replicas.txt`; a sweep is one array over all its replicas,
  replica-major, at most `max_parallel` at once) in `serial_long` (7 days, 1 node per job), `mpi_procs` MPI
  tasks, `slurm_time` (default 3 days, ≈ 2 × the expected 35 h of the 3 µm run) and `slurm_mem` (default
  4G) per replica, email at begin, end and failure (once per array). The job writes the same `status` /
  `pid` files as the local launcher. LAMMPS on LRZ is the same version with `fix bond/break/random` compiled
  in, so the input loads no plugin; otherwise the input is identical to a local one, and the results were
  byte-identical (2 and 8 processes). The folder on LRZ mirrors the local one: `~/qtft-lammps/runs/<group>/
  <name>/`. Setup and the LRZ-specific job settings: `docs/INSTALL.md`, section 9. From the notebook, over
  the SSH connection opened once in a terminal (non-interactive `ssh`/`rsync`, so a closed connection fails
  at once with a hint instead of waiting for 2FA): `start` copies the folder (rsync) and submits `job.slurm`
  (job ID in `slurm_job.json`); `progress`/`summary`/`watch` first fetch `status`, `pid`, `log.lammps`,
  `lammps.out` and `slurm-*.out` (at most every 30 s) and show the Slurm state (running/pending with the
  reason); a replica whose task left Slurm without writing its end status (time limit, node failure,
  `scancel` by hand) gets `failed` (cancelled: `stopped`) from `sacct`, locally and on LRZ; `stop` cancels
  the job (for a run inside a sweep only its array tasks); `fetch` downloads everything, after which the
  folder is a normal run folder for the analysis. Nothing is copied back by itself (you decide when to
  download, ≈ 360 MB per replica of the 3 µm run); the analysis functions that need the trajectory stop
  with a hint until `fetch` (a finished replica without `final.data` here, which only `fetch` copies).
  The logs are fetched by appending the new lines (`rsync --append-verify`; `start` removes the local logs
  of an earlier attempt first), the status files in full. `max_parallel` limits the job array too
  (`%N`); `None` runs all replicas at once, and `check()` warns for an LRZ run with fewer than
  `n_replicas`. `index.csv` lists the `executor` of every run. Checked 2026-10-25 with a 2-replica run (2
  processes each): done in 10 s, fetched results byte-identical to the same replicas run locally,
  `make_report` works.
- Notebooks are committed without outputs (`nbstripout`).
- Commits are authored as Benedikt Pfluger <benedikt.pfluger@tum.de>; no co-author or
  "generated with" lines in commits.
- All work is committed directly on `main`; no feature branches or pull requests.
- LAMMPS documentation is the reference for every command (docs.lammps.org).

## Animation

`animate(run)` (animation notebook) renders one replica as a video (layout from the animation handover, kept
locally): black background, Arial regular (taken from Windows and copied to matplotlib's cache folder:
read from `/mnt/c`, text draws ~10× slower).

- **Resolution:** the layout is defined at 1920 × 1080; `scale` (default 2) multiplies it (2 = 3840 × 2160, 4K):
  text and lines scale, and the spheres are rendered at the full resolution, which they need: at 1080p an Ft
  sphere is only ~3 px wide in the whole-box views (4K: ~7 px; magnification ~20 px). H.264 with CRF 14 and
  `yuv420p` (plays everywhere; 4:2:0 halves the colour resolution, which 4K makes up for). The frame size must be
  even (scale 1, 1.5, 2, 3, ...).
- **Frames:** one video frame per `traj.dump` frame, no interpolation (it would invent Brownian motion and bond
  states). The speed follows from `traj_every` and `fps`: 0.1 s per frame at 30 fps runs 3× faster than real
  time (3 µm run: 1300 frames, 43 s of video); `animate` prints the factor. Rendering takes ≈0.25 s per frame at scale 1
  and ≈0.3 s at scale 2 (3375 particles, without the encoding).
- **Particles:** shaded spheres at physical size (Lambert shading, highlight at the top left), coloured by type
  and bonded (`nbonds > 0`): Qt green / blue, Ft orange / red; painted back to front, the farthest 35 % darker.
  A small numpy renderer pastes one pre-shaded sprite per colour and pixel radius (matplotlib cannot shade
  spheres; OVITO's renderers are slow or need a display).
- **Views:** 3D perspective from a fixed camera (elevation 20°, azimuth 30°, 2.5 box diagonals away), 12 box
  edges (the three at the farthest corner behind the particles); front view orthographic along y (x right, z up)
  over the full depth; magnification of a **central box of a third of every edge**, chosen instead of the full
  depth behind the outlined region, which would stack many particles: particles whose centre lies inside it in
  depth and whose sphere reaches into it in x and z. Periodic runs show wrapped positions (an agglomerate across
  a face appears on both sides).
- **Graph:** the average cluster size of the analysis figures (`p_mean_size`): all particles / clusters, Qt per
  cluster with Qt, Ft per cluster with Ft, free particles counted as clusters of 1; from the thermo table
  (`output_every`), so finer than the frames. Right of the magnification: calcium and calmodulin loading f over
  time (`Params.calcium`, `Params.loading`: the formulas of `template.lmp`), which shows why the agglomerates
  dissolve only slowly after the influx (calcium decays with `ca_tau_s`). Both graphs shade the influx period
  and mark the current time.
- **Statistics** of the shown frame: bonds = Σ nbonds / 2; Qt in the cluster with the most Qt; % of the Qt in
  clusters with ≥ 2 Qt.
- **Header and timeline:** phases and their times from `params.json` (calcium off / on / off at the baseline,
  influx and pump-out); the calcium readout is `Params.calcium(t)`, the formula of `template.lmp`, so it shows
  the rise over `ca_rise_s` and the decay with `ca_tau_s`.

## Status

| Phase | Content | State |
|---|---|---|
| 0 | setup: README, INSTALL, environment | merged |
| 1 | simulation prototype: template, parameters, builder, runner, control notebook | merged |
| 2 | performance benchmark, special-list sizing | merged; benchmark run 2026-10-20 (see Performance) |
| 3 | ensemble statistics, run index, run comparison | merged |
| 4 | structure (OVITO) and overlap analysis, figures, tables | merged |
| 5 | FIB-SEM export, this document | merged |
| 6 | parameter sweeps, walls, Collins–Kimball binding rate, calcium-controlled binding and unbinding | merged |
| 7 | unbinding fix `bond/break/random` (LAMMPS plugin), `bond_break` toggle | merged |
| later | restart files for long runs, HPC cluster runs, continuous k_on (the same for `fix bond/create`) | open |

Open items: size `extra_special` from a long production run (`special_need`); compare the real run speed
with the benchmark projection.

## Output intervals

- `output_every`: thermo table and in-run statistics in `log.lammps` (the time series). `traj_every` (a
  multiple of it): `traj.dump`, `bonds.dump`, `rdf.dat`, `coord_*.hist`. The last four are only read at
  `traj.dump` steps or at the final frame, so writing them at every thermo step would only cost disk space:
  `bonds.dump` alone would be 50–130 GB per replica for a 130 s run of 37 000 particles at 1 ms.
- `check()` warns if the run end or the influx start is not a `traj_every` step (no final frame, or no frame
  for the contact pairs of `binding_rate`).
- Recommended for the full protocol (10/30/90 s = 2.6·10⁸ steps): `output_every` = 2000 (1 ms; resolves the
  initial binding slope for `binding_rate`, ≈ 30 MB of thermo), `traj_every` = 2·10⁵ (100 ms, ≈ 3.4 GB for
  37 000 particles) or 10⁶ (0.5 s) for 240 000 particles, `rate_update_every` = 10⁵ (50 ms; Ca changes by
  0.25 % per segment at τ = 20 s), `break_every` = 10 (p_break per attempt ≈ 5·10⁻⁶ at k_off,free = 1 s⁻¹). Benchmark 2026-10-24 (state at the
  bond peak of a 3 µm run, 675 Qt / 2700 Ft, 7905 bonds, 8 MPI processes, median of 4 rounds): unbinding at
  every step costs 13–14 % (both methods), every 10 or 100 steps nothing measurable (3059 / 3134 steps/s with
  `"random"`, 3055 without unbinding); `"random"` is not slower than `"longest"`.
  The defaults (1000 / 1000 / 20 000 / 10) suit the short test runs.

## Performance (local machine)

Benchmark 2026-10-20 on the local machine (Intel i7-10700K: 8 physical cores, 16 hardware threads, WSL2),
system 4000 Qt / 10 000 Ft in a 2000 nm box, Δt = 500 ns, 2000 production steps per setting
(`runs/benchmark/{16,8}cores/results.csv`).

| Setting | Result | Decision |
|---|---|---|
| neighbour skin | skin 20 is 20–45 % faster than skin 10 in every configuration: list rebuilds every 6.9 instead of 2.7 steps (Brownian jumps ≈ 1.2 nm/step), while the dilute system has < 1 neighbour per particle, so longer lists cost almost nothing | `neighbor_skin = 20` |
| MPI processes per replica | for 14 000 particles more processes per replica pay off (1 → 8 procs: 125 → 1223 steps/s per replica with the machine full); best ensemble throughput with 4 procs × 4 replicas on 16 threads (3540 replica-steps/s) | ensembles: `mpi_procs = 4`, `max_parallel = 4`; a single replica is fastest with 8 procs (≈ 2070 steps/s) |
| hyperthreading | small effect: helps only with 4 procs/replica (3540 vs 2510 replica-steps/s), with 1 proc/replica 16 threads are slower than 8 cores | use all 16 threads with the setting above |
| time budget | ≈ 60 % of the time in `Modify` (`fix brownian`, `fix bond/create` every step, in-run cluster statistics) | candidate for optimisation if runs get long |

Projected time for 500 ms (10⁶ steps) with these settings: ≈ 18 min per replica, 4 at once (≈ 37 min for 8
replicas). The benchmark measures the start of a run with almost no bonds; real runs get slower as the networks
grow, so these are lower bounds.

## Overlap and force constants

The soft repulsion lets particles overlap. Every pair family sits in a harmonic well E = k x² (x = overlap
depth) driven by thermal noise, so the mean depth of overlapping pairs is
√(2/π) · √(k_BT / 2k / (1 − α/2)); the factor 1/(1 − α/2) is the widening by the Euler–Maruyama step of
`fix brownian` (α = 2 k Δt (1/γᵢ + 1/γⱼ), `Params.alpha`). Stiffer constants reduce the overlap, but α grows
with k: at a fixed time step the overlap is smallest at α ≈ 1; at a fixed α it scales with √Δt
(`qtft/tuning.py`: `predicted_overlap`, `constants_for_alpha`).

Sweep 2026-10-20 (`runs/tuning/sweep/`, control notebook section 7): 400 Qt / 1000 Ft in 928.3 nm (production
concentration), 50 ms, 3 replicas per setting, all four constants (k_qq, k_ff, k_qf, bond_k) set to the same α.

| α | Δt (ns) | k Qt–Qt / Ft–Ft / Qt–Ft (zJ/nm²) | mean depth Qt–Qt / Ft–Ft / Qt–Ft (nm) | predicted Qt–Ft (nm) | fraction bound at 10 / 50 ms |
|---|---|---|---|---|---|
| 0.3 | 500 | 3.21 / 1.28 / 1.83 | 0.75 / 1.08 / 0.91 | 0.94 | 0.17 / 0.55 |
| 0.5 | 500 | 5.35 / 2.14 / 3.06 | 0.55 / 0.81 / 0.75 | 0.77 | 0.14 / 0.49 |
| 0.7 | 500 | 7.49 / 3.00 / 4.28 | 0.37 / 0.78 / 0.70 | 0.70 | 0.12 / 0.41 |
| 0.9 | 500 | 9.63 / 3.85 / 5.50 | 0.42 / 0.81 / 0.67 | 0.67 | 0.10 / 0.37 |
| 1.2 | 500 | 12.8 / 5.14 / 7.34 | 0.31 / 0.73 / 0.71 | 0.68 | 0.08 / 0.34 |
| 0.9 | 250 | 19.3 / 7.70 / 11.0 | 0.20 / 0.42 / 0.48 | 0.47 | 0.07 / 0.29 |

Findings:
- Qt–Ft overlap (mostly bonded pairs, 100–200 overlapping pairs per frame) matches the prediction within 5 %;
  at Δt = 500 ns it cannot drop below ≈ 0.67 nm (α ≈ 1). Qt–Qt and Ft–Ft rest on only 1–10 overlapping pairs
  per frame and are noisy; they lie at or below the prediction.
- All overlaps are small against the contact distances (Qt–Qt 55 nm, Ft–Ft 26 nm, Qt–Ft 40.5 nm): < 2 % at the
  current constants, except Ft–Ft (≈ 4 %).
- **The force constants change the binding kinetics.** A pair binds only within R_bind = r_Qt + r_Ft, i.e.
  while it overlaps, and a softer Qt–Ft repulsion lets pairs spend longer inside. The early fraction bound
  scales as 1/√k_qf (α 0.3 → 1.2: k_qf × 4, binding rate ÷ 2.1; Δt 250 ns: k_qf × 6, ÷ 2.4), independent of
  Δt. Stiffer constants therefore need a k_on raised by √(k_qf,new / k_qf,old) to keep the same kinetics;
  k_on has to be calibrated against experiment after the constants are fixed.

With harmonic walls the same estimate holds for the wall with α = 2 k Δt / γᵢ (only the particle moves):
`constants_for_alpha` also returns `wall_k_qt` and `wall_k_ft`, and `overlap_frame` adds the rows `Qt-wall` and
`Ft-wall` (how far spheres stick out of the box; always 0 for reflecting walls). Test 2026-10-21 (50 Qt / 200 Ft,
400 × 500 × 600 nm, 0.5 ms): largest wall depth 0.37 nm, reflect 0.

Decision (2026-10-20): α ≈ 0.7 at Δt = 500 ns, rounded: k_qq = 7.5, k_ff = 3.0, k_qf = bond_k = 4.3 zJ/nm²
(α 0.70 / 0.70 / 0.70 / 0.70; predicted overlap 0.53 / 0.84 / 0.70 nm), k_on = 1.86·10⁻⁶ ns⁻¹. Most of the
reduction possible at this time step, with margin to the α > 1 warning; the run time is unchanged.

## Binding model (Collins–Kimball)

A Qt–Ft pair can bond only within R_bind = r_Qt + r_Ft, i.e. while the two spheres overlap (`fix bond/create`
requires R_bind ≤ pair cutoff, and the repulsion ends at contact). Binding is therefore a reaction at contact
with a finite rate: the **Collins–Kimball** model (partially reactive surface). Particles diffuse with
D = D_Qt + D_Ft relative to each other and react at the contact surface R = 40.5 nm with a surface
reactivity κ:

- intrinsic (reaction-limited) rate κ = (rate per pair) · (contact pairs) · V / (N_Qt N_Ft), with the rate per
  pair p_bind / (bind_every Δt) ≈ k_on and the contact pairs = Qt–Ft pairs within R_bind at equilibrium; the
  product (contact pairs) · V / (N_Qt N_Ft) is the volume of the overlap layer ("contact volume")
- diffusion-limited (Smoluchowski) rate k_D = 4π D R
- association rate constant (steady state) k_a = k_D κ / (k_D + κ)

`binding_rate(run_dir)` measures during the calcium influx: it counts the contact pairs in the `traj.dump` frame at
the influx start (unbonded pairs; equilibrium distribution, since binding is weak before) and measures k_a as the
initial slope of the bond count from there; the rate per pair is that at peak calcium, k_on f_peak. The numbers
below were measured before the calcium control with a constant k_on (an influx-only protocol with f = 1); with
the defaults (f_peak = 0.89) the same k_on gives an 11 % smaller κ.
Defaults (4000 Qt / 10 000 Ft, 2000 nm box, k_QF = 4.3 zJ/nm², Δt = 500 ns, k_on = 1.86·10⁻⁶ ns⁻¹, 3 replicas,
2026-10-21): **measured k_a = (1.76 ± 0.05)·10⁷ M⁻¹ s⁻¹, predicted (1.75 ± 0.10)·10⁷ M⁻¹ s⁻¹**; κ = 1.9·10⁷,
k_D = 2.1·10⁸ M⁻¹ s⁻¹ (1 nm³/ns = 6.02·10⁸ M⁻¹ s⁻¹); κ/k_D ≈ 0.09, so binding is mostly reaction-limited.
Former constants (k_QF = 1.245, k_on = 10⁻⁶ ns⁻¹): measured 1.91·10⁷, predicted 1.91·10⁷. The measured early
rate starts at κ (equilibrium contact density) and settles to k_a within ≈ R²/D = 2.3 ms; with κ/k_D ≈ 0.09 the
two differ by < 10 %.

Continuum estimate of the contact volume: 4πR² σ √(π/2) with σ² = k_BT / (2 k_QF) / (1 − α/2) (Boltzmann-weighted
overlap layer, `collins_kimball(p)`). It holds only while the relative step per time step is small against σ:
at the former constants (σ ≈ 1.3 nm) it is within 10 % of the counted volume, at the defaults (σ ≈ 0.7 nm, relative
step √(2DΔt) ≈ 0.84 nm per axis) pairs jump through the thin layer within one step, the counted volume is
≈ 25 % smaller (17 100 nm³, replicas 15 800–17 800, instead of 22 600 nm³) and the continuum k_a ≈ 2.3·10⁷ M⁻¹ s⁻¹ is ≈ 29 % too high.
It is listed in `binding_rate.csv` for comparison only. The rule that only mutually closest candidates bond
(`fix bond/create`) removes ≈ 2–3 % of the contact pairs (≥ 97 % are mutual) and is not corrected for.

Consequences:
- k_on is not a physical constant: κ depends on k_QF and Δt through σ. **The physical quantity to report is
  k_a (or κ) in M⁻¹ s⁻¹**, measured from the run as the initial bond formation rate per Qt–Ft pair
  concentration, k_a = (dN_bonds/dt)|₀ · V / (N_Qt N_Ft) (`binding_rate(run_dir)`: fit through the origin
  over the frames up to 10 % bound, mean ± SD over replicas, next to the prediction from the counted contact
  pairs; figure `binding_rate`),
  and compared with experiment. The force constants and
  Δt are fixed (α ≈ 0.7 at 500 ns); changing them requires recalibrating k_on to the same k_a.
- k_a is a rate per Qt–Ft *particle* pair; with up to 60 Ft per Qt, a per-site rate would divide by the
  number of sites.
- Remaining artefact: where particles are pressed together (inside clusters, dense regions) they overlap
  more and bind somewhat faster than at an isolated contact.
- A binding shell beyond contact (R_bind = contact + 2 nm through an extra force-free `pair_style zero`
  cutoff, which would make κ nearly independent of k_QF) was tested on 2026-10-21 and not adopted: the
  contact reaction is a standard model, the constants stay fixed, and the shell would only shift the
  calibration (its neighbour-list cost was 1–2 %).

## Calcium control

The binding of Qt and Ft is mediated by calmodulin–RS20, which binds only with Ca²⁺. In the experiment, a calcium
influx triggers agglomeration and the cell then pumps the calcium out again (≈ 30 s high calcium, ≈ 90 s
deagglomeration). Every production follows such a protocol, computed inside LAMMPS from the step (`template.lmp`,
section 11):

    Ca(t) = Ca_rest                                              baseline   (0 ≤ t < t_baseline)
          = Ca_peak − (Ca_peak − Ca_rest) · exp(−λ_on t')        influx     (t' from its start, until t_baseline + t_influx;
                                                                             λ_on = ln(100) / ca_rise_s; a step if ca_rise_s = 0)
          = Ca_rest + (Ca_start − Ca_rest) · exp(−t'/τ)            pump-out   (t' from its start; Ca_start = Ca reached at
                                                                             the end of the influx, or Ca_rest without one)
    f(t)  = Caⁿ / (K_Caⁿ + Caⁿ)                                  calmodulin loading (Hill)
    k_on(t)  = k_on · f                                          binding rate constant per Qt–Ft pair in contact
    k_off(t) = f · k_off,Ca + (1 − f) · k_off,free               unbinding rate constant per bond

f is the fraction of calmodulin with calcium bound, i.e. able to hold RS20 (mean field: every bond sees the
average loading; calcium binding to calmodulin takes milliseconds, so f follows Ca(t) instantly).

**Rise and decay: optoCRAC one-equation model.** Calcium enters through ORAI1 channels gated by optoCRAC (LOV2–STIM1),
switched on by blue light (the influx phase = light on) and off by the thermal dark reversion of LOV2 (pump-out phase
= light off). With A the fraction of active channels, dA/dt = k_act I (1 − A) − k_dark A, and c = c0 + (c_max − c0) A.
For constant light this is exactly the formula above: the rise relaxes with λ_on = k_act I + k_dark to the plateau
Ca_peak = c0 + (c_max − c0) k_act I / λ_on, the decay with k_dark = 1/τ. The parameters are the observables:
plateau `ca_peak_uM`, rise time `ca_rise_s` (99 % of the rise; defines λ_on), decay `ca_tau_s`; `derived()` shows the
model constants (defaults: τ_on = 1.09 s, k_act I = 0.87 s⁻¹, k_dark = 0.05 s⁻¹, effective c_max = 1.05 µM).
`ca_rise_s` must stay below τ ln 100 (≈ 92 s), otherwise k_act < 0. The model assumes that Ca²⁺ clearance by the
pumps (PMCA/SERCA, ≈ 1–3 s, not verified) is much faster than the channel kinetics, so that Ca follows A at once;
the decay after light-off is the channels closing, not the pumping. With a rise of 5 s to 99 % (τ_on ≈ 1.1 s) this
holds only for clearance faster than ≈ 0.5 s; otherwise the real rise is slower and S-shaped, and the two-equation
gate + pump model (dc/dt = J A − k (c − c0)) would be needed. As a curve shape the formula is exact.
`binding_rate` needs a calcium step (`ca_rise_s = 0`): its fit assumes the peak rate from the influx start on.
For any protocol, `bond_rates(run_dir)` gives the time course per calcium segment (or per `bin_s`): bonds formed
per second (increase of Bonds + n_broken), bonds broken per second, and the apparent association rate constant
k_a(t) = formed per second · V / (N_Qt N_Ft), normalized by the total particle numbers like the measured k_a of
`binding_rate` (bonds also form between particles that are already bound). It is apparent: it follows k_on(t) but
also falls as particles fill their bond slots (`maxbond_*`) and as free partners are used up, even at constant calcium.

| Parameter | Default | Source |
|---|---|---|
| `ca_rest_uM`, `ca_peak_uM` | 0.1, 1.0 µM | typical resting level and global calcium signal in cells (plateau under light) |
| `ca_rise_s` | 5 s | plateau (99 %) reached 5 s after light-on; to be fitted to a measured calcium trace |
| `ca_tau_s` | 20 s | decay after light-off (LOV2 dark reversion, tens of seconds); calibrate against the 90 s deagglomeration |
| `ca_K_uM`, `ca_hill` | 0.5 µM, 3 | target peptides raise the calcium affinity of calmodulin; cooperative loading of 4 sites |
| `k_on` | 1.86·10⁻⁶ ns⁻¹ (at f = 1) | k_a at full loading ≈ 2·10⁷ M⁻¹ s⁻¹, typical for calmodulin–peptide association |
| `k_off_ca_per_s`, `k_off_free_per_s` | 10⁻³, 1 s⁻¹ | calmodulin–RS20 with Ca²⁺ binds with nM affinity; without Ca²⁺ it releases the peptide within ~1 s |

These are order-of-magnitude literature estimates, to be checked and calibrated. With them f(rest) = 0.008, so the
baseline binds weakly (k_on < 1 % of its maximum) and the few bonds live ~1 s; f(peak) = 0.89, so binding runs
near its maximum, but the 11 % unloaded calmodulin dominates unbinding even at the peak: k_off(peak) = 0.11 s⁻¹
(bond lifetime ≈ 9 s), not 10⁻³ s⁻¹.

**Segments.** `fix bond/create` (and `fix bond/break`) take a fixed probability, so the production runs in segments
of `rate_update_every` steps (default 20 000 = 10 ms): each segment re-creates the binding fix (with
`bond_break = "longest"` also the unbinding fix) with the probability at its start and a new seed (seed + segment
number). A phase boundary inside a segment takes effect for these at the next segment (warning in `check()`).
`fix bond/break/random` (`"random"`) is defined once for the whole production and evaluates p_off at every attempt,
so unbinding needs no segments. `fix bond/create` counts the bonds per atom (for `maxbond_*`) only when it is
created, so a bond broken within a segment still counts until the next segment (with both methods). The segment starts write "Inconsistent image flags" (bonds across the periodic boundary,
harmless; the analysis uses the minimum image) and "Using fix bond/create … with fix bond/break may not work as
expected" (≈ 2 log lines per segment, ≈ 26 000 lines for 130 s; not limited, so that other warnings stay
visible). Checked: after a full protocol the special lists are
consistent (energy unchanged after rebuilding them), the bond count in the thermo output equals `bonds.dump`, and
no bond is duplicated.

**Unbinding: limitation of `fix bond/break` (`bond_break = "longest"`).** Every `break_every` steps each atom picks its *longest* bond as
its only candidate, and a bond breaks (with probability p_off) only if it is the longest bond of both atoms. With
Rmax = 0 every bond is a candidate, but bonds of highly connected atoms break less often. Measured 2026-10-21
(150 Qt / 600 Ft, 964 bonds, unbinding only): per-bond rate 0.37 / 0.27 / 0.17 / 0.12 / 0.09 × k_off for Qt with
1–2 / 3–4 / 5–6 / 7–9 / ≥ 10 bonds (0.14 for all bonds; by Ft bonds 0.17 for 1, 0.07 for ≥ 4). Consequences:
the effective k_off is much smaller than set, cluster cores (well-connected Qt) dissolve too slowly, so
deagglomeration peels clusters from the outside, and stretched bonds break preferentially (no calibration with a
single factor can remove this). `fix bond/react` was tested and is unsuitable: its templates must match the exact number of bonds of both atoms
of the breaking bond (edge atoms are not modified), so every combination of Qt and Ft bond counts would need its
own template (up to 60 × 10).

**Unbinding: `fix bond/break/random` (`bond_break = "random"`, default).** A LAMMPS plugin in
`lammps_plugins/bond_break_random/` (C++, based on `fix bond/break` of the installed version; built with
`build.sh` against the LAMMPS source of the same version, loaded with `plugin load` unless `lmp` has the style built
in): `fix ID group bond/break/random Nevery btype prob seed`, `prob` a number or an equal-style variable
(`v_p_off`), evaluated at every attempt. Every bond (stored once with `newton_bond on`) is tested once per attempt
with its own random number, an atom can lose several bonds in one attempt; the broken pairs are gathered from all
MPI ranks, the special lists are updated as in `fix bond/break` (1-2 removed, 1-3/1-4 rebuilt), and the neighbour
lists are rebuilt in the same step. Output vector: [1] broken in the last attempt, [2] since the fix was defined.
Checked 2026-10-24 (150 Qt / 600 Ft): per-bond rate 0.97 ± 0.03 × k_off, flat over the coordination (Qt 1–2 …
≥ 10 bonds: 0.91–1.17, Ft 1–3: 0.89–0.99; `"longest"` in the same test 0.15); decay 991 → 142 bonds (theory 134);
during a pump-out in which k_off rises 345 → 1974 s⁻¹ within the segments, broken / expected = 0.99 / 1.01 / 1.01
at 1 / 2 / 4 MPI processes (`"longest"` 0.06); with binding and breaking in the same steps (p up to 0.3): energy
unchanged after rebuilding the special lists, bonds = created − broken, `bonds.dump` = thermo, no duplicates; the
`"longest"` input reproduces the former one bit for bit. **Calibration:** with `"random"` k_off is the real rate
per bond; runs with `"longest"` (all runs before 2026-10-24) had an effective k_off of ≈ 0.1 × the set value
(coordination dependent), so `k_off_ca_per_s` and `k_off_free_per_s` must be recalibrated. Calibration 2026-10-25
against the 3 µm `"longest"` run (675 Qt / 2700 Ft, harmonic walls, 10/30/90 s, k_off 10⁻³ / 1 s⁻¹): broken bonds /
∫ bonds · k_off dt = 0.098 over the whole run (0.075–0.09 at 15–80 s with large clusters, 0.12–0.19 at 100–130 s while
they dissolve). `control.ipynb` used k_off × 0.098 = 9.8·10⁻⁵ / 0.098 s⁻¹ with `"random"` up to and including the
130 s 3 µm run, to stay comparable with the older `"longest"` runs. **Since 2026-10-26 it uses the unscaled
literature values 10⁻³ / 1 s⁻¹** (`break_every` = 10): with `"random"` k_off is the true per-bond rate, so the
0.098 only reproduced an artifact of the superseded method. Runs before and after this change are therefore not
directly comparable in their deagglomeration kinetics (unbinding is ≈ 10× faster now). The literature basis for
the two values is in `docs/calmodulin_rs20_literature.md`.
1 s restarts from that run's states: broken new / old = 1.26 ± 0.09 at 51.5 s (peak, 7905 bonds; bond count and
largest cluster unchanged) and 0.99 ± 0.06 at 100 s (pump-out, 4640 bonds); whether the deagglomeration half-times
match needs a full run. The `Params` defaults stay at the literature values (table above). Old `params.json` files
(without `bond_break`) are read as `"longest"`.

**Analysis.** `ensemble_stats` adds per replica the peaks of the bond count and the largest cluster and the
deagglomeration half-times (`t_half_unbind_bonds_ms`, `t_half_unbind_largest_ms`: time after the end of the influx
until the value has fallen to half of its value at that moment); "final" values are those at the end of the whole
protocol. `fig_calcium` shows calcium, loading, the rate constants k_on and k_off, the binding and unbinding rates (bonds formed
and broken per second) with the apparent k_a, bonds, fraction bound and cluster sizes. k_on (per contact pair) and
k_off (per bond) are not comparable with each other: the per-particle binding rate is k_a·c_Ft, which includes the
encounters; all time-series figures mark
the start and end of the influx with dotted vertical lines.

