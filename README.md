# qtft-lammps

Coarse-grained Brownian dynamics simulation of the agglomeration of **Qt encapsulin**
(large protein cage, r = 27.5 nm) and **ferritin** (Ft, r = 13 nm) in LAMMPS.
Qt and Ft diffuse in implicit solvent, repel each other softly, and form harmonic
Qt–Ft bonds stochastically when they touch. Clusters grow into Qt–Ft–Qt networks.

## Model in short

| Component | LAMMPS implementation |
|---|---|
| Units | `units nano` (nm, ns, energy in zJ, force in pN) |
| Box | cubic (`box_length=2000.0`) or rectangular (`box_length=[2000.0, 3000.0, 4000.0]`); periodic (`boundary="periodic"`, default) or with walls on all faces: reflecting (`"reflect"`, `fix wall/reflect`) or soft (`"harmonic"`, `fix wall/harmonic`) |
| Dynamics | overdamped Brownian dynamics, `fix brownian`, γ = k_BT/D per species |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j |
| Bonds | `bond_style harmonic`, E = K (r − r₀)², r₀ = r_Qt + r_Ft |
| Binding | `fix bond/create`, Qt–Ft only, at contact (R_bind = r_Qt + r_Ft), probability 1 − exp(−k_on f Δt) per step: a Collins–Kimball contact reaction; report the association rate k_a in M⁻¹ s⁻¹ (see `docs/decisions.md`) |
| Unbinding | `fix bond/break/random` (own plugin in `lammps_plugins/`): every bond breaks independently with probability 1 − exp(−k_off Δt) per attempt, k_off following the calcium at every attempt (`bond_break="random"`, default); `bond_break="longest"`: the built-in `fix bond/break` (only the longest bond of both atoms can break, see `docs/decisions.md`) |
| Calcium control | every production follows a calcium protocol (baseline → influx with an exponential rise to the plateau → exponential pump-out, each phase optional); calmodulin loading f = Caⁿ/(K_Caⁿ + Caⁿ) sets k_on f and k_off = f k_off,Ca + (1 − f) k_off,free; computed inside LAMMPS (k_on per segment, k_off at every unbinding attempt) |
| Start | overlap-free random insertion, short equilibration without binding |

## Usage

1. Install everything as described in [docs/INSTALL.md](docs/INSTALL.md), including the unbinding plugin
   (`lammps_plugins/bond_break_random/build.sh`; again after every LAMMPS update).
2. `conda activate lammps`, open the repo in VS Code.
3. Open `control.ipynb`: set the parameters, check the derived values, build and run. For a parameter
   study, list values per parameter in section 4 (`build_sweep`): one run per combination.
4. Analyse finished runs in `analysis.ipynb`; render a video of a run in `animation.ipynb` (needs ffmpeg).

Simulation output is written to `runs/` inside the repository (gitignored, never committed):
`runs/single/` for runs with one replica, `runs/ensemble/` for runs with several,
`runs/sweeps/` for parameter sweeps, `runs/benchmark/` for performance measurements, `runs/tuning/` for
force-constant sweeps, and `runs/index.csv` listing all single, ensemble and sweep runs.

## Repository layout

| Path | Purpose |
|---|---|
| `control.ipynb` | set all parameters, build the LAMMPS input, run, monitor |
| `analysis.ipynb` | ensemble statistics, structure and overlap analysis, figures and tables, comparison of runs, FIB-SEM export |
| `animation.ipynb` | video of one replica: set the run folder name, `animate` writes `rep_XX/animation/animation.mp4` |
| `docs/decisions.md` | model and project decisions with reasons, status of the implementation |
| `docs/outputs.md` | every output file: content and what creates it |
| `template.lmp` | the LAMMPS input template (model structure, `{{placeholders}}`, no numbers) |
| `lammps_plugins/bond_break_random/` | LAMMPS plugin `fix bond/break/random` (C++ source, `build.sh`; the built `.so` is not committed) |
| `qtft/params.py` | all parameters with defaults, derived values (γ = k_BT/D, α, p, …) and checks |
| `qtft/build.py` | fills the template → `rep_XX/in.lammps`, writes `params.json`, derives seeds |
| `qtft/run.py`, `qtft/launcher.py` | start runs or whole sweeps in the background (one queue, at most `max_parallel` replicas at once), live progress (`watch`, `launcher.log`), stop |
| `qtft/remote.py`, `job_template.slurm` | runs on the LRZ Linux cluster (`Params(executor="lrz")`): LRZ settings (host, folders, Slurm partition, modules; email from the gitignored `lrz_local.json`), the Slurm job script (one array task per replica), and submit / sync / cancel / `fetch` over SSH, used by `start`, `watch`, `stop` (setup: `docs/INSTALL.md`, section 9) |
| `qtft/read.py` | readers for `log.lammps` (YAML thermo), dumps, `rdf.dat`, histograms |
| `qtft/quicklook.py` | quick overview plot of a run: thermo time series, potential energy, virial pressure, mean overlap per pair family |
| `qtft/ensemble.py` | mean ± SD over replicas on a common time grid, half-times, percolation, final-state table, association rate k_a (`binding_rate`, measured and Collins–Kimball; needs a calcium step), bond formation and breaking rates and apparent k_a over time (`bond_rates`, any protocol), number of agglomerates ≥ n over time for several n, in the box or per 1000 µm³, with the size counted in all particles, Qt or Ft (`agglomerates_above_size`, from `traj.dump`), `index.csv` of all runs, comparison of runs |
| `qtft/structure.py` | cluster geometry with OVITO (clusters from bonds, unwrapped across the periodic box; not needed with walls): Rg, compactness, centres, nearest-neighbour distances, composition; percolation (a cluster spanning the box); solid and envelope volume of every agglomerate at the end of the influx (for confocal volumes; numpy/scipy) |
| `qtft/overlap.py` | interpenetration diagnostics per pair family (with walls also particle–wall) |
| `qtft/figures.py` | Nature-style figures (calcium, bond rates, binding rate, overview, agglomerate size and volume distributions, kinetics, agglomerates above size thresholds (count or per 1000 µm³, size counted in all particles, Qt or Ft, 3 × 2 figure), overlap, comparison; multi-panel figures also as single panels, phase lines), final-state table (CSV + LaTeX), `make_report` |
| `qtft/fibsem.py` | export of the final frame in the FIB-SEM segmentation schema (encapsulins, clusters from bonds; periodic box unwrapped, walls in the box frame) |
| `qtft/animation.py` | video of one replica (H.264 via ffmpeg, 1920 × 1080 × `scale`, default 2 = 4K): whole box in 3D and from the front, fixed central magnification, calcium and calmodulin loading and average cluster size over time, statistics, calcium protocol; shaded spheres by a small numpy renderer, layout with matplotlib |
| `qtft/sweep.py` | parameter sweeps: one run per combination of listed values (grid), same seeds for all variants, results table per variant |
| `qtft/tuning.py` | force-constant sweep: constants for a target α, predicted vs measured overlap per pair family, binding kinetics |
| `qtft/benchmark.py` | performance measurement (MPI processes per replica, replicas in parallel) and sizing of `extra_special` |
| `tests/` | checks (`pytest`); the tests marked `slow` run short LAMMPS simulations, `pytest -m "not slow"` skips them |

## Output

Simulation output goes to `runs/<single|ensemble>/<run name>/` (inside the repository,
gitignored). Every output file,
its content and what creates it is listed in [docs/outputs.md](docs/outputs.md).

In OVITO, `radius`, `cluster` and `nbonds` are read as particle properties; colour by
`cluster` with the *Color coding* modifier.
