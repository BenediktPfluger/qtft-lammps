# qtft-lammps

Coarse-grained Brownian dynamics simulation of the agglomeration of **Qt encapsulin**
(large protein cage, r = 27.5 nm) and **ferritin** (Ft, r = 13 nm) in LAMMPS.
Qt and Ft diffuse in implicit solvent, repel each other softly, and form harmonic
Qt–Ft bonds stochastically when they touch. Clusters grow into Qt–Ft–Qt networks.

## Model in short

| Component | LAMMPS implementation |
|---|---|
| Units | `units nano` (nm, ns, energy in zJ, force in pN) |
| Box | periodic (`boundary p p p`); cubic (`box_length=2000.0`) or rectangular (`box_length=[2000.0, 3000.0, 4000.0]`) |
| Dynamics | overdamped Brownian dynamics, `fix brownian`, γ = k_BT/D per species |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j |
| Bonds | `bond_style harmonic`, E = K (r − r₀)², r₀ = r_Qt + r_Ft |
| Binding | `fix bond/create`, Qt–Ft only, probability 1 − exp(−k_on Δt) per step |
| Start | overlap-free random insertion, short equilibration without binding |

## Usage

1. Install everything as described in [docs/INSTALL.md](docs/INSTALL.md).
2. `conda activate lammps`, open the repo in VS Code.
3. Open `control.ipynb`: set the parameters, check the derived values, build and run. For a parameter
   study, list values per parameter in section 4 (`build_sweep`): one run per combination.

Simulation output is written to `runs/` inside the repository (gitignored, never committed):
`runs/single/` for runs with one replica, `runs/ensemble/` for runs with several,
`runs/sweeps/` for parameter sweeps, `runs/benchmark/` for performance measurements, `runs/tuning/` for
force-constant sweeps, and `runs/index.csv` listing all single, ensemble and sweep runs.

## Repository layout

| Path | Purpose |
|---|---|
| `control.ipynb` | set all parameters, build the LAMMPS input, run, monitor |
| `analysis.ipynb` | ensemble statistics, structure and overlap analysis, figures and tables, comparison of runs, FIB-SEM export |
| `docs/decisions.md` | model and project decisions with reasons, status of the implementation |
| `docs/outputs.md` | every output file: content and what creates it |
| `template.lmp` | the LAMMPS input template (model structure, `{{placeholders}}`, no numbers) |
| `qtft/params.py` | all parameters with defaults, derived values (γ = k_BT/D, α, p, …) and checks |
| `qtft/build.py` | fills the template → `rep_XX/in.lammps`, writes `params.json`, derives seeds |
| `qtft/run.py`, `qtft/launcher.py` | start runs or whole sweeps in the background (one queue, at most `max_parallel` replicas at once), live progress (`watch`, `launcher.log`), stop |
| `qtft/read.py` | readers for `log.lammps` (YAML thermo), dumps, `rdf.dat`, histograms |
| `qtft/quicklook.py` | quick overview plot of a run: thermo time series, potential energy, virial pressure, mean overlap per pair family |
| `qtft/ensemble.py` | mean ± SD over replicas on a common time grid, half-times, percolation, final-state table, `index.csv` of all runs, comparison of runs |
| `qtft/structure.py` | cluster geometry with OVITO (clusters from bonds, unwrapped across the periodic box): Rg, compactness, centres, nearest-neighbour distances, composition |
| `qtft/overlap.py` | interpenetration diagnostics per pair family |
| `qtft/figures.py` | Nature-style figures (overview, kinetics, large agglomerates, overlap, journal panels g–j, comparison), final-state table (CSV + LaTeX), `make_report` |
| `qtft/fibsem.py` | export of the final frame in the FIB-SEM segmentation schema (encapsulins, clusters from bonds, unwrapped) |
| `qtft/sweep.py` | parameter sweeps: one run per combination of listed values (grid), same seeds for all variants, results table per variant |
| `qtft/tuning.py` | force-constant sweep: constants for a target α, predicted vs measured overlap per pair family, binding kinetics |
| `qtft/benchmark.py` | performance measurement (MPI processes per replica, replicas in parallel) and sizing of `extra_special` |
| `tests/` | quick checks (`pytest`); includes a short end-to-end LAMMPS run |

## Output

Simulation output goes to `runs/<single|ensemble>/<run name>/` (inside the repository,
gitignored). Every output file,
its content and what creates it is listed in [docs/outputs.md](docs/outputs.md).

In OVITO, `radius`, `cluster` and `nbonds` are read as particle properties; colour by
`cluster` with the *Color coding* modifier.
