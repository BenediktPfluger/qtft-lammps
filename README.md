# qtft-lammps

Coarse-grained Brownian dynamics simulation of the agglomeration of **Qt encapsulin**
(large protein cage, r = 27.5 nm) and **ferritin** (Ft, r = 13 nm) in LAMMPS.
Qt and Ft diffuse in implicit solvent, repel each other softly, and form harmonic
Qt–Ft bonds stochastically when they touch. Clusters grow into Qt–Ft–Qt networks.

## Model in short

| Component | LAMMPS implementation |
|---|---|
| Units | `units nano` (nm, ns, energy in zJ, force in pN) |
| Box | cubic, periodic (`boundary p p p`) |
| Dynamics | overdamped Brownian dynamics, `fix brownian`, γ = k_BT/D per species |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j |
| Bonds | `bond_style harmonic`, E = K (r − r₀)², r₀ = r_Qt + r_Ft |
| Binding | `fix bond/create`, Qt–Ft only, probability 1 − exp(−k_on Δt) per step |
| Start | overlap-free random insertion, short equilibration without binding |

## Usage

1. Install everything as described in [INSTALL.md](INSTALL.md).
2. `conda activate lammps`, open the repo in VS Code.
3. Open `control.ipynb`: set the parameters, check the derived values, build and run.

Simulation output is written to `~/qtft-data/` (outside the repository).

## Repository layout

| Path | Purpose |
|---|---|
| `control.ipynb` | set all parameters, build the LAMMPS input, run, monitor |
| `analysis.ipynb` | ensemble statistics, structure and overlap analysis, figures and tables, comparison of runs |
| `template.lmp` | the LAMMPS input template (model structure, `{{placeholders}}`, no numbers) |
| `qtft/params.py` | all parameters with defaults, derived values (γ = k_BT/D, α, p, …) and checks |
| `qtft/build.py` | fills the template → `rep_XX/in.lammps`, writes `params.json`, derives seeds |
| `qtft/run.py`, `qtft/launcher.py` | start runs in the background, progress, stop |
| `qtft/read.py` | readers for `log.lammps` (YAML thermo), dumps, `rdf.dat`, histograms |
| `qtft/quicklook.py` | quick overview plot of a run |
| `qtft/ensemble.py` | mean ± SD over replicas on a common time grid, half-times, percolation, final-state table, `index.csv` of all runs, comparison of runs |
| `qtft/structure.py` | cluster geometry with OVITO (clusters from bonds, unwrapped across the periodic box): Rg, compactness, centres, nearest-neighbour distances, composition |
| `qtft/overlap.py` | interpenetration diagnostics per pair family |
| `qtft/figures.py` | Nature-style figures (overview, kinetics, large agglomerates, overlap, journal panels g–j, comparison), final-state table (CSV + LaTeX), `make_report` |
| `qtft/benchmark.py` | performance measurement (MPI processes per replica, replicas in parallel) and sizing of `extra_special` |
| `tests/` | quick checks (`pytest`); includes a short end-to-end LAMMPS run |

## Output of one replica (`~/qtft-data/<run name>/rep_XX/`)

| File | Content |
|---|---|
| `in.lammps` | generated LAMMPS input, all numbers filled in |
| `log.lammps` | LAMMPS log; thermo every `output_every` steps in YAML: energies, bonds, free/bound counts, cluster statistics, coordination, bond lengths, speed |
| `traj.dump` | trajectory for OVITO: `id type x y z ix iy iz radius cluster nbonds` |
| `bonds.dump` | bond list (`atom1 atom2`) per frame |
| `rdf.dat` | Qt–Ft radial distribution function per frame |
| `coord_qt.hist`, `coord_ft.hist` | distribution of bonds per Qt / per Ft per frame |
| `final.data` | final state incl. all bonds (opens directly in OVITO, can restart a run) |
| `status`, `lammps.out` | run status and MPI/launcher messages |

In OVITO, `radius`, `cluster` and `nbonds` are read as particle properties; colour by
`cluster` with the *Color coding* modifier.
