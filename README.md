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
| `template.lmp` | the LAMMPS input template (model structure, no numbers) |
| `qtft/` | Python helpers: parameters, input builder, runner, readers |
| `tests/` | quick checks (`pytest`) |
