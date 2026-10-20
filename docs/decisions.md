# Model and project decisions

Record of how the Qt–Ft agglomeration model is built in LAMMPS and why, plus the project
conventions. Read this first when continuing the work.

## Physical model

| Topic | Decision | Reason |
|---|---|---|
| Units | `units nano`: nm, ns, energy zJ (= ag nm²/ns²), force pN, mass ag, pressure 1 unit = 1 MPa | parameters appear as physical values without a conversion layer; k_BT(310 K) = 4.28 zJ |
| Boundaries | periodic cubic box (`boundary p p p`) | no walls, no lost atoms; clusters crossing the box edge are unwrapped in the analysis (OVITO) |
| Atom types | 2 types: 1 = Qt (r = 27.5 nm), 2 = Ft (r = 13.0 nm), named by `labelmap` (`traj.dump` writes the names Qt/Ft, `final.data` stores the labels); "bound" = at least one bond | the four ReaDDy reactions collapse to one Qt–Ft rule; no re-typing needed (also later for bond breaking); the names make OVITO show Qt/Ft, the readers convert them back to 1/2. No bound-state types (QtC/FtC): colour by `nbonds` in OVITO instead |
| Dynamics | overdamped Brownian dynamics, `fix brownian` (Euler–Maruyama), Gaussian noise, one fix per species, γ = k_BT/D | particle timescales (µs steps) are ~10⁵ × the momentum relaxation time m/γ, so inertia is irrelevant; velocity Verlet + Langevin would need ~10⁶ × more steps |
| Thermostat | none besides `fix brownian` (friction + noise satisfy fluctuation–dissipation) | `fix brownian` is the only integrator; never add `fix nve/langevin/nvt` |
| Diffusion | D_Qt = 2 × 10⁻⁴, D_Ft = 5 × 10⁻⁴ nm²/ns (handover values, not Stokes–Einstein) | keeps Δt = 1 µs stable; Stokes–Einstein would be ~50× faster and need smaller Δt |
| Timestep | Δt = 1000 ns | stability values α = 2kΔt(1/γᵢ + 1/γⱼ): Qt–Qt 0.62, Ft–Ft 0.93, Qt–Ft 0.41, bond 0.54 (all < 1) |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j; k_QQ = 3.32, k_FF = 2.0, k_QF = 1.245 zJ/nm² | soft excluded volume allows the large timestep; LAMMPS k includes the ½ (handover ½k values halved, ×1.6605 zJ per kJ/mol); k_FF lowered from 2.49 to 2.0 to bring α_FF below 1; all three pairs set explicitly (default geometric mixing would give a wrong Qt–Ft cutoff) |
| Bond | `bond_style harmonic`, E = K (r − r₀)², K = 1.66 zJ/nm², r₀ = 40.5 nm | handover value (option A); energy jump at bond formation (K − k_QF)·δ² ≈ 0.4 k_BT for 2 nm overlap, dissipated immediately in BD |
| Bonded pairs | `special_bonds lj 1.0e-50 1.0 1.0` | directly bonded pairs: repulsion off (the bond keeps the distance); pairs 2–3 bonds apart (e.g. two Qt on one Ft) keep the repulsion; 1e-50 instead of 0 keeps bonded pairs in the neighbour list so they appear in the RDF (`fix bond/create` still refuses duplicate bonds) |
| Binding | `fix bond/create` every step, Qt–Ft only, R_bind = 40.5 nm, prob p = 1 − exp(−k_on Δt), k_on = 10⁻⁶ ns⁻¹; max 60 bonds per Qt, 10 per Ft; loops allowed | time-discretised Doi model; difference to ReaDDy accepted: each atom bonds at most once per step, with its closest eligible partner |
| Bond bookkeeping | `extra/bond/per/atom 60`, `extra/special/per/atom 1000` (LAMMPS reserves 2001 slots); `comm_modify cutoff` = max(2.5 r₀, RDF cutoff) + skin = 141.5 nm (skin 20) | bond/create keeps 1-2/1-3/1-4 neighbour lists and needs partners 2 bonds away as ghosts; actual need measured with `benchmark.special_need` |
| Start | overlap-free random insertion (`create_atoms random … overlap`): Qt with 55 nm, then Ft with 40.5 nm; 1000 steps equilibration without binding | dilute system (φ ≈ 6 %); the Ft–Ft start bias (≥ 40.5 instead of 26 nm) is erased in equilibration (~55 nm RMS Ft displacement) |
| Bond breaking | not implemented yet | planned for later (`fix bond/break` is length-based; rate-based breaking needs another approach) |
| Seeds | per replica 5 independent seeds from `numpy.random.SeedSequence(master_seed)`, range 1…900 000 000 | the LAMMPS Marsaglia generator rejects larger seeds; runs are reproducible |

ReaDDy is not used for validation; the LAMMPS model stands on its own.

## Reference parameter set

Full system: 4000 Qt / 16 000 Ft in a 2000 nm box (c_Qt ≈ 0.83 µM, c_Ft ≈ 3.3 µM), T = 310 K,
500 000 steps = 500 ms production. Prototype/test system with the same concentrations:
400 / 1600 in a 928.3 nm box, 50 000 steps.

## What is computed where

| Quantity | Where |
|---|---|
| bonds, free/bound counts, fraction bound, number of clusters, mean/largest size, clusters ≥ 5, size-category fractions, mean/max coordination, bond lengths, energies, virial pressure | LAMMPS during the run (`compute nbond/atom`, `fragment/atom`, `chunk/atom`, …), YAML thermo in `log.lammps` |
| Qt–Ft RDF, coordination histograms | LAMMPS (`rdf.dat`, `coord_*.hist`) |
| mean ± SD over replicas, half-times, percolation | `qtft/ensemble.py` |
| Rg, compactness, cluster centres, nearest-neighbour distances, composition (need unwrapping across the periodic box) | OVITO Python module, `qtft/structure.py` |
| overlap diagnostics | `qtft/overlap.py` (from `traj.dump`) |
| figures, tables | `qtft/figures.py` (Nature style as in the handover) |
| FIB-SEM export | `qtft/fibsem.py` |

`ke`, `temp` and `etotal` are meaningless in overdamped dynamics and are not logged.

## Project conventions

- All parameters are set in one place: the parameter cell of `control.ipynb` (defaults in
  `qtft/params.py`). The LAMMPS input is generated from `template.lmp`; edit the template only
  when the model changes.
- Simulation output lives in `qtft-data/` inside the repository but gitignored: runs with one replica
  in `qtft-data/single/<run name>/`, runs with several in `qtft-data/ensemble/<run name>/`, benchmarks in
  `qtft-data/benchmark/`. `params.json` is the record of a run, `rep_XX/` holds the replicas,
  `analysis/` the results; `qtft-data/index.csv` lists all single and ensemble runs.
- Notebooks are committed without outputs (`nbstripout`).
- Commits are authored as Benedikt Pfluger <benedikt.pfluger@tum.de>; no co-author or
  "generated with" lines in commits.
- All work is committed directly on `main`; no feature branches or pull requests.
- LAMMPS documentation is the reference for every command (docs.lammps.org).

## Status

| Phase | Content | State |
|---|---|---|
| 0 | setup: README, INSTALL, environment | merged |
| 1 | simulation prototype: template, parameters, builder, runner, control notebook | merged |
| 2 | performance benchmark, special-list sizing | merged; benchmark run 2026-10-20 (see Performance) |
| 3 | ensemble statistics, run index, run comparison | merged |
| 4 | structure (OVITO) and overlap analysis, figures, tables | merged |
| 5 | FIB-SEM export, this document | merged |
| later | bond breaking and phase schedules, parameter sweeps, HPC cluster runs | open |

Open items: size `extra_special` from a full 500 ms run (`special_need`); check the full-system results;
compare the real run speed with the benchmark projection.

## Performance (local machine)

Benchmark 2026-10-20 on the local machine (Intel i7-10700K: 8 physical cores, 16 hardware threads, WSL2),
system 4000 Qt / 10 000 Ft in a 2000 nm box, Δt = 500 ns, 2000 production steps per setting
(`qtft-data/benchmark/{16,8}cores/results.csv`).

| Setting | Result | Decision |
|---|---|---|
| neighbour skin | skin 20 is 20–45 % faster than skin 10 in every configuration: list rebuilds every 6.9 instead of 2.7 steps (Brownian jumps ≈ 1.2 nm/step), while the dilute system has < 1 neighbour per particle, so longer lists cost almost nothing | `neighbor_skin = 20` |
| MPI processes per replica | for 14 000 particles more processes per replica pay off (1 → 8 procs: 125 → 1223 steps/s per replica with the machine full); best ensemble throughput with 4 procs × 4 replicas on 16 threads (3540 replica-steps/s) | ensembles: `mpi_procs = 4`, `max_parallel = 4`; a single replica is fastest with 8 procs (≈ 2070 steps/s) |
| hyperthreading | small effect: helps only with 4 procs/replica (3540 vs 2510 replica-steps/s), with 1 proc/replica 16 threads are slower than 8 cores | use all 16 threads with the setting above |
| time budget | ≈ 60 % of the time in `Modify` (`fix brownian`, `fix bond/create` every step, in-run cluster statistics) | candidate for optimisation if runs get long |

Projected time for 500 ms (10⁶ steps) with these settings: ≈ 18 min per replica, 4 at once (≈ 37 min for 8
replicas). The benchmark measures the start of a run with almost no bonds; real runs get slower as the networks
grow, so these are lower bounds.
