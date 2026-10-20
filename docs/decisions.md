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
| Diffusion | D_Qt = 2 × 10⁻⁴, D_Ft = 5 × 10⁻⁴ nm²/ns (handover values, not Stokes–Einstein) | keeps Δt = 500 ns well inside the stable range; Stokes–Einstein would be ~50× faster and need smaller Δt |
| Timestep | Δt = 500 ns (reference) | stability values α = 2kΔt(1/γᵢ + 1/γⱼ): Qt–Qt 0.70, Ft–Ft 0.70, Qt–Ft 0.70, bond 0.70 (force constants chosen for α ≈ 0.7, see Overlap and force constants); 1000 ns would give α ≈ 1.4 (overshooting steps) |
| Repulsion | `pair_style harmonic/cut`, E = k (r_c − r)², r_c = r_i + r_j; k_QQ = 7.5, k_FF = 3.0, k_QF = 4.3 zJ/nm² | soft excluded volume allows the large timestep; LAMMPS k includes the ½; set for α ≈ 0.7 at Δt = 500 ns (least overlap with margin to α = 1, see Overlap and force constants; former handover values 3.32 / 2.0 / 1.245); all three pairs set explicitly (default geometric mixing would give a wrong Qt–Ft cutoff) |
| Bond | `bond_style harmonic`, E = K (r − r₀)², K = 4.3 zJ/nm², r₀ = 40.5 nm | set for α ≈ 0.7 like the repulsion (former handover value 1.66); K = k_QF, so a pair inside contact has the same energy before and after bonding (no jump at bond formation) |
| Bonded pairs | `special_bonds lj 1.0e-50 1.0 1.0` | directly bonded pairs: repulsion off (the bond keeps the distance); pairs 2–3 bonds apart (e.g. two Qt on one Ft) keep the repulsion; 1e-50 instead of 0 keeps bonded pairs in the neighbour list so they appear in the RDF (`fix bond/create` still refuses duplicate bonds) |
| Binding | `fix bond/create` every step, Qt–Ft only, R_bind = 40.5 nm, prob p = 1 − exp(−k_on Δt), k_on = 1.86·10⁻⁶ ns⁻¹ (10⁻⁶ × √(4.3/1.245): same kinetics as with the former k_QF, see Overlap and force constants; to be calibrated against experiment); max 60 bonds per Qt, 10 per Ft; loops allowed | time-discretised Doi model; difference to ReaDDy accepted: each atom bonds at most once per step, with its closest eligible partner |
| Bond bookkeeping | `extra/bond/per/atom 60`, `extra/special/per/atom 1000` (LAMMPS reserves 2001 slots); `comm_modify cutoff` = max(2.5 r₀, RDF cutoff) + skin = 141.5 nm (skin 20) | bond/create keeps 1-2/1-3/1-4 neighbour lists and needs partners 2 bonds away as ghosts; actual need measured with `benchmark.special_need` |
| Start | overlap-free random insertion (`create_atoms random … overlap`): Qt with 55 nm, then Ft with 40.5 nm; 1000 steps equilibration without binding | dilute system (φ ≈ 6 %); the Ft–Ft start bias (≥ 40.5 instead of 26 nm) is erased in equilibration (~55 nm RMS Ft displacement) |
| Bond breaking | not implemented yet | planned for later (`fix bond/break` is length-based; rate-based breaking needs another approach) |
| Seeds | per replica 5 independent seeds from `numpy.random.SeedSequence(master_seed)`, range 1…900 000 000 | the LAMMPS Marsaglia generator rejects larger seeds; runs are reproducible |

ReaDDy is not used for validation; the LAMMPS model stands on its own.

## Reference parameters

Only the time step has a reference value: **Δt = 500 ns** (the `Params` default). Particle numbers, box size
and run length are chosen per run in the control notebook; there is no reference system.

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
- Simulation output lives in `runs/` inside the repository but gitignored: runs with one replica
  in `runs/single/<run name>/`, runs with several in `runs/ensemble/<run name>/`, benchmarks in
  `runs/benchmark/`, force-constant sweeps in `runs/tuning/`. `params.json` is the record of a run, `rep_XX/` holds the replicas,
  `analysis/` the results; `runs/index.csv` lists all single and ensemble runs.
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

Open items: size `extra_special` from a long production run (`special_need`); compare the real run speed
with the benchmark projection.

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

Sweep 2026-10-20 (`runs/tuning/sweep/`, control notebook section 6): 400 Qt / 1000 Ft in 928.3 nm (production
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

Decision (2026-10-20): α ≈ 0.7 at Δt = 500 ns, rounded: k_qq = 7.5, k_ff = 3.0, k_qf = bond_k = 4.3 zJ/nm²
(α 0.70 / 0.70 / 0.70 / 0.70; predicted overlap 0.53 / 0.84 / 0.70 nm), k_on = 1.86·10⁻⁶ ns⁻¹. Most of the
reduction possible at this time step, with margin to the α > 1 warning; the run time is unchanged.
