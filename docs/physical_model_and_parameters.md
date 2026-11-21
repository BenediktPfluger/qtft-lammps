# Physical model and simulation parameters

This page describes the coarse-grained model of Qt–Ft agglomeration that the simulation solves, and the parameters
of the current runs:

- **Part I, physical model:** particles, dynamics, interactions, bonds, binding, unbinding and the calcium control,
  with the equations and the LAMMPS commands that implement them.
- **Part II, simulation parameters:** the values of the current runs in two variants, an **optoCRAC cycle in
  HEK293T cells** (`01_simulate_hek.ipynb`) and a **spontaneous network event in a neuron culture**
  (`02_simulate_neurons.ipynb`), and the values derived from them.

Units are those of LAMMPS `units nano` (section 1); the calcium protocol is given in s, 1/s and µM. Parameter names
are the fields of `Params` in [`qtft/params.py`](../qtft/params.py), the single source of all values; the LAMMPS
input is generated from [`template.lmp`](../template.lmp). For the files a run writes, see
[`repo_structure.md`](repo_structure.md); for the setup, [`install.md`](install.md).

| Component | Choice |
|---|---|
| Particles | two species of hard-core-like spheres: Qt encapsulins (r = 27.5 nm) and Ft ferritins (r = 13.0 nm) |
| Dynamics | overdamped Brownian dynamics (Euler–Maruyama), translation and, with anchored bonds, rotation |
| Excluded volume | soft harmonic repulsion between all particles |
| Bonds | Qt–Ft only; **anchored** (default): each bond end fixed in its particle, bending penalty; **free**: central harmonic bond |
| Binding | reaction at contact with a finite rate (Collins–Kimball), rate ∝ calmodulin loading |
| Unbinding | every bond breaks independently (Poisson process), rate set by the calmodulin loading |
| Calcium control | baseline → influx → pump-out, calmodulin loading by a Hill equation, computed inside LAMMPS |
| Integrators | `fix brownian`, with anchored bonds `fix brownian/asphere`; no other integrator or thermostat |
| Own LAMMPS styles | plugin `qtftstyles.so`: `bond_style harmonic/anchor`, `fix bond/break/random` |

Contents:

- Part I: physical model
  1. [System and units](#1-system-and-units)
  2. [Brownian dynamics](#2-brownian-dynamics)
  3. [Excluded volume](#3-excluded-volume)
  4. [Bonds](#4-bonds)
  5. [Binding](#5-binding)
  6. [Unbinding](#6-unbinding)
  7. [Calcium and calmodulin](#7-calcium-and-calmodulin)
  8. [Start, equilibration and seeds](#8-start-equilibration-and-seeds)
  9. [Assumptions and limits](#9-assumptions-and-limits)
- Part II: simulation parameters
  10. [Overview](#10-overview)
  11. [Common parameters](#11-common-parameters)
  12. [HEK293T cells: optoCRAC cycle](#12-hek293t-cells-optocrac-cycle)
  13. [Neurons: spontaneous network event](#13-neurons-spontaneous-network-event)
  14. [Derived values](#14-derived-values)
  15. [Differences to the `Params` defaults](#15-differences-to-the-params-defaults)
- Reference: [Symbols](#symbols), [Literature](#literature)

---

# Part I: physical model

## 1. System and units

Two protein cages are simulated as spheres:

| Particle | LAMMPS type | Construct | Radius | Binding partner on the surface |
|---|---|---|---|---|
| Qt | 1 (`Qt`) | Qt encapsulin, Qt–M–eUnaG–NES–RS20 | `radius_qt` = 27.5 nm | RS20 peptide (240 per Qt) |
| Ft | 2 (`Ft`) | calmodulin–mScarlet–ferritin | `radius_ft` = 13.0 nm | calmodulin (24 per Ft) |

The radii are effective radii from AlphaFold 3 models of the constructs, measured in ChimeraX. A Qt and an Ft bind
when the calmodulin of the Ft, loaded with Ca²⁺, binds an RS20 peptide of the Qt; in the model this is one bond
between the two particle centres (sections 4 to 6). Qt–Qt and Ft–Ft pairs never bind.

**Units** (`units nano`; every parameter appears as a physical value, there is no conversion layer):

| Quantity | Unit |
|---|---|
| length | nm |
| time | ns (the calcium protocol and the rates in `Params`: s and 1/s) |
| energy | zJ = 10⁻²¹ J = ag nm²/ns² |
| force | pN |
| mass | ag (attogram; irrelevant in overdamped dynamics) |
| temperature | K; k_BT = 4.28 zJ at 310 K |
| concentration | µM (calcium), particles per µm³ (Qt, Ft) |

**Box.** A periodic box by default, cubic (`box_length` = one edge) or rectangular (`[Lx, Ly, Lz]`). The particle
amounts are number densities per µm³ of the box (`qt_per_um3`, `ft_per_um3`), the experimental quantity; the
numbers are

```
N_Qt = round(ρ_Qt · V_box),    N_Ft = round(ρ_Ft · V_box)
```

so a change of the box size keeps the densities. Molar concentrations c = N / (N_A V) and volume fractions
φ = N · (4/3)π r³ / V are among the derived values (section 14).

**Walls** (`boundary`, alternative to the periodic box, on all six faces):

- `"reflect"` (`fix wall/reflect`, per species at the distance r_i from each face): a centre crossing the plane is
  mirrored back, so the sphere touches the face; the no-flux boundary of diffusion, without parameters.
- `"harmonic"` (`fix wall/harmonic`, per species, cutoff r_i): E = k (r_i − d)² for a centre–face distance d < r_i,
  `wall_k_qt` = 15, `wall_k_ft` = 6 zJ/nm² (stability α ≈ 0.7 like the pair constants, section 2).

---

## 2. Brownian dynamics

### Translation

At the length and time scales of the model (nm, µs) the momentum of a particle relaxes within m/γ (picoseconds or
less), more than 10⁵ times faster than a time step. Inertia is therefore irrelevant, and every particle follows the **overdamped
Langevin equation**

```
γ dr/dt = F(r) + √(2 k_BT γ) · η(t),        γ = k_BT / D
```

with the force F from all interactions, the friction γ and Gaussian white noise η (zero mean, ⟨ηᵢ(t) ηⱼ(t')⟩ =
δᵢⱼ δ(t − t')). `fix brownian` integrates it with the **Euler–Maruyama** step

```
r(t + Δt) = r(t) + F Δt / γ + √(2 D Δt) · ξ,        ξ ~ N(0, 1) per axis
```

Friction and noise satisfy the fluctuation–dissipation relation, so the fix also acts as the thermostat; there is
no other integrator or thermostat. The noise is Gaussian (`rng gaussian`). One fix per species (`fix bd_qt Qt
brownian …`, `fix bd_ft Ft brownian …`) sets its γ.

The diffusion coefficients `D_qt` = 2·10⁻⁴ and `D_ft` = 5·10⁻⁴ nm²/ns are taken over from the earlier ReaDDy model
of the project. They are ≈ 60 × below Stokes–Einstein in water (≈ 1.2·10⁻² nm²/ns for Qt at 310 K); with
Stokes–Einstein values the time step would have to be ≈ 60 × smaller for the same stability (below).

### Rotation (anchored bonds)

With anchored bonds (section 4) every particle has an **orientation**, a unit quaternion q, and R(q) is the
rotation matrix from the particle's body frame to the lab frame. The particles are spheres stored as ellipsoids
with three equal axes (`atom_style hybrid bond ellipsoid`, `set type … shape 2r 2r 2r`): only ellipsoids carry a
quaternion in LAMMPS, and three equal axes keep the shape a sphere. `fix brownian/asphere` integrates translation
and rotation together: at every step the angular velocity

```
ω = τ / γ_rot + √(2 k_BT / (γ_rot Δt)) · ξ,        γ_rot = k_BT / D_rot,    D_rot = 3 D / (4 r²)
```

(torque τ from the bonds, ξ Gaussian per body axis) turns the quaternion by ω Δt, and q is normalised again.
D_rot follows from D with the Stokes–Einstein ratio of rotational and translational diffusion of a sphere,
D_rot / D = 3 / (4 r²). The start orientations are random (`set type … quat/random`, own seeds per species). Free
runs have no orientation and use `fix brownian`.

### Time step and stability

The step Δt = `timestep` = 500 ns is limited by the stiffest spring. For a harmonic spring of constant k between two
particles (energy k x²), one Euler–Maruyama step multiplies the deviation x by 1 − α with the **stability number**

```
α = 2 k Δt (1/γᵢ + 1/γⱼ)
```

The step is stable for α < 2, without overshoot for α < 1. All force constants are chosen for **α ≈ 0.7**
(sections 3 and 4), a margin to 1 at the least overlap (section 3); Δt = 1000 ns would give α ≈ 1.4. A finite α
also widens the thermal fluctuations of a spring by the factor 1 / (1 − α/2) in ⟨x²⟩ (≈ 1.54 at α = 0.7).

For the rotation, the bending torques of all bonds of a particle add up; the corresponding number at the
coordination cap (section 5) is

```
α_rot = maxbond · K_anchor · Δt / γ_rot          (0.36 for Qt, 0.30 for Ft)
```

`check()` warns above 0.7 and refuses above 1.5. The RMS step per axis, √(2DΔt), is 0.45 nm (Qt) and 0.71 nm (Ft).

---

## 3. Excluded volume

All pairs repel each other softly when their spheres overlap (`pair_style harmonic/cut`):

```
E(r) = k (r_c − r)²    for r < r_c = rᵢ + rⱼ,      0 beyond
```

(LAMMPS's k includes the ½). The three pair families are set explicitly (geometric mixing would give a wrong Qt–Ft
cutoff):

| Pair | Constant | r_c | α |
|---|---|---|---|
| Qt–Qt | `k_qq` = 7.5 zJ/nm² | 55.0 nm | 0.70 |
| Ft–Ft | `k_ff` = 3.0 zJ/nm² | 26.0 nm | 0.70 |
| Qt–Ft | `k_qf` = 4.3 zJ/nm² | 40.5 nm | 0.70 |

A soft repulsion allows the large time step.

**Overlap.** Every overlapping pair sits in a harmonic well E = k x² (x = overlap depth) driven by thermal noise.
The mean depth of overlapping pairs is

```
⟨x⟩ = √(2/π) · √( k_BT / (2k) / (1 − α/2) )
```

with the Euler–Maruyama widening 1/(1 − α/2). Stiffer constants reduce the overlap, but α grows with k: at a fixed
time step the overlap is smallest at α ≈ 1, at a fixed α it scales with √Δt. The constants were chosen from an
overlap sweep for α ≈ 0.7 (`qtft/tuning.py`: `predicted_overlap`, `constants_for_alpha`).

**The repulsion sets the binding kinetics.** A Qt–Ft pair binds only within R_bind = r_Qt + r_Ft, that is while it
overlaps (section 5). A softer Qt–Ft repulsion lets pairs stay inside longer: the early binding rate scales as
1/√k_QF. k_on is therefore calibrated after the constants are fixed, and stiffer constants would need k_on raised
by √(k_QF,new / k_QF,old) for the same kinetics.

**Bonded pairs** feel no repulsion (`special_bonds lj 1.0e-50 1.0 1.0`): the bond keeps their distance. Pairs two
or three bonds apart (for example two Qt on the same Ft) still repel each other. The factor 10⁻⁵⁰ instead of 0
keeps bonded pairs in the neighbour list and in the radial distribution function.

---

## 4. Bonds

A bond joins one Qt and one Ft. Its **radial part** is the same in both bond geometries:

```
E_radial = K (r − r₀)²,      K = bond_k = 4.3 zJ/nm²,   r₀ = r_Qt + r_Ft = 40.5 nm
```

K equals k_QF, so a pair inside contact has the same energy before and after bonding; α ≈ 0.70 like the
repulsion.

`bond_geometry` chooses how the bond is attached to the particles:

- **Anchored** (default): each bond end is fixed at a point of its particle. A bound Ft cannot slide over the Qt
  surface; it can only wobble about its attachment point.
- **Free**: point particles and a central bond (`bond_style harmonic`). Nothing fixes where on the surfaces two
  particles are joined, so a bound Ft slides freely over the Qt surface, and particles with several bonds rearrange
  until the bonds pull them into compact, multiply bridged clusters. Free runs are ≈ 1.6 × faster.

### Anchored bonds

`bond_style harmonic/anchor` (own plugin, `lammps_plugins/qtft_styles/`) adds a bending term for each end:

```
E = K (r − r₀)² + K_qt (1 − cos θ_qt) + K_ft (1 − cos θ_ft)
```

θ is the angle between the bond and the anchor of that end. 1 − cos θ is smooth, bounded and ≈ θ²/2 for small
angles. There is no twist term: a particle with one bond spins freely about it, as a flexible linker would allow.

### Anchors and quaternions

```
              a_Qt                                  a_Ft
             ↗                                        ↖
   Qt  ●──────────────────────── b̂ ────────────────────→  ●  Ft
     (q_Qt)                                             (q_Ft)

   θ_Qt: angle between a_Qt and b̂          θ_Ft: angle between a_Ft and −b̂
   a = R(q) · s, the anchor s fixed in the body frame of its particle
```

- **When a bond forms** (its first evaluation), the unit bond vector b̂ (from Qt to Ft) is stored once in the body
  frame of each particle:

  ```
  s_Qt = R(q_Qt)ᵀ · b̂,        s_Ft = R(q_Ft)ᵀ · (−b̂)
  ```

  These **anchors** are fixed relative to their particle and turn with it. RS20 and calmodulin sit all over the
  surfaces, so a pair binds where it touches.
- **At every step** the anchors are turned back into the lab frame, a = R(q) · s, and compared with the current bond
  direction:

  ```
  cos θ_Qt = a_Qt · b̂,        cos θ_Ft = −a_Ft · b̂
  ```

  The energy gives forces perpendicular to the bond on both particles, which bend the bond back towards the anchors,
  and torques that turn the particles back:

  ```
  τ_Qt = K_qt (a_Qt × b̂),        τ_Ft = K_ft (a_Ft × (−b̂))
  ```

  The torques enter the rotation of `fix brownian/asphere` (section 2).
- **Storage:** the two anchors are 6 numbers per bond in a per-bond history (`fix BOND_HISTORY`). When a bond breaks,
  the later bonds of that atom shift down in its bond list and their anchors move with them, and the freed slot is
  cleared, so a bond formed there later gets new anchors. Binary restart files keep the anchors; `read_data` sets new
  ones from the current geometry (a data file has no section for them).
- **Output:** `bonds.dump` of anchored runs holds θ_qt and θ_ft of every bond (`compute bond/local b1 b2`), from which
  the analysis measures the wobble δ (`anchor_wobble`).

### Wobble δ and share

The bond may tilt against the anchor of each end by an angle θ, at the cost E = K (1 − cos θ) ≈ K θ²/2. Two angular
degrees of freedom per end give, in thermal equilibrium,

```
⟨θ²⟩ = 2 k_BT / K
```

A tilt θ moves the attachment point sideways over the surface by u = r θ (r the radius of that particle). The
**wobble δ** (`anchor_wobble_nm`) is the RMS sideways displacement of both attachment points together:

```
δ² = ⟨u_Qt²⟩ + ⟨u_Ft²⟩ = 2 k_BT (r_Qt² / K_qt + r_Ft² / K_ft)
```

δ measures how much play the binding link has at the contact; free bonds are the limit δ → ∞. The **share**
(`anchor_wobble_share_qt`) is the part of δ² that comes from the Qt end, the rest from the Ft end:

```
⟨u_Qt²⟩ = share · δ²,                     ⟨u_Ft²⟩ = (1 − share) · δ²
K_qt = 2 k_BT r_Qt² / (share · δ²),       K_ft = 2 k_BT r_Ft² / ((1 − share) · δ²)
```

δ and the share are set; `Params` derives the stiffnesses. With δ = 5 nm and share 0.5: K_qt = 517.9 zJ,
K_ft = 115.7 zJ, both points move 3.54 nm (RMS), and the smaller Ft needs about twice the tilt angle for that
(θ_rms 15.6° against 7.4° for Qt). The basis of δ ≈ 5 nm: the constructs join the folded domains by short GS
linkers, which leave a few nm of play at the contact. The share has hardly any effect on the agglomerates, δ a
moderate one (larger δ: more bonds per particle, more compact clusters, towards the free limit).

### Bond energy per bond

Thermo `ebond_per_bond` (bond energy / number of bonds). By equipartition a relaxed bond has ½ k_BT per degree of
freedom: ½ k_BT for a free bond (one radial degree), 5/2 k_BT for an anchored bond (one radial and two angular per
end). At Δt = 500 ns the radial fluctuations are widened by 1/(1 − α/2) ≈ 1.54 (α = 0.70), the angular ones by
≈ 1.02 (α per bond ≈ 0.035), so ≈ 0.77 k_BT free and ≈ 2.8 k_BT anchored are expected; measured: 0.85 and
2.84–2.90 k_BT. The small excess in free runs comes from the collective modes of particles with several bonds.

---

## 5. Binding

Bonds form with `fix bond/create` (Qt–Ft only) between pairs closer than R_bind = `bind_radius` = r_Qt + r_Ft =
40.5 nm, that is, at contact. Every `bind_every` steps each eligible pair binds with the probability

```
p_on = 1 − exp(−k_on · f · Δt · bind_every)
```

with the calmodulin loading f(t) (section 7) and `k_on` = 1.86·10⁻⁶ ns⁻¹ (1860 s⁻¹) at full loading f = 1.

- **Coordination caps:** at most `maxbond_qt` = 30 bonds per Qt and `maxbond_ft` = 10 per Ft. Rings of bonds are
  allowed.
- **Pair choice** (a rule of `fix bond/create`): each atom forms at most one bond per attempt, and only with a
  partner that also has it as its closest eligible partner. This removes 2–3 % of the contact pairs per attempt.
- R_bind cannot exceed the Qt–Ft pair cutoff (a requirement of `fix bond/create`).

### Collins–Kimball kinetics

Binding is a reaction at contact with a finite rate, the **Collins–Kimball** model: a partially reactive surface
at R = 40.5 nm, approached by relative diffusion with D = D_Qt + D_Ft.

```
κ   = (rate per pair) · (contact pairs) · V / (N_Qt N_Ft)     intrinsic, reaction-limited rate
k_D = 4π D R                                                    diffusion-limited (Smoluchowski) rate
k_a = k_D κ / (k_D + κ)                                         association rate constant (steady state)
```

The rate per pair is p_on / (`bind_every` Δt) ≈ k_on f; (contact pairs) · V / (N_Qt N_Ft) is the volume of the
overlap layer, the "contact volume". Here k_D ≈ 2.1·10⁸ M⁻¹ s⁻¹, so binding is mostly reaction-limited.

- **k_on is not a physical constant:** κ depends on k_QF and Δt through the overlap layer (section 3). The quantity
  to compare with experiment is **k_a in M⁻¹ s⁻¹** (`binding_rate`).
- k_a is a rate per Qt–Ft *particle* pair; a rate per binding site would divide by the number of sites.
- **Measured** at 20 Qt / 80 Ft per µm³ after a calcium step to f = 0.89: k_a ≈ 1.3·10⁷ M⁻¹ s⁻¹, the same for
  anchored and free bonds within the statistics (±10 %), ≈ 40 % below the continuum Collins–Kimball estimate
  (2.0·10⁷).

---

## 6. Unbinding

Every bond breaks independently, a **Poisson process per bond** with the rate

```
k_off(t) = f · k_off,Ca + (1 − f) · k_off,free
```

(`k_off_ca_per_s`: bond with Ca²⁺-loaded calmodulin; `k_off_free_per_s`: Ca²⁺-free calmodulin). Every
`break_every` = 10 steps each bond breaks with

```
p_off = 1 − exp(−k_off(t) · Δt · break_every)
```

evaluated at every attempt, so k_off follows the calcium continuously.

The default, `bond_break = "random"`, is `fix bond/break/random` (own plugin): every bond gets its own random
number per attempt, an atom can lose several bonds at once, the broken pairs are gathered from all MPI ranks, and the
special lists and neighbour lists are updated in the same step. With anchored bonds it moves the per-bond anchors
with their bonds (section 4).

`bond_break = "longest"` selects the built-in `fix bond/break`, kept for comparison with older free runs: each atom
can only lose its longest bond, and only if it is also the partner's longest, so the effective rate is ≈ 0.1 ×
k_off and falls with the coordination. It is not allowed with anchored bonds.

---

## 7. Calcium and calmodulin

The Qt–Ft link is calmodulin–RS20, which forms only with Ca²⁺-loaded calmodulin. In the experiment a calcium influx
triggers agglomeration, and after the influx the cell pumps the calcium out again. Every production run follows such
a **calcium protocol**, computed inside LAMMPS from the step number:

```
Ca(t) = Ca_rest                                          baseline
      = Ca_peak − (Ca_peak − Ca_rest) · exp(−λ_on t')    influx,   λ_on = ln(100) / ca_rise_s  (a step if ca_rise_s = 0)
      = Ca_rest + (Ca_start − Ca_rest) · exp(−t'/τ)      pump-out, Ca_start = calcium at the end of the influx

f(t)     = Caⁿ / (Kⁿ + Caⁿ)                              calmodulin loading (Hill)
k_on(t)  = k_on · f                                      binding rate per Qt–Ft pair in contact
k_off(t) = f · k_off,Ca + (1 − f) · k_off,free           unbinding rate per bond
```

t' is the time since the start of the phase; the phases last `t_baseline_s`, `t_influx_s` and `t_pumpout_s` (each
may be 0) and start at whole steps. Ca_rest, Ca_peak, K, n, ca_rise_s and τ are `ca_rest_uM`, `ca_peak_uM`,
`ca_K_uM`, `ca_hill`, `ca_rise_s` and `ca_tau_s`.

- **Loading f** is the fraction of calmodulin with calcium bound, a mean field: every bond sees the average loading.
  Calcium binds calmodulin within milliseconds, so f follows Ca(t) at once. With K = 0.5 µM and n = 3:
  f = 0.008 at 0.1 µM (binding below 1 % of its maximum) and f = 0.89 at 1 µM.
- **Unloaded calmodulin dominates unbinding** even at the peak: k_off = 0.029 s⁻¹ at 1 µM (bond lifetime ≈ 34 s)
  and 0.198 s⁻¹ at rest. The speed of deagglomeration is set mainly by k_off,free and τ.
- **Segments.** `fix bond/create` takes a fixed probability, so the production runs in segments of
  `rate_update_every` steps; each segment re-creates the binding fix with p_on at its start. k_on therefore follows
  the calcium in steps of one segment (50 ms in the current runs), k_off at every unbinding attempt (5 µs).

**optoCRAC rise and decay (HEK293T).** Calcium enters through ORAI1 channels gated by optoCRAC (LOV2–STIM1): light
on opens them, the thermal dark reversion of LOV2 closes them after light-off. With A the fraction of active
channels,

```
dA/dt = k_act I (1 − A) − k_dark A,        Ca = Ca_0 + (Ca_max − Ca_0) A
```

For constant light this gives exactly the protocol above: a rise with λ_on = k_act I + k_dark to the plateau
Ca_peak, and a decay with k_dark = 1/τ. The observables (plateau, rise time to 99 %, decay τ) are the parameters;
the model constants k_act I, k_dark and the effective Ca_max are derived values. The model assumes that the pumps
clear calcium much faster than the channels change.

**Network event (neurons).** A spontaneous burst of action potentials in a neuron culture raises the calcium within
less than 200 ms, holds it for 1–5 s while the neurons fire, and lets it decay with τ ≈ 1–2 s (Mitroshina 2018,
Pozzi 2019, Opitz 2002); the model uses a step rise, a short plateau and an exponential decay.

---

## 8. Start, equilibration and seeds

- **Insertion** without overlap (`create_atoms random … overlap`): first the Qt with a minimum distance of 55 nm,
  then the Ft at 40.5 nm from all particles; with walls, inside the box shrunk by r_i.
- **Equilibration:** `equil_steps` = 5000 steps (2.5 ms) without binding. The system is dilute (volume fraction
  ≈ 0.2 %) and starts close to an equilibrium fluid; only the contact pairs needed for binding are missing at the
  start.
- **Seeds:** every replica has its own seeds for the placement of Qt and Ft, the Brownian noise of Qt and Ft, binding,
  unbinding and the start orientations of Qt and Ft, all derived from `master_seed` (`numpy.random.SeedSequence`).
  The binding seed changes with every segment. Runs are reproducible: the same parameters give the same trajectory
  with the same LAMMPS version and number of MPI processes.

---

## 9. Assumptions and limits

- **Diffusion:** the D values are ≈ 60 × below Stokes–Einstein in water (section 2). Absolute times of the
  diffusion-driven processes scale with D; the binding and unbinding rates are set separately.
- **Binding as one bond per pair:** a Qt–Ft contact can hold several RS20–calmodulin links at once (240 RS20 per Qt,
  24 calmodulin per Ft); the model has at most one bond per pair. Possible effects (slower unbinding of a contact,
  stiffer joints) are not included.
- **Calmodulin loading** is a mean field and instant (section 7).
- **Contact reaction:** where particles are pressed together (inside clusters) they overlap more and bind somewhat
  faster than at an isolated contact (section 3).
- **k_a** is ≈ 40 % below the continuum Collins–Kimball estimate (section 5), a property of the discrete contact
  layer; k_on is not calibrated against an experimental association rate yet.
- **Rates** (k_off, calcium levels, Hill parameters) are literature estimates for calmodulin with RS20-like peptides,
  to be calibrated against the experiment.

---

# Part II: simulation parameters

As of 2026-11-21: the parameter cell of `01_simulate_hek.ipynb` (current working setup) and of
`02_simulate_neurons.ipynb`. All derived values were computed with `Params`.

## 10. Overview

| | HEK293T (optoCRAC) | Neurons (network event) |
|---|---|---|
| Notebook | `01_simulate_hek.ipynb` | `02_simulate_neurons.ipynb` |
| Particles | 540 Qt, 2160 Ft (20 / 80 per µm³) | 960 Qt, 3840 Ft (15 / 60 per µm³) |
| Box | 3000 nm cube (27 µm³), periodic | 4000 nm cube (64 µm³), periodic |
| Bonds | anchored, δ = 5 nm, share 0.5 | anchored, δ = 5 nm, share 0.5 |
| Calcium | 0.1 → 1.0 µM, K = 0.5 µM, Hill 3 | 0.1 → 1.0 µM, K = 0.5 µM, Hill 3 |
| Protocol (baseline / influx / pump-out) | 1 / 30 / 0.5 s | 10 / 2 / 30 s |
| Rise (to 99 %) / decay τ | 5 s / 30 s | step / 1.5 s |
| Production time | 31.5 s (6.3·10⁷ steps) | 42 s (8.4·10⁷ steps) |
| Replicas × MPI processes | 1 × 8 | 10 × 4 |
| Run folder name | `Qt20_Ft80_L3000_kon1860_koff0.008-0.2_anch5nm_Ca0.1-1uM_K0.5_rise5tau30_t1-30-0.5s_dt500ns_periodic` | `Qt15_Ft60_L4000_kon1860_koff0.008-0.2_anch5nm_Ca0.1-1uM_K0.5_rise0tau1.5_t10-2-30s_dt500ns_periodic` |

---

## 11. Common parameters

The same in both variants unless the table says otherwise.

### System and box

| Parameter | HEK293T | Neurons | Meaning |
|---|---|---|---|
| `qt_per_um3` | 20 | 15 | Qt per µm³ of box |
| `ft_per_um3` | 80 | 60 | Ft per µm³ of box (4 × Qt) |
| `box_length` | 3000 nm | 4000 nm | cube edge |
| `boundary` | `"periodic"` | `"periodic"` | no walls (`wall_k_qt` = 15, `wall_k_ft` = 6 zJ/nm² apply only to harmonic walls) |

### Particles and dynamics

| Parameter | Value | Meaning |
|---|---|---|
| `radius_qt` | 27.5 nm | effective Qt radius |
| `radius_ft` | 13.0 nm | effective Ft radius |
| `D_qt` | 2·10⁻⁴ nm²/ns | Qt diffusion coefficient → γ_Qt = 2.14·10⁴ ag/ns |
| `D_ft` | 5·10⁻⁴ nm²/ns | Ft diffusion coefficient → γ_Ft = 8.56·10³ ag/ns |
| `temperature` | 310 K | k_BT = 4.28 zJ |
| `timestep` | 500 ns | Δt |

### Repulsion and bonds

| Parameter | Value | Meaning |
|---|---|---|
| `k_qq` | 7.5 zJ/nm² | Qt–Qt repulsion, r_c = 55 nm |
| `k_ff` | 3.0 zJ/nm² | Ft–Ft repulsion, r_c = 26 nm |
| `k_qf` | 4.3 zJ/nm² | Qt–Ft repulsion, r_c = 40.5 nm |
| `bond_k` | 4.3 zJ/nm² | radial bond constant K |
| `bond_r0` | `None` → 40.5 nm | r₀ = r_Qt + r_Ft |
| `bond_geometry` | `"anchored"` | each bond end anchored in its particle |
| `anchor_wobble_nm` | 5.0 nm | wobble δ → K_qt = 517.9 zJ, K_ft = 115.7 zJ |
| `anchor_wobble_share_qt` | 0.5 | share of δ² from the Qt end |

### Binding and unbinding

| Parameter | Value | Meaning |
|---|---|---|
| `k_on` | 1.86·10⁻⁶ ns⁻¹ (1860 s⁻¹) | binding rate per Qt–Ft pair in contact at f = 1 |
| `bind_radius` | `None` → 40.5 nm | R_bind = r_Qt + r_Ft |
| `bind_every` | 1 | binding attempt every step |
| `maxbond_qt` | 30 | maximum bonds per Qt |
| `maxbond_ft` | 10 | maximum bonds per Ft |
| `k_off_ca_per_s` | 8·10⁻³ s⁻¹ | unbinding with Ca²⁺-loaded calmodulin |
| `k_off_free_per_s` | 0.2 s⁻¹ | unbinding with Ca²⁺-free calmodulin |
| `bond_break` | `"random"` | every bond breaks independently (`fix bond/break/random`) |
| `break_every` | 10 | unbinding attempt every 10 steps (5 µs) |

### Calcium

| Parameter | Value | Meaning |
|---|---|---|
| `ca_rest_uM` | 0.1 µM | resting calcium |
| `ca_peak_uM` | 1.0 µM | calcium plateau during the influx |
| `ca_K_uM` | 0.5 µM | calcium at half calmodulin loading |
| `ca_hill` | 3 | Hill coefficient |
| `rate_update_every` | 10⁵ steps (50 ms) | segment length: k_on follows the calcium in 50 ms steps |

The phase lengths and the shape of the protocol differ (sections 12 and 13).

### Equilibration, output and technical settings

| Parameter | Value | Meaning |
|---|---|---|
| `equil_steps` | 5000 (2.5 ms) | equilibration after the insertion, binding off |
| `output_every` | 2000 steps (1 ms) | thermo and in-run analysis in `log.lammps` |
| `traj_every` | 2·10⁵ steps (100 ms) | `traj.dump`, `bonds.dump`, RDF and histograms |
| `rdf_bins` | 200 | RDF bins |
| `rdf_cutoff` | `None` → 121.5 nm | 3 (r_Qt + r_Ft) |
| `neighbor_skin` | 20 nm | neighbour list skin |
| `extra_special` | 1000 | reserved 1-2/1-3/1-4 neighbours per atom |
| `comm_cutoff` | `None` → 141.5 nm | ghost cutoff, max(2.5 r₀, RDF cutoff) + skin |
| `insert_maxtry` | 1000 | insertion attempts per particle |

### Execution

| Parameter | HEK293T | Neurons | Meaning |
|---|---|---|---|
| `n_replicas` | 1 | 10 | independent replicas (1 → `runs/single/`, more → `runs/ensemble/`) |
| `mpi_procs` | 8 | 4 | MPI processes per replica |
| `max_parallel` | `None` | `None` | LRZ: all replicas at once |
| `master_seed` | 12345 | 12345 | all seeds of a run are derived from it |
| `executor` | `"lrz"` | `"lrz"` | Slurm job on the LRZ Linux cluster |
| `slurm_time` | `1-20:00:00` | `4-20:00:00` | time limit per replica (at most 7 days) |
| `slurm_mem` | 4G | 4G | memory per replica |

---

## 12. HEK293T cells: optoCRAC cycle

One light-induced calcium cycle (section 7): optoCRAC opens ORAI1 channels while the light is on; after light-off the
channels close by the dark reversion of LOV2 and the calcium is pumped out.

| Parameter | Value | Meaning |
|---|---|---|
| `t_baseline_s` | 1 s | resting calcium |
| `t_influx_s` | 30 s | light on: calcium rises to the plateau |
| `t_pumpout_s` | 0.5 s | light off: calcium decays towards rest |
| `ca_rise_s` | 5 s | 99 % of the plateau reached 5 s after light-on (λ_on = 0.92 s⁻¹, rise τ = 1.09 s) |
| `ca_tau_s` | 30 s | decay time constant (measured optoCRAC decay t½ 25–35 s, He 2015) |

- Phases: influx from 1 s to 31 s, pump-out from 31 s to 31.5 s (2·10⁶ / 6·10⁷ / 10⁶ steps).
- Calcium 0.99 µM 5 s after light-on; at the end of the run 0.985 µM (f = 0.88), as the pump-out is short against τ.
- Calmodulin more than half loaded (f > 0.5) for ≈ 29.9 s.
- optoCRAC model constants: k_act I = 0.89 s⁻¹, k_dark = 0.033 s⁻¹, effective Ca_max = 1.03 µM.
- Output: 315 `traj.dump` frames, 31 500 thermo rows; 630 segments.

---

## 13. Neurons: spontaneous network event

One spontaneous calcium event of a neuron network (section 7), without optogenetic stimulation. Only one event is
simulated: after the 30 s pump-out the calcium and the bonds are back at rest before a next event would start
(events come every 10–60 s), so further events would repeat the first. The calcium levels are those of the HEK
cells; the papers report only ΔF/F.

| Parameter | Value | Meaning |
|---|---|---|
| `t_baseline_s` | 10 s | resting calcium |
| `t_influx_s` | 2 s | plateau during the burst |
| `t_pumpout_s` | 30 s | recovery after the event |
| `ca_rise_s` | 0 | instantaneous step (the measured rise is below the imaging resolution) |
| `ca_tau_s` | 1.5 s | decay time constant |

- Phases: event from 10 s to 12 s, decay from 12 s to 42 s (2·10⁷ / 4·10⁶ / 6·10⁷ steps).
- Calcium at the decay start: 1.0 µM; back at rest (0.1 µM) at the end of the run.
- Calmodulin more than half loaded for ≈ 3.2 s; calcium above 0.2 µM for ≈ 5.3 s (the event duration).
- **Time resolution of binding:** k_on is updated every 50 ms; during the decay (τ = 1.5 s) the calcium changes by up
  to ≈ 3 % per segment, so binding follows it with a delay of at most 50 ms. Unbinding follows it at every attempt.
- Output: 420 `traj.dump` frames, 42 000 thermo rows; 840 segments.

---

## 14. Derived values

From `Params.summary()`; identical for both variants:

| Quantity | Value |
|---|---|
| k_BT | 4.28 zJ |
| γ_Qt / γ_Ft | 2.14·10⁴ / 8.56·10³ ag/ns |
| r₀ bond, R_bind | 40.5 nm |
| α Qt–Qt, Ft–Ft, Qt–Ft, bond | 0.701, 0.701, 0.703, 0.703 |
| K_anchor Qt / Ft | 517.9 / 115.7 zJ |
| θ_rms Qt / Ft | 7.37° / 15.58° |
| D_rot Qt / Ft | 1.98·10⁻⁷ / 2.22·10⁻⁶ rad²/ns |
| γ_rot Qt / Ft | 2.16·10⁷ / 1.93·10⁶ zJ ns |
| α_rot Qt / Ft (at the coordination cap) | 0.36 / 0.30 |
| RMS step Qt / Ft per axis | 0.45 / 0.71 nm |
| loading f at rest / peak | 0.008 / 0.889 |
| k_on per pair at rest / peak | 14.8 / 1.65·10³ s⁻¹ |
| k_off per bond at rest / peak | 0.198 / 0.029 s⁻¹ |
| p_bind per attempt (f = 1) | 9.3·10⁻⁴ |
| p_break per attempt (max) | 1.0·10⁻⁶ |
| comm cutoff | 141.5 nm |

Different between the variants:

| Quantity | HEK293T | Neurons |
|---|---|---|
| particles Qt / Ft | 540 / 2160 | 960 / 3840 |
| box volume | 27 µm³ | 64 µm³ |
| c_Qt / c_Ft | 0.033 / 0.133 µM | 0.025 / 0.100 µM |
| φ_Qt / φ_Ft | 0.17 % / 0.074 % | 0.13 % / 0.055 % |
| production time | 31.5 s | 42 s |
| phase steps (baseline / influx / pump-out) | 2·10⁶ / 6·10⁷ / 10⁶ | 2·10⁷ / 4·10⁶ / 6·10⁷ |
| segments (rate updates) | 630 | 840 |
| `traj.dump` frames | 315 | 420 |
| f > 0.5 | ≈ 29.9 s | ≈ 3.2 s |
| size classes (small / medium / large upper bound) | 54 / 270 / 675 particles | 96 / 480 / 1200 particles |

`check()` reports "all OK" for both.

---

## 15. Differences to the `Params` defaults

The defaults in `qtft/params.py` suit short test runs. The current runs change:

| Parameter | Default | HEK293T | Neurons |
|---|---|---|---|
| `qt_per_um3`, `ft_per_um3` | 500, 2000 | 20, 80 | 15, 60 |
| `box_length` | 928.3 nm | 3000 nm | 4000 nm |
| `t_baseline_s` | 10 s | 1 s | 10 s |
| `t_influx_s` | 30 s | 30 s | 2 s |
| `t_pumpout_s` | 90 s | 0.5 s | 30 s |
| `ca_rise_s` | 5 s | 5 s | 0 |
| `ca_tau_s` | 20 s | 30 s | 1.5 s |
| `k_off_ca_per_s` | 10⁻³ s⁻¹ | 8·10⁻³ s⁻¹ | 8·10⁻³ s⁻¹ |
| `k_off_free_per_s` | 1 s⁻¹ | 0.2 s⁻¹ | 0.2 s⁻¹ |
| `rate_update_every` | 2·10⁴ | 10⁵ | 10⁵ |
| `output_every` | 1000 | 2000 | 2000 |
| `traj_every` | 1000 | 2·10⁵ | 2·10⁵ |
| `n_replicas` | 1 | 1 | 10 |
| `mpi_procs` | 1 | 8 | 4 |
| `executor` | `"local"` | `"lrz"` | `"lrz"` |
| `slurm_time` | `3-00:00:00` | `1-20:00:00` | `4-20:00:00` |

All other parameters are at their defaults.

---

# Reference

## Symbols

| Symbol | Meaning | Unit | `Params` field |
|---|---|---|---|
| r_Qt, r_Ft | particle radii | nm | `radius_qt`, `radius_ft` |
| D, D_rot | translational, rotational diffusion coefficient | nm²/ns, rad²/ns | `D_qt`, `D_ft` (D_rot derived) |
| γ, γ_rot | translational, rotational friction | ag/ns, zJ ns | derived |
| k_BT | thermal energy | zJ | `temperature` |
| Δt | time step | ns | `timestep` |
| α, α_rot | stability numbers | – | derived |
| k_QQ, k_FF, k_QF | repulsion constants | zJ/nm² | `k_qq`, `k_ff`, `k_qf` |
| K, r₀ | radial bond constant, rest length | zJ/nm², nm | `bond_k`, `bond_r0` |
| K_qt, K_ft | anchor stiffnesses | zJ | derived from δ and the share |
| θ_qt, θ_ft | tilt of the bond against the anchors | rad | – (in `bonds.dump`) |
| q, R(q) | orientation quaternion, its rotation matrix | – | – |
| s, a | anchor in the body frame, in the lab frame | – | – |
| δ | RMS wobble of both attachment points | nm | `anchor_wobble_nm` |
| share | part of δ² from the Qt end | – | `anchor_wobble_share_qt` |
| R_bind | binding distance | nm | `bind_radius` |
| k_on | binding rate per pair in contact at f = 1 | 1/ns | `k_on` |
| k_off,Ca, k_off,free | unbinding rates per bond | 1/s | `k_off_ca_per_s`, `k_off_free_per_s` |
| κ, k_D, k_a | intrinsic, diffusion-limited and overall association rate constants | M⁻¹ s⁻¹ | measured (`binding_rate`) |
| Ca, f | calcium, calmodulin loading | µM, – | `ca_rest_uM`, `ca_peak_uM` |
| K, n (Hill) | half-loading calcium, Hill coefficient | µM, – | `ca_K_uM`, `ca_hill` |
| τ | pump-out time constant | s | `ca_tau_s` |

## Literature

- **Collins FC, Kimball GE (1949).** "Diffusion-controlled reaction rates." *J Colloid Sci* 4, 425–437. → binding
  at contact with a finite rate (section 5).
- **He L, Zhang Y, Ma G, Tan P, Li Z, Zang S, Wu X, Jing J, Fang S, Zhou L, Wang Y, Huang Y, Hogan PG, Han G, Zhou Y
  (2015).** "Near-infrared photoactivatable control of Ca²⁺ signaling and optogenetic immunomodulation." *eLife* 4,
  e10024. <https://elifesciences.org/articles/10024> → optoCRAC: cytosolic Ca²⁺ rise t½ 23–36 s, decay t½ ≈ 25–35 s,
  amplitude ≈ 500–800 nM.
- **Török K, Trentham DR (1994).** *Biochemistry* 33(43), 12807–12820. → calmodulin–MLCK peptide: off-rate
  0.0079 s⁻¹ (k_off,Ca).
- **Török K, Cowley DJ, Brandmeier BD, Howell S, Aitken A, Trentham DR (1998).** *Biochemistry* 37(17), 6188–6198.
  <https://doi.org/10.1021/bi972773e> → affinity of the RS20-like 20-mer peptide.
- **Johnson JD, Snyder C, Walsh M, Flynn M (1996).** *J Biol Chem* 271(2), 761–767. → Ca²⁺ release from
  calmodulin–peptide complexes 0.1–2.9 s⁻¹ (k_off,free).
- **Persechini A, White HD, Gansz KJ (1996).** *J Biol Chem* 271(1), 62–67. → Ca²⁺ dissociation from
  calmodulin–MLCK complexes.
- **Helassa N et al. (2015)**, *Sci Rep* 5, 15978; **Helassa N, Podor B, Fine A, Török K (2016)**, *Sci Rep* 6,
  38276. → Ca²⁺ affinity of calmodulin–RS20 (GCaMP): K 0.09–0.33 µM, Hill 2.8–4.3.
- **Mitroshina et al. (2018), Pozzi et al. (2019), Opitz et al. (2002).** → calcium events of spontaneous network
  activity in hippocampal and cortical cultures (section 13).
- LAMMPS documentation of the styles used: [`fix brownian`](https://docs.lammps.org/fix_brownian.html) (also
  `brownian/asphere`), [`pair_style harmonic/cut`](https://docs.lammps.org/pair_harmonic_cut.html),
  [`bond_style harmonic`](https://docs.lammps.org/bond_harmonic.html),
  [`fix bond/create`](https://docs.lammps.org/fix_bond_create.html),
  [`fix bond/break`](https://docs.lammps.org/fix_bond_break.html),
  [`special_bonds`](https://docs.lammps.org/special_bonds.html),
  [`atom_style`](https://docs.lammps.org/atom_style.html) (ellipsoid).
