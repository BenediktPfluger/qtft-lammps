"""All simulation parameters (LAMMPS `units nano`) and the values derived from them.

Units: length nm, time ns, energy zJ (= ag nm^2/ns^2), force pN, mass ag,
diffusion coefficient nm^2/ns, friction coefficient ag/ns, temperature K.
Exception: the calcium protocol is given in s, 1/s and uM (unit in the field name).

Production is always calcium-controlled: baseline (resting calcium) -> influx (peak calcium) ->
exponential pump-out; each phase may have duration 0. The calmodulin loading
f = Ca^n / (K^n + Ca^n) sets the binding rate k_on f and the unbinding rate
f k_off,Ca + (1 - f) k_off,free.
"""

from __future__ import annotations

import json
import math
import numbers
import os
import re
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np

K_B = 1.380649e-2        # Boltzmann constant in zJ/K
N_A = 6.02214076e23      # Avogadro constant in 1/mol
NM3_TO_L = 1e-24         # 1 nm^3 in litres

REPO_DIR = Path(__file__).resolve().parent.parent

# settings that do not change the physics: shared by all runs of a sweep (not sweepable) and
# ignored when listing or labelling parameter differences
EXECUTION_SETTINGS = {"n_replicas", "mpi_procs", "max_parallel", "master_seed", "data_dir", "group",
                      "run_name", "lmp_command", "mpirun_command", "executor", "slurm_time", "slurm_mem"}
EXECUTORS = ("local", "lrz")
SLURM_MAX_DAYS = 7      # time limit of the LRZ partition serial_long


def slurm_days(time: str) -> float | None:
    """A Slurm time limit "D-HH:MM:SS" or "HH:MM:SS" in days; None if it has another form."""
    m = re.fullmatch(r"(?:(\d+)-)?(\d+):(\d\d):(\d\d)", time)
    if not m:
        return None
    d, h, mi, sec = (int(x or 0) for x in m.groups())
    return d + (h + (mi + sec / 60) / 60) / 24


def data_path(data_dir: str | Path = "runs") -> Path:
    """Absolute output folder; a relative data_dir is taken relative to the repository."""
    path = Path(os.path.expanduser(data_dir))
    return path if path.is_absolute() else REPO_DIR / path


@dataclass
class Params:
    # --- system --------------------------------------------------------------
    n_qt: int = 400                  # number of Qt encapsulins
    n_ft: int = 1600                 # number of Ft ferritins
    box_length: float | list[float] = 928.3   # box [nm]: one edge (cube) or [Lx, Ly, Lz]
    boundary: str = "periodic"       # "periodic", "reflect" (fix wall/reflect) or "harmonic" (fix wall/harmonic)
    wall_k_qt: float = 15.0          # harmonic wall E = k (r_Qt - d)^2, d = centre-wall distance [zJ/nm^2]
    wall_k_ft: float = 6.0           # same for Ft [zJ/nm^2]

    # --- particles -----------------------------------------------------------
    radius_qt: float = 27.5          # effective Qt radius [nm]
    radius_ft: float = 13.0          # effective Ft radius [nm]
    D_qt: float = 2e-4               # Qt diffusion coefficient [nm^2/ns]
    D_ft: float = 5e-4               # Ft diffusion coefficient [nm^2/ns]
    temperature: float = 310.0       # [K]

    # --- soft repulsion, pair_style harmonic/cut: E = k (r_c - r)^2 ----------
    k_qq: float = 7.5                # Qt-Qt [zJ/nm^2], r_c = 2 r_Qt
    k_ff: float = 3.0                # Ft-Ft [zJ/nm^2], r_c = 2 r_Ft
    k_qf: float = 4.3                # Qt-Ft [zJ/nm^2], r_c = r_Qt + r_Ft

    # --- bonds, bond_style harmonic: E = K (r - r0)^2 ------------------------
    bond_k: float = 4.3              # K [zJ/nm^2]
    bond_r0: float | None = None     # r0 [nm]; None -> r_Qt + r_Ft

    # --- binding, fix bond/create --------------------------------------------
    k_on: float = 1.86e-6            # binding rate per pair at full calmodulin loading (f = 1) [1/ns]
    bind_radius: float | None = None  # [nm]; None -> r_Qt + r_Ft
    bind_every: int = 1              # attempt bond creation every N steps
    maxbond_qt: int = 60             # max bonds per Qt
    maxbond_ft: int = 10             # max bonds per Ft

    # --- calcium control (production) ----------------------------------------
    t_baseline_s: float = 10.0       # baseline at resting calcium [s]
    t_influx_s: float = 30.0         # influx (light on): calcium rises to its plateau [s]
    t_pumpout_s: float = 90.0        # pump-out (light off): calcium decays exponentially to rest [s]
    ca_rest_uM: float = 0.1          # resting calcium [uM]
    ca_peak_uM: float = 1.0          # calcium plateau during the influx [uM]
    ca_rise_s: float = 5.0           # time to 99 % of the rise after the influx start [s]; 0 = step
    ca_tau_s: float = 20.0           # pump-out time constant [s]
    ca_K_uM: float = 0.5             # calcium at half calmodulin loading [uM]
    ca_hill: float = 3.0             # Hill coefficient of the calmodulin loading
    k_off_ca_per_s: float = 1e-3     # unbinding rate of a bond with Ca-loaded calmodulin [1/s]
    k_off_free_per_s: float = 1.0    # unbinding rate of a bond with Ca-free calmodulin [1/s]
    rate_update_every: int = 20_000  # segment length [steps]: k_on is updated every segment ("longest": k_off too)
    break_every: int = 10            # unbinding attempts every N steps
    bond_break: str = "random"       # "random": every bond breaks independently, k_off follows f at every attempt
                                     # (fix bond/break/random, lammps_plugins/); "longest": fix bond/break (only
                                     # the longest bond of both atoms can break, k_off per segment)

    # --- run -----------------------------------------------------------------
    timestep: float = 500.0          # [ns]
    equil_steps: int = 5000          # equilibration, binding off

    # --- output and in-run analysis ------------------------------------------
    output_every: int = 1000         # thermo/analysis in log.lammps [steps]
    traj_every: int = 1000           # traj.dump, bonds.dump, rdf, histograms [steps]
    large_cluster_min: int = 5       # "large cluster" = at least this many particles
    rdf_bins: int = 200
    rdf_cutoff: float | None = None  # [nm]; None -> 3 (r_Qt + r_Ft)

    # --- technical -----------------------------------------------------------
    neighbor_skin: float = 20.0      # neighbor list skin [nm]
    extra_special: int = 1000        # reserved 1-2/1-3/1-4 neighbors per atom
    comm_cutoff: float | None = None  # ghost cutoff [nm]; None -> automatic
    insert_maxtry: int = 1000        # insertion attempts per particle

    # --- execution -----------------------------------------------------------
    n_replicas: int = 1
    mpi_procs: int = 1               # MPI processes per replica
    max_parallel: int | None = None  # replicas running at once; None -> local: cores // mpi_procs, lrz: all
    master_seed: int = 12345
    data_dir: str = "runs"           # relative -> inside the repository (gitignored)
    group: str | None = None         # subfolder; None -> "single" (1 replica) or "ensemble"
    run_name: str | None = None      # None -> generated from the parameters
    lmp_command: str = "lmp"
    mpirun_command: str = "mpirun --bind-to none"   # no core pinning: replicas run side by side
    executor: str = "local"          # "local": this computer; "lrz": Slurm job on the LRZ Linux cluster (qtft/remote.py)
    slurm_time: str = "3-00:00:00"   # lrz: time limit per replica [D-HH:MM:SS], at most 7 days
    slurm_mem: str = "4G"            # lrz: memory per replica (~150 MB per MPI process are used)

    # ---------------------------------------------------------------- derived
    @property
    def box(self) -> tuple[float, float, float]:
        """Box edges (Lx, Ly, Lz) [nm]; a single box_length is a cube."""
        if isinstance(self.box_length, numbers.Real):
            return (float(self.box_length),) * 3
        return tuple(float(v) for v in self.box_length)

    @property
    def periodic(self) -> bool:
        return self.boundary == "periodic"

    def reachable_fraction(self, radius: float) -> float:
        """Fraction of the box volume a particle centre can reach (1 for a periodic box)."""
        if self.periodic:
            return 1.0
        return math.prod(max(L - 2 * radius, 0.0) / L for L in self.box)

    @property
    def r0(self) -> float:
        return self.bond_r0 if self.bond_r0 is not None else self.radius_qt + self.radius_ft

    @property
    def r_bind(self) -> float:
        return self.bind_radius if self.bind_radius is not None else self.radius_qt + self.radius_ft

    @property
    def cut_qq(self) -> float:
        return 2 * self.radius_qt

    @property
    def cut_ff(self) -> float:
        return 2 * self.radius_ft

    @property
    def cut_qf(self) -> float:
        return self.radius_qt + self.radius_ft

    @property
    def rdf_cut(self) -> float:
        return self.rdf_cutoff if self.rdf_cutoff is not None else 3 * self.cut_qf

    @property
    def kT(self) -> float:
        return K_B * self.temperature

    @property
    def gamma_qt(self) -> float:
        return self.kT / self.D_qt

    @property
    def gamma_ft(self) -> float:
        return self.kT / self.D_ft

    # --------------------------------------------------- calcium control
    @property
    def phase_steps(self) -> tuple[int, int, int]:
        """Steps of the baseline, influx and pump-out phases."""
        return tuple(round(t * 1e9 / self.timestep) for t in (self.t_baseline_s, self.t_influx_s, self.t_pumpout_s))

    @property
    def prod_steps(self) -> int:
        """Production steps: all three phases."""
        return sum(self.phase_steps)

    @property
    def t_on_s(self) -> float:
        """Start of the influx [s]."""
        return self.t_baseline_s

    @property
    def t_off_s(self) -> float:
        """Start of the pump-out [s]."""
        return self.t_baseline_s + self.t_influx_s

    @property
    def influx_end_step(self) -> int:
        """Production step at the end of the influx, where the agglomerates are analysed (cluster size,
        volume and coordination distributions); the run end if there is no influx."""
        return sum(self.phase_steps[:2]) if self.t_influx_s > 0 else self.prod_steps

    @property
    def ca_rate_on_per_s(self) -> float:
        """Relaxation rate of the rise, lambda_on = ln(100) / ca_rise_s [1/s] (inf for a step)."""
        return math.log(100) / self.ca_rise_s if self.ca_rise_s > 0 else math.inf

    def _rise(self, dt_s):
        """Calcium [uM] a time dt_s >= 0 after the influx start."""
        if self.ca_rise_s <= 0:
            return np.full_like(np.asarray(dt_s, dtype=float), self.ca_peak_uM)
        return self.ca_peak_uM - (self.ca_peak_uM - self.ca_rest_uM) * np.exp(-self.ca_rate_on_per_s * dt_s)

    @property
    def ca_start_uM(self) -> float:
        """Calcium at the start of the pump-out: reached at the end of the influx, rest without one."""
        return float(self._rise(self.t_influx_s)) if self.t_influx_s > 0 else self.ca_rest_uM

    def calcium(self, t_s):
        """Calcium [uM] at production time t [s] (scalar or array); the same formula as in template.lmp."""
        t = np.asarray(t_s, dtype=float)
        rise = self._rise(np.maximum(t - self.t_on_s, 0.0))
        decay = self.ca_rest_uM + (self.ca_start_uM - self.ca_rest_uM) * np.exp(
            -np.maximum(t - self.t_off_s, 0.0) / self.ca_tau_s)
        return np.where(t < self.t_on_s, self.ca_rest_uM, np.where(t < self.t_off_s, rise, decay))

    def loading(self, ca_uM):
        """Calmodulin loading f = Ca^n / (K^n + Ca^n)."""
        ca = np.asarray(ca_uM, dtype=float) ** self.ca_hill
        return ca / (self.ca_K_uM ** self.ca_hill + ca)

    def k_off_per_s(self, f):
        """Unbinding rate per bond [1/s] at loading f."""
        return f * self.k_off_ca_per_s + (1 - f) * self.k_off_free_per_s

    @property
    def p_bind(self) -> float:
        """Binding probability per attempt at full loading (f = 1), the upper bound."""
        return 1.0 - math.exp(-self.k_on * self.timestep * self.bind_every)

    @property
    def p_break_max(self) -> float:
        """Largest unbinding probability per attempt."""
        k = max(self.k_off_ca_per_s, self.k_off_free_per_s) * 1e-9
        return 1.0 - math.exp(-k * self.timestep * self.break_every)

    @property
    def comm_cut(self) -> float:
        # fix bond/create needs partners up to 2 bonds away as ghosts,
        # compute rdf needs rdf_cutoff + skin
        if self.comm_cutoff is not None:
            return self.comm_cutoff
        return max(2.5 * self.r0, self.rdf_cut) + self.neighbor_skin

    def alpha(self, k: float, gamma_i: float, gamma_j: float) -> float:
        """Per-step overshoot of a pair separation for force constant 2k (E = k x^2)."""
        return 2 * k * self.timestep * (1 / gamma_i + 1 / gamma_j)

    def wall_alpha(self, k: float, gamma: float) -> float:
        """Per-step overshoot against a fixed wall (E = k x^2, only the particle moves)."""
        return 2 * k * self.timestep / gamma

    def size_bounds(self) -> tuple[int, int, int]:
        """Upper bounds of the 'small', 'medium' and 'large' cluster size classes."""
        n = self.n_qt + self.n_ft
        return (max(5, round(0.02 * n)), max(20, round(0.10 * n)), max(50, round(0.25 * n)))

    def derived(self) -> dict:
        vol = math.prod(self.box)
        to_uM = 1e6 / (vol * NM3_TO_L * N_A)
        sphere = lambda r: 4 / 3 * math.pi * r ** 3
        gq, gf = self.gamma_qt, self.gamma_ft
        walls = {}
        if self.boundary == "harmonic":
            walls |= {"alpha wall Qt": self.wall_alpha(self.wall_k_qt, gq),
                      "alpha wall Ft": self.wall_alpha(self.wall_k_ft, gf)}
        if not self.periodic:
            walls |= {"reachable volume Qt": self.reachable_fraction(self.radius_qt),
                      "reachable volume Ft": self.reachable_fraction(self.radius_ft)}
        f_rest, f_peak = float(self.loading(self.ca_rest_uM)), float(self.loading(self.ca_peak_uM))
        calcium = {
            "production time [s]": self.t_baseline_s + self.t_influx_s + self.t_pumpout_s,
            "phase steps (baseline/influx/pump-out)": self.phase_steps,
            "segments (rate updates)": self.prod_steps // self.rate_update_every if self.rate_update_every else 0,
            "calcium at the pump-out start [uM]": round(self.ca_start_uM, 4),
            "loading f at rest / peak": (round(f_rest, 4), round(f_peak, 4)),
            "k_on per pair at rest / peak [1/s]": (f"{self.k_on * 1e9 * f_rest:.3g}", f"{self.k_on * 1e9 * f_peak:.3g}"),
            "k_off per bond at rest / peak [1/s]": (f"{self.k_off_per_s(f_rest):.3g}", f"{self.k_off_per_s(f_peak):.3g}"),
            "p_bind per attempt (f = 1)": self.p_bind,
            "p_break per attempt (max)": self.p_break_max,
        }
        if self.ca_rise_s > 0:                    # optoCRAC one-equation model (docs/decisions.md)
            lam, k_dark = self.ca_rate_on_per_s, 1 / self.ca_tau_s
            calcium |= {
                "tau rise / decay [s]": (round(1 / lam, 4), self.ca_tau_s),
                "k_act*I / k_dark [1/s]": (round(lam - k_dark, 4), round(k_dark, 4)),
                "effective c_max [uM]": round(self.ca_rest_uM + (self.ca_peak_uM - self.ca_rest_uM)
                                              * lam / (lam - k_dark), 4) if lam > k_dark else float("nan"),
            }
        return {
            "kT [zJ]": self.kT,
            "gamma_Qt [ag/ns]": gq,
            "gamma_Ft [ag/ns]": gf,
            "r0 bond [nm]": self.r0,
            "R_bind [nm]": self.r_bind,
            "alpha Qt-Qt": self.alpha(self.k_qq, gq, gq),
            "alpha Ft-Ft": self.alpha(self.k_ff, gf, gf),
            "alpha Qt-Ft": self.alpha(self.k_qf, gq, gf),
            "alpha bond": self.alpha(self.bond_k, gq, gf),
            **walls,
            "RMS step Qt [nm/axis]": math.sqrt(2 * self.D_qt * self.timestep),
            "RMS step Ft [nm/axis]": math.sqrt(2 * self.D_ft * self.timestep),
            "c_Qt [uM]": self.n_qt * to_uM,
            "c_Ft [uM]": self.n_ft * to_uM,
            "phi_Qt": self.n_qt * sphere(self.radius_qt) / vol,
            "phi_Ft": self.n_ft * sphere(self.radius_ft) / vol,
            **calcium,
            "comm cutoff [nm]": self.comm_cut,
            "size classes (small/medium/large upper bound)": self.size_bounds(),
        }

    # ----------------------------------------------------------------- checks
    def check(self) -> list[tuple[str, str]]:
        """Return a list of (level, message) with level 'ERROR' or 'WARNING'."""
        out = []
        if isinstance(self.box_length, (list, tuple)) and len(self.box_length) != 3:
            return [("ERROR", "box_length must be one number (cube) or three numbers [Lx, Ly, Lz]")]
        if min(self.box) <= 0:
            return [("ERROR", "box edges must be positive")]
        if min(self.t_baseline_s, self.t_influx_s, self.t_pumpout_s) < 0 or self.prod_steps <= 0:
            return [("ERROR", "phase durations must be >= 0 and add up to more than one step")]
        if min(self.ca_rest_uM, self.ca_peak_uM, self.ca_K_uM, self.ca_tau_s, self.ca_hill) <= 0:
            return [("ERROR", "ca_rest_uM, ca_peak_uM, ca_K_uM, ca_tau_s and ca_hill must be positive")]
        if self.ca_rise_s < 0:
            return [("ERROR", "ca_rise_s must be >= 0 (0 = instantaneous rise)")]
        if self.ca_rise_s >= self.ca_tau_s * math.log(100):
            return [("ERROR", f"ca_rise_s must be < ca_tau_s * ln(100) = {self.ca_tau_s * math.log(100):.3g} s: "
                              "the rise must be faster than the decay (k_act > 0)")]
        if min(self.k_off_ca_per_s, self.k_off_free_per_s) < 0 or self.break_every < 1:
            return [("ERROR", "k_off values must be >= 0 and break_every >= 1")]
        if self.bond_break not in ("random", "longest"):
            return [("ERROR", f"bond_break '{self.bond_break}' unknown: use 'random' or 'longest'")]
        if self.executor not in EXECUTORS:
            return [("ERROR", f"executor '{self.executor}' unknown: use 'local' or 'lrz'")]
        if self.executor == "lrz":
            days = slurm_days(self.slurm_time)
            if days is None or not 0 < days <= SLURM_MAX_DAYS:
                return [("ERROR", f"slurm_time '{self.slurm_time}' must be D-HH:MM:SS or HH:MM:SS, "
                                  f"at most {SLURM_MAX_DAYS} days")]
            if not re.fullmatch(r"\d+[KMGT]?", self.slurm_mem):
                return [("ERROR", f"slurm_mem '{self.slurm_mem}' must be a number with K, M, G or T, e.g. 4G")]
        if self.boundary not in ("periodic", "reflect", "harmonic"):
            return [("ERROR", f"boundary '{self.boundary}' unknown: use 'periodic', 'reflect' or 'harmonic'")]
        if not self.periodic and min(self.box) <= 2 * max(self.radius_qt, self.radius_ft):
            return [("ERROR", "with walls every box edge must exceed the largest particle diameter")]
        d = self.derived()
        if self.r_bind > self.cut_qf + 1e-9:
            out.append(("ERROR", f"R_bind {self.r_bind} nm exceeds the Qt-Ft pair cutoff {self.cut_qf} nm "
                                 "(fix bond/create requires R_bind <= pair cutoff)"))
        for name in ("alpha Qt-Qt", "alpha Ft-Ft", "alpha Qt-Ft", "alpha bond", "alpha wall Qt", "alpha wall Ft"):
            if name not in d:
                continue
            a = d[name]
            if a >= 2:
                out.append(("ERROR", f"{name} = {a:.2f} >= 2: unstable, reduce dt or k"))
            elif a > 1:
                out.append(("WARNING", f"{name} = {a:.2f} > 1: steps overshoot, consider smaller dt or k"))
        if self.p_bind > 0.1:
            out.append(("WARNING", f"p_bind = {self.p_bind:.3f} > 0.1: per-step probability starts to saturate"))
        if self.p_break_max >= 0.1:
            out.append(("ERROR", f"p_break = {self.p_break_max:.3f} >= 0.1 per attempt: reduce break_every or k_off"))
        elif self.p_break_max >= 0.01:
            out.append(("WARNING", f"p_break = {self.p_break_max:.3f} >= 0.01 per attempt: unbinding starts to saturate"))
        if d["phi_Qt"] / self.reachable_fraction(self.radius_qt) \
                + d["phi_Ft"] / self.reachable_fraction(self.radius_ft) > 0.3:
            out.append(("WARNING", "total volume fraction > 0.3: overlap-free insertion may fail"))
        if self.comm_cut < self.rdf_cut + self.neighbor_skin:
            out.append(("ERROR", "comm cutoff must be >= rdf cutoff + neighbor skin"))
        if min(self.box) < 2 * self.rdf_cut:
            why = ("the RDF sees periodic images of the same particles" if self.periodic else
                   "many particles are near a wall, where the RDF (normalised for an infinite system) is biased low")
            out.append(("WARNING", f"shortest box edge {min(self.box):g} nm < 2 x rdf cutoff ({self.rdf_cut:g} nm): {why}"))
        if self.rate_update_every < 1 or self.rate_update_every % self.output_every:
            out.append(("ERROR", "rate_update_every must be a positive multiple of output_every"))
        elif self.prod_steps % self.rate_update_every:
            out.append(("ERROR", f"the production ({self.prod_steps} steps) must be a whole number of segments "
                                 f"(rate_update_every = {self.rate_update_every})"))
        elif any(n % self.rate_update_every for n in self.phase_steps[:2]):
            rates = "k_on changes" if self.bond_break == "random" else "rates change"
            out.append(("WARNING", f"a phase boundary falls inside a segment: {rates} at the next segment"))
        if 0 < self.t_influx_s < self.ca_rise_s:
            out.append(("WARNING", f"the influx ({self.t_influx_s:g} s) is shorter than ca_rise_s "
                                   f"({self.ca_rise_s:g} s): calcium does not reach its plateau"))
        if self.ca_rise_s > 0 and 1 / self.ca_rate_on_per_s < 10 * self.rate_update_every * self.timestep * 1e-9:
            out.append(("WARNING", "the calcium rise time constant is shorter than 10 segments: "
                                   "the rate updates do not resolve the rise"))
        if self.t_baseline_s > 0 and self.t_influx_s > 0 and self.phase_steps[0] % self.traj_every:
            out.append(("WARNING", "the influx start is not a traj_every step: binding_rate needs a frame there"))
        if self.t_influx_s > 0 and self.influx_end_step % self.traj_every:
            out.append(("WARNING", "the influx end is not a traj_every step: the cluster distributions use the "
                                   "last frame before it"))
        if self.traj_every % self.output_every:
            out.append(("ERROR", "traj_every must be a multiple of output_every"))
        elif self.prod_steps % self.traj_every:
            out.append(("WARNING", "the run end is not a traj_every step: the last frame, rdf and histograms "
                                   "are not the final state"))
        if self.maxbond_qt < 1 or self.maxbond_ft < 1:
            out.append(("ERROR", "maxbond values must be >= 1"))
        if self.executor == "lrz" and self.max_parallel and self.max_parallel < self.n_replicas:
            out.append(("WARNING", f"on LRZ only {self.max_parallel} of {self.n_replicas} replicas run at once "
                                   "(max_parallel); max_parallel=None runs all at once"))
        return out

    def raise_on_error(self) -> None:
        errors = [m for lvl, m in self.check() if lvl == "ERROR"]
        if errors:
            raise ValueError("Invalid parameters:\n  " + "\n  ".join(errors))

    # ---------------------------------------------------------------- display
    def summary(self) -> str:
        lines = ["Derived values", "-" * 60]
        for key, val in self.derived().items():
            lines.append(f"{key:<46} {_fmt(val)}")
        issues = self.check()
        lines += ["", "Checks", "-" * 60]
        lines += [f"{lvl}: {msg}" for lvl, msg in issues] or ["all OK"]
        return "\n".join(lines)

    def auto_name(self) -> str:
        return (f"{self.n_qt}Qt_{self.n_ft}Ft_L{box_label(self.box_length)}"
                f"_kQQ{self.k_qq:g}_kFF{self.k_ff:g}_kQF{self.k_qf:g}_K{self.bond_k:g}"
                f"_kon{self.k_on:g}_dt{self.timestep:g}ns"
                f"_ca{self.t_baseline_s:g}-{self.t_influx_s:g}-{self.t_pumpout_s:g}s_{self.boundary}")

    @property
    def name(self) -> str:
        return self.run_name or self.auto_name()

    @property
    def group_name(self) -> str:
        return self.group or ("single" if self.n_replicas == 1 else "ensemble")

    @property
    def run_dir(self) -> Path:
        return data_path(self.data_dir) / self.group_name / self.name

    # ------------------------------------------------------------ persistence
    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Params":
        """Params of a stored run (params.json); runs from before bond_break existed used fix bond/break."""
        known = {f.name for f in fields(cls)}
        data = {"bond_break": "longest", **data}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data.get("params", data))


def box_label(box_length) -> str:
    """'2000' for a cube, '2000x3000x4000' for a rectangular box (run names, sweep labels)."""
    if isinstance(box_length, numbers.Real):
        return f"{box_length:g}"
    return "x".join(f"{v:g}" for v in box_length)


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)
