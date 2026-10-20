"""All simulation parameters (LAMMPS `units nano`) and the values derived from them.

Units: length nm, time ns, energy zJ (= ag nm^2/ns^2), force pN, mass ag,
diffusion coefficient nm^2/ns, friction coefficient ag/ns, temperature K.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path

K_B = 1.380649e-2        # Boltzmann constant in zJ/K
N_A = 6.02214076e23      # Avogadro constant in 1/mol
NM3_TO_L = 1e-24         # 1 nm^3 in litres

REPO_DIR = Path(__file__).resolve().parent.parent


def data_path(data_dir: str | Path = "qtft-data") -> Path:
    """Absolute output folder; a relative data_dir is taken relative to the repository."""
    path = Path(os.path.expanduser(data_dir))
    return path if path.is_absolute() else REPO_DIR / path


@dataclass
class Params:
    # --- system --------------------------------------------------------------
    n_qt: int = 400                  # number of Qt encapsulins
    n_ft: int = 1600                 # number of Ft ferritins
    box_length: float = 928.3        # cubic periodic box edge [nm]

    # --- particles -----------------------------------------------------------
    radius_qt: float = 27.5          # effective Qt radius [nm]
    radius_ft: float = 13.0          # effective Ft radius [nm]
    D_qt: float = 2e-4               # Qt diffusion coefficient [nm^2/ns]
    D_ft: float = 5e-4               # Ft diffusion coefficient [nm^2/ns]
    temperature: float = 310.0       # [K]

    # --- soft repulsion, pair_style harmonic/cut: E = k (r_c - r)^2 ----------
    k_qq: float = 3.32               # Qt-Qt [zJ/nm^2], r_c = 2 r_Qt
    k_ff: float = 2.0                # Ft-Ft [zJ/nm^2], r_c = 2 r_Ft
    k_qf: float = 1.245              # Qt-Ft [zJ/nm^2], r_c = r_Qt + r_Ft

    # --- bonds, bond_style harmonic: E = K (r - r0)^2 ------------------------
    bond_k: float = 1.66             # K [zJ/nm^2]
    bond_r0: float | None = None     # r0 [nm]; None -> r_Qt + r_Ft

    # --- binding, fix bond/create --------------------------------------------
    k_on: float = 1e-6               # microscopic binding rate per pair [1/ns]
    bind_radius: float | None = None  # [nm]; None -> r_Qt + r_Ft
    bind_every: int = 1              # attempt bond creation every N steps
    maxbond_qt: int = 60             # max bonds per Qt
    maxbond_ft: int = 10             # max bonds per Ft

    # --- run -----------------------------------------------------------------
    timestep: float = 500.0          # [ns]
    equil_steps: int = 1000          # equilibration, binding off
    prod_steps: int = 50_000         # production, binding on

    # --- output and in-run analysis ------------------------------------------
    output_every: int = 1000         # thermo/analysis, bonds.dump, rdf, histograms [steps]
    traj_every: int = 1000           # traj.dump [steps]
    large_cluster_min: int = 5       # "large cluster" = at least this many particles
    rdf_bins: int = 200
    rdf_cutoff: float | None = None  # [nm]; None -> 3 (r_Qt + r_Ft)

    # --- technical -----------------------------------------------------------
    neighbor_skin: float = 10.0      # neighbor list skin [nm]
    extra_special: int = 1000        # reserved 1-2/1-3/1-4 neighbors per atom
    comm_cutoff: float | None = None  # ghost cutoff [nm]; None -> automatic
    insert_maxtry: int = 1000        # insertion attempts per particle

    # --- execution -----------------------------------------------------------
    n_replicas: int = 1
    mpi_procs: int = 1               # MPI processes per replica
    max_parallel: int | None = None  # replicas running at once; None -> cores // mpi_procs
    master_seed: int = 12345
    data_dir: str = "qtft-data"      # relative -> inside the repository (gitignored)
    group: str | None = None         # subfolder; None -> "single" (1 replica) or "ensemble"
    run_name: str | None = None      # None -> generated from the parameters
    lmp_command: str = "lmp"
    mpirun_command: str = "mpirun --bind-to none"   # no core pinning: replicas run side by side

    # ---------------------------------------------------------------- derived
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

    @property
    def p_bind(self) -> float:
        return 1.0 - math.exp(-self.k_on * self.timestep * self.bind_every)

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

    def size_bounds(self) -> tuple[int, int, int]:
        """Upper bounds of the 'small', 'medium' and 'large' cluster size classes."""
        n = self.n_qt + self.n_ft
        return (max(5, round(0.02 * n)), max(20, round(0.10 * n)), max(50, round(0.25 * n)))

    def derived(self) -> dict:
        vol = self.box_length ** 3
        to_uM = 1e6 / (vol * NM3_TO_L * N_A)
        sphere = lambda r: 4 / 3 * math.pi * r ** 3
        gq, gf = self.gamma_qt, self.gamma_ft
        return {
            "kT [zJ]": self.kT,
            "gamma_Qt [ag/ns]": gq,
            "gamma_Ft [ag/ns]": gf,
            "r0 bond [nm]": self.r0,
            "R_bind [nm]": self.r_bind,
            "p_bind per attempt": self.p_bind,
            "alpha Qt-Qt": self.alpha(self.k_qq, gq, gq),
            "alpha Ft-Ft": self.alpha(self.k_ff, gf, gf),
            "alpha Qt-Ft": self.alpha(self.k_qf, gq, gf),
            "alpha bond": self.alpha(self.bond_k, gq, gf),
            "RMS step Qt [nm/axis]": math.sqrt(2 * self.D_qt * self.timestep),
            "RMS step Ft [nm/axis]": math.sqrt(2 * self.D_ft * self.timestep),
            "c_Qt [uM]": self.n_qt * to_uM,
            "c_Ft [uM]": self.n_ft * to_uM,
            "phi_Qt": self.n_qt * sphere(self.radius_qt) / vol,
            "phi_Ft": self.n_ft * sphere(self.radius_ft) / vol,
            "production time [ms]": self.prod_steps * self.timestep * 1e-6,
            "comm cutoff [nm]": self.comm_cut,
            "size classes (small/medium/large upper bound)": self.size_bounds(),
        }

    # ----------------------------------------------------------------- checks
    def check(self) -> list[tuple[str, str]]:
        """Return a list of (level, message) with level 'ERROR' or 'WARNING'."""
        out = []
        d = self.derived()
        if self.r_bind > self.cut_qf + 1e-9:
            out.append(("ERROR", f"R_bind {self.r_bind} nm exceeds the Qt-Ft pair cutoff {self.cut_qf} nm "
                                 "(fix bond/create requires R_bind <= pair cutoff)"))
        for name in ("alpha Qt-Qt", "alpha Ft-Ft", "alpha Qt-Ft", "alpha bond"):
            a = d[name]
            if a >= 2:
                out.append(("ERROR", f"{name} = {a:.2f} >= 2: unstable, reduce dt or k"))
            elif a > 1:
                out.append(("WARNING", f"{name} = {a:.2f} > 1: steps overshoot, consider smaller dt or k"))
        if self.p_bind > 0.1:
            out.append(("WARNING", f"p_bind = {self.p_bind:.3f} > 0.1: per-step probability starts to saturate"))
        if d["phi_Qt"] + d["phi_Ft"] > 0.3:
            out.append(("WARNING", "total volume fraction > 0.3: overlap-free insertion may fail"))
        if self.comm_cut < self.rdf_cut + self.neighbor_skin:
            out.append(("ERROR", "comm cutoff must be >= rdf cutoff + neighbor skin"))
        if self.prod_steps % self.output_every:
            out.append(("WARNING", "prod_steps should be a multiple of output_every"))
        if self.traj_every % self.output_every:
            out.append(("ERROR", "traj_every must be a multiple of output_every"))
        if self.maxbond_qt < 1 or self.maxbond_ft < 1:
            out.append(("ERROR", "maxbond values must be >= 1"))
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
        return (f"{self.n_qt}Qt_{self.n_ft}Ft_L{self.box_length:g}"
                f"_kQQ{self.k_qq:g}_kFF{self.k_ff:g}_kQF{self.k_qf:g}_K{self.bond_k:g}"
                f"_kon{self.k_on:g}_dt{self.timestep:g}ns_{self.prod_steps * self.timestep * 1e-6:g}ms")

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
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    @classmethod
    def load(cls, path: str | Path) -> "Params":
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data.get("params", data))


def _fmt(val) -> str:
    if isinstance(val, float):
        return f"{val:.4g}"
    return str(val)
