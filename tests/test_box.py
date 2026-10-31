"""Rectangular (non-cubic) boxes: box_length = [Lx, Ly, Lz]."""

import math
import shutil

import numpy as np
import pytest

from qtft import Params, build_run, build_sweep, read_dump, start, sweep_runs
from conftest import influx


def test_box_property_and_derived():
    assert Params(box_length=2000.0).box == (2000.0, 2000.0, 2000.0)
    p = Params(qt_per_um3=4000 / 24, ft_per_um3=10000 / 24, box_length=[2000.0, 3000.0, 4000.0])
    assert p.box == (2000.0, 3000.0, 4000.0)
    cube = Params(qt_per_um3=500.0, ft_per_um3=1250.0, box_length=2000.0)
    # three times the volume of the 2000 nm cube -> a third of the concentration
    assert p.derived()["c_Qt [uM]"] == pytest.approx(cube.derived()["c_Qt [uM]"] / 3)
    assert p.derived()["phi_Qt"] == pytest.approx(4000 * 4 / 3 * math.pi * 27.5**3 / 24e9)
    assert "_L2000x3000x4000_" in p.auto_name() and "_L2000_" in cube.auto_name()
    assert p.density_per_um3 == pytest.approx((4000 / 24, 10000 / 24))
    assert p.auto_name().startswith("Qt167_Ft417_") and cube.auto_name().startswith("Qt500_Ft1250_")
    assert p.check() == []


def test_particle_numbers_from_densities():
    p = Params(qt_per_um3=20.0, ft_per_um3=80.0, box_length=4000.0)              # 64 um^3
    assert (p.n_qt, p.n_ft, p.volume_um3) == (1280, 5120, pytest.approx(64.0))
    assert p.derived()["particles Qt / Ft"] == (1280, 5120) and p.summary().count("(1280, 5120)") == 1
    # the same density in a larger box: more particles, same density (a box sweep keeps the density)
    assert Params(qt_per_um3=20.0, ft_per_um3=80.0, box_length=[4000.0, 4000.0, 8000.0]).n_qt == 2560
    small = Params(qt_per_um3=20.3, ft_per_um3=80.0, box_length=500.0)            # 0.125 um^3 -> 2.54 Qt -> 3
    assert small.n_qt == 3 and any(lvl == "WARNING" and "3 Qt give 24" in m for lvl, m in small.check())
    assert Params(qt_per_um3=-1.0).check()[0][0] == "ERROR"
    assert Params(qt_per_um3=0.0, ft_per_um3=0.0).check()[0][0] == "ERROR"
    assert "at least one Qt and one Ft" in Params(qt_per_um3=0.0, ft_per_um3=80.0, box_length=1000.0).check()[0][1]
    assert Params(qt_per_um3=20.0, ft_per_um3=0.4, box_length=1000.0).check()[0][0] == "ERROR"   # 0.4 Ft -> 0
    assert "n_qt" not in Params().to_dict() and "qt_per_um3" in Params().to_dict()
    old = {k: v for k, v in Params().to_dict().items() if k not in ("qt_per_um3", "ft_per_um3")}
    with pytest.raises(ValueError, match="built with particle numbers"):     # runs from before the densities
        Params.from_dict({**old, "n_qt": 1280, "n_ft": 5120})


def test_box_checks():
    assert Params(box_length=[1000.0, 2000.0]).check()[0][0] == "ERROR"
    assert Params(box_length=[1000.0, 0.0, 1000.0]).check()[0][0] == "ERROR"
    warnings = [m for lvl, m in Params(box_length=[200.0, 1000.0, 1000.0]).check() if lvl == "WARNING"]
    assert any("rdf cutoff" in m for m in warnings)


def test_box_in_input_and_sweep_label(tmp_path):
    p = Params(box_length=[400.0, 500.0, 600.0], data_dir=str(tmp_path), run_name="box")
    text = (build_run(p) / "rep_01" / "in.lammps").read_text()
    assert "region          box block 0 400 0 500 0 600" in text
    sweep = build_sweep(Params(data_dir=str(tmp_path)), {"box_length": [928.3, [800.0, 900.0, 1200.0]]}, "box")
    assert [r.name for r in sweep_runs(sweep)] == ["box_length928.3", "box_length800x900x1200"]


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_rectangular_box_run(tmp_path):
    from qtft.overlap import overlap_frame

    p = Params(qt_per_um3=50 / 0.12, ft_per_um3=200 / 0.12, box_length=[400.0, 500.0, 600.0], equil_steps=200, **influx(1000),
               output_every=500, traj_every=500, k_on=1e-4, data_dir=str(tmp_path), run_name="box")
    run = build_run(p)
    start(run, wait=True)
    rep = run / "rep_01"
    assert (rep / "status").read_text().strip() == "done"
    lines = (rep / "traj.dump").read_text().splitlines()
    bounds = [lines[i + 1 + k].split() for i, l in enumerate(lines) if l.startswith("ITEM: BOX BOUNDS")
              for k in range(3)][:3]
    assert [float(b[1]) - float(b[0]) for b in bounds] == pytest.approx([400.0, 500.0, 600.0])

    _, atoms = list(read_dump(rep / "traj.dump"))[-1]
    box = np.array(p.box)
    pos, types = atoms[["x", "y", "z"]].to_numpy(), atoms["type"].to_numpy()
    # atoms are wrapped back into the periodic box only when the neighbour list is rebuilt
    assert (pos.min(axis=0) >= -p.neighbor_skin).all() and (pos.max(axis=0) <= box + p.neighbor_skin).all()
    r = np.where(types == 1, p.radius_qt, p.radius_ft)
    d = pos[:, None, :] - pos[None, :, :]
    d -= box * np.round(d / box)                                   # minimum image per axis
    depth = np.maximum(r[:, None] + r[None, :] - np.linalg.norm(d, axis=2), 0)
    iu = np.triu_indices(len(pos), 1)
    result = overlap_frame(atoms, p).set_index("family")
    assert result["n overlapping"].sum() == (depth[iu] > 0).sum()


def test_run_name():
    p = Params(qt_per_um3=20.0, ft_per_um3=80.0, box_length=4000.0, k_on=1.86e-6, k_off_ca_per_s=8e-3, k_off_free_per_s=0.2,
               ca_rest_uM=0.1, ca_peak_uM=1.0, ca_K_uM=0.5, ca_rise_s=5.0, ca_tau_s=30.0,
               t_baseline_s=10.0, t_influx_s=30.0, t_pumpout_s=150.0, timestep=500.0)
    assert p.auto_name() == ("Qt20_Ft80_L4000_kon1860_koff0.008-0.2_Ca0.1-1uM_K0.5_rise5tau30_t10-30-150s"
                             "_dt500ns_periodic")
    assert p.name == p.auto_name() and Params(run_name="mine").name == "mine"
    # the force constants, the bond constant and the Hill coefficient are not in the name
    assert Params(k_qq=1.0, k_ff=1.0, k_qf=1.0, bond_k=1.0, ca_hill=2.0).auto_name() == Params().auto_name()
