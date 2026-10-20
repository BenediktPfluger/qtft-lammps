import os

from qtft.benchmark import loop_timing, physical_cores, special_need

LOG = """\
Loop time of 0.5 on 1 procs for 1000 steps with 250 atoms
MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.1        | 0.1        | 0.1        |   0.0 | 20.00
Neigh   | 0.4        | 0.4        | 0.4        |   0.0 | 80.00
Nlocal:            250 ave         250 max         250 min
Loop time of 2.0 on 4 procs for 3000 steps with 250 atoms
MPI task timing breakdown:
Section |  min time  |  avg time  |  max time  |%varavg| %total
---------------------------------------------------------------
Pair    | 0.5        | 0.5        | 0.5        |   0.0 | 25.00
Modify  | 1.5        | 1.5        | 1.5        |   0.0 | 75.00
Nlocal:            250 ave         250 max         250 min
"""


def test_loop_timing_uses_last_run(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(LOG)
    t = loop_timing(log)
    assert (t["wall_s"], t["procs"], t["steps"]) == (2.0, 4, 3000)
    assert t["Pair [%]"] == 25.0 and t["Modify [%]"] == 75.0 and "Neigh [%]" not in t


def _write_frame(rep, atoms, bonds):
    box = "ITEM: BOX BOUNDS pp pp pp\n0 100\n0 100\n0 100\n"
    (rep / "traj.dump").write_text(
        f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ATOMS\n{len(atoms)}\n{box}"
        "ITEM: ATOMS id type x y z ix iy iz radius cluster nbonds\n"
        + "".join(f"{i} {t} 0 0 0 0 0 0 1 1 0\n" for i, t in atoms))
    (rep / "bonds.dump").write_text(
        f"ITEM: TIMESTEP\n0\nITEM: NUMBER OF ENTRIES\n{len(bonds)}\n{box}"
        "ITEM: ENTRIES index atom1 atom2\n"
        + "".join(f"{k} {a} {b}\n" for k, (a, b) in enumerate(bonds, 1)))


def test_special_need_chain(tmp_path):
    # chain Qt1 - Ft2 - Qt3 - Ft4 plus a free Ft5
    _write_frame(tmp_path, [(1, 1), (2, 2), (3, 1), (4, 2), (5, 2)], [(1, 2), (3, 2), (3, 4)])
    need = special_need(tmp_path)
    assert need["max up to 1-2"] == 2          # atoms 2 and 3
    assert need["max up to 1-3"] == 3          # atom 2: {1,3} + {4}; atom 3: {2,4} + {1}
    assert need["max up to 1-4"] == 3          # atom 1: {2}, {3}, {4}


def test_physical_cores_within_threads():
    assert 1 <= physical_cores() <= (os.cpu_count() or 1)
