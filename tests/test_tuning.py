import shutil

import pytest

from qtft import Params, start, sweep_runs
from qtft.tuning import alpha_sweep, constants_for_alpha, predicted_overlap, sweep_table
from conftest import influx


def test_constants_reach_target_alpha():
    p = Params()
    d = Params(**constants_for_alpha(p, 0.8)).derived()
    for name in ("alpha Qt-Qt", "alpha Ft-Ft", "alpha Qt-Ft", "alpha bond"):
        assert d[name] == pytest.approx(0.8)


def test_predicted_overlap():
    # former constants: matches the measured mean depths of the 500 ms production run (0.70 / 0.95 / 0.95 nm)
    old = Params(k_qq=3.32, k_ff=2.0, k_qf=1.245, bond_k=1.66)
    assert predicted_overlap(old).to_dict() == pytest.approx(
        {"Qt-Qt": 0.697, "Ft-Ft": 0.943, "Qt-Ft": 0.974}, abs=1e-3)


@pytest.mark.slow
@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
def test_sweep_end_to_end(tmp_path):
    base = Params(qt_per_um3=500.0, ft_per_um3=2000.0, box_length=464.2, equil_steps=200, **influx(2000),
                  output_every=500, traj_every=500, n_replicas=1, data_dir=str(tmp_path))
    sweep = alpha_sweep(base, {500.0: [0.3, 0.9], 250.0: [0.9]})
    runs = sweep_runs(sweep)
    assert [r.name for r in runs] == ["alpha0.3_dt500ns", "alpha0.9_dt500ns", "alpha0.9_dt250ns"]
    assert sweep == tmp_path / "tuning" / "sweep"
    assert Params.load(runs[2] / "params.json").prod_steps == 4000                # same simulated time
    start(sweep, wait=True)
    table = sweep_table(sweep, every=1, last_frames=2)
    assert len(table) == 9 and (sweep / "sweep_summary.csv").exists()
    assert table.groupby("run")["alpha"].first().to_dict() == pytest.approx(
        {"alpha0.3_dt500ns": 0.3, "alpha0.9_dt500ns": 0.9, "alpha0.9_dt250ns": 0.9})
