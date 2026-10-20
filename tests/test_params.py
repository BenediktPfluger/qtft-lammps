import math

import pytest

from qtft import Params


def test_kT_and_gamma():
    p = Params()
    assert p.kT == pytest.approx(4.280, abs=1e-3)               # zJ at 310 K
    assert p.gamma_qt == pytest.approx(p.kT / 2e-4)              # ag/ns
    assert p.gamma_ft == pytest.approx(p.kT / 5e-4)


def test_geometry_defaults():
    p = Params()
    assert (p.cut_qq, p.cut_ff, p.cut_qf) == (55.0, 26.0, 40.5)
    assert p.r0 == 40.5 and p.r_bind == 40.5
    assert p.rdf_cut == pytest.approx(121.5)
    assert p.comm_cut == pytest.approx(131.5)


def test_binding_probability():
    p = Params()
    assert p.p_bind == pytest.approx(1 - math.exp(-5e-4))                # k_on dt = 1e-6 x 500
    assert Params(bind_every=10).p_bind == pytest.approx(1 - math.exp(-5e-3))


def test_stability_values_of_reference_timestep():
    d = Params().derived()                                               # dt = 500 ns
    assert d["alpha Qt-Qt"] == pytest.approx(0.31, abs=0.01)
    assert d["alpha Ft-Ft"] == pytest.approx(0.47, abs=0.01)
    assert d["alpha Qt-Ft"] == pytest.approx(0.20, abs=0.01)
    assert d["alpha bond"] == pytest.approx(0.27, abs=0.01)
    assert Params().check() == []


def test_concentrations():
    small, large = Params(), Params(n_qt=4000, n_ft=16000, box_length=2000)
    assert small.derived()["c_Qt [uM]"] == pytest.approx(large.derived()["c_Qt [uM]"], rel=1e-3)
    assert large.derived()["c_Qt [uM]"] == pytest.approx(0.83, abs=0.01)


def test_checks_catch_bad_values():
    assert any(lvl == "ERROR" for lvl, _ in Params(bind_radius=45.0).check())
    assert any(lvl == "ERROR" for lvl, _ in Params(timestep=5000.0).check())      # alpha >= 2
    assert any(lvl == "WARNING" for lvl, _ in Params(k_ff=3.0, timestep=1000.0).check())   # alpha > 1
    with pytest.raises(ValueError):
        Params(bind_radius=45.0).raise_on_error()


def test_size_bounds():
    assert Params().size_bounds() == (40, 200, 500)
    assert Params(n_qt=4000, n_ft=16000).size_bounds() == (400, 2000, 5000)
