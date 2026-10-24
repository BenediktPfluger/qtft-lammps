"""Calcium-controlled binding and unbinding: baseline -> influx -> exponential pump-out."""

import json
import shutil

import numpy as np
import pytest

import pandas as pd

from qtft import Params, build_run, load_thermo, read_bonds, start
from qtft.ensemble import TO_PER_M_S, bond_rates, deagglomeration


def test_phases_and_calcium_formula():
    p = Params(t_baseline_s=10.0, t_influx_s=30.0, t_pumpout_s=90.0, ca_tau_s=20.0, ca_rise_s=0.0)
    assert p.phase_steps == (20_000_000, 60_000_000, 180_000_000) and p.prod_steps == 260_000_000
    ca = p.calcium([0.0, 9.99, 10.0, 39.99, 40.0, 60.0])
    assert ca == pytest.approx([0.1, 0.1, 1.0, 1.0, 1.0, 0.1 + 0.9 * np.exp(-1.0)])
    f = p.loading([0.5, 1.0])
    assert f[0] == pytest.approx(0.5) and f[1] == pytest.approx(1 / (1 + 0.5 ** 3))
    assert p.k_off_per_s(1.0) == p.k_off_ca_per_s and p.k_off_per_s(0.0) == p.k_off_free_per_s
    no_influx = Params(t_baseline_s=1.0, t_influx_s=0.0, t_pumpout_s=1.0)
    assert no_influx.ca_start_uM == no_influx.ca_rest_uM                      # pump-out stays at rest
    assert "_ca10-30-90s_periodic" in p.auto_name()


def test_calcium_rise():
    p = Params(t_baseline_s=10.0, t_influx_s=30.0, t_pumpout_s=90.0, ca_rise_s=5.0)
    ca = p.calcium([9.99, 10.0, 15.0, 40.0, 60.0])
    assert ca[0] == pytest.approx(0.1) and ca[1] == pytest.approx(0.1)            # continuous at t_on
    assert (ca[2] - 0.1) / (1.0 - 0.1) == pytest.approx(0.99)                     # 99 % of the rise after 5 s
    assert ca[3] == pytest.approx(p.ca_start_uM) and p.ca_start_uM == pytest.approx(1.0, abs=1e-9)
    assert ca[4] == pytest.approx(0.1 + (p.ca_start_uM - 0.1) * np.exp(-1.0))
    short = Params(t_baseline_s=1.0, t_influx_s=1.0, t_pumpout_s=1.0, ca_rise_s=5.0)
    assert short.ca_start_uM == pytest.approx(1.0 - 0.9 * 0.01 ** (1 / 5))        # plateau not reached
    d = p.derived()
    assert d["k_act*I / k_dark [1/s]"] == pytest.approx((round(np.log(100) / 5 - 0.05, 4), 0.05))
    assert d["effective c_max [uM]"] == pytest.approx(1.0517, abs=1e-4)


def test_deagglomeration_half_time():
    p = Params(t_baseline_s=1e-3, t_influx_s=2e-3, t_pumpout_s=3e-3)            # pump-out starts at 3 ms
    t = np.linspace(0, 6, 61)
    bonds = np.where(t < 3, 100 * t / 3, 100 * np.exp(-(t - 3) / 1.0))           # half-time ln 2 ms
    df = pd.DataFrame({"time_ms": t, "Bonds": bonds, "largest": bonds / 2})
    out = deagglomeration(df, p)
    assert out["bonds_peak"] == pytest.approx(100) and out["t_bonds_peak_ms"] == pytest.approx(3.0)
    assert out["t_half_unbind_bonds_ms"] == pytest.approx(np.log(2), abs=0.01)
    assert out["t_half_unbind_largest_ms"] == pytest.approx(np.log(2), abs=0.01)
    assert np.isnan(deagglomeration(df, Params(t_baseline_s=1e-3, t_influx_s=2e-3, t_pumpout_s=0.0))
                    ["t_half_unbind_bonds_ms"])


def test_bond_rates(tmp_path, monkeypatch):
    # 10 bonds formed and 4 broken per segment (1 ms), known exactly; two identical replicas
    p = Params(n_qt=10, n_ft=20, box_length=100.0, output_every=500, rate_update_every=2000)
    (tmp_path / "params.json").write_text(json.dumps({"params": p.__dict__}))
    step = np.arange(0, 20_001, 500)
    broken = 4 * step / 2000
    df = pd.DataFrame({"Step": step, "time_ms": step * p.timestep * 1e-6, "Bonds": 6 * step / 2000,
                       "n_broken": broken})
    monkeypatch.setattr("qtft.ensemble.replica_series", lambda run_dir: {"rep_01": df, "rep_02": df})
    r = bond_rates(tmp_path)
    assert r["bin_ms"] == pytest.approx(1.0) and r["time_ms"] == pytest.approx(np.arange(10) + 0.5)
    assert r["mean"]["formed_per_s"] == pytest.approx(np.full(10, 1e4))
    assert r["mean"]["broken_per_s"] == pytest.approx(np.full(10, 4e3)) and not r["sd"]["formed_per_s"].any()
    assert r["mean"]["k_a_per_M_s"] == pytest.approx(np.full(10, 1e-5 * 1e6 / 200 * TO_PER_M_S))
    saved = pd.read_csv(tmp_path / "analysis" / "bond_rates.csv")
    assert list(saved.columns[:3]) == ["time_ms", "formed_per_s_mean", "formed_per_s_sd"]
    assert len(bond_rates(tmp_path, bin_s=2e-3, save=False)["time_ms"]) == 5
    with pytest.raises(ValueError):
        bond_rates(tmp_path, bin_s=1e-4)                                         # not a multiple of output_every


def test_calcium_checks():
    errors = lambda p: [m for lvl, m in p.check() if lvl == "ERROR"]
    assert errors(Params(t_influx_s=-1.0))
    assert errors(Params(t_baseline_s=0.0, t_influx_s=0.0, t_pumpout_s=0.0))
    assert errors(Params(ca_K_uM=0.0))
    assert any("whole number of segments" in m for m in errors(Params(t_influx_s=30.0001)))
    assert any("multiple of output_every" in m for m in errors(Params(rate_update_every=1500)))
    warnings = lambda p: [m for lvl, m in p.check() if lvl == "WARNING"]
    assert any("run end" in m for m in warnings(Params(traj_every=7000)))
    assert not any("run end" in m for m in warnings(Params(traj_every=200_000)))
    assert any("p_break" in m for m in errors(Params(k_off_free_per_s=1e5)))
    assert errors(Params(ca_rise_s=-1.0)) and errors(Params(ca_rise_s=93.0))
    assert any("plateau" in m for m in warnings(Params(t_influx_s=3.0)))
    assert any("10 segments" in m for m in warnings(Params(ca_rise_s=0.2)))
    assert not errors(Params())


def test_input_has_segment_loop(tmp_path):
    p = Params(data_dir=str(tmp_path), run_name="ca")
    text = (build_run(p) / "rep_01" / "in.lammps").read_text()
    assert "variable        seg loop 13000" in text and "jump            SELF segment" in text
    assert "fix             unbind all bond/break 10 1 0.0" in text
    assert "v_ca v_f v_k_on_s v_k_off_s v_n_broken" in text
    seeds = json.loads((tmp_path / "single" / "ca" / "params.json").read_text())["seeds"][0]
    assert "seed_break" in seeds


@pytest.mark.skipif(shutil.which("lmp") is None, reason="LAMMPS executable 'lmp' not found")
@pytest.mark.parametrize("rise", [0.0, 2e-3])
def test_compressed_protocol_run(tmp_path, rise):
    # milliseconds instead of seconds and fast rates, so that bonds form and break within the test
    p = Params(n_qt=150, n_ft=600, box_length=500.0, equil_steps=200, output_every=500, traj_every=1000,
               t_baseline_s=2e-3, t_influx_s=4e-3, t_pumpout_s=6e-3, ca_tau_s=2e-3, ca_rise_s=rise,
               rate_update_every=1000,
               k_on=2e-3, k_off_ca_per_s=1.0, k_off_free_per_s=2000.0, data_dir=str(tmp_path), run_name="ca")
    run = build_run(p)
    rep = run / "rep_01"
    # after the run: rebuild the special lists from the bonds; the energy must not change
    text = (rep / "in.lammps").read_text().replace(
        'print           "QTFT_DONE"',
        'variable e1 equal pe\nprint "PE_BEFORE ${e1}"\nspecial_bonds lj 1.0e-50 1.0 1.0\nrun 0\n'
        'variable e2 equal pe\nprint "PE_AFTER ${e2}"\nprint "QTFT_DONE"')
    (rep / "in.lammps").write_text(text)
    start(run, wait=True)
    assert (rep / "status").read_text().strip() == "done"

    df = load_thermo(rep)
    assert df["Step"].is_unique and df["Step"].is_monotonic_increasing and df["Step"].iloc[-1] == p.prod_steps
    t = df["time_ms"].to_numpy() * 1e-3
    assert np.allclose(df["ca"], p.calcium(t)) and np.allclose(df["f"], p.loading(p.calcium(t)))
    assert np.allclose(df["k_off_s"], p.k_off_per_s(df["f"]))
    influx_end = df["time_ms"] <= 6.0
    assert df.loc[influx_end, "Bonds"].iloc[-1] > 3 * df.loc[df["time_ms"] <= 2.0, "Bonds"].iloc[-1]   # influx binds
    assert df["Bonds"].iloc[-1] < df["Bonds"].max()                                                    # pump-out unbinds
    assert df["n_broken"].is_monotonic_increasing and df["n_broken"].iloc[-1] > 0

    log = (rep / "log.lammps").read_text()
    before = float(log.split("\nPE_BEFORE ")[1].split()[0])
    after = float(log.split("\nPE_AFTER ")[1].split()[0])
    assert after == pytest.approx(before, rel=1e-12)
    dumped = {s: len(b) for s, b in read_bonds(rep / "bonds.dump")}
    thermo = dict(zip(df["Step"], df["Bonds"]))
    assert all(thermo[s] == n for s, n in dumped.items() if s in thermo)

    import matplotlib
    matplotlib.use("Agg")
    from qtft import make_report
    out = make_report(run, every=2, structure_analysis=False)
    assert (out / "figures" / "calcium.pdf").exists()
    assert (out / "figures" / "binding_rate.pdf").exists() == (rise == 0)      # only with a calcium step
    assert (out / "bond_rates.csv").exists()
    final = pd.read_csv(out / "final_state.csv", index_col=0)
    assert {"bonds_peak", "t_half_unbind_bonds_ms"} <= set(final.index)
