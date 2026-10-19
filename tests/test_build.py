import json
import re

import pytest

from qtft import Params, build_run
from qtft.build import PLACEHOLDER, TEMPLATE, render, replica_seeds, template_values


def test_template_placeholders_all_have_values():
    p = Params()
    values = template_values(p, 1, replica_seeds(1, 1)[0])
    missing = set(PLACEHOLDER.findall(TEMPLATE.read_text())) - set(values)
    assert not missing


def test_render_fails_on_unknown_placeholder():
    with pytest.raises(KeyError):
        render({}, "pair_coeff 1 1 {{nope}}")


def test_seeds_independent_and_valid():
    seeds = replica_seeds(12345, 3)
    flat = [v for s in seeds for v in s.values()]
    assert len(set(flat)) == len(flat)
    assert all(1 <= v <= 900_000_000 for v in flat)
    assert replica_seeds(12345, 3) == seeds          # reproducible


def test_build_run(tmp_path):
    p = Params(n_replicas=2, data_dir=str(tmp_path), run_name="test")
    run_dir = build_run(p)
    text = (run_dir / "rep_01" / "in.lammps").read_text()
    assert "{{" not in text
    assert "pair_coeff      1 2 1.245 40.5" in text
    assert "bond_coeff      1 1.66 40.5" in text
    gammas = [float(g) for g in re.findall(r"gamma_t (\S+)", text)]
    assert gammas == pytest.approx([p.kT / p.D_qt, p.kT / p.D_ft])
    assert (run_dir / "rep_01" / "in.lammps").read_text() != (run_dir / "rep_02" / "in.lammps").read_text()
    record = json.loads((run_dir / "params.json").read_text())
    assert record["params"]["n_replicas"] == 2
    with pytest.raises(FileExistsError):
        build_run(p)
    build_run(p, overwrite=True)
