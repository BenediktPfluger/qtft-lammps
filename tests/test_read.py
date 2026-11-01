"""Thermo readers on hand-made log.lammps files: numeric parsing equals the YAML content."""

import yaml

from qtft.read import last_thermo, read_thermo

EQUIL = """LAMMPS (30 Sep 2026)
---
keywords: ['Step', 'PotEng', 'Bonds']
data:
  - [0, 0, 0, ]
  - [100, 1.5, 0, ]
...
Loop time of 1.0
"""
PROD = """---
keywords: ['Step', 'PotEng', 'Bonds', 'v_ca']
data:
  - [0, 2.25, 0, 0.1, ]
WARNING: Inconsistent image flags (src/domain.cpp:1051)
  - [500, 3.5e-05, 7, 1, ]
...
---
keywords: ['Step', 'PotEng', 'Bonds', 'v_ca']
data:
  - [500, 3.5e-05, 7, 1, ]
  - [1000, -4, 12, 0.55, ]
...
"""


def test_read_thermo_matches_yaml(tmp_path):
    log = tmp_path / "log.lammps"
    log.write_text(EQUIL + PROD)
    blocks = read_thermo(log)
    assert [len(b) for b in blocks] == [2, 2, 2]
    assert list(blocks[1].columns) == ["Step", "PotEng", "Bonds", "v_ca"]
    expected = yaml.safe_load(PROD.split("...")[0].replace("WARNING: Inconsistent image flags (src/domain.cpp:1051)\n", ""))
    assert blocks[1].to_numpy().tolist() == expected["data"]
    assert str(blocks[1]["Bonds"].dtype) == "int64" and str(blocks[1]["PotEng"].dtype) == "float64"
    assert str(blocks[0]["Step"].dtype) == "int64"


def test_last_thermo_while_running(tmp_path):
    log = tmp_path / "log.lammps"
    assert last_thermo(log) is None
    log.write_text("LAMMPS (30 Sep 2026)\n")
    assert last_thermo(log) is None
    log.write_text(EQUIL)
    assert last_thermo(log) == ("equilibration", {"Step": 100, "PotEng": 1.5, "Bonds": 0})
    log.write_text(EQUIL + PROD + "---\nkeywords: ['Step', 'PotEng', 'Bonds', 'v_ca']\ndata:\n  - [1000, -4, 12, 0.55, ]\n"
                   "  - [1500, -5.5, 1")                                       # last row half-written
    assert last_thermo(log) == ("production", {"Step": 1000, "PotEng": -4, "Bonds": 12, "v_ca": 0.55})
    log.write_text(EQUIL + "---\nkeywords: ['Step', 'PotEng', 'Bonds', 'v_ca']\n")   # production without a row
    assert last_thermo(log) == ("equilibration", {"Step": 100, "PotEng": 1.5, "Bonds": 0})
