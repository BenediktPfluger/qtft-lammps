"""Shared helpers for the tests."""

import os

import pytest


@pytest.fixture(autouse=True)
def no_lrz(tmp_path_factory, monkeypatch):
    """Tests never reach the real LRZ: ssh and rsync fail (an open ssh connection to LRZ would otherwise
    let start() of an executor="lrz" run submit a real job). The fake_lrz fixture puts its own in front."""
    bin_dir = tmp_path_factory.getbasetemp() / "no_lrz_bin"
    if not bin_dir.exists():
        bin_dir.mkdir()
        for name in ("ssh", "rsync"):
            (bin_dir / name).write_text('#!/bin/sh\necho "tests must not reach LRZ" >&2\nexit 255\n')
            (bin_dir / name).chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")


def influx(steps: int, segment: int = 500, timestep: float = 500.0) -> dict:
    """Calcium protocol with only an influx phase of `steps` production steps (constant rates, the
    agglomeration of the tests before the calcium control), updated every `segment` steps."""
    return {"t_baseline_s": 0.0, "t_influx_s": steps * timestep * 1e-9, "t_pumpout_s": 0.0,
            "ca_rise_s": 0.0, "rate_update_every": segment}
