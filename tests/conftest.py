"""Shared helpers for the tests."""


def influx(steps: int, segment: int = 500, timestep: float = 500.0) -> dict:
    """Calcium protocol with only an influx phase of `steps` production steps (constant rates, the
    agglomeration of the tests before the calcium control), updated every `segment` steps."""
    return {"t_baseline_s": 0.0, "t_influx_s": steps * timestep * 1e-9, "t_pumpout_s": 0.0,
            "ca_rise_s": 0.0, "rate_update_every": segment}
