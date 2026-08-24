"""serve_after: the pipeline serves the crosscoder it just trained when the run asks for it."""

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from diffing.serving.launch import (
    SERVE_AFTER_KEYS,
    serve_after_binding,
    serve_after_diffing,
)


def config(serve_after=None, method="crosscoder"):
    body = {"diffing": {"method": {"name": method}}}
    if serve_after is not None:
        body["diffing"]["method"]["serve_after"] = serve_after
    return OmegaConf.create(body)


def test_a_method_declaring_no_serve_after_block_serves_nothing():
    assert serve_after_binding(config()) is None
    assert serve_after_diffing(config(), "diffing") is False
    assert serve_after_binding(config(method="diff_mining")) is None


def test_a_disabled_serve_after_block_serves_nothing():
    disabled = config({"enabled": False, "host": "0.0.0.0", "port": 8000, "queue": 32})

    assert serve_after_binding(disabled) is None
    assert serve_after_diffing(disabled, "diffing") is False


def test_an_enabled_block_gives_the_binding_the_server_runs_on():
    enabled = config({"enabled": True, "host": "0.0.0.0", "port": 8123, "queue": 8})

    assert serve_after_binding(enabled) == {"host": "0.0.0.0", "port": 8123, "queue": 8}


def test_a_serve_after_block_missing_its_binding_stops_the_run():
    with pytest.raises(AssertionError, match=r"lacks \['port', 'queue'\]"):
        serve_after_binding(config({"enabled": True, "host": "0.0.0.0"}))

    with pytest.raises(AssertionError, match=r"lacks \['enabled'\]"):
        serve_after_binding(config({"host": "0.0.0.0", "port": 8000, "queue": 32}))


def test_only_a_mode_that_ran_the_diffing_stage_serves(monkeypatch):
    served = []
    monkeypatch.setattr(
        "diffing.serving.launch.serve_crosscoder",
        lambda cfg, host, port, queue: served.append(mode),
    )
    enabled = config({"enabled": True, "host": "0.0.0.0", "port": 8123, "queue": 8})

    for mode in ("preprocessing", "evaluation"):
        assert serve_after_diffing(enabled, mode) is False
    for mode in ("full", "diffing", "no_evaluation"):
        assert serve_after_diffing(enabled, mode) is True
    assert served == ["full", "diffing", "no_evaluation"]


def test_the_shipped_crosscoder_config_does_not_serve_after_diffing():
    shipped = OmegaConf.load(
        Path(__file__).resolve().parents[2] / "configs/diffing/method/crosscoder.yaml"
    )

    assert set(shipped.serve_after) == SERVE_AFTER_KEYS
    assert shipped.serve_after.enabled is False


def test_the_pipeline_serves_the_run_it_just_trained(monkeypatch):
    served = {}
    monkeypatch.setattr(
        "diffing.serving.launch.serve_crosscoder",
        lambda cfg, host, port, queue: served.update(host=host, port=port, queue=queue),
    )
    enabled = config({"enabled": True, "host": "10.0.0.3", "port": 8123, "queue": 8})

    assert serve_after_diffing(enabled, "diffing") is True
    assert served == {"host": "10.0.0.3", "port": 8123, "queue": 8}


def test_a_binding_written_as_text_reaches_the_server_as_numbers():
    quoted = config({"enabled": True, "host": "0.0.0.0", "port": "8123", "queue": "8"})

    assert serve_after_binding(quoted) == {"host": "0.0.0.0", "port": 8123, "queue": 8}
