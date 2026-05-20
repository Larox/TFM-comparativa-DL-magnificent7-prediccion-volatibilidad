"""Tests that YAML config files parse and have the expected fields."""

import yaml

from tfm_volatility import config


def _load(name: str) -> dict:
    return yaml.safe_load((config.CONFIGS_DIR / name).read_text())


def test_garch_config_parses_with_expected_fields():
    cfg = _load("garch.yaml")
    assert cfg["p"] == 1
    assert cfg["q"] == 1
    assert cfg["mean"] == "Constant"
    assert cfg["distributions"] == ["normal", "t", "ged"]
    assert cfg["rescale"] is True


def test_deepar_config_parses_with_expected_fields():
    cfg = _load("deepar.yaml")
    assert cfg["target"] == "log_rv"
    assert cfg["encoder_length"] == 60
    assert cfg["prediction_length"] == 21
    assert cfg["hidden_size"] == 32
    assert cfg["likelihood"] == "Normal"
