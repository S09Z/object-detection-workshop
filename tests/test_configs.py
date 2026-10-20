from pathlib import Path

import pytest

from vehicle_detect.config import load_config

TRAIN_CONFIGS = sorted(Path("configs/train").glob("*.yaml"))
REQUIRED = {"model", "dataset_dir", "output_dir", "epochs", "batch_size", "grad_accum_steps", "lr"}


@pytest.mark.parametrize("path", TRAIN_CONFIGS, ids=lambda p: p.stem)
def test_train_config_has_required_keys(path):
    cfg = load_config(path)
    assert cfg.keys() >= REQUIRED, REQUIRED - cfg.keys()
    assert cfg["model"] in {"nano", "small", "medium"}


@pytest.mark.parametrize(
    "path", [p for p in TRAIN_CONFIGS if "rtx5070" in p.stem], ids=lambda p: p.stem
)
def test_rtx5070_configs_keep_effective_batch_16(path):
    cfg = load_config(path)
    assert cfg["batch_size"] * cfg["grad_accum_steps"] == 16
    assert cfg["train_kwargs"]["amp_dtype"] == "bf16"
