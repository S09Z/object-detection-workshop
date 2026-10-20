"""DVC stage `train`: fine-tune RF-DETR with standard checkpoint practice, logged to MLflow.

    uv run vd-train --config configs/train/rfdetr_small_rtx5070.yaml
    uv run vd-train --config ... --resume auto          # continue from <output_dir>/last.ckpt
    uv run vd-train --config ... --resume path/to/last.ckpt

After training:
  1. load-test checkpoint_best_total.pth (catches truncated/corrupt files)
  2. prune old interval archives (keep `keep_last_checkpoints`)
  3. log to the same MLflow run: checkpoint, configs, checksums; tags git SHA + DVC data rev
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from pathlib import Path

from vehicle_detect.checkpoints import (
    BEST,
    dvc_data_rev,
    git_info,
    prune_archives,
    resolve_resume,
    sha256,
)
from vehicle_detect.config import load_config
from vehicle_detect.models import build_rfdetr

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RF-DETR")
    parser.add_argument("--config", default="configs/train/rfdetr_small.yaml")
    parser.add_argument("--resume", default=None, help="'auto' or a path to last.ckpt")
    return parser.parse_args()


def find_run_id(experiment: str, run_name: str) -> str | None:
    """RF-DETR's Lightning MLFlowLogger owns the run; find it by name to attach artifacts."""
    import mlflow

    runs = mlflow.search_runs(
        experiment_names=[experiment],
        filter_string=f"tags.mlflow.runName = '{run_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    return runs[0].info.run_id if runs else None


def log_lineage(
    run_id: str, cfg_path: str, cfg: dict, best: Path, resumed_from: str | None
) -> None:
    import mlflow

    classes = load_config("configs/data/thai_traffic.yaml")["classes"]
    meta = {
        "checkpoint": best.name,
        "sha256": sha256(best),
        "size_mb": round(best.stat().st_size / 2**20, 1),
        "classes": classes,
        "config": cfg_path,
        **git_info(),
        "dvc_data_rev": dvc_data_rev(),
        "resumed_from": resumed_from or "",
    }
    with mlflow.start_run(run_id=run_id):
        mlflow.set_tags({k: str(v) for k, v in meta.items() if k not in {"classes"}})
        mlflow.log_params(
            {f"cfg.{k}": v for k, v in cfg.items() if not isinstance(v, dict)}
            | {f"cfg.train_kwargs.{k}": v for k, v in cfg.get("train_kwargs", {}).items()}
        )
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "model_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            shutil.copy2(cfg_path, Path(tmp, "train_config.yaml"))
            shutil.copy2("configs/data/thai_traffic.yaml", Path(tmp, "data_config.yaml"))
            mlflow.log_artifacts(tmp, artifact_path="model")
        if cfg.get("log_checkpoint_artifact", True):
            mlflow.log_artifact(str(best), artifact_path="model")
    print(json.dumps({"mlflow_run_id": run_id, **meta}, indent=2))


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_dir = Path(cfg["output_dir"])

    # Must be set before rfdetr/lightning are imported (MLFlowLogger reads it at import).
    os.environ.setdefault("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)
    mlflow_cfg = cfg.get("mlflow", {})
    experiment = mlflow_cfg.get("experiment", "vehicle-detect")
    run_name = mlflow_cfg.get("run_name") or f"{cfg['model']}-{time.strftime('%Y%m%d-%H%M%S')}"
    resume = resolve_resume(args.resume, out_dir)
    if resume:
        run_name = f"{run_name}-resume"
        print(f"Resuming from {resume}")

    model = build_rfdetr(cfg["model"], **cfg.get("model_kwargs", {}))
    model.train(
        dataset_dir=cfg["dataset_dir"],
        output_dir=str(out_dir),
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        grad_accum_steps=cfg["grad_accum_steps"],
        lr=cfg["lr"],
        seed=cfg.get("seed", 42),
        resume=resume,
        best_model_metric=cfg.get("best_model_metric", "map"),
        early_stopping=cfg.get("early_stopping", False),
        early_stopping_patience=cfg.get("early_stopping_patience", 10),
        checkpoint_interval=cfg.get("checkpoint_interval", 10),
        run_test=cfg.get("run_test", True),
        mlflow=True,
        project=experiment,
        run=run_name,
        tensorboard=False,
        **cfg.get("train_kwargs", {}),
    )

    best = out_dir / BEST
    if not best.exists():
        raise SystemExit(f"Training finished but {best} is missing")
    from rfdetr import RFDETR

    RFDETR.from_checkpoint(str(best))  # load test: fails loudly on a corrupt/truncated file
    removed = prune_archives(out_dir, cfg.get("keep_last_checkpoints", 2))
    if removed:
        print(f"Pruned {len(removed)} old interval checkpoints")

    if run_id := find_run_id(experiment, run_name):
        log_lineage(run_id, args.config, cfg, best, resume)
    else:
        print(f"WARNING: MLflow run '{run_name}' not found — lineage not logged")


if __name__ == "__main__":
    main()
