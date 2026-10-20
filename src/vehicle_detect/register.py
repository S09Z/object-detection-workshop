"""Register a trained run's `model/` artifacts in the MLflow Model Registry and set an alias.

    uv run vd-register --run-id <id>                       # -> alias "candidate"
    uv run vd-register --run-id <id> --alias production    # only after eval + 60 FPS bench pass

RF-DETR checkpoints aren't MLflow "flavor" models, so we register the artifact folder as the
version source (MlflowClient.create_model_version) instead of mlflow.register_model.
"""

from __future__ import annotations

import argparse
import contextlib
import os

DEFAULT_NAME = "vehicle-detect-rfdetr"


def main() -> None:
    parser = argparse.ArgumentParser(description="Register model version + alias")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", default=DEFAULT_NAME)
    parser.add_argument("--alias", default="candidate", choices=["candidate", "production"])
    args = parser.parse_args()

    os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    from mlflow import MlflowClient
    from mlflow.exceptions import MlflowException

    client = MlflowClient()
    run = client.get_run(args.run_id)
    with contextlib.suppress(MlflowException):  # already exists
        client.create_registered_model(args.name)

    version = client.create_model_version(
        name=args.name,
        source=f"{run.info.artifact_uri}/model",
        run_id=args.run_id,
        tags={k: v for k, v in run.data.tags.items() if k in {"git_sha", "dvc_data_rev", "sha256"}},
    )
    client.set_registered_model_alias(args.name, args.alias, version.version)
    print(f"{args.name} v{version.version} -> @{args.alias}  (run {args.run_id})")


if __name__ == "__main__":
    main()
