"""Model factory — keeps the framework choice in one place."""

from __future__ import annotations

from typing import Any


def build_rfdetr(size: str, **kwargs: Any) -> Any:
    """Return an RF-DETR model (Apache-2.0 for nano/small/medium).

    kwargs are forwarded, e.g. pretrain_weights="runs/.../checkpoint_best_total.pth", num_classes=9.
    """
    import rfdetr  # imported lazily so CPU-only tooling (tests, ruff) stays fast

    classes = {
        "nano": rfdetr.RFDETRNano,
        "small": rfdetr.RFDETRSmall,
        "medium": rfdetr.RFDETRMedium,
    }
    if size not in classes:
        raise ValueError(f"Unknown RF-DETR size {size!r}; choose from {sorted(classes)}")
    return classes[size](**kwargs)
