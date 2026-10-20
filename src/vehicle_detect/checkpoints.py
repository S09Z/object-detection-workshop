"""Checkpoint hygiene + run lineage: resume lookup, pruning, checksums, git/DVC provenance.

RF-DETR writes into `output_dir`:
  checkpoint_best_total.pth   best of EMA/regular by val `best_model_metric` (the one we ship)
  checkpoint_best_ema.pth / checkpoint_best_regular.pth
  last.ckpt                   latest full training state (overwritten every epoch) -> resume
  checkpoint_<epoch>.ckpt     interval archives every `checkpoint_interval` epochs (kept forever)
"""

from __future__ import annotations

import hashlib
import re
import subprocess
from pathlib import Path

BEST = "checkpoint_best_total.pth"
LAST = "last.ckpt"
_ARCHIVE = re.compile(r"checkpoint_(?:epoch=)?(\d+)\.ckpt$")


def resolve_resume(resume: str | None, output_dir: str | Path) -> str | None:
    """None -> fresh run; "auto" -> <output_dir>/last.ckpt if present; else an explicit path."""
    if not resume:
        return None
    if resume == "auto":
        last = Path(output_dir) / LAST
        return str(last) if last.exists() else None
    path = Path(resume)
    if not path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {path}")
    return str(path)


def prune_archives(output_dir: str | Path, keep_last: int) -> list[Path]:
    """Delete all but the newest `keep_last` interval archives. Never touches best/last files."""
    archives = []
    for p in Path(output_dir).glob("checkpoint_*.ckpt"):
        if m := _ARCHIVE.search(p.name):
            archives.append((int(m.group(1)), p))
    archives.sort()
    doomed = [p for _, p in archives[: max(len(archives) - keep_last, 0)]]
    for p in doomed:
        p.unlink()
    return doomed


def sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def git_info() -> dict[str, str]:
    """Commit SHA + dirty flag, so every run can be traced back to the exact code."""

    def run(*args: str) -> str:
        return subprocess.run(args, capture_output=True, text=True, check=False).stdout.strip()

    sha = run("git", "rev-parse", "HEAD")
    if not sha:
        return {"git_sha": "unknown", "git_dirty": "unknown"}
    dirty = bool(run("git", "status", "--porcelain", "--untracked-files=no"))
    return {"git_sha": sha, "git_dirty": str(dirty).lower()}


def dvc_data_rev(lock_path: str | Path = "dvc.lock", out: str = "data/processed/coco") -> str:
    """md5 of the `prepare` output recorded in dvc.lock (dataset version used for training)."""
    path = Path(lock_path)
    if not path.exists():
        return "unknown"
    import yaml

    lock = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    for o in lock.get("stages", {}).get("prepare", {}).get("outs", []):
        if o.get("path") == out:
            return str(o.get("md5", "unknown"))
    return "unknown"
