import hashlib

import pytest

from vehicle_detect.checkpoints import (
    BEST,
    LAST,
    dvc_data_rev,
    prune_archives,
    resolve_resume,
    sha256,
)


def test_resolve_resume(tmp_path):
    assert resolve_resume(None, tmp_path) is None
    assert resolve_resume("auto", tmp_path) is None  # nothing to resume yet -> fresh run
    (tmp_path / LAST).write_bytes(b"x")
    assert resolve_resume("auto", tmp_path) == str(tmp_path / LAST)
    with pytest.raises(FileNotFoundError):
        resolve_resume(str(tmp_path / "missing.ckpt"), tmp_path)


def test_prune_keeps_newest_archives_and_never_best_or_last(tmp_path):
    for epoch in (9, 19, 29, 39):
        (tmp_path / f"checkpoint_{epoch}.ckpt").write_bytes(b"x")
    (tmp_path / BEST).write_bytes(b"best")
    (tmp_path / LAST).write_bytes(b"last")

    removed = prune_archives(tmp_path, keep_last=2)

    assert sorted(p.name for p in removed) == ["checkpoint_19.ckpt", "checkpoint_9.ckpt"]
    left = sorted(p.name for p in tmp_path.iterdir())
    assert left == sorted([BEST, "checkpoint_29.ckpt", "checkpoint_39.ckpt", LAST])


def test_sha256_matches_hashlib(tmp_path):
    f = tmp_path / "w.pth"
    f.write_bytes(b"weights" * 1000)
    assert sha256(f, chunk=7) == hashlib.sha256(b"weights" * 1000).hexdigest()


def test_dvc_data_rev(tmp_path):
    assert dvc_data_rev(tmp_path / "dvc.lock") == "unknown"
    lock = tmp_path / "dvc.lock"
    lock.write_text(
        "schema: '2.0'\nstages:\n  prepare:\n    outs:\n"
        "    - path: data/processed/coco\n      md5: abc123.dir\n"
    )
    assert dvc_data_rev(lock) == "abc123.dir"
