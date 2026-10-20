import json
from pathlib import Path

import pytest

from vehicle_detect.benchmark import summarize


def test_passes_when_fast_and_stable():
    report = summarize([10.0] * 1000, {"detect": [8.0] * 1000})
    assert report.fps == pytest.approx(100.0)
    assert report.passed


def test_fails_on_tail_latency_even_if_mean_fps_is_high():
    # mean ~ 11.9 ms (84 FPS) but 2% of frames take 110 ms -> p99 blows the 16.7 ms budget
    frames = [10.0] * 980 + [110.0] * 20
    report = summarize(frames, {})
    assert report.fps > 60
    assert report.p99_ms > 16.7
    assert not report.passed


@pytest.mark.gpu
def test_real_engine_meets_budget():
    """On the target device: `make bench` then `uv run pytest -m gpu`."""
    report_path = Path("reports/bench.json")
    if not report_path.exists():
        pytest.skip("run `make bench` on the target device first")
    report = json.loads(report_path.read_text())
    assert report["passed"], f"FPS={report['fps']} p99={report['p99_ms']}ms"
