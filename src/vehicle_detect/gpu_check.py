"""Check that PyTorch can really train on this GPU (e.g. RTX 5070 / Blackwell sm_120).

Run:  uv run vd-gpu-check
Fails (exit 1) if CUDA is missing or the installed wheel wasn't compiled for this GPU's arch —
the classic RTX 50xx error is "no kernel image is available for execution on the device".
"""

from __future__ import annotations

import sys


def main() -> None:
    import torch

    print(f"torch {torch.__version__} · CUDA build {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("FAIL: CUDA not available — check driver (nvidia-smi) and the cu130 torch wheel")
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    major, minor = torch.cuda.get_device_capability(0)
    arch = f"sm_{major}{minor}"
    compiled = torch.cuda.get_arch_list()
    print(f"GPU: {props.name} · {props.total_memory / 2**30:.1f} GB · {arch}")
    print(f"wheel archs: {' '.join(compiled)}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    if arch not in compiled and f"compute_{major}{minor}" not in compiled:
        print(f"FAIL: wheel has no kernels for {arch}; reinstall with the cu130 index (uv sync)")
        sys.exit(1)

    # real kernel launch + bf16 matmul (catches 'no kernel image' errors early)
    x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
    torch.cuda.synchronize()
    (x @ x).sum().item()
    print("OK: CUDA kernels run on this GPU")


if __name__ == "__main__":
    main()
