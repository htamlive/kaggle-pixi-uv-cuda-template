"""Fast, actionable PyTorch/CUDA smoke test."""

import sys
import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}; CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is unavailable; check the driver and GPU allocation.", file=sys.stderr)
        return 1
    result = (torch.ones(8, device="cuda") * 2).sum().item()
    if result != 16:
        print(f"ERROR: CUDA computation returned {result}, expected 16.", file=sys.stderr)
        return 1
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("CUDA smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
