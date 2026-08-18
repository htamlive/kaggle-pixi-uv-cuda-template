"""Fast, actionable PyTorch/CUDA smoke test."""

import sys
import warnings

warnings.filterwarnings("ignore", message="Failed to initialize NumPy")
import torch


def main() -> int:
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"PyTorch CUDA runtime: {torch.version.cuda}")
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. Check the NVIDIA driver and GPU allocation.", file=sys.stderr)
        return 1
    device = torch.device("cuda")
    value = (torch.ones(8, device=device) * 2).sum().item()
    if value != 16:
        print(f"ERROR: CUDA computation returned {value}, expected 16.", file=sys.stderr)
        return 1
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} (compute capability {props.major}.{props.minor})")
    print("CUDA smoke test passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
