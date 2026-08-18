# Research stack example

This is a standalone, opt-in example containing the heavier FAISS GPU,
torchvision/audio, PyTorch3D, Detectron2, Ninja, and FlashAttention stack. It
does not affect the repository-root environment.

From this directory:

```bash
pixi install
pixi run uv sync
pixi run check-gpu
```

Several dependencies compile CUDA/C++ extensions. Expect a long first build,
substantial disk use, and failures on low-memory machines. Keep the NVIDIA
driver new enough for CUDA 12.4. The nested environment is an intentional
research-only tradeoff: Pixi supplies uv, Toolkit/NVCC, and Ninja, while uv
manages Python from `.python-version` and installs packages into `.venv`.
Generated locks are ignored.

> **Important:** if you remove `cuda-toolkit` or `cuda-nvcc`, also remove
> `CUDA_HOME` from `pixi.toml`.
