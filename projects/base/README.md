# Base CUDA project

Minimal Pixi-only Linux x86-64 PyTorch 2.4.1 + CUDA 12.4 project for the Kaggle
template. Python and PyTorch are configured once in `pixi.toml`; there is no uv
environment. Use `pixi install`, then `pixi run check-gpu` on a GPU runner.
