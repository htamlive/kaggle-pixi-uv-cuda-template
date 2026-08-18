# Research stack example

This is a standalone, opt-in example containing the heavier FAISS GPU,
torchvision/audio, PyTorch3D, Detectron2, Ninja, and FlashAttention stack. It
does not affect the repository-root environment.

From this directory:

```bash
pixi install --locked
pixi run setup
pixi run check-gpu
```

Several dependencies compile CUDA/C++ extensions. Expect a long first build,
substantial disk use, and failures on low-memory machines. Keep the NVIDIA
driver new enough for CUDA 12.4. To change Python, CUDA, or PyTorch, update the
constraints together and regenerate both locks with `pixi lock` and
`pixi run uv lock`.
