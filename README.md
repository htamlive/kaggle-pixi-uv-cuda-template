# Kaggle Pixi CUDA template

This repository stages and runs a manually created PyTorch/CUDA project archive
from a GPU-enabled Kaggle script. No lockfiles are committed.

## Choose and ZIP a profile

The independent projects live in `projects/base/` and `projects/research-stack/`.
Choose what your Dataset should contain and create `upload_payload/repo.zip`
yourself. Either a flat archive or one top-level project directory is accepted:

```bash
cd projects/base
zip -r ../../upload_payload/repo.zip pixi.toml README.md scripts

# Or preserve one wrapper directory for the research profile:
cd ../..
zip -r upload_payload/repo.zip projects/research-stack \
  -x '*/.pixi/*' '*/.venv/*' '*/pixi.lock' '*/uv.lock'
```

The ZIP is ignored by Git. The entrypoint also accepts an unpacked project.

## Publish to Kaggle

Replace every `YOUR_KAGGLE_USERNAME` in `upload_payload/dataset-metadata.json`
and `kernel-metadata.json`. Both resources are private by default. With Kaggle
API credentials configured:

```bash
cd upload_payload
kaggle datasets create -p .
# For later revisions:
kaggle datasets version -p . -m "Update CUDA project"
cd ..
kaggle kernels push -p .
```

The kernel needs Internet enabled to install Pixi and packages, and GPU enabled
for `check-gpu`. The entrypoint requires exactly one discoverable Pixi project
and rejects absolute paths, traversal, backslashes, symlinks, and malformed
layouts. It runs `pixi install`, conditionally runs `pixi run uv sync` when a
`pyproject.toml` exists, then runs the shared `check-gpu` task.

## Profiles and local use

The base profile is Pixi-only: Pixi installs Python 3.12 and GPU-enabled
PyTorch, and `check-gpu` runs directly in that environment. Use `pixi install`
then `pixi run check-gpu`.

The research profile intentionally nests uv inside Pixi because compiled
extensions need Toolkit/NVCC. Pixi supplies uv and the build toolchain; uv
reads `.python-version`, downloads managed Python, creates `.venv`, and installs
the Python packages. Run `pixi install`, `pixi run uv sync`, then
`pixi run check-gpu`.

> **Important:** the research profile's `CUDA_HOME` points to Pixi's Toolkit.
> If you remove `cuda-toolkit` or `cuda-nvcc`, also remove `CUDA_HOME`.

If CUDA is unavailable, verify Kaggle's accelerator setting and `nvidia-smi`.
Research extension builds can require considerable RAM, disk, and time.

Copy, fork, or use this as a GitHub template. It is CC0: use and adapt it
freely; no credit is required. This repository does not publish anything to
Kaggle on your behalf.
