# Kaggle Pixi + uv CUDA template

This repository packages a locked PyTorch/CUDA project as a small, reviewable
Kaggle Dataset, then stages and runs it from a GPU-enabled Kaggle script. The
default is Python 3.12, CUDA Toolkit/NVCC 12.4, and PyTorch 2.4.1 from the
explicit CUDA 12.4 wheel index.

## Choose and build a profile

The independently locked projects live in `projects/base/` and
`projects/research-stack/`. Review one, then generate the deterministic dataset
archive:

```bash
python scripts/build_payload.py --profile base
# or: python scripts/build_payload.py --profile research
```

Only `.python-version`, manifests, locks, the profile README, and CUDA check
script are allowed into `upload_payload/repo.zip`. The generated ZIP is ignored
by Git.

## Publish to Kaggle

Replace every `YOUR_KAGGLE_USERNAME` in `upload_payload/dataset-metadata.json`
and `kernel-metadata.json`. Both resources are private by default. With Kaggle
API credentials configured:

```bash
cd upload_payload
kaggle datasets create -p .
# For later revisions:
kaggle datasets version -p . -m "Update locked CUDA project"
cd ..
kaggle kernels push -p .
```

The kernel needs Internet enabled to install Pixi and download locked packages,
and GPU enabled for `check-gpu`. The entrypoint discovers exactly one unpacked
project or `repo.zip`, rejects unsafe ZIP paths, runs `pixi install --locked`,
syncs uv through the frozen `setup` task, and runs `check-gpu`.

## Local setup and customization

Inside either project, run `pixi install --locked` and `pixi run setup`; use
`pixi run check-gpu` on a GPU host. Keep Python, CUDA, PyTorch, and its wheel
index aligned when changing versions, then regenerate `pixi.lock` with
`pixi lock` and `uv.lock` with `pixi run uv lock`.

If CUDA is unavailable, verify Kaggle's accelerator setting and `nvidia-smi`.
For driver/runtime errors, use a driver compatible with CUDA 12.4 or select a
matching CUDA/PyTorch combination. Research-profile extension builds may need
considerable RAM, disk, and time; inspect compiler output and `CUDA_HOME`.

Copy, fork, or use this as a GitHub template. It is CC0: use and adapt it
freely; no credit is required. This repository does not publish anything to
Kaggle on your behalf.
