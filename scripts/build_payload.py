"""Build a deterministic, allowlisted Kaggle dataset archive."""

from __future__ import annotations

import argparse
from pathlib import Path, PurePosixPath
import zipfile

FILES = (
    ".python-version",
    "README.md",
    "pixi.lock",
    "pixi.toml",
    "pyproject.toml",
    "scripts/check_cuda.py",
    "uv.lock",
)
PROFILES = {"base", "research"}


def build(profile: str, output: Path) -> Path:
    if profile not in PROFILES:
        raise ValueError(f"unknown profile: {profile}")
    root = Path(__file__).resolve().parents[1]
    source = root / "projects" / ("base" if profile == "base" else "research-stack")
    missing = [name for name in FILES if not (source / name).is_file()]
    if missing:
        raise FileNotFoundError(f"profile {profile} is missing: {', '.join(missing)}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name in FILES:
            path = source / name
            arcname = PurePosixPath(name).as_posix()
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, choices=sorted(PROFILES))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output or root / "upload_payload" / "repo.zip"
    print(build(args.profile, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
