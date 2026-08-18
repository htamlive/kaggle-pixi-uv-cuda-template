"""Safely stage and run a Pixi project on Kaggle."""

from __future__ import annotations

import argparse
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import tempfile
import urllib.request
import zipfile

REQUIRED = {"pixi.toml", "scripts/check_cuda.py"}


def _safe_members(archive: zipfile.ZipFile) -> list[zipfile.ZipInfo]:
    members = archive.infolist()
    for member in members:
        path = PurePosixPath(member.filename)
        mode = member.external_attr >> 16
        if (not member.filename or path.is_absolute() or ".." in path.parts
                or "\\" in member.filename or stat.S_ISLNK(mode)):
            raise ValueError(f"unsafe archive member: {member.filename!r}")
    return members


def _valid_project(path: Path) -> bool:
    return all((path / name).is_file() for name in REQUIRED)


def _archive_project_root(members: list[zipfile.ZipInfo]) -> PurePosixPath | None:
    files = {PurePosixPath(item.filename) for item in members if not item.is_dir()}
    roots = {path.parent for path in files if path.name == "pixi.toml" and len(path.parent.parts) <= 1}
    valid = [root for root in roots if all(root / name in files for name in REQUIRED)]
    return valid[0] if len(valid) == 1 else None


def discover(input_root: Path) -> tuple[str, Path]:
    """Return exactly one ('directory'|'zip', path) project candidate."""
    candidates: list[tuple[str, Path]] = []
    for manifest in input_root.rglob("pixi.toml"):
        if _valid_project(manifest.parent):
            candidates.append(("directory", manifest.parent))
    for archive_path in input_root.rglob("*.zip"):
        with zipfile.ZipFile(archive_path) as archive:
            if _archive_project_root(_safe_members(archive)) is not None:
                candidates.append(("zip", archive_path))
    if len(candidates) != 1:
        raise RuntimeError(f"expected exactly one project input, found {len(candidates)}")
    return candidates[0]


def stage(input_root: Path, working_dir: Path) -> Path:
    kind, source = discover(input_root)
    if working_dir.exists():
        shutil.rmtree(working_dir)
    working_dir.mkdir(parents=True)
    if kind == "directory":
        shutil.copytree(source, working_dir, dirs_exist_ok=True)
    else:
        with zipfile.ZipFile(source) as archive:
            members = _safe_members(archive)
            root = _archive_project_root(members)
            if root is None:
                raise RuntimeError("archive must contain exactly one Pixi project")
            archive.extractall(working_dir, members=members)
        if root.parts:
            extracted = working_dir / root
            for item in extracted.iterdir():
                shutil.move(str(item), working_dir / item.name)
            extracted.rmdir()
    if not _valid_project(working_dir):
        raise RuntimeError("staged project is incomplete")
    return working_dir


def find_or_install_pixi() -> Path:
    found = shutil.which("pixi")
    if found:
        return Path(found)
    expected = Path.home() / ".pixi" / "bin" / "pixi"
    if expected.is_file():
        return expected
    with tempfile.NamedTemporaryFile(suffix=".sh") as installer:
        urllib.request.urlretrieve("https://pixi.sh/install.sh", installer.name)
        subprocess.run(["sh", installer.name], check=True)
    if not expected.is_file():
        raise RuntimeError("Pixi installer completed but the binary was not found")
    return expected


def run(input_root: Path, working_dir: Path, *, execute: bool = True) -> Path:
    project = stage(input_root.resolve(), working_dir.resolve())
    if execute:
        pixi = find_or_install_pixi()
        env = os.environ.copy()
        subprocess.run([str(pixi), "install"], cwd=project, env=env, check=True)
        if (project / "pyproject.toml").is_file():
            subprocess.run([str(pixi), "run", "uv", "sync"], cwd=project, env=env, check=True)
        subprocess.run([str(pixi), "run", "check-gpu"], cwd=project, env=env, check=True)
    return project


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--working-dir", type=Path, default=Path("/kaggle/working/pixi_project"))
    parser.add_argument("--stage-only", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    run(args.input_root, args.working_dir, execute=not args.stage_only)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
