from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from kaggle_entrypoint import discover, run, stage


ROOT = Path(__file__).resolve().parents[1]


def write_project(path: Path, *, uv: bool = False) -> None:
    path.mkdir(parents=True)
    (path / "scripts").mkdir()
    (path / "pixi.toml").write_text("[workspace]\nname='test'\nchannels=[]\nplatforms=['linux-64']\n")
    (path / "scripts/check_cuda.py").write_text("pass\n")
    if uv:
        (path / "pyproject.toml").write_text("[project]\nname='test'\nversion='0.1.0'\n")
        (path / ".python-version").write_text("3.12\n")


def zip_project(source: Path, output: Path, wrapper: str = "") -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w") as archive:
        for path in source.rglob("*"):
            if path.is_file():
                name = path.relative_to(source).as_posix()
                archive.write(path, f"{wrapper}/{name}" if wrapper else name)


class EntrypointTests(unittest.TestCase):
    def test_unpacked_pixi_only_and_uv_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for uv in (False, True):
                source = temp / f"source-{uv}"
                write_project(source, uv=uv)
                self.assertEqual(discover(source), ("directory", source))
                result = stage(source, temp / f"work-{uv}")
                self.assertEqual((result / "pyproject.toml").exists(), uv)

    def test_flat_and_single_directory_zips(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for wrapper in ("", "my-project"):
                source = temp / f"source-{wrapper or 'flat'}"
                write_project(source, uv=True)
                input_root = temp / f"input-{wrapper or 'flat'}"
                zip_project(source, input_root / "manual-name.zip", wrapper)
                self.assertEqual(discover(input_root)[0], "zip")
                result = stage(input_root, temp / f"work-{wrapper or 'flat'}")
                self.assertTrue((result / "pixi.toml").is_file())
                self.assertTrue((result / "pyproject.toml").is_file())

    def test_run_syncs_only_uv_projects(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            for uv in (False, True):
                source = temp / f"run-{uv}"
                write_project(source, uv=uv)
                with patch("kaggle_entrypoint.find_or_install_pixi", return_value=Path("/pixi")), \
                     patch("kaggle_entrypoint.subprocess.run") as execute:
                    run(source, temp / f"run-work-{uv}")
                commands = [call.args[0] for call in execute.call_args_list]
                self.assertEqual(commands[0], ["/pixi", "install"])
                self.assertEqual(commands[-1], ["/pixi", "run", "check-gpu"])
                self.assertEqual(["/pixi", "run", "uv", "sync"] in commands, uv)

    def test_missing_ambiguous_and_malformed_inputs_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            with self.assertRaises(RuntimeError):
                discover(temp)
            write_project(temp / "one")
            write_project(temp / "two")
            with self.assertRaises(RuntimeError):
                discover(temp)
            malformed = temp / "malformed.zip"
            malformed.write_bytes(b"not a zip")
            with self.assertRaises(zipfile.BadZipFile):
                discover(malformed.parent)

    def test_unsafe_archive_members_are_rejected(self):
        cases = ("../escape", "/absolute", "bad\\windows")
        for index, name in enumerate(cases):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as tmp:
                archive_path = Path(tmp) / f"bad-{index}.zip"
                with zipfile.ZipFile(archive_path, "w") as archive:
                    archive.writestr(name, b"bad")
                with self.assertRaises(ValueError):
                    discover(Path(tmp))

        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "symlink.zip"
            info = zipfile.ZipInfo("link")
            info.create_system = 3
            info.external_attr = (stat.S_IFLNK | 0o777) << 16
            with zipfile.ZipFile(archive_path, "w") as archive:
                archive.writestr(info, "target")
            with self.assertRaises(ValueError):
                discover(Path(tmp))

    def test_zip_with_multiple_projects_is_not_discoverable(self):
        with tempfile.TemporaryDirectory() as tmp:
            archive_path = Path(tmp) / "ambiguous.zip"
            with zipfile.ZipFile(archive_path, "w") as archive:
                for root in ("one", "two"):
                    archive.writestr(f"{root}/pixi.toml", "[workspace]\n")
                    archive.writestr(f"{root}/scripts/check_cuda.py", "pass\n")
            with self.assertRaises(RuntimeError):
                discover(Path(tmp))


if __name__ == "__main__":
    unittest.main()
