from pathlib import Path
import tempfile
import unittest
import zipfile

from kaggle_entrypoint import discover, stage
from scripts.build_payload import FILES, build


class PayloadTests(unittest.TestCase):
    def test_profiles_are_exact_and_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            for profile in ("base", "research"):
                first = build(profile, Path(tmp) / f"{profile}-first.zip")
                second = build(profile, Path(tmp) / f"{profile}-second.zip")
                self.assertEqual(first.read_bytes(), second.read_bytes())
                with zipfile.ZipFile(first) as archive:
                    self.assertEqual(archive.namelist(), list(FILES))
                    self.assertTrue(all(i.date_time == (1980, 1, 1, 0, 0, 0) for i in archive.infolist()))

    def test_directory_and_zip_staging(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            directory_input = temp / "dir-input"
            directory_input.mkdir()
            source = root / "projects" / "base"
            for name in FILES:
                target = directory_input / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / name).read_bytes())
            self.assertEqual(discover(directory_input)[0], "directory")
            self.assertTrue((stage(directory_input, temp / "work-one") / "pixi.lock").is_file())
            zip_input = temp / "zip-input"
            zip_input.mkdir()
            build("base", zip_input / "repo.zip")
            self.assertEqual(discover(zip_input)[0], "zip")
            self.assertTrue((stage(zip_input, temp / "work-two") / "uv.lock").is_file())

    def test_missing_ambiguous_and_traversal_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            temp = Path(tmp)
            with self.assertRaises(RuntimeError):
                discover(temp)
            build("base", temp / "repo.zip")
            build("base", temp / "nested" / "repo.zip")
            with self.assertRaises(RuntimeError):
                discover(temp)
            bad = temp / "bad" / "repo.zip"
            bad.parent.mkdir()
            with zipfile.ZipFile(bad, "w") as archive:
                archive.writestr("../escape", b"bad")
            with self.assertRaises(ValueError):
                discover(bad.parent)


if __name__ == "__main__":
    unittest.main()
