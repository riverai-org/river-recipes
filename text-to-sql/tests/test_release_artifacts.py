"""Builds the release artifacts and checks what they publish.

The sdist holds only the allowlist in pyproject.toml; the wheel holds the package with its presets; and no file in
either carries the vocabulary that test_public_vocabulary bans. This scans what would actually ship, not the tree.
"""
import importlib.util
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from test_public_vocabulary import scan

from text_to_sql_recipe import config

ROOT = Path(__file__).resolve().parents[1]
SDIST_TOP_LEVEL = {".dockerignore", ".gitignore", "Dockerfile", "LICENSE", "PKG-INFO", "README.md", "docs", "pyproject.toml", "src",
                   "tests"}


@unittest.skipUnless(importlib.util.find_spec("hatchling"), "building the artifacts needs hatchling: pip install -e '.[test]'")
class ReleaseArtifactsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        out = Path(cls.tmp.name)
        subprocess.run([sys.executable, "-m", "hatchling", "build", "-d", str(out)], cwd=ROOT, check=True,
                       capture_output=True)
        [sdist] = out.glob("*.tar.gz")
        [wheel] = out.glob("*.whl")
        with tarfile.open(sdist) as archive:
            cls.sdist = {member.name.split("/", 1)[1]: archive.extractfile(member).read()
                         for member in archive.getmembers() if member.isfile()}
        with zipfile.ZipFile(wheel) as archive:
            cls.wheel = {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_the_sdist_publishes_only_the_allowlist(self):
        self.assertEqual({name.split("/", 1)[0] for name in self.sdist}, SDIST_TOP_LEVEL)
        self.assertEqual([name for name in self.sdist if "__pycache__" in name or name.endswith(".pyc")], [])
        self.assertIn("src/text_to_sql_recipe/config.py", self.sdist)

    def test_the_wheel_ships_the_package_and_its_presets(self):
        top = {name.split("/", 1)[0] for name in self.wheel}
        self.assertEqual({name for name in top if not name.endswith(".dist-info")}, {"text_to_sql_recipe"})
        presets = {Path(name).stem for name in self.wheel if name.startswith("text_to_sql_recipe/recipes/")}
        self.assertEqual(presets, set(config.presets()))
        self.assertEqual(len(presets), 7)

    def test_both_artifacts_carry_the_license(self):
        self.assertIn("Apache License", self.sdist["LICENSE"].decode())
        licenses = [name for name in self.wheel if ".dist-info/" in name and name.endswith("LICENSE")]
        self.assertEqual(len(licenses), 1, sorted(self.wheel))
        self.assertIn("Apache License", self.wheel[licenses[0]].decode())

    def test_no_published_file_carries_private_vocabulary(self):
        hits = []
        for artifact, files in (("sdist", self.sdist), ("wheel", self.wheel)):
            for name, raw in files.items():
                # Only the wheel's RECORD is skipped: it lists the same files with base64 digests.
                if name.endswith(".dist-info/RECORD"):
                    continue
                hits += scan(f"{artifact}:{name}", raw.decode("utf-8", errors="replace"))
        self.assertEqual(hits, [], "\n" + "\n".join(hits))


if __name__ == "__main__":
    unittest.main()
