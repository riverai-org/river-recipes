"""Fails on anything in the example's own files that looks private: local or shared-storage paths, in-cluster hosts,
e-mail addresses and credentials.

Scans every file of the example, this one included, but not the data and run directories a user creates in it. The
patterns are written so that none matches its own source line, so this file needs no exemption here or in the
release-artifact scan.
"""
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# Caches and build output, plus what `setup data` and `train` write into the example by default.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv", "venv", "build", "dist", "data", "runs"}
PATTERNS = {
    "absolute home path": r"/(?:Users|home)/[A-Za-z]",
    "shared storage path": r"/shared(?=/)",
    "in-cluster host": r"\.svc\.cluster\.local|[.]internal\b",
    "e-mail address": r"[\w.+-]+@[\w-]+\.[a-z]{2,}",
    "private key": r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
    "access token": r"\b(?:sk|ghp|xox[bp])-[A-Za-z0-9]{20,}",
}


def exported_files():
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_dir() or any(part in SKIP_DIRS or part.endswith(".egg-info") for part in rel.parts):
            continue
        yield path


def scan(name: str, text: str) -> list[str]:
    hits = []
    for label, pattern in PATTERNS.items():
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            line = text.count("\n", 0, match.start()) + 1
            hits.append(f"{name}:{line}: {label}: {match.group(0)!r}")
    return hits


class PublicVocabularyTest(unittest.TestCase):
    def test_no_private_vocabulary(self):
        hits = []
        for path in exported_files():
            hits += scan(str(path.relative_to(ROOT)), path.read_text(errors="replace"))
        self.assertEqual(hits, [], "\n" + "\n".join(hits))

    def test_scan_covers_the_package_including_this_file(self):
        names = {p.relative_to(ROOT).as_posix() for p in exported_files()}
        self.assertIn("src/text_to_sql_recipe/config.py", names)
        self.assertIn("tests/test_public_vocabulary.py", names)
        self.assertTrue(any(n.startswith("src/text_to_sql_recipe/recipes/") for n in names))

    def test_the_patterns_catch_what_they_are_for(self):
        samples = ["/" + "Users/someone/data", "/" + "shared/runs", "api.svc" + ".cluster.local", "a" + "@example.com",
                   "-----BEGIN " + "RSA PRIVATE KEY-----"]
        for sample in samples:
            with self.subTest(sample=sample):
                self.assertTrue(scan("sample", sample), sample)
        for legal in ("diamond-numeric-v1", "DeepSeek V4.1 Flash", "batch/v1", "/workspace/data"):
            with self.subTest(legal=legal):
                self.assertEqual(scan("legal", legal), [])


if __name__ == "__main__":
    unittest.main()
