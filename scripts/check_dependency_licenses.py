#!/usr/bin/env python3
"""Fail if any installed dependency carries a non-permissive license.

    python scripts/check_dependency_licenses.py --python <venv>/bin/python

Examples are embedded into customer codebases wholesale, so every dependency
must be permissively licensed. Package license metadata is heterogeneous —
Trove classifiers, SPDX expressions (``Apache-2.0 OR BSD-2-Clause``), or the
full license text pasted into the License field — so a package passes if
ANY of its declared forms normalizes into the allowlist:

- exact match against known permissive names (case-insensitive);
- SPDX ``OR`` expressions where at least one branch is allowed;
- SPDX ``AND`` expressions where every part is allowed;
- Trove classifiers, when the license field itself is unusable.

Requires ``pip-licenses`` on the interpreter running this script.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

ALLOWED = {
    "apache software license",
    "apache license 2.0",
    "apache-2.0",
    "apache 2.0",
    "mit",
    "mit license",
    "mit-cmu",
    "cmu license (mit-cmu)",
    "bsd",
    "bsd license",
    "bsd-2-clause",
    "bsd-3-clause",
    "2-clause bsd",
    "3-clause bsd",
    "0bsd",
    "cnri-python",
    "isc",
    "isc license (iscl)",
    "python software foundation license",
    "python-2.0",
    "psf-2.0",
    "mozilla public license 2.0 (mpl 2.0)",
    "mpl-2.0",
    "the unlicense (unlicense)",
    "unlicense",
    "zlib",
    # Zero-obligation public-domain dedications, fine to consume as
    # dependencies (numpy's SPDX expression includes CC0 for vendored parts).
    "cc0-1.0",
    "public domain",
}

IGNORED_PACKAGES = {"pip", "setuptools", "wheel"}


def _normalized_allowed(value: str) -> bool:
    text = value.strip().strip(";").strip()
    if not text:
        return False
    lowered = text.casefold()
    if lowered in ALLOWED:
        return True
    # "Apache 2.0 License", "3-Clause BSD License", ... — a trailing
    # " license" is formatting, not information.
    if lowered.endswith(" license") and lowered.removesuffix(" license") in ALLOWED:
        return True
    # SPDX expressions: any OR-branch may be chosen; AND requires every part.
    if " or " in lowered:
        return any(_normalized_allowed(part) for part in lowered.split(" or "))
    if " and " in lowered:
        return all(_normalized_allowed(part) for part in lowered.split(" and "))
    return False


def _candidates(value: str) -> list[str]:
    # pip-licenses joins multiple classifiers with "; ". Some packages paste
    # their entire license text into the field; its first line is usually the
    # license name ("MIT License\n\nCopyright ..."), so judge that alone.
    if len(value) > 120:
        first_line = value.splitlines()[0] if value.splitlines() else ""
        return [first_line] if first_line.strip() else []
    return [part for part in value.split(";") if part.strip()]


def _pip_licenses(python: str, source: str) -> dict[str, str]:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "piplicenses",
            "--python",
            python,
            "--format=json",
            f"--from={source}",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    rows = json.loads(result.stdout)
    return {row["Name"].casefold(): row.get("License") or "" for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--python", required=True, help="Interpreter of the audited venv.")
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help="Additional package names to skip (repeatable).",
    )
    args = parser.parse_args()

    ignored = IGNORED_PACKAGES | {name.casefold() for name in args.ignore}
    by_field = _pip_licenses(args.python, "mixed")
    by_classifier = _pip_licenses(args.python, "classifier")

    failures: list[str] = []
    for name in sorted(by_field):
        if name in ignored:
            continue
        forms = _candidates(by_field.get(name, "")) + _candidates(by_classifier.get(name, ""))
        if not any(_normalized_allowed(form) for form in forms):
            field = (by_field.get(name) or "")[:80].replace("\n", " ")
            classifier = (by_classifier.get(name) or "")[:80]
            failures.append(f"  {name}: field={field!r} classifier={classifier!r}")

    if failures:
        print("dependencies with licenses outside the permissive allowlist:")
        print("\n".join(failures))
        print(
            "\nIf a license is genuinely permissive, add its exact form to "
            "ALLOWED in scripts/check_dependency_licenses.py (that edit is "
            "the review record)."
        )
        return 1
    checked = len([name for name in by_field if name not in ignored])
    print(f"dependency licenses OK ({checked} packages checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
