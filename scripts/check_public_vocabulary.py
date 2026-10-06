#!/usr/bin/env python3
"""Public-vocabulary guard: tracked content may only reference public infrastructure.

Secret scanners match things that look like credentials. This guard is the
complement: it enforces an ALLOWLIST of infrastructure vocabulary, so an
internal hostname, registry, cluster name, or username-bearing path fails the
build even if no scanner pattern knows about it. Allowlists catch the
identifier nobody thought to denylist.

Rules (each failure names the rule; matched values are REDACTED so a violation
never broadcasts the leaked value through public CI logs):

  hostname     Hostnames on watched TLDs must be on the public allowlist.
  ip-address   Literal IPs must be loopback, private-range, documentation
               (RFC 5737), unspecified, or well-known public resolvers.
  image        Container images must come from allowlisted public registries
               (or be bare Docker official images).
  namespace    Kubernetes namespaces must be generic, never personal.
  k8s-secret   Secret manifests with data are only allowed as *.example.yaml.
  home-path    No absolute paths under a user home directory.

Modes: default scans all tracked files; --staged scans staged content
(what the commit would introduce), for use from pre-commit.

Adding vocabulary: if a rule blocks something genuinely public (a new public
registry, a public dataset host), extend the allowlist in the same PR and say
so in the PR description.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys

# --- Allowed public vocabulary -------------------------------------------------

ALLOWED_HOST_SUFFIXES = {
    "river.ai",
    "localhost",
    "cluster.local",
    "svc.local",
    "example.com",
    "example.org",
    "example.net",
    "github.com",
    "githubusercontent.com",
    "ghcr.io",
    "docker.io",
    "docker.com",
    "registry.k8s.io",
    "quay.io",
    "public.ecr.aws",
    "pypi.org",
    "python.org",
    "kubernetes.io",
    "k8s.io",
    "apache.org",
    "opensource.org",
    "creativecommons.org",  # licence links (text-to-sql names VeriEQL's CC BY-NC-SA licence)
}

# Hostname-shaped tokens are only interesting when they end in a TLD that real
# infrastructure uses; this keeps file names like config.yaml out of scope.
WATCHED_TLDS = {
    "ai", "com", "io", "dev", "net", "org", "cloud", "app",
    "internal", "corp", "lan", "local",
}

ALLOWED_REGISTRIES = {
    "docker.io",
    "ghcr.io",
    "registry.k8s.io",
    "quay.io",
    "public.ecr.aws",
}

ALLOWED_NAMESPACES = {"default", "example", "river-examples", "kube-system"}

ALLOWED_IP_PREFIXES = (
    "127.", "10.", "192.168.", "0.0.0.0", "255.",
    "192.0.2.", "198.51.100.", "203.0.113.",  # RFC 5737 documentation ranges
    "169.254.",
)
ALLOWED_IPS = {"8.8.8.8", "8.8.4.4", "1.1.1.1"}

# Four-part package versions that are also valid dotted quads; each names the
# package it pins.
ALLOWED_VERSION_QUADS = {
    "4.13.0.0",  # z3-solver, pinned by text-to-sql's VeriEQL setup
}

SKIP_FILES = {".gitleaks.toml"}  # pattern definitions, not content
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".ico", ".woff", ".woff2", ".pdf"}

# --- Detection patterns --------------------------------------------------------

HOSTNAME_RE = re.compile(
    r"\b(?:[a-z0-9](?:[a-z0-9-]{0,62})\.)+([a-z]{2,})\b", re.IGNORECASE
)
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
IMAGE_RE = re.compile(r"""\bimage:\s*["']?([^\s"']+)""")
NAMESPACE_RE = re.compile(r"""\bnamespace:\s*["']?([A-Za-z0-9._-]+)""")
# Assembled from fragments so this file's own source never matches the rule.
HOME_PATH_RE = re.compile(
    r"[/\\](?:Us" + r"ers|ho" + r"me)[/\\]+[A-Za-z0-9._-]+", re.IGNORECASE
)
HOME_PATH_ALLOWED_RE = re.compile(
    r"[/\\](?:Us" + r"ers|ho" + r"me)[/\\]+(?:runner|<[^>]+>|\$)", re.IGNORECASE
)


def redact(value: str) -> str:
    """Show enough to locate the problem locally, never the value itself."""
    if len(value) <= 4:
        return "*" * len(value)
    return f"{value[:2]}…({len(value)} chars)"


def is_allowed_host(host: str) -> bool:
    host = host.lower().rstrip(".")
    return any(
        host == allowed or host.endswith("." + allowed)
        for allowed in ALLOWED_HOST_SUFFIXES
    )


def check_text(rel_path: str, text: str) -> list[str]:
    violations: list[str] = []
    lines = text.splitlines()

    is_yaml = rel_path.endswith((".yaml", ".yml"))
    is_example = ".example." in rel_path.rsplit("/", 1)[-1]
    is_k8s_secret = is_yaml and re.search(r"^kind:\s*Secret\s*$", text, re.MULTILINE)

    for lineno, line in enumerate(lines, start=1):
        for match in HOSTNAME_RE.finditer(line):
            host, tld = match.group(0), match.group(1).lower()
            if tld not in WATCHED_TLDS or is_allowed_host(host):
                continue
            if IP_RE.fullmatch(host):  # dotted quads are handled by ip-address
                continue
            violations.append(
                f"{rel_path}:{lineno}: hostname: {redact(host)} not in public allowlist"
            )

        for match in IP_RE.finditer(line):
            ip = match.group(0)
            if ip in ALLOWED_IPS or ip in ALLOWED_VERSION_QUADS or ip.startswith(ALLOWED_IP_PREFIXES):
                continue
            if any(int(octet) > 255 for octet in ip.split(".")):
                continue  # version-like token, not an address
            violations.append(
                f"{rel_path}:{lineno}: ip-address: {redact(ip)} outside allowed ranges"
            )

        for match in IMAGE_RE.finditer(line):
            image = match.group(1)
            registry = image.split("/", 1)[0]
            if "/" not in image or "." not in registry:
                continue  # bare Docker official image (postgres:16)
            if registry.lower() not in ALLOWED_REGISTRIES:
                violations.append(
                    f"{rel_path}:{lineno}: image: registry {redact(registry)} not allowlisted"
                )

        for match in NAMESPACE_RE.finditer(line):
            ns = match.group(1)
            if rel_path.endswith(".py") and ns == "str":
                continue  # a Python annotation (`namespace: str = ...`), not a manifest value
            if ns.lower() not in ALLOWED_NAMESPACES and not ns.startswith("<"):
                violations.append(
                    f"{rel_path}:{lineno}: namespace: {redact(ns)} — use a generic namespace"
                )

        for match in HOME_PATH_RE.finditer(line):
            if HOME_PATH_ALLOWED_RE.match(match.group(0)):
                continue
            violations.append(
                f"{rel_path}:{lineno}: home-path: {redact(match.group(0))} — use relative paths"
            )

    if (
        is_k8s_secret
        and not is_example
        and re.search(r"^\s*(data|stringData):", text, re.MULTILINE)
    ):
        violations.append(
            f"{rel_path}: k8s-secret: Secret manifests with data must be "
            f"*.example.yaml templates with placeholder values"
        )

    return violations


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], capture_output=True, text=True, check=True
    ).stdout


def iter_files(staged: bool):
    if staged:
        names = _git(
            ["diff", "--staged", "--name-only", "--diff-filter=ACMR", "-z"]
        ).split("\0")
        for rel in filter(None, names):
            yield rel, _git(["show", f":{rel}"])
    else:
        for rel in filter(None, _git(["ls-files", "-z"]).split("\0")):
            try:
                with open(rel, encoding="utf-8", errors="ignore") as handle:
                    yield rel, handle.read()
            except OSError:
                continue


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged", action="store_true", help="scan staged content only")
    args = parser.parse_args()

    violations: list[str] = []
    for rel, text in iter_files(args.staged):
        name = rel.rsplit("/", 1)[-1]
        if name in SKIP_FILES or any(rel.endswith(s) for s in SKIP_SUFFIXES):
            continue
        violations.extend(check_text(rel, text))

    if violations:
        print("public-vocabulary guard FAILED (values redacted; inspect the file locally):")
        for violation in violations:
            print(f"  {violation}")
        print(
            "\nIf a flagged value is genuinely public, extend the allowlist in "
            "scripts/check_public_vocabulary.py in this same change and note it "
            "in the PR. Never widen the allowlist to admit internal infrastructure."
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
