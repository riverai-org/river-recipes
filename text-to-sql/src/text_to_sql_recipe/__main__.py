"""Command line: `python -m text_to_sql_recipe <command>`.

  plan           print the resolved recipe for a preset (with any --set overrides)
  train          run training locally (the local runner)
  render-job     print a Kubernetes Job that runs `train` on a cluster (the cluster runner); with --setup-data, the
                 Job that fills the cluster volume with the data release first
  setup data     download the recipe's data release and check every file (see `data_setup`)
  setup verieql  install VeriEQL, which is licensed separately, and write the runtime manifest the verifier reads
                 (see `verieql_setup`)
"""
from __future__ import annotations

import argparse
import json
import sys

from . import cluster, config


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def _node_selector(text: str) -> tuple[str, str]:
    key, separator, value = text.partition("=")
    if not separator or not key:
        raise argparse.ArgumentTypeError("must be KEY=VALUE")
    return key, value


def _recipe_args(p: argparse.ArgumentParser, *, required: bool = True) -> None:
    p.add_argument("--recipe", required=required,
                   help="a preset name (" + ", ".join(config.presets()) + ") or the path to a recipe TOML")
    p.add_argument("--set", dest="overrides", action="append", default=[], metavar="SECTION.KEY=VALUE",
                   help="override one setting; repeatable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="text_to_sql_recipe")
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="print the resolved recipe")
    _recipe_args(plan)

    train = sub.add_parser("train", help="train locally")
    _recipe_args(train)
    train.add_argument("--run-name", default=None, help="run directory name under TEXT_TO_SQL_OUTPUT_DIR")
    train.add_argument("--max-steps", type=_positive_int, default=None,
                       help="stop after this many training batches (a short trial run); a batch whose groups all "
                            "carry no learning signal makes no optimizer step")

    job = sub.add_parser("render-job", help="print a Kubernetes Job manifest")
    _recipe_args(job, required=False)
    job.add_argument("--setup-data", action="store_true",
                     help="print the Job that downloads the data release onto the volume instead (run it once, "
                          "before the first training Job; needs no --recipe)")
    job.add_argument("--name", required=True, help="the Job's name, which is also the run's name")
    job.add_argument("--image", required=True, help="an image built from the package's Dockerfile")
    job.add_argument("--volume-claim", required=True,
                     help="a PersistentVolumeClaim of at least 80 GB for the data, the runs and the database copies")
    job.add_argument("--namespace", default="default")
    job.add_argument("--api-key-secret", default="river-api-key", help="the Secret holding the River API key")
    job.add_argument("--api-key-secret-key", default="RIVER_API_KEY")
    job.add_argument("--base-url", default=None)
    job.add_argument("--mount-path", default="/workspace")
    job.add_argument("--cpu", default=None, help="CPU request and limit (default: sized from the recipe)")
    job.add_argument("--memory", default=None, help="memory request and limit (default: sized from the recipe)")
    job.add_argument("--node-selector", action="append", default=[], type=_node_selector, metavar="KEY=VALUE")
    job.add_argument("--verieql-manifest", default=None,
                     help="VeriEQL runtime manifest inside the image or volume (required when the recipe uses VeriEQL)")
    job.add_argument("--verieql-manifest-sha256", default=None)

    setup = sub.add_parser("setup", help="download the recipe's data, or install a separately licensed tool")
    targets = setup.add_subparsers(dest="target", required=True)
    data = targets.add_parser("data", help="download the datasets the recipe trains and evaluates on (about 34 GB) "
                                           "and check every file")
    data.add_argument("--data-dir", default=None,
                      help="where to put the data (default: TEXT_TO_SQL_DATA_DIR, else ./data)")
    data.add_argument("--source", default=None, help="copy a local copy of the release instead of downloading it")
    data.add_argument("--repo", default=None, help="the Hugging Face dataset repository to download")
    data.add_argument("--revision", default=None, help="the repository revision to download")
    verieql = targets.add_parser("verieql", help="install VeriEQL (CC BY-NC-SA 4.0, non-commercial) and write the "
                                                 "runtime manifest the verifier reads")
    verieql.add_argument("--prefix", default=None,
                         help="installation directory (default: a directory under your user cache directory)")
    verieql.add_argument("--accept-license", action="store_true",
                         help="accept VeriEQL's licence (CC BY-NC-SA 4.0, non-commercial use only) without a prompt")
    verieql.add_argument("--python", default=None,
                         help="a CPython 3.11 interpreter to build VeriEQL's environment from; required without uv, "
                              "which otherwise supplies one")

    args = parser.parse_args(argv)
    if args.command == "setup":   # needs no recipe
        if args.target == "data":
            from . import data_setup

            return data_setup.run(data_dir=args.data_dir, source=args.source,
                                  repo=args.repo or data_setup.DATA_REPO,
                                  revision=args.revision or data_setup.DATA_REVISION)
        from . import verieql_setup

        return verieql_setup.run(prefix=args.prefix, accept=args.accept_license, python=args.python)
    if args.command == "render-job" and args.setup_data:   # needs no recipe
        spec = cluster.JobSpec(name=args.name, image=args.image, volume_claim=args.volume_claim,
                               namespace=args.namespace, mount_path=args.mount_path, cpu=args.cpu, memory=args.memory,
                               node_selector=dict(args.node_selector))
        try:
            sys.stdout.write(cluster.render(spec, setup_data=True))
        except ValueError as error:
            parser.error(str(error))
        return 0
    if args.recipe is None:
        parser.error("--recipe is required")
    try:
        recipe = config.load(args.recipe, args.overrides)
    except (ValueError, FileNotFoundError) as error:   # a bad preset, key or value is a message, not a traceback
        parser.error(str(error))

    if args.command == "plan":
        print(json.dumps(recipe.as_dict(), indent=2))
        return 0
    if args.command == "render-job":
        uses_verieql = recipe.reward.verifier == "verieql"
        spec = cluster.JobSpec(name=args.name, image=args.image, recipe=args.recipe, namespace=args.namespace,
                               api_key_secret=args.api_key_secret, api_key_secret_key=args.api_key_secret_key,
                               base_url=args.base_url, volume_claim=args.volume_claim, mount_path=args.mount_path,
                               cpu=args.cpu, memory=args.memory, overrides=args.overrides,
                               node_selector=dict(args.node_selector), uses_verieql=uses_verieql,
                               verifier_workers=recipe.execution.verifier_workers if uses_verieql else 0,
                               verieql_manifest=args.verieql_manifest,
                               verieql_manifest_sha256=args.verieql_manifest_sha256)
        try:
            manifest = cluster.render(spec)
        except ValueError as error:   # e.g. VeriEQL without its runtime manifest
            parser.error(str(error))
        sys.stdout.write(manifest)
        return 0
    if args.command == "train":
        connection = config.connection()   # fail fast on a missing key before any work
        from . import train

        try:
            train.preflight(recipe, connection)   # the client, the data and the verifier runtime, in seconds
        except (RuntimeError, ValueError) as error:
            raise SystemExit(f"error: {error}") from None
        run_dir = train.main(recipe, args.run_name, connection, max_steps=args.max_steps)
        print(json.dumps({"training_complete": str(run_dir)}))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
