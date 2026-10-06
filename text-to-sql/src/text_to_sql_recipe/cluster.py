"""Cluster runner: render the Kubernetes Jobs that fill the data volume and run the recipe in a container.

Training runs on River; a Job only hosts the driver process, so it needs CPU, memory and a volume, and no GPUs. Every
cluster-specific value comes from the caller: image, namespace, the Secret holding RIVER_API_KEY, the volume claim,
and optional node placement. Nothing here assumes a particular cluster.

Both Jobs mount one PersistentVolumeClaim at `mount_path`, which holds everything that must outlive a pod:

    data/               the data release, filled once by the data Job (`data_manifest`), about 34 GB
    runs/<job name>/    a training Job's run directory, which a restarted pod resumes
    sqlite-snapshots/   private, read-only copies of the databases the runs query, about 34 GB, shared by every run

so the claim needs about 70 GB plus what the runs write.
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field

NAME = re.compile(r"^[a-z0-9]([-a-z0-9]*[a-z0-9])?$")
WEEK_SECONDS = 7 * 24 * 3600


def driver_resources(verifier_workers: int) -> dict[str, dict[str, str]]:
    """Kubernetes resources for a training driver whose recipe runs `verifier_workers` VeriEQL checks at once.

    The driver itself (SQL execution, grading, rollout bookkeeping) stayed under 20 GiB on the full release. Each
    VeriEQL check is a process of its own that can grow past 1.4 GB, and a CPU-bound one racing a 30-second deadline,
    so each worker adds 1.5 GiB of memory and a core of CPU limit (half a core of request). The pod's own disk holds
    the temporary indexes of streamed grading, for which a run needs 20 GB free when it starts.
    """
    memory = f"{24 + math.ceil(1.5 * verifier_workers)}Gi"
    return {"requests": {"cpu": str(4 + verifier_workers // 2), "memory": memory, "ephemeral-storage": "24Gi"},
            "limits": {"cpu": str(4 + verifier_workers), "memory": memory}}


@dataclass(frozen=True)
class JobSpec:
    name: str
    image: str
    volume_claim: str                         # PersistentVolumeClaim for data/, runs/ and sqlite-snapshots/
    recipe: str | None = None                 # preset name or recipe path inside the image; unused by the data Job
    namespace: str = "default"
    api_key_secret: str = "river-api-key"     # Secret with the API key
    api_key_secret_key: str = "RIVER_API_KEY"
    base_url: str | None = None               # sets RIVER_BASE_URL when given
    mount_path: str = "/workspace"
    cpu: str | None = None                    # CPU request and limit; default from `driver_resources`
    memory: str | None = None                 # memory request and limit; default from `driver_resources`
    verifier_workers: int = 0                 # the recipe's execution.verifier_workers when it uses VeriEQL, else 0
    overrides: list[str] = field(default_factory=list)
    node_selector: dict[str, str] = field(default_factory=dict)
    uses_verieql: bool = False                # the recipe's reward.verifier is "verieql"
    verieql_manifest: str | None = None       # VeriEQL runtime manifest, a path inside the image or volume
    verieql_manifest_sha256: str | None = None
    backoff_limit: int = 3                    # a restarted pod resumes from the run directory
    ttl_seconds_after_finished: int = WEEK_SECONDS   # keeps a finished Job, and its pod's log, for a week

    def validate(self, *, training: bool = True) -> JobSpec:
        if not NAME.match(self.name) or len(self.name) > 63:
            raise ValueError(f"job name {self.name!r} is not a valid Kubernetes name")
        if not self.image:
            raise ValueError("an image is required")
        if not self.volume_claim:
            raise ValueError("a volume claim is required: the Job keeps its data and run directory on it")
        if training and not self.recipe:
            raise ValueError("a recipe is required")
        if training and self.uses_verieql and not (self.verieql_manifest and self.verieql_manifest_sha256):
            raise ValueError("the recipe uses the VeriEQL verifier: pass its runtime manifest (a path inside the image "
                             "or volume) and the manifest's SHA-256, or set reward.verifier = \"none\"")
        return self

    def resources(self, default: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
        resources = {kind: dict(values) for kind, values in default.items()}
        for key, value in (("cpu", self.cpu), ("memory", self.memory)):
            if value:
                resources["requests"][key] = resources["limits"][key] = value
        return resources


def _job(spec: JobSpec, *, component: str, command: list[str], env: list[dict], resources: dict) -> dict:
    pod = {
        "restartPolicy": "Never",
        "containers": [{
            "name": "recipe",
            "image": spec.image,
            "command": command,
            "env": env,
            "resources": resources,
            "volumeMounts": [{"name": "workspace", "mountPath": spec.mount_path}],
        }],
        "volumes": [{"name": "workspace", "persistentVolumeClaim": {"claimName": spec.volume_claim}}],
    }
    if spec.node_selector:
        pod["nodeSelector"] = dict(spec.node_selector)
    labels = {"app.kubernetes.io/name": "text-to-sql-recipe", "app.kubernetes.io/instance": spec.name,
              "app.kubernetes.io/component": component}
    return {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": spec.name, "namespace": spec.namespace, "labels": labels},
        "spec": {"backoffLimit": spec.backoff_limit, "ttlSecondsAfterFinished": spec.ttl_seconds_after_finished,
                 "template": {"metadata": {"labels": labels}, "spec": pod}},
    }


def manifest(spec: JobSpec) -> dict:
    """The training Job."""
    spec.validate()
    # The run name is the Job name, so a restarted pod resumes the same run directory on the volume.
    command = ["python", "-m", "text_to_sql_recipe", "train", "--recipe", spec.recipe, "--run-name", spec.name]
    for item in spec.overrides:
        command += ["--set", item]
    env = [
        {"name": "RIVER_API_KEY",
         "valueFrom": {"secretKeyRef": {"name": spec.api_key_secret, "key": spec.api_key_secret_key}}},
        {"name": "TEXT_TO_SQL_DATA_DIR", "value": f"{spec.mount_path}/data"},
        {"name": "TEXT_TO_SQL_OUTPUT_DIR", "value": f"{spec.mount_path}/runs"},
        # On the volume rather than in the pod's ephemeral storage, so a restarted pod reuses the copies.
        {"name": "TEXT_TO_SQL_SQLITE_SNAPSHOT_ROOT", "value": f"{spec.mount_path}/sqlite-snapshots"},
    ]
    if spec.base_url:
        env.append({"name": "RIVER_BASE_URL", "value": spec.base_url})
    if spec.verieql_manifest:
        env += [{"name": "TEXT_TO_SQL_VERIEQL_MANIFEST", "value": spec.verieql_manifest},
                {"name": "TEXT_TO_SQL_VERIEQL_MANIFEST_SHA256", "value": spec.verieql_manifest_sha256 or ""}]
    workers = spec.verifier_workers if spec.uses_verieql else 0
    return _job(spec, component="train", command=command, env=env,
                resources=spec.resources(driver_resources(workers)))


def data_manifest(spec: JobSpec) -> dict:
    """The Job that runs `setup data` into the volume's data/ directory; run it once, before the first training Job."""
    spec.validate(training=False)
    command = ["python", "-m", "text_to_sql_recipe", "setup", "data", "--data-dir", f"{spec.mount_path}/data"]
    resources = {"requests": {"cpu": "2", "memory": "8Gi"}, "limits": {"cpu": "2", "memory": "8Gi"}}
    return _job(spec, component="setup-data", command=command, env=[], resources=spec.resources(resources))


def render(spec: JobSpec, *, setup_data: bool = False) -> str:
    """JSON, which kubectl accepts as-is: `kubectl apply -f job.json`."""
    return json.dumps(data_manifest(spec) if setup_data else manifest(spec), indent=2) + "\n"
