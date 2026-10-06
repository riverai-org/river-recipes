# Running on Kubernetes

`render-job` prints Jobs that run the recipe's commands in the image built from the `Dockerfile`. Every
cluster-specific value is a flag: image, namespace, the Secret holding the API key, the PersistentVolumeClaim,
resources and node placement. The Jobs request no GPUs: training runs on River, and the Job only hosts the driver.

## What you need

- The image, pushed where the cluster can pull it.
- A Secret with the River API key (`--api-key-secret`, key `RIVER_API_KEY` unless `--api-key-secret-key` says
  otherwise).
- A PersistentVolumeClaim of at least 80 GB (`--volume-claim`), mounted at `--mount-path` (`/workspace` by default).
  It holds the data release (`data/`), the run directories (`runs/`) and the private database copies
  (`sqlite-snapshots/`).

## Build the image

Building with `ACCEPT_VERIEQL_LICENSE=yes` accepts VeriEQL's licence and installs it in the image under
`/opt/verieql` (see [verieql.md](verieql.md)); `/opt/verieql.env` then holds the runtime manifest's path and SHA-256
that `render-job` needs:

```bash
docker build --build-arg ACCEPT_VERIEQL_LICENSE=yes -t registry.example.com/text-to-sql-recipe:0.1 .
docker push registry.example.com/text-to-sql-recipe:0.1
docker run --rm registry.example.com/text-to-sql-recipe:0.1 cat /opt/verieql.env
```

Without VeriEQL, build without the argument and render the training Job with `--set 'reward.verifier="none"'` in
place of the two `--verieql-*` flags below. `--build-arg RIVER_CLIENT=...` picks the River client the image installs.

## Fill the volume with the data

Run the Job `--setup-data` renders once, and wait for it to complete:

```bash
python -m text_to_sql_recipe render-job --setup-data --name text-to-sql-data \
  --image registry.example.com/text-to-sql-recipe:0.1 --namespace ml --volume-claim text-to-sql \
  | kubectl apply -f -
kubectl -n ml wait --for=condition=complete --timeout=6h job/text-to-sql-data
```

## Start a run

```bash
python -m text_to_sql_recipe render-job --recipe kimi-k2.6 --name kimi-recipe \
  --image registry.example.com/text-to-sql-recipe:0.1 --namespace ml \
  --api-key-secret river-api-key --volume-claim text-to-sql \
  --verieql-manifest /opt/verieql/runtime-manifest.json --verieql-manifest-sha256 <SHA-256 from verieql.env> \
  | kubectl apply -f -
```

## Sizing, restarts and logs

- The training Job's CPU and memory follow the recipe: 4 CPUs and 24 GiB, plus a CPU and 1.5 GiB for each VeriEQL
  check the recipe runs at once (`execution.verifier_workers`); `--cpu` and `--memory` override them.
- It also requests 24 GiB of the node's local disk, where the temporary indexes of streamed grading go.
- The Job passes its own name as the run name, so a pod restarted after a failure (up to three times) resumes the same
  run directory. An error that stops the run is also written to `last-failure.json` in it.
- A finished Job, and its pod's log, is kept for a week.
